from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from app.models import UserProfile
from app.retriever import retrieve


def _normalize(text: str) -> str:
    lowered = (text or "").strip().lower()
    folded = unicodedata.normalize("NFKD", lowered).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", folded))


def _tokens(text: str) -> set[str]:
    return set(_normalize(text).split())


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _name_overlap(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return max(_safe_div(inter, len(ta)), _safe_div(inter, len(tb)))


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _append_beir_log(root_dir: Path, payload: dict[str, Any]) -> str:
    log_dir = root_dir / "data" / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "beir_runs.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return str(log_path.relative_to(root_dir)).replace("\\", "/")


def _build_expected_school_ids(row: dict[str, Any], schools: dict[str, dict]) -> set[str]:
    expected_ids = {
        str(v).strip()
        for v in _as_list(row.get("expected_school_ids")) + _as_list(row.get("expected_school_id"))
        if str(v).strip()
    }
    if expected_ids:
        return expected_ids

    expected_names = _as_list(row.get("expected_school_names")) + _as_list(row.get("expected_school_name"))
    if not expected_names:
        return set()

    resolved: set[str] = set()
    for expected_name in expected_names:
        best_score = 0.0
        best_id = ""
        for school_id, school in schools.items():
            score = _name_overlap(expected_name, str(school.get("name", "")))
            if score > best_score:
                best_score = score
                best_id = str(school_id)
        if best_id and best_score >= 0.45:
            resolved.add(best_id)
    return resolved


def run_beir_eval(
    root_dir: Path,
    schools: dict[str, dict],
    transcripts: list[dict],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    try:
        from beir.retrieval.evaluation import EvaluateRetrieval
    except Exception as exc:
        return {
            "status": "error",
            "error": "beir_not_available",
            "message": f"Install BEIR first: {exc}",
        }

    eval_path = root_dir / "data" / "mock" / "eval_questions.jsonl"
    if not eval_path.exists():
        return {
            "status": "error",
            "error": "missing_eval_questions",
            "message": f"Missing file: {eval_path}",
        }

    chunk_by_id: dict[str, dict[str, Any]] = {}
    chunks_by_school: dict[str, list[dict[str, Any]]] = {}
    corpus: dict[str, dict[str, str]] = {}

    for chunk in transcripts:
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        school_id = str(chunk.get("school_id", "")).strip()
        if not chunk_id or not school_id:
            continue
        school = schools.get(school_id, {})
        text = " ".join(
            [
                str(school.get("name", "")),
                str(chunk.get("program", "")),
                str(chunk.get("text", "")),
            ]
        ).strip()
        chunk_by_id[chunk_id] = chunk
        chunks_by_school.setdefault(school_id, []).append(chunk)
        corpus[chunk_id] = {
            "title": str(school.get("name", "")),
            "text": text,
        }

    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    results: dict[str, dict[str, float]] = {}
    per_query: list[dict[str, Any]] = []
    latencies: list[float] = []

    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = str(row.get("id", "")).strip()
            question = str(row.get("question", "")).strip()
            if not qid or not question:
                continue

            queries[qid] = question
            expected_school_ids = _build_expected_school_ids(row, schools)
            qrels[qid] = {}
            for school_id in expected_school_ids:
                for chunk in chunks_by_school.get(school_id, []):
                    cid = str(chunk.get("chunk_id", "")).strip()
                    if cid:
                        qrels[qid][cid] = 1

            profile = UserProfile.from_dict(row.get("profile", {}))
            t0 = perf_counter()
            hits = retrieve(
                question=question,
                profile=profile,
                schools=schools,
                transcripts=transcripts,
                top_k=max(1, min(20, top_k)),
            )
            latency = perf_counter() - t0
            latencies.append(latency)

            ranked: dict[str, float] = {}
            for hit in hits:
                cid = str(hit.get("chunk", {}).get("chunk_id", "")).strip()
                if not cid:
                    continue
                ranked[cid] = float(hit.get("score", 0.0))

            results[qid] = ranked

            predicted_ids = {str(hit.get("school", {}).get("school_id", "")).strip() for hit in hits}
            predicted_ids.discard("")
            overlap = sorted(expected_school_ids & predicted_ids)
            per_query.append(
                {
                    "id": qid,
                    "expected_school_ids": sorted(expected_school_ids),
                    "predicted_school_ids": sorted(predicted_ids),
                    "overlap_school_ids": overlap,
                    "hits": len(hits),
                    "latency_s": round(latency, 4),
                }
            )

    evaluator = EvaluateRetrieval()
    k_values = [1, 3, 5, 10]
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)
    mrr = evaluator.evaluate_custom(qrels, results, k_values, metric="mrr")

    avg_latency = _safe_div(sum(latencies), len(latencies))
    metrics = {
        "ndcg": {str(k): round(float(v), 4) for k, v in ndcg.items()},
        "map": {str(k): round(float(v), 4) for k, v in _map.items()},
        "recall": {str(k): round(float(v), 4) for k, v in recall.items()},
        "precision": {str(k): round(float(v), 4) for k, v in precision.items()},
        "mrr": {str(k): round(float(v), 4) for k, v in mrr.items()},
        "avg_latency_s": round(avg_latency, 4),
    }

    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(queries),
        "total_corpus_docs": len(corpus),
        "metrics": metrics,
        "details": per_query,
    }
    log_path = _append_beir_log(root_dir, payload)

    return {
        "status": "ok",
        "total_queries": len(queries),
        "total_corpus_docs": len(corpus),
        "metrics": metrics,
        "log_path": log_path,
        "details": per_query,
    }
