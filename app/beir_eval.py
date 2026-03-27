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


def _write_metrics_snapshot(root_dir: Path, file_name: str, payload: dict[str, Any]) -> str:
    log_dir = root_dir / "data" / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / file_name
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return str(out_path.relative_to(root_dir)).replace("\\", "/")


def _f1_at_k(precision: dict[str, float], recall: dict[str, float]) -> dict[str, float]:
    f1: dict[str, float] = {}
    for p_key, p in precision.items():
        p_num = "".join(ch for ch in p_key if ch.isdigit())
        if not p_num:
            continue
        r_key = f"Recall@{p_num}"
        r = recall.get(r_key, 0.0)
        f1[f"F1@{p_num}"] = round(float(_safe_div(2.0 * p * r, p + r)), 4)
    return f1


def _hit_rate_at_k(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> dict[str, float]:
    rates: dict[str, float] = {}
    query_ids = list(qrels.keys())
    for k in k_values:
        hit_count = 0
        for qid in query_ids:
            rel_docs = {doc_id for doc_id, score in qrels.get(qid, {}).items() if score > 0}
            ranked = sorted(results.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
            top_docs = [doc_id for doc_id, _ in ranked[:k]]
            if rel_docs and any(doc_id in rel_docs for doc_id in top_docs):
                hit_count += 1
        rates[f"HitRate@{k}"] = round(float(_safe_div(hit_count, len(query_ids))), 4)
    return rates


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
        # Acronym-first matching (e.g., OFPPT, ENSA, UIR) to prevent weak token overlap mis-resolutions.
        acronym_tokens = {tok.upper() for tok in re.findall(r"\b[A-Za-z]{3,}\b", expected_name) if tok.isupper()}
        if acronym_tokens:
            acronym_hits: set[str] = set()
            for school_id, school in schools.items():
                school_name = str(school.get("name", ""))
                school_upper_tokens = {tok.upper() for tok in re.findall(r"\b[A-Za-z]{3,}\b", school_name)}
                if acronym_tokens & school_upper_tokens:
                    acronym_hits.add(str(school_id))
            if acronym_hits:
                resolved |= acronym_hits
                continue

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

    eval_path = root_dir / "data" / "eval_questions.jsonl"
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
                # Evaluate at school granularity using one representative chunk
                # so chunking strategy does not artificially deflate Recall@k.
                school_chunks = chunks_by_school.get(school_id, [])
                if not school_chunks:
                    continue
                cid = str(school_chunks[0].get("chunk_id", "")).strip()
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
    f1 = _f1_at_k(precision, recall)
    hit_rate = _hit_rate_at_k(qrels, results, k_values)

    avg_latency = _safe_div(sum(latencies), len(latencies))
    metrics = {
        "ndcg": {str(k): round(float(v), 4) for k, v in ndcg.items()},
        "recall": {str(k): round(float(v), 4) for k, v in recall.items()},
        "precision": {str(k): round(float(v), 4) for k, v in precision.items()},
        "f1": f1,
        "mrr": {str(k): round(float(v), 4) for k, v in mrr.items()},
        "hit_rate": hit_rate,
        "map": {str(k): round(float(v), 4) for k, v in _map.items()},
        "avg_latency_s": round(avg_latency, 4),
    }

    retrieval_metrics = {
        "Recall@k": metrics["recall"],
        "Precision@k": metrics["precision"],
        "F1 Score": metrics["f1"],
        "MRR (Mean Reciprocal Rank)": metrics["mrr"],
        "Hit Rate@k": metrics["hit_rate"],
        "nDCG@k": metrics["ndcg"],
    }

    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(queries),
        "total_corpus_docs": len(corpus),
        "metrics": metrics,
        "details": per_query,
    }
    log_path = _append_beir_log(root_dir, payload)
    retrieval_payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(queries),
        "metrics": retrieval_metrics,
    }
    retrieval_metrics_path = _write_metrics_snapshot(root_dir, "retrieval_metrics.json", retrieval_payload)

    return {
        "status": "ok",
        "total_queries": len(queries),
        "total_corpus_docs": len(corpus),
        "metrics": metrics,
        "retrieval_metrics": retrieval_metrics,
        "retrieval_metrics_file": retrieval_metrics_path,
        "log_path": log_path,
        "details": per_query,
    }
