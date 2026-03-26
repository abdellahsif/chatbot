from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from app.chatbot import answer_question
from app.models import EvalResult, EvalSummary, QueryRequest


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _normalize_scores(values: list[float], higher_is_better: bool = True) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [1.0 for _ in values]
    if higher_is_better:
        return [_safe_div(v - lo, hi - lo) for v in values]
    return [_safe_div(hi - v, hi - lo) for v in values]


def _append_eval_log(root_dir: Path, payload: dict) -> str:
    log_dir = root_dir / "data" / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "eval_runs.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return str(log_path.relative_to(root_dir)).replace("\\", "/")


def run_eval(root_dir: Path, schools: dict[str, dict], transcripts: list[dict]) -> EvalSummary:
    eval_path = root_dir / "data" / "mock" / "eval_questions.jsonl"
    results: list[EvalResult] = []
    latencies: list[float] = []
    groundedness_scores: list[float] = []
    relevance_scores: list[float] = []
    compliance_scores: list[float] = []
    hallucination_scores: list[float] = []
    detailed_rows: list[dict] = []

    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            req = QueryRequest.from_dict(
                {
                    "question": row.get("question", ""),
                    "profile": row.get("profile", {}),
                    "top_k": 5,
                }
            )
            t0 = perf_counter()
            res = answer_question(
                question=req.question,
                profile=req.profile,
                schools=schools,
                transcripts=transcripts,
                top_k=req.top_k,
            )
            latency = perf_counter() - t0

            checks = {
                "has_evidence": len(res.evidence) > 0,
                "has_short_answer": bool(res.short_answer.strip()),
                "has_next_action": bool(res.next_action.strip()),
                "grounded_attributes": False,
                "no_external_school": False,
                "relevance_constraints": False,
            }

            evidence_school_names = {str(e.school_name).strip().lower() for e in res.evidence}
            evidence_programs = {str(e.program).strip().lower() for e in res.evidence}
            evidence_cities = {str(e.text).lower() for e in res.evidence}

            rationale = f"{res.short_answer} {res.why_it_fits}".lower()
            checks["grounded_attributes"] = (
                any(name and name in rationale for name in evidence_school_names)
                or any(program and program in rationale for program in evidence_programs)
                or any(str(item.get("city", "")).lower() in rationale for item in [{"city": req.profile.city}] if req.profile.city)
            )

            checks["no_external_school"] = (
                (not evidence_school_names)
                or any(name in res.short_answer.lower() for name in evidence_school_names)
            )

            budget_relevant = True
            if req.profile.budget_band:
                budget_relevant = any("tuition" in e.text.lower() or "mad" in e.text.lower() for e in res.evidence)
            city_relevant = True
            if req.profile.city:
                city_relevant = any(req.profile.city.lower() in e.text.lower() for e in res.evidence) or req.profile.city.lower() in rationale
            checks["relevance_constraints"] = budget_relevant and city_relevant

            passed = all(checks.values())
            results.append(
                EvalResult(
                    id=row["id"],
                    passed=passed,
                    checks=checks,
                    answer_preview=res.short_answer,
                )
            )

            answer_text = " ".join(
                [
                    str(res.short_answer),
                    str(res.why_it_fits),
                    str(res.alternative),
                    str(res.next_action),
                ]
            )
            evidence_text = " ".join(e.text for e in res.evidence)
            answer_tokens = _tokens(answer_text)
            evidence_tokens = _tokens(evidence_text)
            question_tokens = _tokens(req.question)

            groundedness = _safe_div(len(answer_tokens & evidence_tokens), len(answer_tokens))
            relevance = _safe_div(len(answer_tokens & question_tokens), len(question_tokens))
            compliance = 1.0 if all(checks.values()) else _safe_div(sum(1 for v in checks.values() if v), len(checks))
            hallucination = 0.0 if checks["no_external_school"] else 1.0

            latencies.append(latency)
            groundedness_scores.append(groundedness)
            relevance_scores.append(relevance)
            compliance_scores.append(compliance)
            hallucination_scores.append(hallucination)

            detailed_rows.append(
                {
                    "id": row.get("id", ""),
                    "latency_s": round(latency, 3),
                    "groundedness": round(groundedness, 4),
                    "relevance": round(relevance, 4),
                    "compliance": round(compliance, 4),
                    "hallucination": round(hallucination, 4),
                    "hits": len(res.evidence),
                    "short_answer": res.short_answer,
                }
            )

    passed = sum(1 for r in results if r.passed)
    avg_groundedness = _safe_div(sum(groundedness_scores), len(groundedness_scores))
    avg_relevance = _safe_div(sum(relevance_scores), len(relevance_scores))
    avg_compliance = _safe_div(sum(compliance_scores), len(compliance_scores))
    avg_hallucination = _safe_div(sum(hallucination_scores), len(hallucination_scores))
    avg_latency = _safe_div(sum(latencies), len(latencies))

    latency_score = _normalize_scores(latencies, higher_is_better=False)
    avg_latency_score = _safe_div(sum(latency_score), len(latency_score)) if latency_score else 0.0

    # Final quality score in [0, 100], aligned with prior benchmark formula.
    final_score_0_1 = (
        0.35 * avg_groundedness
        + 0.25 * avg_relevance
        + 0.15 * avg_compliance
        + 0.10 * (1.0 - avg_hallucination)
        + 0.10 * avg_latency_score
        + 0.05 * 1.0
    )

    metrics = {
        "formula": "Final=0.35Groundedness+0.25Relevance+0.15Compliance+0.10(1-Hallucination)+0.10LatencyScore+0.05CostScore",
        "avg_latency_s": round(avg_latency, 4),
        "avg_groundedness": round(avg_groundedness, 4),
        "avg_relevance": round(avg_relevance, 4),
        "avg_compliance": round(avg_compliance, 4),
        "avg_hallucination": round(avg_hallucination, 4),
        "avg_latency_score": round(avg_latency_score, 4),
        "final_score": round(final_score_0_1, 4),
        "final_score_100": round(final_score_0_1 * 100.0, 2),
    }

    log_payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "metrics": metrics,
        "details": detailed_rows,
    }
    log_path = _append_eval_log(root_dir, log_payload)

    return EvalSummary(
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        results=results,
        metrics=metrics,
        log_path=log_path,
    )
