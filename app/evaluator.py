from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from app.chatbot import answer_question
from app.models import EvalResult, EvalSummary, QueryRequest
from app.retriever import resolve_effective_profile


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _token_count(text: str) -> int:
    return len(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _normalize_label(text: str) -> str:
    lowered = (text or "").strip().lower()
    folded = unicodedata.normalize("NFKD", lowered).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", folded))


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


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _phrase_tokens(phrase: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "or",
        "the",
        "if",
        "of",
        "to",
        "for",
        "at",
        "least",
        "needed",
        "option",
        "best",
        "match",
        "confidence",
        "question",
        "focus",
        "verdict",
        "alternative",
        "next",
        "action",
        "recommendation",
        "safety",
        "oriented",
        "choose",
        "consider",
        "shortlist",
    }
    return {t for t in _tokens(phrase) if t not in stopwords}


def _extract_claims(text: str) -> list[str]:
    # Split generated text into simple factual claims for grounding checks.
    parts = re.split(r"[\.!?;\n]+", text or "")
    claims: list[str] = []
    for p in parts:
        c = " ".join(str(p).strip().split())
        if not c:
            continue
        if len(_tokens(c)) < 6:
            continue
        claims.append(c)
    return claims


def _claim_supported(claim: str, evidence_tokens: set[str]) -> bool:
    claim_tokens = _phrase_tokens(claim)
    if not claim_tokens:
        return True
    overlap = len(claim_tokens & evidence_tokens)
    overlap_ratio = _safe_div(overlap, len(claim_tokens))
    # Treat a claim as grounded if enough of its content words appear in evidence.
    return overlap >= 1 and overlap_ratio >= 0.2


def _hallucination_stats(answer_text: str, evidence_text: str) -> dict[str, float]:
    claims = _extract_claims(answer_text)
    if not claims:
        return {
            "hallucination": 0.0,
            "supported_claim_rate": 1.0,
            "supported_claims": 0.0,
            "unsupported_claims": 0.0,
            "total_claims": 0.0,
        }

    e_tokens = _tokens(evidence_text)
    supported = sum(1 for c in claims if _claim_supported(c, e_tokens))
    unsupported = len(claims) - supported
    hallucination = _safe_div(unsupported, len(claims))
    return {
        "hallucination": hallucination,
        "supported_claim_rate": _safe_div(supported, len(claims)),
        "supported_claims": float(supported),
        "unsupported_claims": float(unsupported),
        "total_claims": float(len(claims)),
    }


def _must_include_stats(must_include: list[str], answer_text: str, generated_item_count: int) -> dict[str, float]:
    if not must_include:
        return {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "matched": 0.0,
            "generated_items": float(generated_item_count),
        }

    hay_tokens = _tokens(answer_text)
    hits = 0
    for phrase in must_include:
        phrase_toks = _phrase_tokens(str(phrase))
        if not phrase_toks:
            continue
        overlap = _safe_div(len(phrase_toks & hay_tokens), len(phrase_toks))
        if overlap >= 0.5:
            hits += 1

    recall = _safe_div(hits, len(must_include))
    precision = _safe_div(hits, max(1, generated_item_count))
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "matched": float(hits),
        "generated_items": float(generated_item_count),
    }


def _name_overlap(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    # Use containment-style overlap to handle long benchmark labels vs short school names.
    return max(_safe_div(inter, len(ta)), _safe_div(inter, len(tb)))


def _count_name_hits(expected_names: set[str], retrieved_names: set[str], min_overlap: float = 0.45) -> int:
    if not expected_names or not retrieved_names:
        return 0
    hits = 0
    for exp in expected_names:
        best = max((_name_overlap(exp, got) for got in retrieved_names), default=0.0)
        if best >= min_overlap:
            hits += 1
    return hits


def _is_multilingual_query(question: str) -> bool:
    q = (question or "").lower()
    return any(
        marker in q
        for marker in [
            "please answer in english only",
            "reponds en francais",
            "jawbni b darija",
        ]
    )


def _is_noisy_query(question: str) -> bool:
    q_tokens = _tokens(question)
    noisy_markers = {
        "affrodable",
        "whch",
        "optns",
        "internatonal",
        "managment",
        "enginering",
        "optns",
    }
    if q_tokens & noisy_markers:
        return True
    # Repeated prompt prefixes are treated as instruction noise.
    q = (question or "").lower()
    return ("please answer in english only." in q and q.count("please answer in english only") > 1) or (
        "keep answer grounded in evidence only." in q and q.count("keep answer grounded in evidence only") > 1
    )


def _is_conflicting_constraints_query(question: str, req: QueryRequest) -> bool:
    q_tokens = _tokens(question)
    q = (question or "").lower()

    if "edge case: very low budget and uncertain grades" in q:
        return True

    low_budget = req.profile.budget_band in {"zero_public", "tight_25k"} or bool(
        q_tokens & {"affordable", "cheap", "low", "budget", "public"}
    )
    high_aspiration = req.profile.motivation in {"prestige", "expat"} or bool(
        q_tokens & {"prestige", "elite", "international", "global", "abroad", "roi"}
    )
    return low_budget and high_aspiration


def _is_relevant_retrieval_item(
    school_id: str,
    school_name: str,
    expected_school_ids: set[str],
    expected_school_names: set[str],
    min_overlap: float = 0.45,
) -> bool:
    sid = _normalize_label(school_id)
    sname = _normalize_label(school_name)

    if expected_school_ids and sid and sid in expected_school_ids:
        return True

    if expected_school_names and sname:
        best = max((_name_overlap(exp, sname) for exp in expected_school_names), default=0.0)
        if best >= min_overlap:
            return True

    return False


def _dcg_at_k(relevance: list[int], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(relevance[:k]):
        if rel <= 0:
            continue
        score += float(rel) / math.log2(i + 2)
    return score


def _ndcg_at_k(relevance: list[int], ideal_relevant_count: int, k: int) -> float:
    dcg = _dcg_at_k(relevance, k)
    ideal = [1] * min(max(0, ideal_relevant_count), k)
    idcg = _dcg_at_k(ideal, k)
    return _safe_div(dcg, idcg)


def _append_eval_log(root_dir: Path, payload: dict) -> str:
    log_dir = root_dir / "data" / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "eval_runs.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return str(log_path.relative_to(root_dir)).replace("\\", "/")


def _write_metrics_snapshot(root_dir: Path, file_name: str, payload: dict) -> str:
    log_dir = root_dir / "data" / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / file_name
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return str(out_path.relative_to(root_dir)).replace("\\", "/")


def _to_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def run_eval(root_dir: Path, schools: dict[str, dict], transcripts: list[dict]) -> EvalSummary:
    eval_file = os.getenv("EVAL_QUESTIONS_FILE", "data/eval_questions.jsonl")
    eval_path = (root_dir / eval_file).resolve() if not Path(eval_file).is_absolute() else Path(eval_file)
    if not eval_path.exists():
        fallback = root_dir / "data" / "eval_questions_fixed_200.jsonl"
        if fallback.exists():
            eval_path = fallback
        else:
            raise FileNotFoundError(f"Missing evaluation file: {eval_path}")
    results: list[EvalResult] = []
    latencies: list[float] = []
    groundedness_scores: list[float] = []
    relevance_scores: list[float] = []
    compliance_scores: list[float] = []
    hallucination_scores: list[float] = []
    must_include_recall_scores: list[float] = []
    must_include_precision_scores: list[float] = []
    must_include_f1_scores: list[float] = []
    retrieval_recall_scores: list[float] = []
    retrieval_precision_scores: list[float] = []
    retrieval_f1_scores: list[float] = []
    recall_at_5_scores: list[float] = []
    recall_at_10_scores: list[float] = []
    precision_at_5_scores: list[float] = []
    mrr_scores: list[float] = []
    ndcg_at_10_scores: list[float] = []
    hit_rate_at_10_scores: list[float] = []
    faithfulness_scores: list[float] = []
    context_precision_scores: list[float] = []
    correctness_scores: list[float] = []
    completeness_scores: list[float] = []
    input_tokens_per_query: list[int] = []
    output_tokens_per_query: list[int] = []
    total_input_tokens = 0
    total_output_tokens = 0
    input_cost_per_1k = _to_float(os.getenv("COST_PER_1K_INPUT_TOKENS"), 0.0)
    output_cost_per_1k = _to_float(os.getenv("COST_PER_1K_OUTPUT_TOKENS"), 0.0)
    max_queries = 0
    try:
        max_queries = int(os.getenv("EVAL_MAX_QUERIES", "0") or 0)
    except ValueError:
        max_queries = 0
    detailed_rows: list[dict] = []
    robustness_slices: dict[str, dict[str, int]] = {
        "multilingual": {"total": 0, "passed": 0},
        "noisy_query": {"total": 0, "passed": 0},
        "conflicting_constraints": {"total": 0, "passed": 0},
    }

    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_queries > 0 and len(results) >= max_queries:
                break
            row = json.loads(line)
            req = QueryRequest.from_dict(
                {
                    "question": row.get("question", ""),
                    "profile": row.get("profile", {}),
                    "top_k": 10,
                }
            )
            effective_profile = resolve_effective_profile(
                question=req.question,
                profile=req.profile,
                schools=schools,
            )
            t0 = perf_counter()
            res = answer_question(
                question=req.question,
                profile=effective_profile,
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
                or any(str(item.get("city", "")).lower() in rationale for item in [{"city": effective_profile.city}] if effective_profile.city)
            )

            checks["no_external_school"] = (
                (not evidence_school_names)
                or any(name in res.short_answer.lower() for name in evidence_school_names)
            )

            budget_relevant = True
            if effective_profile.budget_band:
                budget_relevant = any("tuition" in e.text.lower() or "mad" in e.text.lower() for e in res.evidence)
            city_relevant = True
            if effective_profile.city:
                city_relevant = any(effective_profile.city.lower() in e.text.lower() for e in res.evidence) or effective_profile.city.lower() in rationale
            checks["relevance_constraints"] = budget_relevant and city_relevant

            passed = all(checks.values())

            if _is_multilingual_query(req.question):
                robustness_slices["multilingual"]["total"] += 1
                if passed:
                    robustness_slices["multilingual"]["passed"] += 1

            if _is_noisy_query(req.question):
                robustness_slices["noisy_query"]["total"] += 1
                if passed:
                    robustness_slices["noisy_query"]["passed"] += 1

            if _is_conflicting_constraints_query(req.question, req):
                robustness_slices["conflicting_constraints"]["total"] += 1
                if passed:
                    robustness_slices["conflicting_constraints"]["passed"] += 1

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
            factual_answer_text = " ".join(
                [
                    str(res.short_answer),
                    str(res.why_it_fits),
                ]
            )
            evidence_text = " ".join(e.text for e in res.evidence)
            combined_text = f"{answer_text} {evidence_text}"
            answer_tokens = _tokens(answer_text)
            evidence_tokens = _tokens(evidence_text)
            question_tokens = _tokens(req.question)

            groundedness = _safe_div(len(answer_tokens & evidence_tokens), len(answer_tokens))
            relevance = _safe_div(len(answer_tokens & question_tokens), len(question_tokens))
            hallu_stats = _hallucination_stats(factual_answer_text, evidence_text)
            hallucination = hallu_stats["hallucination"]
            # Keep backward-compatible check key name but drive it with claim grounding now.
            checks["no_external_school"] = hallucination <= 0.10
            compliance = 1.0 if all(checks.values()) else _safe_div(sum(1 for v in checks.values() if v), len(checks))
            faithfulness = _safe_div(
                len(_tokens(f"{res.short_answer} {res.why_it_fits}") & evidence_tokens),
                len(_tokens(f"{res.short_answer} {res.why_it_fits}")),
            )

            must_include = _as_list(row.get("must_include"))
            generated_item_count = sum(
                1
                for v in [res.short_answer, res.why_it_fits, res.alternative, res.next_action]
                if str(v).strip()
            )
            must_stats = _must_include_stats(must_include, combined_text, generated_item_count)
            must_include_recall = must_stats["recall"]
            must_include_precision = must_stats["precision"]
            must_include_f1 = must_stats["f1"]

            expected_school_ids = {
                _normalize_label(s)
                for s in _as_list(row.get("expected_school_ids")) + _as_list(row.get("expected_school_id"))
                if s
            }
            expected_school_names = {
                _normalize_label(s)
                for s in _as_list(row.get("expected_school_names")) + _as_list(row.get("expected_school_name"))
                if s
            }

            retrieved_ids = {_normalize_label(str(e.school_id)) for e in res.evidence if str(e.school_id).strip()}
            retrieved_names = {_normalize_label(str(e.school_name)) for e in res.evidence if str(e.school_name).strip()}
            retrieved_entities = {
                _normalize_label(str(e.school_id)) if str(e.school_id).strip() else _normalize_label(str(e.school_name))
                for e in res.evidence
                if str(e.school_id).strip() or str(e.school_name).strip()
            }

            relevance_by_rank: list[int] = [
                1
                if _is_relevant_retrieval_item(
                    school_id=str(e.school_id),
                    school_name=str(e.school_name),
                    expected_school_ids=expected_school_ids,
                    expected_school_names=expected_school_names,
                )
                else 0
                for e in res.evidence
            ]

            if relevance_by_rank:
                context_precision_scores.append(_safe_div(sum(relevance_by_rank), len(relevance_by_rank)))
            else:
                context_precision_scores.append(0.0)

            expected_total = max(len(expected_school_names), len(expected_school_ids))
            if expected_total > 0:
                id_hits = len(expected_school_ids & retrieved_ids)
                name_hits = _count_name_hits(expected_school_names, retrieved_names)
                if expected_school_names and expected_school_ids:
                    retrieval_hits = max(id_hits, name_hits)
                elif expected_school_names:
                    retrieval_hits = name_hits
                else:
                    retrieval_hits = id_hits
                retrieval_recall = _safe_div(retrieval_hits, expected_total)
                retrieval_precision = _safe_div(retrieval_hits, max(1, len(retrieved_entities)))
                retrieval_f1 = _safe_div(2.0 * retrieval_precision * retrieval_recall, retrieval_precision + retrieval_recall)

                rel_5 = relevance_by_rank[:5]
                rel_10 = relevance_by_rank[:10]
                hits_5 = min(sum(rel_5), expected_total)
                hits_10 = min(sum(rel_10), expected_total)
                recall_at_5_scores.append(_safe_div(hits_5, expected_total))
                recall_at_10_scores.append(_safe_div(hits_10, expected_total))
                precision_at_5_scores.append(_safe_div(hits_5, 5.0))
                hit_rate_at_10_scores.append(1.0 if hits_10 > 0 else 0.0)

                first_rel_rank = next((idx + 1 for idx, rel in enumerate(relevance_by_rank[:10]) if rel > 0), 0)
                mrr_scores.append(_safe_div(1.0, float(first_rel_rank)))
                ndcg_at_10_scores.append(_ndcg_at_k(relevance_by_rank, expected_total, 10))
            else:
                retrieval_recall = 0.0
                retrieval_precision = 0.0
                retrieval_f1 = 0.0

            prompt_text = " ".join(
                [
                    str(req.question),
                    str(req.profile.bac_stream),
                    str(req.profile.expected_grade_band),
                    str(req.profile.motivation),
                    str(req.profile.budget_band),
                    str(effective_profile.city),
                    str(effective_profile.country),
                ]
            )
            query_input_tokens = _token_count(prompt_text)
            query_output_tokens = _token_count(answer_text)
            total_input_tokens += query_input_tokens
            total_output_tokens += query_output_tokens
            input_tokens_per_query.append(query_input_tokens)
            output_tokens_per_query.append(query_output_tokens)

            latencies.append(latency)
            groundedness_scores.append(groundedness)
            relevance_scores.append(relevance)
            compliance_scores.append(compliance)
            hallucination_scores.append(hallucination)
            must_include_recall_scores.append(must_include_recall)
            must_include_precision_scores.append(must_include_precision)
            must_include_f1_scores.append(must_include_f1)
            retrieval_recall_scores.append(retrieval_recall)
            retrieval_precision_scores.append(retrieval_precision)
            retrieval_f1_scores.append(retrieval_f1)
            faithfulness_scores.append(faithfulness)
            correctness_scores.append(retrieval_recall)
            completeness_scores.append(must_include_recall)

            detailed_rows.append(
                {
                    "id": row.get("id", ""),
                    "latency_s": round(latency, 3),
                    "groundedness": round(groundedness, 4),
                    "relevance": round(relevance, 4),
                    "compliance": round(compliance, 4),
                    "hallucination": round(hallucination, 4),
                    "supported_claim_rate": round(hallu_stats["supported_claim_rate"], 4),
                    "unsupported_claims": int(hallu_stats["unsupported_claims"]),
                    "total_claims": int(hallu_stats["total_claims"]),
                    "faithfulness": round(faithfulness, 4),
                    "must_include_recall": round(must_include_recall, 4),
                    "must_include_precision": round(must_include_precision, 4),
                    "must_include_f1": round(must_include_f1, 4),
                    "retrieval_recall_at_k": round(retrieval_recall, 4),
                    "retrieval_precision_at_k": round(retrieval_precision, 4),
                    "retrieval_f1_at_k": round(retrieval_f1, 4),
                    "answer_correctness": round(retrieval_recall, 4),
                    "answer_completeness": round(must_include_recall, 4),
                    "token_usage": {
                        "input_tokens": query_input_tokens,
                        "output_tokens": query_output_tokens,
                        "total_tokens": query_input_tokens + query_output_tokens,
                    },
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
    avg_must_include_recall = _safe_div(sum(must_include_recall_scores), len(must_include_recall_scores))
    avg_must_include_precision = _safe_div(sum(must_include_precision_scores), len(must_include_precision_scores))
    avg_must_include_f1 = _safe_div(sum(must_include_f1_scores), len(must_include_f1_scores))
    avg_retrieval_recall = _safe_div(sum(retrieval_recall_scores), len(retrieval_recall_scores))
    avg_retrieval_precision = _safe_div(sum(retrieval_precision_scores), len(retrieval_precision_scores))
    avg_retrieval_f1 = _safe_div(sum(retrieval_f1_scores), len(retrieval_f1_scores))
    avg_recall_at_5 = _safe_div(sum(recall_at_5_scores), len(recall_at_5_scores))
    avg_recall_at_10 = _safe_div(sum(recall_at_10_scores), len(recall_at_10_scores))
    avg_precision_at_5 = _safe_div(sum(precision_at_5_scores), len(precision_at_5_scores))
    avg_mrr = _safe_div(sum(mrr_scores), len(mrr_scores))
    avg_ndcg_at_10 = _safe_div(sum(ndcg_at_10_scores), len(ndcg_at_10_scores))
    avg_hit_rate_at_10 = _safe_div(sum(hit_rate_at_10_scores), len(hit_rate_at_10_scores))
    avg_faithfulness = _safe_div(sum(faithfulness_scores), len(faithfulness_scores))
    avg_context_precision = _safe_div(sum(context_precision_scores), len(context_precision_scores))
    avg_correctness = _safe_div(sum(correctness_scores), len(correctness_scores))
    avg_completeness = _safe_div(sum(completeness_scores), len(completeness_scores))

    avg_recall = avg_retrieval_recall
    avg_precision = avg_retrieval_precision
    avg_f1 = avg_retrieval_f1

    latency_score = _normalize_scores(latencies, higher_is_better=False)
    avg_latency_score = _safe_div(sum(latency_score), len(latency_score)) if latency_score else 0.0
    total_tokens = total_input_tokens + total_output_tokens
    total_cost = _safe_div(total_input_tokens, 1000.0) * input_cost_per_1k + _safe_div(total_output_tokens, 1000.0) * output_cost_per_1k
    avg_cost_per_query = _safe_div(total_cost, len(results))
    avg_input_tokens = _safe_div(sum(input_tokens_per_query), len(input_tokens_per_query))
    avg_output_tokens = _safe_div(sum(output_tokens_per_query), len(output_tokens_per_query))
    success_rate = _safe_div(passed, len(results))

    multilingual_pass_rate = _safe_div(
        robustness_slices["multilingual"]["passed"],
        robustness_slices["multilingual"]["total"],
    )
    noisy_query_pass_rate = _safe_div(
        robustness_slices["noisy_query"]["passed"],
        robustness_slices["noisy_query"]["total"],
    )
    conflicting_constraints_pass_rate = _safe_div(
        robustness_slices["conflicting_constraints"]["passed"],
        robustness_slices["conflicting_constraints"]["total"],
    )

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
        "avg_must_include_recall": round(avg_must_include_recall, 4),
        "avg_must_include_precision": round(avg_must_include_precision, 4),
        "avg_must_include_f1": round(avg_must_include_f1, 4),
        "avg_retrieval_recall_at_k": round(avg_retrieval_recall, 4),
        "avg_retrieval_precision_at_k": round(avg_retrieval_precision, 4),
        "avg_retrieval_f1_at_k": round(avg_retrieval_f1, 4),
        "avg_recall": round(avg_recall, 4),
        "avg_precision": round(avg_precision, 4),
        "avg_f1": round(avg_f1, 4),
        "avg_latency_score": round(avg_latency_score, 4),
        "final_score": round(final_score_0_1, 4),
        "final_score_100": round(final_score_0_1 * 100.0, 2),
        "retrieval_metrics": {
            "recall_at_5": round(avg_recall_at_5, 4),
            "recall_at_10": round(avg_recall_at_10, 4),
            "precision_at_5": round(avg_precision_at_5, 4),
            "mrr": round(avg_mrr, 4),
            "ndcg_at_10": round(avg_ndcg_at_10, 4),
            "hit_rate_at_10": round(avg_hit_rate_at_10, 4),
        },
        "answer_quality_metrics": {
            "faithfulness": round(avg_faithfulness, 4),
            "answer_relevance": round(avg_relevance, 4),
            "context_precision": round(avg_context_precision, 4),
        },
        "hallucination": {
            "hallucination_rate": round(avg_hallucination, 4),
        },
        "end_to_end": {
            "success_rate": round(success_rate, 4),
        },
        "performance": {
            "latency_s": round(avg_latency, 4),
        },
        "robustness": {
            "multilingual_pass_rate": round(multilingual_pass_rate, 4),
            "noisy_query_pass_rate": round(noisy_query_pass_rate, 4),
            "conflicting_constraints_pass_rate": round(conflicting_constraints_pass_rate, 4),
        },
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

    generation_metrics = {
        "groundedness": round(avg_groundedness, 4),
        "relevance": round(avg_relevance, 4),
        "faithfulness": round(avg_faithfulness, 4),
        "context_precision": round(avg_context_precision, 4),
        # Claim-level hallucination rate across answers.
        "hallucination_rate": round(avg_hallucination, 4),
        "answer_correctness": round(avg_correctness, 4),
        "answer_completeness": round(avg_completeness, 4),
        "success_rate": round(success_rate, 4),
        "latency_s": round(avg_latency, 4),
        "retrieval_metrics": {
            "recall_at_5": round(avg_recall_at_5, 4),
            "recall_at_10": round(avg_recall_at_10, 4),
            "precision_at_5": round(avg_precision_at_5, 4),
            "mrr": round(avg_mrr, 4),
            "ndcg_at_10": round(avg_ndcg_at_10, 4),
            "hit_rate_at_10": round(avg_hit_rate_at_10, 4),
        },
        "robustness": {
            "multilingual_pass_rate": round(multilingual_pass_rate, 4),
            "noisy_query_pass_rate": round(noisy_query_pass_rate, 4),
            "conflicting_constraints_pass_rate": round(conflicting_constraints_pass_rate, 4),
        },
        "cost_per_query": round(avg_cost_per_query, 8),
        "token_usage": {
            "total_input_tokens": int(total_input_tokens),
            "total_output_tokens": int(total_output_tokens),
            "total_tokens": int(total_tokens),
            "avg_input_tokens_per_query": round(avg_input_tokens, 2),
            "avg_output_tokens_per_query": round(avg_output_tokens, 2),
            "avg_total_tokens_per_query": round(avg_input_tokens + avg_output_tokens, 2),
        },
    }
    generation_payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(results),
        "metrics": generation_metrics,
    }
    generation_metrics_path = _write_metrics_snapshot(root_dir, "generation_metrics.json", generation_payload)
    metrics["generation_metrics_file"] = generation_metrics_path

    return EvalSummary(
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        results=results,
        metrics=metrics,
        log_path=log_path,
    )
