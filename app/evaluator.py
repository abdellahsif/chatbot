from __future__ import annotations

import json
from pathlib import Path

from app.chatbot import answer_question
from app.models import EvalResult, EvalSummary, QueryRequest


def run_eval(root_dir: Path, schools: dict[str, dict], transcripts: list[dict]) -> EvalSummary:
    eval_path = root_dir / "data" / "mock" / "eval_questions.jsonl"
    results: list[EvalResult] = []

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
            res = answer_question(
                question=req.question,
                profile=req.profile,
                schools=schools,
                transcripts=transcripts,
                top_k=req.top_k,
            )

            checks = {
                "has_evidence": len(res.evidence) > 0,
                "has_short_answer": bool(res.short_answer.strip()),
                "has_next_action": bool(res.next_action.strip()),
                "mentions_budget_or_fit": (
                    "budget" in res.why_it_fits.lower()
                    or "fit" in res.why_it_fits.lower()
                ),
            }
            passed = all(checks.values())
            results.append(
                EvalResult(
                    id=row["id"],
                    passed=passed,
                    checks=checks,
                    answer_preview=res.short_answer,
                )
            )

    passed = sum(1 for r in results if r.passed)
    return EvalSummary(
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        results=results,
    )
