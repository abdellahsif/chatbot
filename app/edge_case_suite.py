from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.chatbot import answer_question
from app.data_loader import load_bundle
from app.models import UserProfile


@dataclass
class EdgeCase:
    case_id: str
    question: str
    profile: UserProfile
    top_k: int = 5
    expect_non_empty_answer: bool = True
    expect_evidence: bool = True


def run_edge_case_suite() -> tuple[int, int]:
    bundle = load_bundle(Path("."))

    base = UserProfile(
        bac_stream="science_math",
        expected_grade_band="12_14",
        motivation="career",
        budget_band="comfort_50k",
        city="Rabat",
        country="MA",
    )

    cases = [
        EdgeCase("empty_question", "", base, expect_evidence=False),
        EdgeCase("whitespace_question", "   ", base, expect_evidence=False),
        EdgeCase(
            "unknown_program",
            "I want quantum bioinformatics law robotics in Oujda under 2000 MAD",
            base,
        ),
        EdgeCase(
            "conflicting_city_profile_vs_question",
            "I need a school in Fes for computer science with affordable tuition",
            UserProfile(
                bac_stream="science_math",
                expected_grade_band="12_14",
                motivation="career",
                budget_band="tight_25k",
                city="Rabat",
                country="MA",
            ),
        ),
        EdgeCase(
            "impossible_country_filter",
            "Recommend an engineering school in Rabat",
            UserProfile(
                bac_stream="science_math",
                expected_grade_band="12_14",
                motivation="career",
                budget_band="comfort_50k",
                city="Rabat",
                country="ZZ",
            ),
            expect_evidence=False,
        ),
        EdgeCase(
            "strict_zero_budget_private_city",
            "Best private AI school in Casablanca with zero budget",
            UserProfile(
                bac_stream="science_math",
                expected_grade_band="14_16",
                motivation="career",
                budget_band="zero_public",
                city="Casablanca",
                country="MA",
            ),
        ),
        EdgeCase(
            "mixed_language_query",
            "bghit computer science school f Rabat avec budget moyen",
            base,
        ),
        EdgeCase(
            "very_long_query",
            ("I need guidance " * 80) + "for engineering in Morocco under medium budget in Rabat",
            base,
        ),
        EdgeCase("top_k_boundary_low", "Public engineering schools in Rabat", base, top_k=1),
        EdgeCase("top_k_boundary_high", "Public engineering schools in Rabat", base, top_k=10),
    ]

    passed = 0
    for case in cases:
        response = answer_question(
            question=case.question,
            profile=case.profile,
            schools=bundle.schools,
            transcripts=bundle.transcripts,
            top_k=case.top_k,
        )

        has_answer = bool((response.short_answer or "").strip())
        has_next_action = bool((response.next_action or "").strip())
        has_evidence = len(response.evidence) > 0

        is_ok = has_next_action
        if case.expect_non_empty_answer:
            is_ok = is_ok and has_answer
        if case.expect_evidence:
            is_ok = is_ok and has_evidence
        else:
            is_ok = is_ok and (not has_evidence)

        if is_ok:
            passed += 1

        top_school = response.evidence[0].school_name if response.evidence else "N/A"
        print(
            f"case={case.case_id} pass={is_ok} evidence={len(response.evidence)} "
            f"top={top_school} answer={response.short_answer[:90]}"
        )

    total = len(cases)
    print(f"edge_summary={passed}/{total}")
    return passed, total


if __name__ == "__main__":
    passed, total = run_edge_case_suite()
    raise SystemExit(0 if passed == total else 1)
