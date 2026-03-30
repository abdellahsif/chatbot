from __future__ import annotations

import argparse
from pathlib import Path

from app.chatbot import answer_question
from app.data_loader import load_bundle
from app.models import UserProfile


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive University Chatbot terminal session")
    parser.add_argument("--bac-stream", default="science_math")
    parser.add_argument("--grade-band", default="12_14")
    parser.add_argument("--motivation", default="career")
    parser.add_argument("--budget-band", default="comfort_50k")
    parser.add_argument("--city", default="Fes")
    parser.add_argument("--country", default="MA")
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    bundle = load_bundle(root)
    profile = UserProfile(
        bac_stream=str(args.bac_stream).strip().lower(),
        expected_grade_band=str(args.grade_band).strip().lower(),
        motivation=str(args.motivation).strip().lower(),
        budget_band=str(args.budget_band).strip().lower(),
        city=str(args.city).strip(),
        country=str(args.country).strip().upper(),
    )
    top_k = max(1, min(10, int(args.top_k)))

    print("=== University Chatbot (Interactive) ===")
    print("Type your question and press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print(f"Profile: {profile}")
    print()

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            return

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Session ended.")
            return

        response = answer_question(
            question=question,
            profile=profile,
            schools=bundle.schools,
            transcripts=bundle.transcripts,
            top_k=top_k,
        )

        print(f"Bot: {response.short_answer}")
        print(f"Why: {response.why_it_fits}")
        if response.evidence:
            print("Evidence:")
            for item in response.evidence:
                print(f"  - {item.school_name} | {item.program} | score={item.score:.4f}")
        print("-" * 72)


if __name__ == "__main__":
    main()
