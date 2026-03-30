from __future__ import annotations

import json
import re
from pathlib import Path


TARGET_SIZE = 200


def _load_seed_questions(seed_path: Path) -> list[dict]:
    rows: list[dict] = []
    with seed_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No questions found in {seed_path}")
    return rows


def _inject_typos(text: str) -> str:
    replacements = {
        "which": "whch",
        "options": "optns",
        "affordable": "affrodable",
        "management": "managment",
        "engineering": "enginering",
        "university": "universty",
        "international": "internatonal",
        "program": "progrm",
    }
    out = text
    for old, new in replacements.items():
        out = re.sub(rf"\b{old}\b", new, out, flags=re.IGNORECASE)
    return out


def _question_variant(base_q: str, idx: int) -> str:
    mode = idx % 8
    if mode == 0:
        return base_q
    if mode == 1:
        return f"Please answer in English only. {base_q}"
    if mode == 2:
        return f"Reponds en francais: {base_q}"
    if mode == 3:
        return f"Jawbni b darija: {base_q}"
    if mode == 4:
        return f"Give me a strict recommendation with tradeoffs. {base_q}"
    if mode == 5:
        return f"Edge case: very low budget and uncertain grades. {base_q}"
    if mode == 6:
        return _inject_typos(base_q)
    return f"Keep answer grounded in evidence only. {base_q}"


def _profile_variant(profile: dict, idx: int) -> dict:
    out = dict(profile)
    bands = ["zero_public", "tight_25k", "comfort_50k", "no_limit_70k_plus"]
    motivations = ["cash", "prestige", "passion", "safety", "expat", "employability"]

    out["budget_band"] = bands[idx % len(bands)]
    out["motivation"] = motivations[idx % len(motivations)]
    # Corpus is Morocco-only in this project, so keep country constraint evaluable.
    out["country"] = "MA"
    return out


def _must_include_variant(must_include: list[str], idx: int) -> list[str]:
    extra = [
        "cite evidence",
        "state one alternative",
        "mention cost",
        "mention selectivity",
        "include next action",
    ]
    out = [str(x) for x in must_include if str(x).strip()]
    out.append(extra[idx % len(extra)])
    return out


def build_fixed_set(seed_rows: list[dict], size: int = TARGET_SIZE) -> list[dict]:
    out: list[dict] = []
    i = 0
    while len(out) < size:
        base = seed_rows[i % len(seed_rows)]
        q_idx = len(out) + 1
        row = {
            "id": f"q{q_idx:03d}",
            "question": _question_variant(str(base.get("question", "")), q_idx),
            "profile": _profile_variant(dict(base.get("profile", {})), q_idx),
            "must_include": _must_include_variant(list(base.get("must_include", [])), q_idx),
            "expected_school_names": list(base.get("expected_school_names", [])),
        }
        out.append(row)
        i += 1
    return out


def write_jsonl(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    primary_seed = root_dir / "data" / "eval_questions.jsonl"
    fallback_seed = root_dir / "data" / "eval_questions_fixed_200.jsonl"
    seed_path = primary_seed if primary_seed.exists() else fallback_seed
    out_path = root_dir / "data" / "eval_questions_fixed_200.jsonl"

    seed_rows = _load_seed_questions(seed_path)
    fixed_rows = build_fixed_set(seed_rows, size=TARGET_SIZE)
    write_jsonl(fixed_rows, out_path)

    print(f"Wrote {len(fixed_rows)} fixed questions to {out_path}")


if __name__ == "__main__":
    main()