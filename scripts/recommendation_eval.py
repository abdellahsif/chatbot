import json
import math
import re
import sys
from typing import Any, Dict, List

import requests

BASE_URL = "http://localhost:3001"
ENDPOINT = f"{BASE_URL}/recommendations/query"
TIMEOUT = 25


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _is_relevant(item: Dict[str, Any], targets: List[str]) -> bool:
    if not targets:
        return False
    school_id = _norm(str(item.get("school_id", "")))
    school_name = _norm(str(item.get("school_name", "")))
    for target in targets:
        tgt = _norm(str(target))
        if not tgt:
            continue
        if tgt in school_name or school_name in tgt:
            return True
        if tgt == school_id:
            return True
    return False


def _first_relevant_rank(results: List[Dict[str, Any]], targets: List[str], k: int) -> int:
    for idx, item in enumerate(results[:k], start=1):
        if _is_relevant(item, targets):
            return idx
    return 0


def _ndcg_at_k(rank: int, k: int) -> float:
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def run_eval(case_path: str, top_k: int = 10) -> int:
    with open(case_path, "r", encoding="utf-8") as handle:
        cases = json.load(handle)

    if not isinstance(cases, list):
        print("Cases file must be a list.")
        return 1

    precision_scores: List[float] = []
    hit_scores: List[float] = []
    ndcg_scores: List[float] = []
    unacceptable_hits = 0

    for idx, case in enumerate(cases, 1):
        profile = case.get("profile", {}) if isinstance(case.get("profile", {}), dict) else {}
        career_profile = case.get("career_profile") if isinstance(case.get("career_profile"), dict) else None
        ideal = case.get("ideal_schools", []) if isinstance(case.get("ideal_schools", []), list) else []
        unacceptable = (
            case.get("unacceptable_schools", []) if isinstance(case.get("unacceptable_schools", []), list) else []
        )

        payload = {
            "profile": profile,
            "top_k": top_k,
        }
        if career_profile:
            payload["career_profile"] = career_profile

        try:
            response = requests.post(ENDPOINT, json=payload, timeout=TIMEOUT)
        except Exception as exc:
            print(f"{idx}. ERROR request_failed: {exc}")
            continue

        if response.status_code != 200:
            print(f"{idx}. ERROR status_{response.status_code}: {response.text[:200]}")
            continue

        data = response.json()
        results = data.get("results") or []

        top5 = results[:5]
        hit_top10 = _first_relevant_rank(results, ideal, top_k) > 0
        ndcg_scores.append(_ndcg_at_k(_first_relevant_rank(results, ideal, top_k), top_k))
        hit_scores.append(1.0 if hit_top10 else 0.0)

        if ideal:
            rel_top5 = sum(1 for item in top5 if _is_relevant(item, ideal))
            precision_scores.append(rel_top5 / 5.0)

        if unacceptable:
            unacceptable_hits += sum(1 for item in results[:top_k] if _is_relevant(item, unacceptable))

    if not ndcg_scores:
        print("No valid results to score.")
        return 1

    mean_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    mean_hit = sum(hit_scores) / len(hit_scores)
    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores)

    print("\n=== Recommendation Metrics ===")
    print(f"K: {top_k}")
    print(f"Precision@5: {mean_precision:.3f}")
    print(f"HitRate@10: {mean_hit:.3f}")
    print(f"nDCG@10: {mean_ndcg:.3f}")
    print(f"Unacceptable hits in top {top_k}: {unacceptable_hits}")

    return 0


if __name__ == "__main__":
    case_path = "scripts/recommendation_eval_cases.json"
    top_k = 10
    if len(sys.argv) > 1:
        case_path = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            top_k = int(sys.argv[2])
        except ValueError:
            pass
    sys.exit(run_eval(case_path, top_k=top_k))
