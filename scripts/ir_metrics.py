import json
import math
import re
import sys
from typing import Any, Dict, List, Tuple

import requests

BASE_URL = "http://localhost:3001"
ENDPOINT = f"{BASE_URL}/recommendations/query"
TIMEOUT = 20


def _norm(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "", (text or "").lower())
    return cleaned


def _is_relevant(item: Dict[str, Any], expected: str) -> bool:
    expected_norm = _norm(expected)
    if not expected_norm:
        return False
    school_id = _norm(str(item.get("school_id", "")))
    school_name = _norm(str(item.get("school_name", "")))
    return (
        expected_norm in school_name
        or school_name in expected_norm
        or expected_norm == school_id
    )


def _rank_of_first_relevant(results: List[Dict[str, Any]], expected: str, k: int) -> int:
    for idx, item in enumerate(results[:k], start=1):
        if _is_relevant(item, expected):
            return idx
    return 0


def _ndcg_at_k(rank: int, k: int) -> float:
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _ap_at_k(rank: int, k: int) -> float:
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / float(rank)


def run_eval(val_path: str, top_k: int = 10) -> int:
    with open(val_path, "r", encoding="utf-8") as handle:
        cases = json.load(handle)

    if not isinstance(cases, list):
        print("Validation file must be a list of test cases.")
        return 1

    ndcg_scores: List[float] = []
    recall_scores: List[float] = []
    ap_scores: List[float] = []
    mrr_scores: List[float] = []

    for idx, case in enumerate(cases, 1):
        profile = case.get("profile", {}) if isinstance(case.get("profile", {}), dict) else {}
        expected = str(case.get("expected_school", "")).strip()
        user_id = str(case.get("user_id", case.get("userId", "")) or "").strip()

        payload = {
            "profile": profile,
            "top_k": top_k,
        }
        if user_id:
            payload["userId"] = user_id

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
        rank = _rank_of_first_relevant(results, expected, top_k)

        ndcg_scores.append(_ndcg_at_k(rank, top_k))
        recall_scores.append(1.0 if rank > 0 else 0.0)
        ap_scores.append(_ap_at_k(rank, top_k))
        mrr_scores.append(1.0 / float(rank) if rank > 0 else 0.0)

    if not ndcg_scores:
        print("No valid results to score.")
        return 1

    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    mean_recall = sum(recall_scores) / len(recall_scores)
    mean_map = sum(ap_scores) / len(ap_scores)
    mean_mrr = sum(mrr_scores) / len(mrr_scores)

    print("\n=== IR Metrics (Top K) ===")
    print(f"K: {top_k}")
    print(f"nDCG@{top_k}: {mean_ndcg:.3f}")
    print(f"Recall@{top_k}: {mean_recall:.3f}")
    print(f"MAP: {mean_map:.3f}")
    print(f"MRR: {mean_mrr:.3f}")

    return 0


if __name__ == "__main__":
    val_path = "test_val.json"
    top_k = 10
    if len(sys.argv) > 1:
        val_path = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            top_k = int(sys.argv[2])
        except ValueError:
            pass
    sys.exit(run_eval(val_path, top_k=top_k))
