import json
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List

import requests

BASE_URL = "http://localhost:3001"
ENDPOINT = f"{BASE_URL}/recommendations/query"
TIMEOUT = 20

BAC_STREAMS = [
    "sm",
    "sm_a",
    "sm_b",
    "spc",
    "svt",
    "eco",
    "tgc",
    "lettres",
    "sc_humaines",
    "arts_appliques",
]

BUDGET_BANDS = [
    "zero_public",
    "tight_25k",
    "comfort_50k",
    "no_limit_70k_plus",
]

MOTIVATIONS = [
    "employability",
    "prestige",
    "passion",
    "safety",
    "cash",
    "expat",
]

CITIES = [
    "Rabat",
    "Casablanca",
    "Marrakech",
    "Fes",
    "Tanger",
    "Agadir",
    "Oujda",
    "Tetouan",
    "Meknes",
    "Kenitra",
    "Safi",
    "Ouarzazate",
    "Laayoune",
    "Dakhla",
    "Nador",
]


def build_profile(rng: random.Random) -> Dict[str, str]:
    return {
        "bac_stream": rng.choice(BAC_STREAMS),
        "budget_band": rng.choice(BUDGET_BANDS),
        "city": rng.choice(CITIES),
        "motivation": rng.choice(MOTIVATIONS),
    }


def _score_component(school: dict, key: str) -> float:
    components = school.get("score_components") or {}
    try:
        return float(components.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def run_audit(total: int = 100, sample_size: int = 20, seed: int = 42) -> int:
    rng = random.Random(seed)
    failures: List[str] = []
    status_counts = Counter()

    budget_ok = 0
    city_ok = 0
    valid_budget = 0
    valid_city = 0

    avg_scores = defaultdict(float)
    avg_scores_top3 = defaultdict(float)
    count_scores = 0
    count_scores_top3 = 0

    manual_samples: List[dict] = []

    for idx in range(1, total + 1):
        profile = build_profile(rng)
        payload = {"question": "", "profile": profile, "top_k": 5}
        try:
            response = requests.post(ENDPOINT, json=payload, timeout=TIMEOUT)
        except Exception as exc:
            failures.append(f"{idx}: request_failed: {exc}")
            continue

        status_counts[response.status_code] += 1
        if response.status_code != 200:
            failures.append(f"{idx}: status_{response.status_code}: {response.text[:200]}")
            continue

        data = response.json()
        top_schools = data.get("top_schools") or data.get("ranked_schools") or []
        if not top_schools:
            failures.append(f"{idx}: empty_results")
            continue

        top1 = top_schools[0]
        count_scores += 1
        avg_scores["budget_match"] += _score_component(top1, "budget_match")
        avg_scores["location_match"] += _score_component(top1, "location_match")
        avg_scores["motivation_match"] += _score_component(top1, "motivation_match")
        avg_scores["bac_semantic"] += _score_component(top1, "bac_semantic")

        if profile.get("budget_band"):
            valid_budget += 1
            if _score_component(top1, "budget_match") >= 0.9:
                budget_ok += 1

        if profile.get("city"):
            valid_city += 1
            if _score_component(top1, "location_match") >= 0.9:
                city_ok += 1

        top3 = top_schools[:3]
        if top3:
            for item in top3:
                avg_scores_top3["budget_match"] += _score_component(item, "budget_match")
                avg_scores_top3["location_match"] += _score_component(item, "location_match")
                avg_scores_top3["motivation_match"] += _score_component(item, "motivation_match")
                avg_scores_top3["bac_semantic"] += _score_component(item, "bac_semantic")
            count_scores_top3 += len(top3)

        if len(manual_samples) < sample_size:
            manual_samples.append(
                {
                    "profile": profile,
                    "top_3": [
                        {
                            "name": s.get("name"),
                            "city": s.get("city"),
                            "legal_status": s.get("legal_status"),
                            "tuition_min_mad": s.get("tuition_min_mad"),
                            "tuition_max_mad": s.get("tuition_max_mad"),
                            "match_score": s.get("match_score"),
                            "match_grade": s.get("match_grade"),
                            "score_components": s.get("score_components"),
                        }
                        for s in top3
                    ],
                }
            )

        if idx % 10 == 0:
            print(f"{idx}/{total} tests completed...")

    print("\n=== Constraint Accuracy Summary ===")
    print(f"Total tests: {total}")
    print(f"HTTP status counts: {dict(status_counts)}")
    print(f"Failures: {len(failures)}")

    if valid_budget:
        print(f"Budget pass rate (top1, >=0.9): {budget_ok}/{valid_budget} ({budget_ok/valid_budget:.1%})")
    if valid_city:
        print(f"City pass rate (top1, >=0.9): {city_ok}/{valid_city} ({city_ok/valid_city:.1%})")

    if count_scores:
        print("\nAverage top1 scores:")
        print(f"- budget_match: {avg_scores['budget_match']/count_scores:.3f}")
        print(f"- location_match: {avg_scores['location_match']/count_scores:.3f}")
        print(f"- motivation_match: {avg_scores['motivation_match']/count_scores:.3f}")
        print(f"- bac_semantic: {avg_scores['bac_semantic']/count_scores:.3f}")

    if count_scores_top3:
        print("\nAverage top3 scores:")
        print(f"- budget_match: {avg_scores_top3['budget_match']/count_scores_top3:.3f}")
        print(f"- location_match: {avg_scores_top3['location_match']/count_scores_top3:.3f}")
        print(f"- motivation_match: {avg_scores_top3['motivation_match']/count_scores_top3:.3f}")
        print(f"- bac_semantic: {avg_scores_top3['bac_semantic']/count_scores_top3:.3f}")

    if failures:
        print("\nSample failures (up to 10):")
        for item in failures[:10]:
            print(f"- {item}")

    sample_path = "scripts/quality_sample.json"
    with open(sample_path, "w", encoding="utf-8") as handle:
        json.dump(manual_samples, handle, ensure_ascii=False, indent=2)

    print(f"\nManual review sample saved to {sample_path}")

    return 0 if not failures else 1


if __name__ == "__main__":
    total = 100
    sample_size = 20
    if len(sys.argv) > 1:
        try:
            total = int(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) > 2:
        try:
            sample_size = int(sys.argv[2])
        except ValueError:
            pass
    sys.exit(run_audit(total=total, sample_size=sample_size))
