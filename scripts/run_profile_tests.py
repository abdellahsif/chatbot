import json
import random
import sys
from collections import Counter
from typing import Dict, List

import requests

BASE_URL = "http://localhost:3001"
ENDPOINT = f"{BASE_URL}/recommendations/query"
TIMEOUT = 15

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


def run_tests(total: int = 100, seed: int = 42) -> int:
    rng = random.Random(seed)
    failures: List[str] = []
    status_counts = Counter()
    top_hit_counts = Counter()

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
        schools = data.get("top_schools") or data.get("ranked_schools") or []
        if schools:
            top_name = schools[0].get("name", "unknown")
            top_hit_counts[top_name] += 1

        if idx % 10 == 0:
            print(f"{idx}/{total} tests completed...")

    print("\n=== Profile Recommendation Test Summary ===")
    print(f"Total tests: {total}")
    print(f"HTTP status counts: {dict(status_counts)}")
    print(f"Failures: {len(failures)}")

    if failures:
        print("\nSample failures (up to 10):")
        for item in failures[:10]:
            print(f"- {item}")

    if top_hit_counts:
        print("\nTop 10 most frequent #1 schools:")
        for name, count in top_hit_counts.most_common(10):
            print(f"- {name}: {count}")

    return 0 if not failures else 1


if __name__ == "__main__":
    total = 100
    if len(sys.argv) > 1:
        try:
            total = int(sys.argv[1])
        except ValueError:
            pass
    sys.exit(run_tests(total=total))
