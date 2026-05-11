import json
import re
import sys
from typing import Any, Dict, List

import requests

BASE_URL = "http://localhost:3001"
ENDPOINT = f"{BASE_URL}/recommendations/query"
TIMEOUT = 20


def _norm(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "", (text or "").lower())
    return cleaned


def _extract_school_names(payload: Dict[str, Any]) -> List[str]:
    schools = payload.get("top_schools") or payload.get("ranked_schools") or []
    names: List[str] = []
    for item in schools:
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def run_validation(val_path: str, top_k: int = 5) -> int:
    with open(val_path, "r", encoding="utf-8") as handle:
        cases = json.load(handle)

    if not isinstance(cases, list):
        print("Validation file must be a list of test cases.")
        return 1

    passed = 0
    failed = 0

    print("\n=== Validation Results ===\n")

    for idx, case in enumerate(cases, 1):
        question = str(case.get("question", "")).strip()
        profile = case.get("profile", {}) if isinstance(case.get("profile", {}), dict) else {}
        expected = str(case.get("expected_school", "")).strip()

        payload = {"question": question, "profile": profile, "top_k": top_k}
        try:
            response = requests.post(ENDPOINT, json=payload, timeout=TIMEOUT)
        except Exception as exc:
            print(f"{idx}. ERROR request_failed: {exc}")
            failed += 1
            continue

        if response.status_code != 200:
            print(f"{idx}. ERROR status_{response.status_code}: {response.text[:200]}")
            failed += 1
            continue

        data = response.json()
        names = _extract_school_names(data)[:top_k]

        expected_norm = _norm(expected)
        match = any(expected_norm in _norm(name) or _norm(name) in expected_norm for name in names)

        status = "PASS" if match else "FAIL"
        print(f"{idx}. {status} | expected: {expected}")
        print(f"   top_{top_k}: {', '.join(names) if names else 'NO_RESULTS'}")

        if match:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    print("\n=== Summary ===")
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    val_path = "test_val.json"
    top_k = 5
    if len(sys.argv) > 1:
        val_path = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            top_k = int(sys.argv[2])
        except ValueError:
            pass
    sys.exit(run_validation(val_path, top_k=top_k))
