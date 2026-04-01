from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from time import perf_counter

from app.chatbot import answer_question
from app.data_loader import load_bundle
from app.models import UserProfile


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(p * (len(sorted_values) - 1))
    return sorted_values[idx]


def build_queries(multiplier: int) -> list[str]:
    base = [
        "Best engineering school in Rabat with medium budget",
        "Public CS option in Fes with affordable tuition",
        "Compare ENSA and UM6P for AI with cost focus",
        "Je cherche une ecole informatique a Rabat",
        "bghit chi ecole d ingenieur f rabat",
    ]
    return base * max(1, multiplier)


def run_load_test(*, workers: int, query_multiplier: int, p95_target_s: float, warmup_queries: int) -> int:
    bundle = load_bundle(Path("."))
    profile = UserProfile(
        bac_stream="science_math",
        expected_grade_band="12_14",
        motivation="career",
        budget_band="comfort_50k",
        city="Rabat",
        country="MA",
    )

    queries = build_queries(query_multiplier)
    latencies: list[float] = []
    errors = 0
    success_count = 0

    def run_one(q: str) -> tuple[float, bool]:
        t0 = perf_counter()
        response = answer_question(
            question=q,
            profile=profile,
            schools=bundle.schools,
            transcripts=bundle.transcripts,
            top_k=5,
        )
        dt = perf_counter() - t0
        ok = bool((response.short_answer or "").strip()) and len(response.evidence) > 0
        return dt, ok

    # Warm up model/index caches so measured latency reflects steady-state behavior.
    warmup_n = min(max(0, warmup_queries), max(1, len(queries)))
    if warmup_n > 0:
        warmup_q = queries[:warmup_n]
        for q in warmup_q:
            run_one(q)

    t_all = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_one, q) for q in queries]
        for future in as_completed(futures):
            try:
                dt, ok = future.result()
                latencies.append(dt)
                if ok:
                    success_count += 1
                else:
                    errors += 1
            except Exception:
                errors += 1
    wall_s = perf_counter() - t_all

    latencies.sort()
    total = len(queries)
    completed = len(latencies)
    success_rate = success_count / total if total else 0.0
    avg_s = mean(latencies) if latencies else 0.0
    p50_s = percentile(latencies, 0.50)
    p95_s = percentile(latencies, 0.95)
    p99_s = percentile(latencies, 0.99)

    print(f"workers={workers}")
    print(f"warmup_queries={warmup_n}")
    print(f"total_requests={total}")
    print(f"completed={completed}")
    print(f"errors={errors}")
    print(f"success_rate={success_rate:.4f}")
    print(f"latency_avg_s={avg_s:.4f}")
    print(f"latency_p50_s={p50_s:.4f}")
    print(f"latency_p95_s={p95_s:.4f}")
    print(f"latency_p99_s={p99_s:.4f}")
    print(f"wall_s={wall_s:.2f}")

    passed = (errors == 0) and (success_rate >= 1.0) and (p95_s <= p95_target_s)
    print(f"load_test_pass={passed}")
    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run concurrent load test for chatbot pipeline")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument(
        "--query-multiplier",
        type=int,
        default=20,
        help="Repeats a 5-query base set N times (total requests = 5*N)",
    )
    parser.add_argument(
        "--p95-target-s",
        type=float,
        default=3.0,
        help="Latency p95 pass threshold in seconds",
    )
    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=5,
        help="Number of pre-run queries used to warm model/index caches",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        run_load_test(
            workers=max(1, args.workers),
            query_multiplier=max(1, args.query_multiplier),
            p95_target_s=max(0.1, args.p95_target_s),
            warmup_queries=max(0, args.warmup_queries),
        )
    )
