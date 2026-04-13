import argparse
import importlib
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from app.data_loader import load_bundle
from app.models import UserProfile

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


DEFAULT_PROFILE = UserProfile(
    bac_stream="general",
    expected_grade_band="",
    motivation="",
    budget_band="no_limit_70k_plus",
    city="",
    country="MA",
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def school_aliases(school: dict) -> set[str]:
    aliases = {
        normalize(school.get("school_id", "")),
        normalize(school.get("name", "")),
        normalize(school.get("acronym", "")),
    }
    programs = school.get("programs", [])
    if isinstance(programs, list):
        aliases.update(normalize(p) for p in programs if p)
    return {alias for alias in aliases if alias}


def sample_queries(dataset_path: Path, per_school: int, max_samples: int | None) -> list[dict]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    grouped: dict[str, list[dict]] = defaultdict(list)
    seen_queries: set[str] = set()

    for section in ("positives", "negatives"):
        for item in data.get(section, []):
            query = str(item.get("query", "")).strip()
            gold_school = str(item.get("ecole", "")).strip()
            if not query or not gold_school:
                continue
            key = normalize(query)
            if key in seen_queries:
                continue
            seen_queries.add(key)
            grouped[normalize(gold_school)].append(
                {
                    "query": query,
                    "gold_school": gold_school,
                    "source_id": item.get("source_id", ""),
                    "theme": item.get("theme", ""),
                }
            )

    selected: list[dict] = []
    for school_key in sorted(grouped):
        items = grouped[school_key]
        if len(items) > per_school:
            selected.extend(random.sample(items, per_school))
        else:
            selected.extend(items)

    random.shuffle(selected)
    if max_samples is not None and max_samples > 0:
        selected = selected[:max_samples]
    return selected


def load_retriever_for_model(model_path: str):
    os.environ["CROSS_ENCODER_MODEL"] = model_path
    os.environ["USE_CROSS_ENCODER_RERANKER"] = "1"
    import app.retriever as retriever

    return importlib.reload(retriever)


def find_rank(gold_school: str, ranked: list[dict]) -> int | None:
    gold_norm = normalize(gold_school)
    for idx, item in enumerate(ranked, start=1):
        school = item.get("school", {})
        aliases = school_aliases(school)
        if gold_norm in aliases:
            return idx
        if any(gold_norm == alias or gold_norm in alias or alias in gold_norm for alias in aliases):
            return idx
    return None


def evaluate_model(model_path: str, samples: list[dict], schools: dict, transcripts: list[dict], top_k: int = 10) -> dict:
    retriever = load_retriever_for_model(model_path)

    ranks: list[int | None] = []
    hits_at_1 = 0
    hits_at_2 = 0
    hits_at_10 = 0
    missed = 0
    top1_examples: list[dict] = []
    top2_examples: list[dict] = []
    miss_examples: list[dict] = []

    for sample in samples:
        ranked = retriever.retrieve(
            question=sample["query"],
            profile=DEFAULT_PROFILE,
            schools=schools,
            transcripts=transcripts,
            top_k=top_k,
        )
        rank = find_rank(sample["gold_school"], ranked)
        ranks.append(rank)

        top_names = [str(item.get("school", {}).get("name", "")) for item in ranked[:3]]
        record = {
            "query": sample["query"],
            "gold_school": sample["gold_school"],
            "rank": rank,
            "top3": top_names,
        }

        if rank == 1:
            hits_at_1 += 1
            if len(top1_examples) < 5:
                top1_examples.append(record)
        if rank in {1, 2}:
            hits_at_2 += 1
            if rank == 2 and len(top2_examples) < 5:
                top2_examples.append(record)
        if rank is not None and rank <= top_k:
            hits_at_10 += 1
        else:
            missed += 1
            if len(miss_examples) < 5:
                miss_examples.append(record)

    valid_ranks = [r for r in ranks if r is not None]
    avg_rank = mean(valid_ranks) if valid_ranks else float("nan")
    return {
        "model": model_path,
        "n": len(samples),
        "hit@1": hits_at_1 / len(samples) if samples else float("nan"),
        "hit@2": hits_at_2 / len(samples) if samples else float("nan"),
        "hit@10": hits_at_10 / len(samples) if samples else float("nan"),
        "miss_rate": missed / len(samples) if samples else float("nan"),
        "avg_rank_when_found": avg_rank,
        "ranks": ranks,
        "top1_examples": top1_examples,
        "top2_examples": top2_examples,
        "miss_examples": miss_examples,
    }


def print_summary(base: dict, fine_tuned: dict):
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    print("\nSchool ranking comparison")
    print("=" * 80)
    print(f"Base model      : {base['model']}")
    print(f"Fine-tuned model : {fine_tuned['model']}")
    print("-" * 80)
    print(f"{'Metric':20} {'Base':16} {'Fine-tuned':16} {'Delta(ft-base)':16}")
    print("-" * 80)
    for key in ["n", "hit@1", "hit@2", "hit@10", "miss_rate", "avg_rank_when_found"]:
        b = base[key]
        f = fine_tuned[key]
        delta = f - b if isinstance(b, float) and isinstance(f, float) else None
        print(f"{key:20} {fmt(b):16} {fmt(f):16} {fmt(delta) if delta is not None else '-':16}")
    print("=" * 80)

    if fine_tuned["top1_examples"]:
        print("\nExamples where fine-tuned model got #1:")
        for ex in fine_tuned["top1_examples"][:3]:
            print(f"- gold={ex['gold_school']} | rank={ex['rank']} | top3={ex['top3']} | query={ex['query'][:90]}")

    if fine_tuned["top2_examples"]:
        print("\nExamples where fine-tuned model got #2:")
        for ex in fine_tuned["top2_examples"][:3]:
            print(f"- gold={ex['gold_school']} | rank={ex['rank']} | top3={ex['top3']} | query={ex['query'][:90]}")

    if fine_tuned["miss_examples"]:
        print("\nMissed examples (gold not in top 10):")
        for ex in fine_tuned["miss_examples"][:3]:
            print(f"- gold={ex['gold_school']} | rank={ex['rank']} | top3={ex['top3']} | query={ex['query'][:90]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare school ranking for base vs fine-tuned cross encoder")
    parser.add_argument("--dataset", default="combined_finetune_pairs.json")
    parser.add_argument("--base-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--ft-model", default="checkpoints/model")
    parser.add_argument("--per-school", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    root = Path(__file__).resolve().parent
    if load_dotenv is not None:
        load_dotenv(root / ".env")
    os.environ.setdefault("SUPABASE_STRICT_MODE", "0")
    bundle = load_bundle(root)
    samples = sample_queries(root / args.dataset, per_school=args.per_school, max_samples=args.max_samples)
    if not samples:
        raise RuntimeError("No usable query/school pairs found in dataset")

    print(f"Loaded {len(samples)} queries from {args.dataset}")
    print(f"Data source: {bundle.source} | schools={len(bundle.schools)} | transcripts={len(bundle.transcripts)}")

    base = evaluate_model(args.base_model, samples, bundle.schools, bundle.transcripts, top_k=args.top_k)
    fine_tuned = evaluate_model(args.ft_model, samples, bundle.schools, bundle.transcripts, top_k=args.top_k)
    print_summary(base, fine_tuned)
