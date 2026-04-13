import argparse
import json
from pathlib import Path
from statistics import mean

from sentence_transformers import CrossEncoder


def auc_from_scores(labels, scores):
    """Compute ROC-AUC using rank statistic without external dependencies."""
    pos = [(s, i) for i, (y, s) in enumerate(zip(labels, scores)) if y == 1]
    neg = [(s, i) for i, (y, s) in enumerate(zip(labels, scores)) if y == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranked = sorted([(s, y) for y, s in zip(labels, scores)], key=lambda x: x[0])

    rank_sum_pos = 0.0
    rank = 1
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i))) / 2.0
        for k in range(i, j + 1):
            if ranked[k][1] == 1:
                rank_sum_pos += avg_rank
        rank += (j - i + 1)
        i = j + 1

    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return auc


def load_dataset(path, max_samples=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    positives = []
    negatives = []
    for item in data.get("positives", []):
        q = item.get("query", "").strip()
        r = item.get("positive", "").strip()
        if q and r:
            positives.append((q, r, 1))
    for item in data.get("negatives", []):
        q = item.get("query", "").strip()
        r = item.get("negative", "").strip()
        if q and r:
            negatives.append((q, r, 0))

    if max_samples is not None and max_samples > 0:
        half = max_samples // 2
        pos_take = min(len(positives), half)
        neg_take = min(len(negatives), max_samples - pos_take)
        # If one side is short, fill from the other side.
        if pos_take + neg_take < max_samples:
            remaining = max_samples - (pos_take + neg_take)
            extra_pos = min(len(positives) - pos_take, remaining)
            pos_take += max(0, extra_pos)
            remaining = max_samples - (pos_take + neg_take)
            extra_neg = min(len(negatives) - neg_take, remaining)
            neg_take += max(0, extra_neg)
        samples = positives[:pos_take] + negatives[:neg_take]
    else:
        samples = positives + negatives

    return samples


def evaluate_model(model_name_or_path, samples, batch_size=32):
    model = CrossEncoder(model_name_or_path)
    pairs = [[q, r] for q, r, _ in samples]
    labels = [y for _, _, y in samples]

    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    scores = [float(x) for x in scores]

    pos_scores = [s for s, y in zip(scores, labels) if y == 1]
    neg_scores = [s for s, y in zip(scores, labels) if y == 0]

    threshold = 0.0
    preds = [1 if s >= threshold else 0 for s in scores]
    acc = sum(int(p == y) for p, y in zip(preds, labels)) / len(labels) if labels else float("nan")

    return {
        "model": model_name_or_path,
        "n": len(samples),
        "positives": len(pos_scores),
        "negatives": len(neg_scores),
        "mean_pos": mean(pos_scores) if pos_scores else float("nan"),
        "mean_neg": mean(neg_scores) if neg_scores else float("nan"),
        "margin": (mean(pos_scores) - mean(neg_scores)) if pos_scores and neg_scores else float("nan"),
        "auc": auc_from_scores(labels, scores),
        "acc@0": acc,
    }


def print_report(base_metrics, ft_metrics):
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    keys = ["n", "positives", "negatives", "mean_pos", "mean_neg", "margin", "auc", "acc@0"]

    print("\nComparison Results")
    print("=" * 72)
    print(f"Base model      : {base_metrics['model']}")
    print(f"Fine-tuned model: {ft_metrics['model']}")
    print("-" * 72)
    print(f"{'Metric':20} {'Base':16} {'Fine-tuned':16} {'Delta(ft-base)':16}")
    print("-" * 72)
    for k in keys:
        b = base_metrics[k]
        f = ft_metrics[k]
        if isinstance(b, float) and isinstance(f, float):
            d = f - b
            d_text = f"{d:+.4f}"
        else:
            d_text = "-"
        print(f"{k:20} {fmt(b):16} {fmt(f):16} {d_text:16}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned cross-encoder")
    parser.add_argument("--dataset", default="combined_finetune_pairs.json")
    parser.add_argument("--base-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--ft-model", default="checkpoints/model")
    parser.add_argument("--max-samples", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    samples = load_dataset(str(dataset_path), max_samples=args.max_samples)
    if not samples:
        raise RuntimeError("No valid samples found in dataset")

    print(f"Loaded {len(samples)} labeled pairs from {dataset_path}")

    base_metrics = evaluate_model(args.base_model, samples, batch_size=args.batch_size)
    ft_metrics = evaluate_model(args.ft_model, samples, batch_size=args.batch_size)

    print_report(base_metrics, ft_metrics)
