# Recommendation System Improvement Roadmap

## Step 1 - Lock baselines (v1)

### Retrieval metrics (test_val.json, K=10)
- Recall@10: 0.750
- nDCG@10: 0.338
- MRR: 0.211
- MAP: 0.211

### Recommendation metrics (not yet measured)
- Precision@5: pending
- Precision@10: pending
- Hit Rate@10: pending
- Recommendation nDCG@10: pending
- Bad Recommendation Rate: pending

## Experiment log

| Version | Chunking | Embedding | Retrieval | Reranker | Recall@10 | nDCG@10 | MRR | MAP | Prec@5 | Prec@10 | Hit@10 | Rec nDCG@10 | Bad Rec % | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | current | current | dense | none | 0.750 | 0.338 | 0.211 | 0.211 | - | - | - | - | - | baseline |
