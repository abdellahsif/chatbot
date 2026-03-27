# Full Approaches Report

Date: 2026-03-27

## 1) Objective
This file is the single source of truth for all retrieval/evaluation approaches tried in this workspace, including:
- approach design
- model names
- metric names
- benchmark results
- final decision

## 2) Approaches Tried

### A. Legacy Dense Baseline
- Retrieval: dense semantic retrieval only
- Reranking: none
- Notes: original setup before sparse/hybrid rollout

### B. Dense Tuned
- Retrieval: dense semantic retrieval
- Candidate scoring: dense similarity + bi-encoder style semantic scoring in retriever pipeline
- Notes: quality improved vs baseline, but latency remained high

### C. Sparse + Bi-Encoder (TF-IDF)
- Retrieval: TF-IDF sparse retrieval
- Reranking: semantic bi-score blending
- Notes: strong quality with much lower latency

### D. Hybrid Tuned (Current Default)
- Retrieval: dense candidate union sparse
- Fusion: normalized weighted fusion (dense + sparse + bi-score)
- Notes: best quality/latency tradeoff in current tests

### E. Hybrid + Cross-Encoder Reranker (Optional)
- Retrieval: hybrid candidates first
- Final rerank: cross-encoder top-N reranking with blend factor
- Notes: tested but degraded quality and increased latency in current runs

## 3) Models Used

### Retrieval / Embeddings
- intfloat/multilingual-e5-base

### Optional Cross-Encoder Reranker
- cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

### Generator (documented in project)
- Qwen/Qwen2.5-0.5B-Instruct

## 4) Retrieval Modes and Main Runtime Knobs
- RETRIEVAL_MODE = dense | sparse | hybrid
- BEIR_RETRIEVAL_MODES = dense,sparse,hybrid
- SPARSE_BI_WEIGHT = 0.9
- HYBRID_DENSE_WEIGHT = 0.2
- HYBRID_SPARSE_WEIGHT = 0.1
- HYBRID_BI_WEIGHT = 0.7
- USE_CROSS_ENCODER_RERANKER = 0|1
- CROSS_ENCODER_MODEL = cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
- CROSS_ENCODER_TOP_N = 8
- CROSS_ENCODER_BLEND = 0.6

## 5) Metrics Used

### Retrieval Benchmarks (BEIR)
- NDCG@k
- MAP@k
- Recall@k
- Precision@k
- MRR@k
- avg_latency_s

### End-to-End Evaluation (latest logged formula)
Formula:
Final = 0.30*Groundedness + 0.20*Relevance + 0.15*Compliance + 0.10*(1-Hallucination) + 0.05*LatencyScore + 0.15*Recall + 0.05*CostScore

Recall composite:
Recall = 0.4*MustIncludeRecall + 0.6*RetrievalRecall@k

Latest metric keys in logs:
- avg_latency_s
- avg_groundedness
- avg_groundedness_faithfulness
- avg_relevance
- avg_relevance_appropriateness
- avg_compliance
- avg_hallucination
- hallucination_rate
- avg_coherence_fluency
- avg_must_include_recall
- avg_must_include_precision
- avg_must_include_f1
- avg_recall
- avg_retrieval_recall_at_k
- avg_precision
- avg_retrieval_precision_at_k
- avg_f1
- avg_retrieval_f1_at_k
- avg_recall_composite
- avg_retrieve_seconds
- avg_generate_seconds
- avg_post_filter_seconds
- fallback_rate
- avg_latency_score
- final_score
- final_score_100

## 6) Benchmark Results (from runs)

### Retrieval (BEIR) Summary
| Approach | NDCG@10 | MRR@10 | Avg Latency (s) |
|---|---:|---:|---:|
| Legacy dense baseline | 0.5747 | 0.5410 | 0.5785 |
| Dense tuned | 0.6732 | 0.7077 | 0.4736 |
| Sparse tuned | 0.6761 | 0.7077 | 0.0612 |
| Hybrid tuned | 0.6761 | 0.7077 | 0.0814 |
| Hybrid + cross (blend=0.6) | 0.5672 | 0.5769 | 0.7431 |
| Hybrid + cross (blend=0.2) | 0.6526 | 0.6308 | 0.7132 |

### Latest End-to-End Logged Run
- run_at: 2026-03-26T16:30:05.016215+00:00
- total: 13
- passed: 9
- failed: 4
- final_score_100: 47.17
- avg_latency_s: 7.8711
- avg_retrieve_seconds: 0.0229
- avg_generate_seconds: 7.8478
- fallback_rate: 0.3846

## 7) Final Decision
- Keep hybrid tuned as default production retrieval mode.
- Keep sparse tuned as low-latency fallback option.
- Keep cross-encoder reranking disabled by default.

## 8) Why This Decision
- Sparse and hybrid currently match top retrieval quality with significantly better latency than dense-only.
- Cross-encoder reranking increased latency and reduced retrieval quality in tested settings.
- End-to-end latency is dominated by generation time, not retrieval.

