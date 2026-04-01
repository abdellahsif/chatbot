# Metrics Improvement TODO

Date: 2026-04-01
Owner: Team
Status: Active

## Goal (Do Not Ship Before These Are Stable)
- Faithfulness >= 0.85
- Answer Relevance >= 0.85
- Context Precision >= 0.70
- Hallucination Rate <= 0.05
- End-to-End Success Rate >= 0.85
- Multilingual Pass Rate >= 0.85
- Noisy Query Pass Rate >= 0.85
- Conflicting Constraints Pass Rate >= 0.85
- Precision@5 >= 0.60

## Current Baseline (from latest agreed reference)
- Faithfulness: 0.512
- Answer Relevance: 0.291
- Context Precision: 0.334
- Hallucination Rate: 0.08
- End-to-End Success Rate: 0.17
- Multilingual Pass Rate: 0.16
- Noisy Query Pass Rate: 0.1045
- Conflicting Constraints Pass Rate: 0.1714
- Precision@5: 0.399

## Phase 0 - Baseline Lock
- [x] Run one clean full evaluation (fixed eval set, no code edits during run)
- [x] Save output paths for eval log and generation metrics snapshot
- [x] Copy baseline values into this file as LOCKED_BASELINE

### LOCKED_BASELINE (2026-04-01, full set, USE_CROSS_ENCODER_RERANKER=0)
- Total: 200
- Passed: 40
- Failed: 160
- Retrieval:
	- recall_at_5: 0.85
	- recall_at_10: 0.9175
	- precision_at_5: 0.411
	- mrr: 0.8283
	- ndcg_at_10: 0.9133
	- hit_rate_at_10: 1.0
- Answer quality:
	- faithfulness: 0.5009
	- answer_relevance: 0.3266
	- context_precision: 0.339
- Hallucination rate: 0.08
- End-to-end success rate: 0.2
- Performance latency_s: 0.2146
- Robustness:
	- multilingual_pass_rate: 0.1867
	- noisy_query_pass_rate: 0.1642
	- conflicting_constraints_pass_rate: 0.1714
- Final score (100): 59.73
- Metrics file: data/eval_logs/generation_metrics.json
- Eval log: data/eval_logs/eval_runs.jsonl

## Phase 1 - Retrieval Quality (Precision@5, Context Precision)
- [ ] Enable cross-encoder reranker with controlled top_n and blend
- [ ] Tune blend weights dense/sparse/reranker
- [ ] Add stricter relevance filtering before final top-5
- [ ] Keep fallback path for recall safety
- [ ] Re-run eval and compare Precision@5 and Context Precision deltas

## Phase 2 - Grounding and Hallucination
- [ ] Enforce evidence-only response construction for final answer fields
- [ ] Keep only top grounded evidence chunks for generation context (top 3 to 5)
- [ ] Add post-generation claim verification pass
- [ ] Rewrite or remove unsupported claims automatically
- [ ] Re-run eval and verify Hallucination <= 0.05

## Phase 3 - Answer Quality (Faithfulness, Relevance)
- [ ] Improve rationale generation to reference evidence tokens explicitly
- [ ] Add conflict-aware response strategy (budget vs prestige, etc.)
- [ ] Add low-confidence fallback response instead of weak definitive answer
- [ ] Validate faithfulness and relevance on full eval

## Phase 4 - Robustness (Multilingual, Noisy, Conflicting)
- [ ] Expand query normalization for typo and noisy prompts
- [ ] Add multilingual instruction handling and language-preserving responses
- [ ] Add soft-constraint handling with explicit trade-off explanation
- [ ] Run targeted robustness slices and full eval

## Phase 5 - End-to-End Success
- [ ] Combine best retrieval config + grounding checks + robust response policy
- [ ] Run 3 full evaluation runs back-to-back
- [ ] Confirm all threshold metrics pass in all 3 runs

## Operating Rules While Improving
- [ ] Change one small thing at a time
- [ ] Re-run evaluation after each change
- [ ] Keep change only if key metric improves without major regression
- [ ] Log each accepted change in docs/full_approaches_report.md

## Execution Log
- 2026-04-01: Checklist created.
- 2026-04-01: Baseline locked with full 200-query run and recorded in LOCKED_BASELINE.
- 2026-04-01: Phase 1 quick A/B (10 queries).
	- mode=off -> precision_at_5=0.44, faithfulness=0.4831, hallucination=0.10, latency_s=1.4025, final=57.16
	- mode=on_b035_n8 -> precision_at_5=0.40, faithfulness=0.5122, hallucination=0.05, latency_s=2.3263, final=58.26
	- temporary decision: keep reranker disabled for now and tune retrieval filtering + grounding first; revisit reranker after top-5 filtering is improved.
- 2026-04-01: Phase 1 experiment - top-result relevance prioritizer in retriever.
	- 15-query checkpoint regressed retrieval (precision_at_5=0.3333, recall_at_10=0.8556).
	- action taken: change rolled back.
	- rollback sanity check (10 queries): precision_at_5=0.44, recall_at_10=0.9667, ndcg_at_10=0.9848 restored.
