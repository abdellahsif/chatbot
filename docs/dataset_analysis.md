# Full Dataset Analysis

## Scope
- Sources analyzed: `data/mock/schools.csv`, `data/mock/transcripts.jsonl`, `data/mock/eval_questions.jsonl`.
- Auto-generated summary JSON: `docs/dataset_analysis.json`.

## Core Counts
- Schools: 10
- Transcript chunks: 12
- Eval questions: 13

## School Dataset
- Tuition max median (MAD): 48500
- Tuition max range (MAD): 0 to 140000
- Employability score median: 4.2
- International double degree flagged yes: 4 (40.0%)
- Top cities by number of schools:
  - Rabat: 3
  - Casablanca: 2
  - Benguerir: 1
  - Ifrane: 1
  - Fes: 1
  - Agadir: 1
  - Tanger: 1

## Transcript Dataset
- Median text length: 106 chars
- Language distribution: [{'name': 'fr', 'count': 8}, {'name': 'darija', 'count': 2}, {'name': 'en', 'count': 2}]
- Duplicate chunk_id count: 0
- Unknown school_id rows: 0
- Schools without transcripts: 0

## Evaluation Dataset
- Expected school names total: 31
- Expected school names resolved exactly: 0 (0.0%)
- Expected school names resolved with fuzzy matching: 0 (0.0%)
- Profile bac distribution: [{'name': 'spc', 'count': 5}, {'name': 'sm', 'count': 4}, {'name': 'eco', 'count': 2}, {'name': 'svt', 'count': 1}, {'name': 'l', 'count': 1}]
- Profile budget distribution: [{'name': 'tight_25k', 'count': 6}, {'name': 'comfort_50k', 'count': 4}, {'name': 'no_limit_70k_plus', 'count': 1}, {'name': 'zero_public', 'count': 1}, {'name': 'no_limit', 'count': 1}]
- Profile motivation distribution: [{'name': 'cash', 'count': 4}, {'name': 'prestige', 'count': 2}, {'name': 'safety', 'count': 2}, {'name': 'passion', 'count': 2}, {'name': 'expat', 'count': 1}, {'name': 'stability', 'count': 1}, {'name': 'employability', 'count': 1}]

## Data Quality Findings
- Expected-school-name exact resolution is low (0.0%).

## Recommendations
- Add canonical `expected_school_ids` in eval questions to remove name-matching ambiguity.
- Ensure every school has at least one transcript chunk for fair retrieval coverage.
- Add per-program transcript balancing for underrepresented domains.
- Keep chunk_id unique and enforce schema validation in data ingestion.

## Evaluation Run Diagnostics (Latest)
- Latest run timestamp: 2026-03-26T16:30:05.016215+00:00
- Pass rate: 9/13 (69.23%)
- Fallback rate: 38.46%
- Generation dominates latency: generate 7.8478s vs retrieve 0.0229s (average)
- Grounding is strong: 100% grounding_ok at item level
- Main issue is retrieval/intent coverage on difficult queries:
  - zero retrieval recall@k in 10/13 queries
  - fallback reasons: intent_coverage_failed (4), no_candidates (1)
  - low relevance ids: q001, q004, q005, q010, q012

## Priority Action Plan
- P0: Align eval expected schools with mock dataset IDs (`expected_school_ids`) and keep names as display-only.
- P1: Expand transcript coverage for underperforming intents (healthcare non-medical, military engineering, OFPPT vocational, architecture public/private comparisons).
- P2: Add targeted lexical aliases in retrieval for high-miss entities (ENSA network, OFPPT, IFCS, INAU, ERM, ENA variants).
- P3: Reduce generation latency separately (model/provider/runtime optimization), because retrieval is not the runtime bottleneck.
