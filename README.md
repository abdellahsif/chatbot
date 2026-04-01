# Chatbot

This workspace supports direct ingestion from `BDD_MCD_Universites.xlsx` when present at workspace root.

## Goal
Build a chatbot that:
- answers from structured school data + transcript snippets
- personalizes by user profile (bac stream, budget, motivation, city, country)
- returns evidence with each answer

## Files
- `config/policy_rules.yaml`: chatbot behavior and compliance rules
- `docs/data_schema.md`: required fields and validation rules

## Quick Start
1. If `BDD_MCD_Universites.xlsx` exists at workspace root, the server ingests it automatically.
2. If no local dataset exists, the API still starts in clean-start mode with empty data.
3. Apply metadata filters (`country`, `budget_band`, `program`, `level`) before vector search.
4. For each answer, return:
   - short answer
   - rationale
   - evidence snippets (with `video_id` and `recorded_at`)
   - one alternative option

## Retrieval Stack
By default, retrieval runs in semantic mode with one production path:
- Embeddings: `intfloat/multilingual-e5-base`
- Generator LLM: `Qwen/Qwen2.5-0.5B-Instruct`

Notes:
- First run requires internet to download model weights.
- After cache is populated, it can run offline.
- Production default is 0.5B based on benchmark winner for best quality/latency/cost trade-off.

## Run Local API
1. Install dependencies:
   - `c:/Users/abdos/OneDrive/Documents/pfe/.venv/Scripts/python.exe -m pip install -r requirements.txt`

1. Start server:
   - `c:/Users/abdos/OneDrive/Documents/pfe/.venv/Scripts/python.exe -m app.main`
2. Health check:
   - `GET http://127.0.0.1:8000/health`
3. Query chatbot:
   - `POST http://127.0.0.1:8000/chat/query`
4. Run quick evaluation:
   - `POST http://127.0.0.1:8000/chat/evaluate`
   - Optional fixed set: set `EVAL_QUESTIONS_FILE=data/eval_questions_fixed_200.jsonl`
5. Run retrieval-focused BEIR evaluation:
   - `POST http://127.0.0.1:8000/chat/evaluate_beir`
   - Logs are saved in `data/eval_logs/beir_runs.jsonl`
   - Optional mode comparison env var: `BEIR_RETRIEVAL_MODES=dense,sparse,hybrid`
## Build Fixed 200-Question Test Set
- Generate deterministic set:
  - `c:/Users/abdos/OneDrive/Documents/pfe/.venv/Scripts/python.exe -m app.generate_fixed_eval_set`
- Output file:
  - `data/eval_questions_fixed_200.jsonl`

## Retrieval Modes
- `RETRIEVAL_MODE=dense`: semantic retrieval only
- `RETRIEVAL_MODE=sparse`: TF-IDF retrieval + bi-encoder reranking
- `RETRIEVAL_MODE=hybrid` (default): dense + sparse candidate union + bi-encoder fusion
- Optional sparse fusion weight: `SPARSE_BI_WEIGHT=0.9` (range 0..1)
- Optional hybrid fusion weights:
   - `HYBRID_DENSE_WEIGHT=0.2`
   - `HYBRID_SPARSE_WEIGHT=0.1`
   - `HYBRID_BI_WEIGHT=0.7`

### Optional Cross-Encoder Reranking
- `USE_CROSS_ENCODER_RERANKER=1` to enable final reranking
- `CROSS_ENCODER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- `CROSS_ENCODER_TOP_N=8` (rerank only top-N candidates)
- `CROSS_ENCODER_BLEND=0.6` (0..1 blend between base score and cross-encoder score)

## Important
Do not index `Projet ORION -TECH.docx` as retrieval knowledge.
Use it only as product guidance and policy extraction source.
