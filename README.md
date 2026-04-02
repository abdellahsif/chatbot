# Chatbot

This workspace is configured to use Supabase as the primary data source.

## Goal
Build a chatbot that:
- answers from structured school data + transcript snippets
- personalizes by user profile (bac stream, budget, motivation, city, country)
- returns evidence with each answer

## Files
- `config/policy_rules.yaml`: chatbot behavior and compliance rules
- `docs/data_schema.md`: required fields and validation rules
- `docs/metrics_todo.md`: metric targets and execution checklist before production deployment

## Quick Start
1. Configure Supabase environment variables before starting the API.
2. By default, `SUPABASE_STRICT_MODE=1`, so startup fails if DB schools are not loaded.
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
   - `GET http://127.0.0.1:3001/health`
3. Query chatbot:
   - `POST http://127.0.0.1:3001/chat/query`
4. Run quick evaluation:
   - `POST http://127.0.0.1:3001/chat/evaluate`
   - Optional fixed set: set `EVAL_QUESTIONS_FILE=data/eval_questions_fixed_200.jsonl`
5. Run retrieval-focused BEIR evaluation:
   - `POST http://127.0.0.1:3001/chat/evaluate_beir`
   - Logs are saved in `data/eval_logs/beir_runs.jsonl`
   - Optional mode comparison env var: `BEIR_RETRIEVAL_MODES=dense,sparse,hybrid`

## Supabase (Read-Only)
Use Supabase only as a read source for existing data (no insert/update/delete from this app).

Quick setup with `.env`:
1. Copy `.env.example` to `.env` (already scaffolded in this workspace).
2. Fill `SUPABASE_URL` and `SUPABASE_ANON_KEY`.
3. Start the API normally; `.env` is loaded automatically at startup.

Runtime behavior:
- Chatbot startup uses Supabase `schools` as the source of truth for `schools` and retrieval chunks.
- Strict mode is enabled by default (`SUPABASE_STRICT_MODE=1`).
- Optional dev fallback exists only when explicitly setting `SUPABASE_STRICT_MODE=0`.

Environment variables:
- `SUPABASE_URL=https://<project-ref>.supabase.co`
- `SUPABASE_ANON_KEY=<anon-or-readonly-key>`
- `SUPABASE_STRICT_MODE=1` (optional, default strict DB-only mode)
- `APP_PORT=3001` (optional, API port; defaults to 3001)
- `GENERATOR_MODEL=Qwen/Qwen2.5-0.5B-Instruct` (optional, choose answer generation model)
- `USE_CROSS_ENCODER_RERANKER=1` (optional, enable/disable reranker)
- `CROSS_ENCODER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (optional, choose reranker model)

Advanced optional overrides (only if you need custom table/query behavior):
- `SUPABASE_TABLE_SCHOOLS=schools`
- `SUPABASE_SCHOOLS_LIMIT=500`
- `SUPABASE_SCHOOLS_ORDER_COLUMN=created_at`
- `SUPABASE_SCHOOLS_SELECT=*`
- `SUPABASE_TIMEOUT_SECONDS=10`
- `SUPABASE_EVAL_TABLE=eval_runs`
- `SUPABASE_EVAL_PAYLOAD_COLUMN=payload`
- `SUPABASE_EVAL_ORDER_COLUMN=created_at`

Read endpoint:
- `GET /chat/schools?limit=100`
- `GET /chat/eval_runs?limit=20`

Notes:
- The API fetches rows from Supabase REST and returns payloads from the configured column.
- If Supabase env vars are missing, endpoint returns `enabled=false` with no data.
- For production, prefer a key that is read-only through RLS policies (SELECT allowed, INSERT/UPDATE/DELETE denied).
- Evaluation still writes local logs under `data/eval_logs/`.
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
