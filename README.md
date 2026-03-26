# Chatbot

This workspace contains mock data and now also supports direct ingestion from `BDD_MCD_Universites.xlsx` when present at workspace root.

## Goal
Build a chatbot that:
- answers from structured school data + transcript snippets
- personalizes by user profile (bac stream, budget, motivation, city, country)
- returns evidence with each answer

## Files
- `data/mock/schools.csv`: mock school facts
- `data/mock/transcripts.jsonl`: mock transcript snippets linked to schools
- `data/mock/eval_questions.jsonl`: evaluation questions with expected constraints
- `config/policy_rules.yaml`: chatbot behavior and compliance rules
- `docs/data_schema.md`: required fields and validation rules

## Quick Start
1. If `BDD_MCD_Universites.xlsx` exists at workspace root, the server ingests it automatically.
2. Otherwise, it falls back to `data/mock/schools.csv` and `data/mock/transcripts.jsonl`.
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
5. Run retrieval-focused BEIR evaluation:
   - `POST http://127.0.0.1:8000/chat/evaluate_beir`
   - Logs are saved in `data/eval_logs/beir_runs.jsonl`

## Important
Do not index `Projet ORION -TECH.docx` as retrieval knowledge.
Use it only as product guidance and policy extraction source.
