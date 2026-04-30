# Chatbot

Production API for school recommendations backed by Supabase.

## Runtime
- `app/main.py` starts the HTTP server.
- `app/recommendation_system.py` is the decision engine: builds the retrieval query, applies profile constraints, ranks schools, and returns structured top schools.
- `app/chatbot.py` is the advisor layer: classifies chat vs recommendation intent and explains recommendation results.
- `app/retriever.py` provides hybrid retrieval, scoring, and cross-encoder reranking.
- `app/data_loader.py` loads Supabase or local fallback data.

## Start
1. Set `SUPABASE_URL` and `SUPABASE_ANON_KEY` in `.env`.
2. Keep `SUPABASE_STRICT_MODE=1` for production.
3. Run the API with the virtualenv Python:
   - `c:/Users/abdos/OneDrive/Documents/pfe/.venv/Scripts/python.exe -m app.main`

## Endpoints
- `GET /health`
- `POST /chat/query`
- `POST /recommendations/query`
- `GET /chat/schools?limit=100`

`/recommendations/query` returns structured ranked schools only. `/chat/query` calls the same recommendation system, then explains the result in natural language.

## Notes
- The app uses Supabase as the primary read source.
- Local fallback mode is still available with `SUPABASE_STRICT_MODE=0`.
- Policy rules live in `config/policy_rules.yaml`.
