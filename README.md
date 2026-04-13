# Chatbot

Production API for school recommendations backed by Supabase.

## Runtime
- `app/main.py` starts the HTTP server.
- `app/chatbot.py` formats answers.
- `app/retriever.py` ranks schools and applies profile filters.
- `app/data_loader.py` loads Supabase or local fallback data.

## Start
1. Set `SUPABASE_URL` and `SUPABASE_ANON_KEY` in `.env`.
2. Keep `SUPABASE_STRICT_MODE=1` for production.
3. Run the API with the virtualenv Python:
   - `c:/Users/abdos/OneDrive/Documents/pfe/.venv/Scripts/python.exe -m app.main`

## Endpoints
- `GET /health`
- `POST /chat/query`
- `GET /chat/schools?limit=100`

## Notes
- The app uses Supabase as the primary read source.
- Local fallback mode is still available with `SUPABASE_STRICT_MODE=0`.
- Policy rules live in `config/policy_rules.yaml`.
