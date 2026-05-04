# Improve the Chatbot (Step-by-step)

This project has two modes:
- **Chat mode**: friendly advisor conversation.
- **Recommendation mode**: retrieves + ranks schools, then explains the ranking.

## 1) Start with reliability (no empty answers)
- Ensure chat responses never come back empty when the local HF model cannot load.
- Add safe, deterministic replies for common small-talk/out-of-scope questions (ex: **name**, **weather**, **what can you do**).
- Optional: set `ASSISTANT_NAME` in `.env` to control what the bot calls itself.

## 2) Make routing predictable
- Use the request field `mode`:
  - `auto` (default): tries to classify intent.
  - `chat`: force chat mode.
  - `recommendation`: force recommendation mode.
- When debugging routing, start by forcing `mode` to isolate problems.

## 3) Improve answer quality iteratively
- Collect real user queries (good + bad), then fix one failure category at a time:
  1. intent/routing errors
  2. missing-profile handling (ask 1 focused question)
  3. retrieval relevance (top-k contains the right schools)
  4. explanation tone (no database-dump style)

## 4) Add a tiny regression suite
- Add tests for your most common failures (small-talk, language, routing).
- Run: `python -m unittest discover -s tests -p "test_*.py"`

## 5) Keep a benchmark set
See `docs/hybrid_chatbot_todo.md` for a suggested phased plan and evaluation targets.
