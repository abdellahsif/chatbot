from __future__ import annotations

import json
import os
from urllib.parse import parse_qs, urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from app.chatbot import answer_question
from app.data_loader import DataBundle, load_bundle
from app.models import QueryRequest
from app.recommendation_system import recommend_schools
from app.supabase_store import fetch_schools

ROOT_DIR = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(ROOT_DIR / ".env")

DATA: DataBundle | None = None
WEB_FILE = ROOT_DIR / "app" / "web" / "index.html"


def ensure_data_loaded() -> DataBundle:
    global DATA
    if DATA is None:
        DATA = load_bundle(ROOT_DIR)
    return DATA


class ChatbotHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self) -> None:
        # Allow local browser UIs (file://, localhost, VS Code webview/live-server).
        origin = self.headers.get("Origin", "*")
        self.send_header("Access-Control-Allow-Origin", origin if origin else "*")
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            return

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_html(self, status: int, html_text: str) -> None:
        body = html_text.encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        data = ensure_data_loaded()
        parsed = urlparse(self.path)
        if self.path == "/":
            if WEB_FILE.exists():
                self._send_html(200, WEB_FILE.read_text(encoding="utf-8"))
                return
            self._send_html(
                200,
                "<h1>Chatbot API</h1><p>Use POST /chat/query or open the UI at / when app/web/index.html exists.</p>",
            )
            return

        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "schools": len(data.schools),
                    "transcript_chunks": len(data.transcripts),
                    "data_source": getattr(data, "source", "unknown"),
                },
            )
            return

        if parsed.path == "/chat/schools":
            params = parse_qs(parsed.query)
            try:
                limit = int((params.get("limit") or ["100"])[0])
            except ValueError:
                self._send_json(400, {"error": "invalid_limit"})
                return
            try:
                payload = fetch_schools(limit=limit)
                self._send_json(200, payload)
            except Exception as exc:
                self._send_json(500, {"error": "supabase_fetch_failed", "message": str(exc)})
            return

        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        global DATA
        data = ensure_data_loaded()

        if self.path == "/ingest/reload":
            DATA = load_bundle(ROOT_DIR)
            self._send_json(
                200,
                {
                    "status": "reloaded",
                    "schools": len(DATA.schools),
                    "transcript_chunks": len(DATA.transcripts),
                },
            )
            return

        if self.path == "/chat/query":
            self._send_json(410, {"error": "chat_disabled"})
            return

        if self.path == "/recommendations/query":
            try:
                payload = self._read_json_body()
                if not isinstance(payload, dict):
                    payload = {}
                profile_payload = payload.get("profile", {})
                if not isinstance(profile_payload, dict):
                    profile_payload = {}
                if not profile_payload:
                    profile_payload = payload
                profile = QueryRequest.from_dict({"profile": profile_payload}).profile
                try:
                    top_k = int(payload.get("top_k", 5))
                except (TypeError, ValueError):
                    top_k = 5
                top_k = max(1, min(10, top_k))
                user_id = str(payload.get("user_id", payload.get("userId", "")) or "").strip()
                career_profile = payload.get("career_profile")
                if not isinstance(career_profile, dict):
                    career_profile = None
                chat_history = payload.get("chat_history", [])
                if not isinstance(chat_history, list):
                    chat_history = []
                chat_history = [
                    {
                        "role": str(item.get("role", "")).strip().lower(),
                        "content": str(item.get("content", "")).strip(),
                    }
                    for item in chat_history
                    if isinstance(item, dict)
                ]
                debug_mode = str(payload.get("debug", "")).strip().lower() in {"1", "true", "yes", "on"}

                result = recommend_schools(
                    question="",
                    profile=profile,
                    schools=data.schools,
                    transcripts=data.transcripts,
                    top_k=top_k,
                    chat_history=chat_history,
                    user_id=user_id,
                    career_profile=career_profile,
                )
                ranked = result.top_schools or result.ranked_schools
                minimal_results = [
                    self._build_recommendation_payload(item, debug_mode=debug_mode)
                    for item in ranked
                ]
                self._send_json(200, {"results": minimal_results, "count": len(minimal_results)})
                return
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid_json"})
                return

        self._send_json(404, {"error": "not_found"})

    def _build_recommendation_payload(self, item: dict, *, debug_mode: bool) -> dict:
        components = item.get("score_components") or {}
        raw_score = item.get("match_score")
        try:
            normalized_score = round(float(raw_score) / 100.0, 2)
        except (TypeError, ValueError):
            normalized_score = None

        def _match(value: object, threshold: float) -> bool:
            try:
                return float(value) >= threshold
            except (TypeError, ValueError):
                return False

        bac_match = _match(components.get("bac_semantic"), 0.5)
        city_match = _match(components.get("location_match"), 0.6)
        budget_match = _match(components.get("budget_match"), 0.6)
        motivation_match = _match(components.get("motivation_match"), 0.6)
        domain_match = any(
            [
                _match(components.get("career_domain_match"), 0.5),
                _match(components.get("domain_alignment"), 0.5),
                _match(components.get("program_match"), 0.4),
                _match(components.get("bac_semantic"), 0.5),
            ]
        )

        reasons: list[str] = []
        if bac_match:
            reasons.append("Matches your bac stream")
        if city_match:
            reasons.append("Available in your preferred city")
        if budget_match:
            reasons.append("Fits your budget band")
        if domain_match:
            reasons.append("Aligned with your domain")
        if motivation_match:
            reasons.append("Aligned with your motivation")

        payload = {
            "school_id": item.get("school_id"),
            "school_name": item.get("name"),
            "match_score": normalized_score,
            "match_reasons": reasons,
            "matched_fields": {
                "city": city_match,
                "budget": budget_match,
                "domain": domain_match,
                "motivation": motivation_match,
            },
        }

        if debug_mode:
            payload["score_breakdown"] = {
                "profile_priority": round(float(components.get("profile_priority", 0.0)), 2),
                "career_profile_boost": round(float(components.get("career_domain_match", 0.0)), 2),
                "final_score": normalized_score,
            }

        return payload


def run_server(host: str = "127.0.0.1", port: int = 3001) -> None:
    env_port = os.getenv("APP_PORT") or os.getenv("PORT")
    if env_port:
        try:
            port = int(env_port)
        except ValueError:
            pass
    ensure_data_loaded()
    server = ThreadingHTTPServer((host, port), ChatbotHandler)
    print(f"Server running on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
