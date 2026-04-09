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

from app.beir_eval import run_beir_eval
from app.chatbot import answer_question
from app.data_loader import DataBundle, load_bundle
from app.evaluator import run_eval
from app.models import QueryRequest
from app.supabase_store import fetch_recent_eval_runs, fetch_schools

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

        if parsed.path == "/chat/eval_runs":
            params = parse_qs(parsed.query)
            try:
                limit = int((params.get("limit") or ["20"])[0])
            except ValueError:
                self._send_json(400, {"error": "invalid_limit"})
                return
            try:
                payload = fetch_recent_eval_runs(limit=limit)
                self._send_json(200, payload)
            except Exception as exc:
                self._send_json(500, {"error": "supabase_fetch_failed", "message": str(exc)})
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
            try:
                payload = self._read_json_body()
                req = QueryRequest.from_dict(payload)

                result = answer_question(
                    question=req.question,
                    profile=req.profile,
                    schools=data.schools,
                    transcripts=data.transcripts,
                    top_k=req.top_k,
                    chat_history=req.chat_history,
                )
                self._send_json(200, result.to_dict())
                return
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid_json"})
                return

        if self.path == "/chat/evaluate":
            try:
                summary = run_eval(ROOT_DIR, data.schools, data.transcripts)
                self._send_json(200, summary.to_dict())
            except Exception as exc:
                self._send_json(500, {"error": "eval_failed", "message": str(exc)})
            return

        if self.path == "/chat/evaluate_beir":
            try:
                summary = run_beir_eval(ROOT_DIR, data.schools, data.transcripts)
                self._send_json(200, summary)
            except Exception as exc:
                self._send_json(500, {"error": "beir_eval_failed", "message": str(exc)})
            return

        self._send_json(404, {"error": "not_found"})


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
