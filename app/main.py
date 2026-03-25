from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from app.chatbot import answer_question
from app.data_loader import DataBundle, load_bundle
from app.evaluator import run_eval
from app.models import QueryRequest

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA: DataBundle | None = None


def ensure_data_loaded() -> DataBundle:
    global DATA
    if DATA is None:
        DATA = load_bundle(ROOT_DIR)
    return DATA


class ChatbotHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        data = ensure_data_loaded()
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "schools": len(data.schools),
                    "transcript_chunks": len(data.transcripts),
                },
            )
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
                if not req.question:
                    self._send_json(400, {"error": "question is required"})
                    return

                result = answer_question(
                    question=req.question,
                    profile=req.profile,
                    schools=data.schools,
                    transcripts=data.transcripts,
                    top_k=req.top_k,
                )
                self._send_json(200, result.to_dict())
                return
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid_json"})
                return

        if self.path == "/chat/evaluate":
            summary = run_eval(ROOT_DIR, data.schools, data.transcripts)
            self._send_json(200, summary.to_dict())
            return

        self._send_json(404, {"error": "not_found"})


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    ensure_data_loaded()
    server = ThreadingHTTPServer((host, port), ChatbotHandler)
    print(f"Server running on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
