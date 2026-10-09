"""Development mock of llama-server (OpenAI-compatible) for LocalGPT E2E checks.

Streams a fixed sample reply over SSE on 127.0.0.1:8080. Stdlib only.

    python fake_llama.py
"""
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SAMPLE = (
    "LocalGPT streaming works. Here is **bold** text, `inline code`, "
    "and a short list:\n\n- first item\n- second item\n\n```python\nprint('hello')\n```"
)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json({"status": "ok", "mock": True})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except ValueError:
            payload = {}

        if self.path != "/v1/chat/completions":
            self._json({"error": "not found"}, status=404)
            return

        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for token in SAMPLE.split(" "):
                chunk = {"choices": [{"delta": {"content": token + " "}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.02)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        self._json(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": SAMPLE}}
                ]
            }
        )


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8080), Handler)
    print("mock llama-server on 127.0.0.1:8080")
    server.serve_forever()
