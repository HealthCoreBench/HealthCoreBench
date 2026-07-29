"""A minimal OpenAI-compatible HTTP server for integration tests.

Implements ``GET /v1/models`` and ``POST /v1/chat/completions`` using only the stdlib
http.server, so tests exercise the real AsyncOpenAI client over real HTTP. Behaviour is
controlled via a small in-process config object shared with the test.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MockServerState:
    def __init__(self):
        self.served_model = "mock-served-model"
        self.fail_first_n = 0          # return 500 for the first N chat calls
        self._call_count = 0
        self.return_usage = True
        self.answer_letter = "A"       # letter to answer with
        self.lock = threading.Lock()

    def next_should_fail(self) -> bool:
        with self.lock:
            self._call_count += 1
            return self._call_count <= self.fail_first_n


def make_handler(state: MockServerState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # silence

        def do_GET(self):
            if self.path.rstrip("/").endswith("/v1/models"):
                body = {"object": "list", "data": [{"id": state.served_model, "object": "model"}]}
                self._send(200, body)
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            _ = self.rfile.read(length)
            if not self.path.rstrip("/").endswith("/v1/chat/completions"):
                self._send(404, {"error": "not found"})
                return
            if state.next_should_fail():
                self._send(500, {"error": {"message": "mock server error"}})
                return
            usage = ({"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}
                     if state.return_usage else None)
            body = {
                "id": "chatcmpl-mock", "object": "chat.completion", "created": 1710000000,
                "model": state.served_model, "system_fingerprint": "fp_mock",
                "choices": [{"index": 0, "message": {"role": "assistant",
                             "content": f"The answer is {state.answer_letter}."},
                             "finish_reason": "stop"}],
                "usage": usage,
            }
            self._send(200, body)

        def _send(self, code, body):
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


class MockServer:
    def __init__(self, state: MockServerState):
        self.state = state
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()
