#!/usr/bin/env python3
"""
Real analysis-engine server — wraps Person B's functions from
analysis_engine.py (copied verbatim from the notebook) behind the same
three HTTP routes mock_server.py used, so BASE_URL can point here instead
without any change on the voice side.

This is a thin pass-through: whatever analysis_engine.py's functions
return is what goes over the wire, unmodified. That's deliberate — this
server exists to test his real code against the real contract, bugs and
all, not to paper over them.

Fits the model once at startup (not on first conversation call) so every
caller reuses the same pipeline — closer to how a real deployment should
behave than "fit on first turn of every conversation."

Needs pandas / scikit-learn installed. Run:
    python3 real_server.py            # serves on :8000
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import analysis_engine as engine

DATASET_ID = "med_insurance"


def handle_get_dataset_context(_body):
    return 200, engine.get_dataset_context(DATASET_ID)


def handle_run_analysis(_body):
    return 200, engine.run_analysis(dataset_id=DATASET_ID)


def handle_predict(body):
    return 200, engine.predict_targetval(dataset_id=DATASET_ID, supplied_inputs=body)


ROUTES = {
    "/tools/get_dataset_context": handle_get_dataset_context,
    "/tools/run_analysis": handle_run_analysis,
    "/tools/predict": handle_predict,
}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        handler = ROUTES.get(self.path)
        if handler is None:
            self._send(404, {"ok": False, "reason": "not_found", "message": "No such tool."})
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"ok": False, "reason": "bad_json", "message": "Body wasn't valid JSON."})
            return

        try:
            status, payload = handler(body)
        except Exception as exc:  # surfaced as a value, not a crash, per the contract's own rule
            self._send(500, {"ok": False, "reason": "server_error", "message": str(exc)})
            return

        self._send(status, payload)

    def _send(self, status, payload):
        data = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print(f"{self.command} {self.path} -> {args[-1] if args else ''}")


if __name__ == "__main__":
    print("Fitting the model once at startup...")
    startup_result = engine.run_analysis(dataset_id=DATASET_ID)
    print("run_analysis at startup:", "ok" if startup_result.get("ok") else startup_result)

    port = 8000
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Real analysis engine on http://localhost:{port}")
    print("Routes:", ", ".join(ROUTES))
    server.serve_forever()
