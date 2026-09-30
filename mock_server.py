#!/usr/bin/env python3
"""
Mock analysis-engine — Person A's testing aid, not Person B's real work.

Implements the three tools exactly to the team's tool contract, so the voice
side (agents/talk-to-your-data.jsonc) can be built and tested end to end
today, without waiting on the real pandas/scikit-learn service. Swap
BASE_URL over to that service once it exists; nothing here changes on the
voice side when you do, because the JSON shape is identical.

Standard library only — same rule the voice-agent-starter itself follows, so
there's nothing to install to run this.

predict() below uses a made-up linear formula loosely shaped like the real
dataset's known effects (smoker >> bmi > age), only so numbers respond
sensibly to different inputs while testing conversation flow. It is NOT a
trained model and must not be mistaken for one.

Run:
    python3 mock_server.py            # serves on :8000

Then tunnel it (ngrok http 8000, or cloudflared tunnel --url http://localhost:8000)
and put the HTTPS URL in .env as BASE_URL for the voice agent.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

TARGET_DESCRIPTION = "Annual medical insurance cost, in US dollars"

FEATURES = {
    "age":      {"required": True,  "type": "number"},
    "bmi":      {"required": True,  "type": "number"},
    "smoker":   {"required": True,  "type": "enum", "values": ["yes", "no"]},
    "sex":      {"required": False, "type": "enum", "values": ["male", "female"]},
    "children": {"required": False, "type": "integer"},
    "region":   {"required": False, "type": "enum",
                 "values": ["northeast", "northwest", "southeast", "southwest"]},
}

# Placeholder "most common value" defaults. The real service computes these
# from the actual training data; these are just plausible stand-ins.
DEFAULTS = {"sex": "female", "children": 0, "region": "southeast"}

MOCK_TEST_MAE = 4210.53

TOP_DRIVERS = [
    {"feature": "smoker", "effect": "Smokers pay about $23,000 more per year on average"},
    {"feature": "bmi",    "effect": "Each extra BMI point adds about $340/year"},
    {"feature": "age",    "effect": "Each extra year adds about $260/year"},
]


def get_dataset_context(_body):
    return 200, {
        "ok": True,
        "target": "charges",
        "target_description": TARGET_DESCRIPTION,
        "features": FEATURES,
        "message": (
            "This dataset has 1,338 people — age, sex, BMI, children, smoking "
            "status, region — used to predict their annual insurance charges "
            "in dollars."
        ),
    }


def run_analysis(_body):
    return 200, {
        "ok": True,
        "test_mae": MOCK_TEST_MAE,
        "top_drivers": TOP_DRIVERS,
        "message": (
            "Smoking status is by far the strongest driver, followed by BMI "
            "and age. On held-out data, predictions were off by about "
            "$4,200 on average."
        ),
    }


def predict(body):
    for field in ("age", "bmi", "smoker"):
        if body.get(field) is None:
            return 200, {
                "ok": False,
                "reason": "missing_required_field",
                "field": field,
                "message": f"Ask the caller for {field} — it's required for a prediction.",
            }

    try:
        age = float(body["age"])
        bmi = float(body["bmi"])
    except (TypeError, ValueError):
        return 200, {
            "ok": False,
            "reason": "invalid_value",
            "message": "age and bmi need to be numbers — ask the caller to clarify.",
        }

    smoker = body["smoker"]
    if smoker not in ("yes", "no"):
        return 200, {
            "ok": False,
            "reason": "invalid_value",
            "field": "smoker",
            "message": "smoker must be yes or no — ask the caller to confirm.",
        }

    assumptions_used = []
    children = body.get("children")
    if children is None:
        children = DEFAULTS["children"]
        assumptions_used.append(f"children defaulted to {children} (most common in this data)")

    region = body.get("region")
    if region is None:
        region = DEFAULTS["region"]
        assumptions_used.append(f"region defaulted to {region} (most common in this data)")

    if body.get("sex") is None:
        assumptions_used.append(f"sex defaulted to {DEFAULTS['sex']} (most common in this data)")

    # Toy formula, not a trained model — see module docstring.
    base = 2500
    smoker_effect = 23000 if smoker == "yes" else 0
    estimate = base + age * 260 + bmi * 340 + smoker_effect + children * 500

    message = (
        f"For a {int(age)}-year-old {'smoker' if smoker == 'yes' else 'non-smoker'} "
        f"with a BMI of {bmi:g}, the estimated annual charge is about "
        f"${estimate:,.0f}. On held-out test data, predictions differed from "
        f"actual charges by about ${MOCK_TEST_MAE:,.0f} on average — individual "
        f"errors can be larger."
    )

    return 200, {
        "ok": True,
        "predicted_charge": round(estimate, 2),
        "assumptions_used": assumptions_used,
        "message": message,
    }


ROUTES = {
    "/tools/get_dataset_context": get_dataset_context,
    "/tools/run_analysis": run_analysis,
    "/tools/predict": predict,
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

        status, payload = handler(body)
        self._send(status, payload)

    def _send(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print(f"{self.command} {self.path} -> {args[-1] if args else ''}")


if __name__ == "__main__":
    port = 8000
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Mock analysis engine on http://localhost:{port}")
    print("Routes:", ", ".join(ROUTES))
    server.serve_forever()
