import hashlib
import hmac
import os
import re
import secrets
import textwrap
import threading

import psycopg2
import requests
from flask import Flask, jsonify, request
from psycopg2.extras import RealDictCursor

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ENROLLMENT_SECRET = os.environ.get("ENROLLMENT_SECRET", "0791554f1186185f6bc11b97e0f744092b3336ad9586cdb058a4296a7dd32547")
DATABASE_URL = os.environ.get("DATABASE_URL", "https://studytool-backend.onrender.com")
MAX_GEMINI_CONCURRENCY = int(os.environ.get("MAX_GEMINI_CONCURRENCY", "20"))
GEMINI_SERVICE_TIER = os.environ.get("GEMINI_SERVICE_TIER", "standard").lower()

MODEL_CHAIN = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
]

RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}
SYSTEM_INSTRUCTION = (
    "Summarize this educational material in English as concise plain-text lines. "
    "Each line must be at most 22 "
    "characters including spaces. Do not start lines with bullet symbols."
)

_db_init_lock = threading.Lock()
_db_initialized = False
_gemini_slots = threading.BoundedSemaphore(MAX_GEMINI_CONCURRENCY)


def db_connect():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured")
    return psycopg2.connect(DATABASE_URL, connect_timeout=5, sslmode="require")


def ensure_db():
    global _db_initialized

    if _db_initialized:
        return

    with _db_init_lock:
        if _db_initialized:
            return

        with db_connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS devices (
                        device_id TEXT PRIMARY KEY,
                        token_hash TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )

        _db_initialized = True


def hash_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def valid_device_id(device_id):
    return (
        isinstance(device_id, str)
        and 1 <= len(device_id) <= 32
        and re.fullmatch(r"[A-Fa-f0-9]+", device_id) is not None
    )


def is_authorized_device(device_id, token):
    if not valid_device_id(device_id) or not token:
        return False

    ensure_db()

    with db_connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT token_hash FROM devices WHERE device_id = %s",
                (device_id,),
            )
            row = cursor.fetchone()

    return row is not None and hmac.compare_digest(
        row[0], hash_token(token)
    )


def format_lines(text, width=22):
    result = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        for marker in ("- ", "* ", "• "):
            if line.startswith(marker):
                line = line[len(marker):].strip()
                break

        if line:
            result.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
            )

    return "\n".join(result)


def call_gemini(payload):
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    last_error = None

    for model in MODEL_CHAIN:
        url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{model}:generateContent"
        )
        model_payload = dict(payload)

        if GEMINI_SERVICE_TIER in {"standard", "priority"}:
            model_payload["service_tier"] = GEMINI_SERVICE_TIER

        try:
            response = requests.post(
                url,
                json=model_payload,
                headers=headers,
                timeout=(3, 12),
            )
        except requests.RequestException as error:
            last_error = {
                "model": model,
                "message": str(error),
            }
            continue

        if response.ok:
            try:
                data = response.json()
                text = (
                    data["candidates"][0]
                    ["content"]["parts"][0]["text"]
                )
                return {"model": model, "text": text}, None
            except (ValueError, KeyError, IndexError, TypeError) as error:
                return None, {
                    "status": 502,
                    "details": {
                        "model": model,
                        "message": "Gemini response contained no text",
                        "parse_error": str(error),
                    },
                }

        try:
            error_details = response.json()
        except ValueError:
            error_details = {"message": response.text[:1000]}

        last_error = {
            "model": model,
            "upstream_status": response.status_code,
            "details": error_details,
        }

        # Switch model for temporary capacity/rate errors or unknown model.
        if response.status_code not in RETRYABLE_STATUSES | {404}:
            return None, {
                "status": 502,
                "details": last_error,
            }

    return None, {
        "status": 503,
        "details": {
            "message": "All configured Gemini models failed",
            "last_error": last_error,
        },
    }


@app.get("/")
def index():
    return "StudyTool backend is running."


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/register")
def register_device():
    if not ENROLLMENT_SECRET or not DATABASE_URL:
        return jsonify({
            "error": "Server registration is not configured"
        }), 503

    supplied_secret = request.headers.get("x-enrollment-secret", "")
    if not hmac.compare_digest(supplied_secret, ENROLLMENT_SECRET):
        return jsonify({"error": "Invalid enrollment credential"}), 401

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON object"}), 400

    device_id = data.get("device_id")
    if not valid_device_id(device_id):
        return jsonify({"error": "Invalid device_id"}), 400

    token = secrets.token_urlsafe(32)
    token_hash = hash_token(token)

    try:
        ensure_db()
        with db_connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO devices (device_id, token_hash)
                    VALUES (%s, %s)
                    ON CONFLICT (device_id)
                    DO UPDATE SET token_hash = EXCLUDED.token_hash,
                                  created_at = NOW()
                    """,
                    (device_id, token_hash),
                )
    except Exception:
        app.logger.exception("Device registration failed")
        return jsonify({"error": "Device registration failed"}), 503

    # Return the plaintext token only once. The database stores its hash.
    return jsonify({"device_token": token})


@app.post("/process")
def process_media():
    if not GEMINI_API_KEY:
        return jsonify({"error": "Server is missing GEMINI_API_KEY"}), 503
    if not DATABASE_URL:
        return jsonify({"error": "Server is missing DATABASE_URL"}), 503

    device_id = request.headers.get("x-device-id", "")
    device_token = request.headers.get("x-device-token", "")

    try:
        authorized = is_authorized_device(device_id, device_token)
    except Exception:
        app.logger.exception("Device authorization failed")
        return jsonify({"error": "Device authorization unavailable"}), 503

    if not authorized:
        return jsonify({"error": "Invalid device credentials"}), 401

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON object"}), 400

    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "JSON field 'text' is required"}), 400

    # Do not build an unbounded server-side queue. If all slots are occupied,
    # tell the device to keep its chunk and retry later.
    if not _gemini_slots.acquire(blocking=False):
        response = jsonify({
            "error": "Server is at its configured parallel capacity",
            "retry_after_seconds": 10,
        })
        response.status_code = 429
        response.headers["Retry-After"] = "10"
        return response

    try:
        payload = {
            "systemInstruction": {
                "parts": [{"text": SYSTEM_INSTRUCTION}]
            },
            "contents": [{
                "role": "user",
                "parts": [{"text": text.strip()}],
            }],
            "generationConfig": {
                "maxOutputTokens": 500,
                "temperature": 0.3,
            },
        }

        result, error = call_gemini(payload)

        if error:
            return jsonify({
                "error": "Gemini request failed",
                "details": error["details"],
            }), error["status"]

        return jsonify({
            "text": format_lines(result["text"]),
            "model": result["model"],
        })
    finally:
        _gemini_slots.release()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3000"))
    app.run(host="0.0.0.0", port=port)
