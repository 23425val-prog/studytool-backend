import hmac
import os
import textwrap

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DEVICE_TOKEN = os.environ.get("DEVICE_TOKEN", "")

MODEL_CHAIN = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
]

RETRYABLE_STATUSES = {
    408, 429, 500, 502, 503, 504,
}

SYSTEM_INSTRUCTION = (
    "Summarize the educational material as concise bullet points. "
    "Each line must be at most 23 characters including spaces."
)


def format_lines(text, width=22):
    """Убирает маркеры списка и переносит строки до width символов."""
    output_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        for marker in ("- ", "* ", "• "):
            if line.startswith(marker):
                line = line[len(marker):].strip()
                break

        if not line:
            continue

        output_lines.extend(
            textwrap.wrap(
                line,
                width=width,
                break_long_words=True,
                break_on_hyphens=False,
            )
        )

    return "\n".join(output_lines)


def call_gemini(payload):
    """
    Пробует модели по очереди.
    При временной ошибке или неизвестной модели переходит к следующей.
    """
    last_status = 503
    last_details = {"message": "No Gemini model responded"}

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    for model in MODEL_CHAIN:
        url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{model}:generateContent"
        )

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=(3, 10),
            )
        except requests.RequestException as err:
            last_status = 503
            last_details = {
                "model": model,
                "message": str(err),
            }
            continue

        if response.ok:
            try:
                data = response.json()
                generated_text = (
                    data["candidates"][0]
                    ["content"]["parts"][0]["text"]
                )

                return {
                    "ok": True,
                    "model": model,
                    "text": generated_text,
                }, None

            except (ValueError, KeyError, IndexError, TypeError) as err:
                return None, {
                    "status": 502,
                    "details": {
                        "model": model,
                        "message": "Gemini returned an unexpected response",
                        "parse_error": str(err),
                    },
                }

        try:
            details = response.json()
        except ValueError:
            details = {"message": response.text[:1000]}

        last_status = response.status_code
        last_details = {
            "model": model,
            "details": details,
        }

        # При ошибке ключа или неправильном запросе смена модели
        # не поможет — сразу возвращаем ошибку.
        if response.status_code not in RETRYABLE_STATUSES and \
           response.status_code != 404:
            return None, {
                "status": response.status_code,
                "details": last_details,
            }

    return None, {
        "status": 503,
        "details": {
            "message": "All configured Gemini models failed",
            "last_error": last_details,
        },
    }


@app.route("/process", methods=["POST"])
def process_media():
    if not GEMINI_API_KEY:
        return jsonify({
            "error": "Server is missing GEMINI_API_KEY"
        }), 503

    if not DEVICE_TOKEN:
        return jsonify({
            "error": "Server is missing DEVICE_TOKEN"
        }), 503

    received_token = request.headers.get("x-device-token", "")

    if not hmac.compare_digest(received_token, DEVICE_TOKEN):
        return jsonify({
            "error": "Invalid device token"
        }), 401

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Expected a JSON object"
        }), 400

    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "error": "JSON field 'text' is required"
        }), 400

    payload = {
        "systemInstruction": {
            "parts": [
                {"text": SYSTEM_INSTRUCTION}
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": text.strip()}
                ],
            }
        ],
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

    formatted_text = format_lines(result["text"])

    return jsonify({
        "text": formatted_text,
        "model": result["model"],
    })


@app.route("/", methods=["GET"])
def index():
    return "StudyTool backend is running."


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3000"))
    app.run(host="0.0.0.0", port=port)
