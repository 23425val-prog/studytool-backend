import base64
import os
import textwrap

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/"
    f"v1beta/models/{GEMINI_MODEL}:generateContent"
)

SYSTEM_INSTRUCTION = (
    "Summarize the educational material as concise bullet points. "
    "Each output line must contain at most 23 characters, including spaces. "
    "Keep the meaning clear. Use Russian if the input is in Russian."
)


def wrap_bullets(text, width=22):
    """Форматирует ответ: каждая строка — не длиннее width."""
    result = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        # Убираем возможный маркер списка от модели.
        for marker in ("- ", "* ", "• "):
            if line.startswith(marker):
                line = line[len(marker):].strip()
                break

        wrapped = textwrap.wrap(
            line,
            width=width,
            initial_indent="- ",
            subsequent_indent="  ",
            break_long_words=True,
            break_on_hyphens=False,
        )

        result.extend(wrapped or ["-"])

    return "\n".join(result)


@app.route("/process", methods=["POST"])
def process_media():
    api_key = request.headers.get("x-api-key", "").strip()

    if not api_key:
        return jsonify({"error": "Missing x-api-key header"}), 401

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Expected a JSON object"
        }), 400

    text = data.get("text", "").strip()

    if not text and not data.get("image_base64"):
        return jsonify({
            "error": "Provide 'text' or 'image_base64'"
        }), 400

    parts = []

    if text:
        parts.append({"text": text})

    image_base64 = data.get("image_base64")

    if image_base64:
        mime_type = data.get("mime_type", "image/jpeg")

        allowed_types = {
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/gif",
        }

        if mime_type not in allowed_types:
            return jsonify({
                "error": "Unsupported image mime_type"
            }), 400

        try:
            # Проверяем, что строка — корректный Base64.
            decoded = base64.b64decode(
                image_base64,
                validate=True,
            )
        except (ValueError, TypeError):
            return jsonify({
                "error": "Invalid image_base64"
            }), 400

        parts.append({
            "inlineData": {
                "mimeType": mime_type,
                "data": base64.b64encode(decoded).decode("ascii"),
            }
        })

    payload = {
        "systemInstruction": {
            "parts": [
                {"text": SYSTEM_INSTRUCTION}
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 500,
            "temperature": 0.3,
        },
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    try:
        response = requests.post(
            GEMINI_URL,
            json=payload,
            headers=headers,
            timeout=30,
        )
    except requests.RequestException as err:
        app.logger.exception("Gemini request failed")
        return jsonify({
            "error": "Could not reach Gemini API",
            "details": str(err),
        }), 502

    try:
        response_data = response.json()
    except ValueError:
        return jsonify({
            "error": "Gemini returned a non-JSON response",
            "status_code": response.status_code,
            "details": response.text[:1000],
        }), 502

    if not response.ok:
        return jsonify({
            "error": "Gemini API request failed",
            "status_code": response.status_code,
            "details": response_data,
        }), 502

    try:
        generated_text = (
            response_data["candidates"][0]
            ["content"]["parts"][0]["text"]
        )
    except (KeyError, IndexError, TypeError):
        return jsonify({
            "error": "Gemini response did not contain text",
            "details": response_data,
        }), 502

    formatted_text = wrap_bullets(generated_text)

    return jsonify({"text": formatted_text})


@app.route("/", methods=["GET"])
def index():
    return "VElectronics StudyTool backend is running."


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3000"))
    app.run(host="0.0.0.0", port=port)
