import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_media():
    # Вытаскиваем твой легитимный ключ AQ. из заголовков платы
    user_api_key = request.headers.get('x-api-key')
    if not user_api_key:
        return jsonify({"error": "Missing personal Google Gemini API Key in Headers"}), 401

    try:
        # Жесткий системный промпт форматирования (Твой коммерческий Moat на 23 символа)
        system_instruction = (
            "You are the backend core of VElectronics StudyTool v1. "
            "Analyze the incoming educational material. "
            "Provide a high-density academic summary strictly formatted as bullet points. "
            "CRITICAL CONSTRAINT: Each line in your response MUST be strictly under 23 characters long! "
            "Do not exceed 23 characters per line including spaces. Cut or wrap words if needed."
        )

        prompt_context = "Summarize this educational input."
        
        # Конструируем чистый JSON по официальной спецификации Google REST API
        json_payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt_context}
                    ]
                }
            ],
            "systemInstruction": {
                "parts": [
                    {"text": system_instruction}
                ]
            },
            "generationConfig": {
                "maxOutputTokens": 500,
                "temperature": 0.3
            }
        }

        # ЖЕСТКИЙ, НЕИЗМЕНЯЕМЫЙ СТАТИЧНЫЙ URL-АДРЕС GOOGLE (Никаких f-строк и подстановок в хост!)
        url = "https://googleapis.com"
        
        # Передаем ключ в легитимном заголовке x-goog-api-key
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": str(user_api_key).strip()
        }
        
        # Выстреливаем прямым HTTP POST запросом
        response = requests.post(url, json=json_payload, headers=headers)
        
        if response.status_code == 200:
            res_data = response.json()
            try:
                result_text = res_data['candidates'][0]['content']['parts'][0]['text']
                return jsonify({"text": result_text})
            except (KeyError, IndexError, TypeError) as parse_err:
                return jsonify({"error": "Malformed JSON structure from Google", "raw": res_data, "details": str(parse_err)}), 500
        else:
            try:
                error_data = response.json()
                return jsonify({"error": "Google API Reject", "status_code": response.status_code, "details": error_data}), 500
            except Exception:
                return jsonify({"error": "Google API Raw Error", "status_code": response.status_code, "text": response.text}), 500

    except Exception as e:
        return jsonify({"error": "Internal AI Gateway Processing Error", "details": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return "VElectronics StudyTool Backend Operating at REST Gold Master Status."

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
