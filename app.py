import os
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_media():
    user_api_key = request.headers.get('x-api-key')
    if not user_api_key:
        return jsonify({"error": "Missing personal Google Gemini API Key in Headers"}), 401

    try:
        client = genai.Client(api_key=user_api_key)
        prompt_context = ""
        media_contents = []

        system_instruction = (
            "You are the backend core of VElectronics StudyTool v1. "
            "Analyze the incoming educational material (audio transcription or macro-image text/diagrams). "
            "Provide a high-density academic summary strictly formatted as bullet points. "
            "CRITICAL CONSTRAINT: Each line in your response MUST be strictly under 23 characters long! "
            "Do not exceed 23 characters per line including spaces. Cut or wrap words if needed."
        )

        if 'image' in request.files:
            image_file = request.files['image'].read()
            media_contents.append(
                types.Part.from_bytes(
                    data=image_file,
                    mime_type="image/jpeg",
                )
            )
            prompt_context += " Analyze this macro photograph of a textbook/board. Extract and summarize text/diagrams."

        if 'audio' in request.files:
            audio_file = request.files['audio'].read()
            media_contents.append(
                types.Part.from_bytes(
                    data=audio_file,
                    mime_type="audio/wav",
                )
            )
            prompt_context += " Transcribe, clean, and summarize this captured lecture audio segment."

        if not media_contents:
            return jsonify({"error": "No media payload received"}), 400

        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[prompt_context, *media_contents],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=500,
                temperature=0.3
            )
        )

        result_text = response.text if response.text else "Error: Empty AI Response"
        return jsonify({"text": result_text})

    except Exception as e:
        return jsonify({"error": "Internal AI Gateway Processing Error", "details": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return "VElectronics StudyTool Backend Operating at Python Gold Master Status."

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
