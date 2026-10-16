{\rtf1\ansi\ansicpg1251\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import os\
import base64\
from flask import Flask, request, jsonify\
from google import genai\
from google.genai import types\
\
app = Flask(__name__)\
\
# \uc0\u1043 \u1083 \u1072 \u1074 \u1085 \u1099 \u1081  \u1048 \u1048 -\u1096 \u1083 \u1102 \u1079  \u1086 \u1073 \u1088 \u1072 \u1073 \u1086 \u1090 \u1082 \u1080  \u1072 \u1091 \u1076 \u1080 \u1086  \u1080  \u1084 \u1072 \u1082 \u1088 \u1086 -\u1092 \u1086 \u1090 \u1086 \u1075 \u1088 \u1072 \u1092 \u1080 \u1081  (Section 6.1 \u1087 \u1072 \u1090 \u1077 \u1085 \u1090 \u1072 )\
@app.route('/process', methods=['POST'])\
def process_media():\
    # \uc0\u1042 \u1099 \u1090 \u1072 \u1089 \u1082 \u1080 \u1074 \u1072 \u1077 \u1084  \u1087 \u1077 \u1088 \u1089 \u1086 \u1085 \u1072 \u1083 \u1100 \u1085 \u1099 \u1081  API-\u1082 \u1083 \u1102 \u1095  \u1087 \u1086 \u1083 \u1100 \u1079 \u1086 \u1074 \u1072 \u1090 \u1077 \u1083 \u1103  \u1080 \u1079  \u1079 \u1072 \u1075 \u1086 \u1083 \u1086 \u1074 \u1082 \u1086 \u1074  (Section 6.2 \u1087 \u1072 \u1090 \u1077 \u1085 \u1090 \u1072 )\
    user_api_key = request.headers.get('x-api-key')\
    if not user_api_key:\
        return jsonify(\{"error": "Missing personal Google Gemini API Key in Headers"\}), 401\
\
    try:\
        # \uc0\u1048 \u1085 \u1080 \u1094 \u1080 \u1072 \u1083 \u1080 \u1079 \u1080 \u1088 \u1091 \u1077 \u1084  \u1082 \u1083 \u1080 \u1077 \u1085 \u1090  Google AI Studio \u1087 \u1086 \u1076  \u1082 \u1083 \u1102 \u1095  \u1082 \u1086 \u1085 \u1082 \u1088 \u1077 \u1090 \u1085 \u1086 \u1075 \u1086  \u1087 \u1086 \u1083 \u1100 \u1079 \u1086 \u1074 \u1072 \u1090 \u1077 \u1083 \u1103 \
        client = genai.Client(api_key=user_api_key)\
        \
        prompt_context = ""\
        media_contents = []\
\
        # \uc0\u1046 \u1077 \u1089 \u1090 \u1082 \u1080 \u1081  \u1089 \u1080 \u1089 \u1090 \u1077 \u1084 \u1085 \u1099 \u1081  \u1087 \u1088 \u1086 \u1084 \u1087 \u1090  \'97 \u1090 \u1074 \u1086 \u1081  \u1075 \u1083 \u1072 \u1074 \u1085 \u1099 \u1081  \u1082 \u1086 \u1084 \u1084 \u1077 \u1088 \u1095 \u1077 \u1089 \u1082 \u1080 \u1081  Moat\
        system_instruction = (\
            "You are the backend core of VElectronics StudyTool v1. "\
            "Analyze the incoming educational material (audio transcription or macro-image text/diagrams). "\
            "Provide a high-density academic summary strictly formatted as bullet points. "\
            "CRITICAL CONSTRAINT: Each line in your response MUST be strictly under 23 characters long! "\
            "Do not exceed 23 characters per line including spaces. Cut or wrap words if needed."\
        )\
\
        # \uc0\u1045 \u1089 \u1083 \u1080  \u1087 \u1088 \u1080 \u1083 \u1077 \u1090 \u1077 \u1083 \u1072  \u1084 \u1072 \u1082 \u1088 \u1086 -\u1092 \u1086 \u1090 \u1086 \u1075 \u1088 \u1072 \u1092 \u1080 \u1103  (Section 6.5 \u1087 \u1072 \u1090 \u1077 \u1085 \u1090 \u1072 )\
        if 'image' in request.files:\
            image_file = request.files['image'].read()\
            media_contents.append(\
                types.Part.from_bytes(\
                    data=image_file,\
                    mime_type="image/jpeg",\
                )\
            )\
            prompt_context += " Analyze this macro photograph of a textbook/board. Extract and summarize text/diagrams."\
\
        # \uc0\u1045 \u1089 \u1083 \u1080  \u1087 \u1088 \u1080 \u1083 \u1077 \u1090 \u1077 \u1083  transient \u1072 \u1091 \u1076 \u1080 \u1086 -\u1089 \u1077 \u1075 \u1084 \u1077 \u1085 \u1090  \u1080 \u1079  RAM (Section 6.2 \u1087 \u1072 \u1090 \u1077 \u1085 \u1090 \u1072 )\
        if 'audio' in request.files:\
            audio_file = request.files['audio'].read()\
            media_contents.append(\
                types.Part.from_bytes(\
                    data=audio_file,\
                    mime_type="audio/wav",\
                )\
            )\
            prompt_context += " Transcribe, clean, and summarize this captured lecture audio segment."\
\
        if not media_contents:\
            return jsonify(\{"error": "No media payload received"\}), 400\
\
        # \uc0\u1059 \u1088 \u1072 \u1075 \u1072 \u1085 \u1085 \u1086  \u1074 \u1099 \u1089 \u1090 \u1088 \u1077 \u1083 \u1080 \u1074 \u1072 \u1077 \u1084  \u1074  \u1073 \u1077 \u1089 \u1087 \u1083 \u1072 \u1090 \u1085 \u1099 \u1081  \u1096 \u1083 \u1102 \u1079  Gemini 1.5 Flash\
        response = client.models.generate_content(\
            model='gemini-1.5-flash',\
            contents=[prompt_context, *media_contents],\
            config=types.GenerateContentConfig(\
                system_instruction=system_instruction,\
                max_output_tokens=500,\
                temperature=0.3\
            )\
        )\
\
        result_text = response.text if response.text else "Error: Empty AI Response"\
        print("VElectronics Core: Processed summary successfully generated.")\
        \
        return jsonify(\{"text": result_text\})\
\
    except Exception as e:\
        print(f"AI Generation Failed: \{e\}")\
        return jsonify(\{"error": "Internal AI Gateway Processing Error", "details": str(e)\}), 500\
\
# \uc0\u1041 \u1072 \u1079 \u1086 \u1074 \u1099 \u1081  \u1101 \u1085 \u1076 \u1087 \u1086 \u1080 \u1085 \u1090  \u1076 \u1083 \u1103  \u1087 \u1088 \u1086 \u1074 \u1077 \u1088 \u1082 \u1080  \u1072 \u1082 \u1090 \u1080 \u1074 \u1085 \u1086 \u1089 \u1090 \u1080  \u1089 \u1077 \u1088 \u1074 \u1077 \u1088 \u1072  \u1085 \u1072  Render\
@app.route('/', methods=['GET'])\
def index():\
    return "VElectronics StudyTool Backend Operating at Python Gold Master Status."\
\
if __name__ == '__main__':\
    port = int(os.environ.get('PORT', 3000))\
    app.run(host='0.0.0.0', port=port)\
}