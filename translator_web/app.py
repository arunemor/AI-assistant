from flask import Flask, render_template, request, jsonify
from deep_translator import GoogleTranslator
import requests

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get('text', '').strip()
    target_lang = data.get('language', 'english').lower()
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return jsonify({'translated': translated})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ollama', methods=['POST'])
def ollama():
    data = request.json
    text = data.get('text', '').strip()
    question = data.get('question', '').strip()
    target_lang = data.get('language', 'english').lower()

    if not text or not question:
        return jsonify({'error': 'Text or question missing'}), 400

    system_prompt = (
        f"You are an AI assistant. ONLY use the following selected text to answer user's question. "
        f"Do not provide any external information. Always reply in {target_lang}.\n\n"
        f"Selected text:\n{text}"
    )

    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": "llama3.2",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                "stream": False
            },
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            reply = data.get('message', {}).get('content', '')
            return jsonify({'reply': reply})
        else:
            return jsonify({'error': f'Ollama returned {response.status_code}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
