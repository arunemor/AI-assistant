#!/usr/bin/env python3
"""
Unified AI Voice Assistant - Complete Application
Combines PyQt5 desktop app + Flask web server with voice capabilities
Features:
- Voice input and audio output
- Document Q&A with PDF upload
- Real-time translation
- Clipboard monitoring
- S3 integration
- Web interface with ChatGPT-style UI
"""

import sys
import os
import json
import threading
import webbrowser
from pathlib import Path
from time import time
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "ai-assistant-docs")
AWS_EXTRACT_BUCKET = os.getenv("AWS_EXTRACT_BUCKET", "ai-assistant-extracts")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1")

# Imports
import boto3
import requests
from flask import Flask, request, send_from_directory, Response, stream_with_context, jsonify
from flask_cors import CORS
from botocore.exceptions import ClientError, NoCredentialsError
from PyQt5.QtWidgets import (
    QApplication, QWidget, QTextEdit, QVBoxLayout, QPushButton, 
    QHBoxLayout, QLineEdit, QMenu, QFileDialog, QLabel, QFrame, 
    QSystemTrayIcon, QAction, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPainter, QColor, QLinearGradient, QBrush, QIcon
from PyPDF2 import PdfReader

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except:
    HAS_TRANSLATOR = False

try:
    import pyperclip
    HAS_PYPERCLIP = True
except:
    HAS_PYPERCLIP = False

# ============================================
# FLASK WEB SERVER
# ============================================

app = Flask(__name__)
CORS(app)

latest_answer = ""

def create_polly_client():
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("⚠️  AWS credentials not configured")
        return None
    try:
        client = boto3.client(
            "polly",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        client.describe_voices(Engine="standard", MaxResults=1)
        print("✅ AWS Polly connected")
        return client
    except Exception as e:
        print(f"⚠️  Polly error: {e}")
        return None

polly_client = create_polly_client()

def get_best_voice():
    if not polly_client:
        return None, None
    try:
        voices = polly_client.describe_voices(Engine="neural")["Voices"]
        for voice in voices:
            if voice["LanguageCode"] == "en-IN":
                return voice["Id"], "neural"
        return voices[0]["Id"], "neural" if voices else (None, None)
    except:
        return None, None

def get_ollama_response(question: str, context: str = "", language: str = "english"):
    global latest_answer
    
    if context:
        prompt = f"""You are an expert document analyst. Answer based ONLY on the document below.

DOCUMENT:
{context}

USER QUESTION: {question}

Provide a detailed answer in {language} with clear points. If the answer isn't in the document, say so."""
    else:
        prompt = f"""Answer this question clearly in {language}:

{question}

Provide a comprehensive answer with examples."""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        if response.status_code == 200:
            answer = response.json().get("response", "").strip()
            answer = answer.replace("\\n", " ").replace("\n", " ")
            answer = " ".join(answer.split())
            latest_answer = answer
            return answer
        return None
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return None

@app.route("/api/ask-stream", methods=["POST"])
def stream_text():
    data = request.get_json()
    question = data.get("question", "").strip()
    context = data.get("context", "")
    language = data.get("language", "english")
    
    if not question:
        return Response("Please enter a question", status=400)

    answer = get_ollama_response(question, context, language)
    if not answer:
        return Response("Failed to get response", status=500)

    def generate():
        words = answer.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            import time
            time.sleep(0.05)
    
    return Response(stream_with_context(generate()), mimetype="text/plain")

@app.route("/api/ask-audio-stream", methods=["POST"])
def stream_audio():
    data = request.get_json()
    question = data.get("question", "").strip()
    
    if not question or not polly_client:
        return Response("Audio not available", status=500)

    answer = latest_answer if latest_answer else get_ollama_response(question)
    if not answer:
        return Response("Failed to generate audio", status=500)

    voice_id, engine = get_best_voice()
    if not voice_id:
        return Response("No voice available", status=500)

    try:
        response = polly_client.synthesize_speech(
            Text=answer,
            OutputFormat="mp3",
            VoiceId=voice_id,
            Engine=engine
        )

        def generate():
            chunk = response["AudioStream"].read(1024)
            while chunk:
                yield chunk
                chunk = response["AudioStream"].read(1024)

        return Response(stream_with_context(generate()), mimetype="audio/mpeg")
    except Exception as e:
        print(f"❌ Audio error: {e}")
        return Response("Audio generation failed", status=500)

@app.route("/")
def serve_frontend():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Voice Assistant</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
    }
    .container {
      background: rgba(255, 255, 255, 0.1);
      backdrop-filter: blur(10px);
      border-radius: 20px;
      padding: 40px;
      max-width: 700px;
      width: 90%;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    h1 { text-align: center; margin-bottom: 30px; font-size: 32px; }
    .chat-box {
      background: rgba(0, 0, 0, 0.3);
      border-radius: 15px;
      height: 400px;
      overflow-y: auto;
      padding: 20px;
      margin-bottom: 20px;
    }
    .message {
      margin: 15px 0;
      padding: 12px 16px;
      border-radius: 12px;
      max-width: 80%;
    }
    .user { background: #667eea; margin-left: auto; text-align: right; }
    .assistant { background: rgba(255, 255, 255, 0.2); }
    .input-wrapper {
      display: flex;
      gap: 10px;
      align-items: center;
    }
    input {
      flex: 1;
      padding: 15px;
      border: none;
      border-radius: 25px;
      background: rgba(255, 255, 255, 0.9);
      font-size: 16px;
    }
    button {
      padding: 15px 25px;
      border: none;
      border-radius: 25px;
      background: #667eea;
      color: white;
      font-weight: 600;
      cursor: pointer;
      transition: 0.3s;
    }
    button:hover { background: #5568d3; transform: scale(1.05); }
    .mic-btn { width: 50px; height: 50px; border-radius: 50%; font-size: 20px; }
    .audio-indicator {
      text-align: center;
      margin-top: 10px;
      font-size: 14px;
      color: #ffd700;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1> AI Voice Assistant</h1>
    <div class="chat-box" id="chatBox">
      <div class="message assistant">Hi! Ask me anything or click the mic to speak.</div>
    </div>
    <div class="input-wrapper">
      <button class="mic-btn" onclick="toggleMic()">🎤</button>
      <input id="questionInput" placeholder="Type your question..." onkeypress="handleEnter(event)">
      <button onclick="sendQuestion()">Send</button>
    </div>
    <div class="audio-indicator" id="audioStatus"></div>
  </div>

  <script>
    let recognition = null;
    
    if ('webkitSpeechRecognition' in window) {
      recognition = new webkitSpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-IN';
      recognition.onresult = (e) => {
        document.getElementById('questionInput').value = e.results[0][0].transcript;
        sendQuestion();
      };
    }

    function toggleMic() {
      if (recognition) recognition.start();
    }

    function handleEnter(e) {
      if (e.key === 'Enter') sendQuestion();
    }

    async function sendQuestion() {
      const input = document.getElementById('questionInput');
      const question = input.value.trim();
      if (!question) return;

      const chatBox = document.getElementById('chatBox');
      chatBox.innerHTML += `<div class="message user">${question}</div>`;
      input.value = '';
      chatBox.scrollTop = chatBox.scrollHeight;

      const assistantDiv = document.createElement('div');
      assistantDiv.className = 'message assistant';
      assistantDiv.textContent = '...';
      chatBox.appendChild(assistantDiv);

      // Stream text
      const response = await fetch('/api/ask-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let text = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        text += decoder.decode(value);
        assistantDiv.textContent = text;
        chatBox.scrollTop = chatBox.scrollHeight;
      }

      // Play audio
      document.getElementById('audioStatus').textContent = '🔊 Playing audio...';
      const audioResp = await fetch('/api/ask-audio-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });
      
      const audioBlob = await audioResp.blob();
      const audio = new Audio(URL.createObjectURL(audioBlob));
      audio.onended = () => document.getElementById('audioStatus').textContent = '';
      audio.play();
    }
  </script>
</body>
</html>"""
    return html_content

def run_flask():
    print("🌐 Starting web server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)

# ============================================
# PYQT5 DESKTOP APPLICATION
# ============================================

class UploadThread(QThread):
    progress = pyqtSignal(str)
    extracted_text = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )

    def run(self):
        try:
            filename = os.path.basename(self.file_path)
            
            # Extract text
            text = ""
            reader = PdfReader(self.file_path)
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n\n"
            
            # Upload to S3
            if AWS_BUCKET_NAME:
                self.s3.upload_file(self.file_path, AWS_BUCKET_NAME, filename)
                self.progress.emit(f"✅ Uploaded: {filename}")
                
                if text and AWS_EXTRACT_BUCKET:
                    key = f"{Path(filename).stem}.txt"
                    self.s3.put_object(Bucket=AWS_EXTRACT_BUCKET, Key=key, Body=text.encode("utf-8"))
                    self.progress.emit(f"✅ Extracted text saved")
            
            self.extracted_text.emit(text)
        except Exception as e:
            self.progress.emit(f"❌ Error: {e}")

class FloatingButton(QWidget):
    clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(80, 80)
        self.dragging = False
        self.offset = None
        
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.width() - 120, screen.height() - 120)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        gradient = QLinearGradient(0, 0, 80, 80)
        gradient.setColorAt(0, QColor(102, 126, 234))
        gradient.setColorAt(1, QColor(118, 75, 162))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, 80, 80)
        
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 20, QFont.Bold))
        painter.drawText(self.rect(), Qt.AlignCenter, "AI")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.offset = event.pos()

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.move(self.mapToParent(event.pos() - self.offset))

    def mouseReleaseEvent(self, event):
        if self.dragging and event.pos() == self.offset:
            self.clicked.emit()
        self.dragging = False

class AIAssistant(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setStyleSheet("background:#1a1d2e; color:white; border-radius:15px;")
        self.resize(700, 600)
        self.pdf_text = ""
        self.last_clip = ""
        self.init_ui()
        
        self.clipboard_timer = QTimer()
        self.clipboard_timer.timeout.connect(self.check_clipboard)

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Header
        header = QFrame()
        header.setStyleSheet("background:#2e3040; border-radius:15px 15px 0 0; padding:15px;")
        header_layout = QHBoxLayout()
        
        title = QLabel("✨ AI Assistant")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(32, 32)
        close_btn.clicked.connect(self.hide)
        close_btn.setStyleSheet("background:#ff5252; border-radius:16px; font-weight:bold;")
        
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(close_btn)
        header.setLayout(header_layout)
        
        # Mode selector
        mode_frame = QFrame()
        mode_layout = QHBoxLayout()
        
        self.web_btn = QPushButton("🌐 Open Web Interface")
        self.web_btn.clicked.connect(self.open_web)
        self.web_btn.setStyleSheet("background:#667eea; padding:10px; border-radius:8px; font-weight:600;")
        
        self.translate_btn = QPushButton("🔄 Auto Translate")
        self.translate_btn.setCheckable(True)
        self.translate_btn.toggled.connect(self.toggle_translate)
        self.translate_btn.setStyleSheet("background:#4caf50; padding:10px; border-radius:8px; font-weight:600;")
        
        mode_layout.addWidget(self.web_btn)
        mode_layout.addWidget(self.translate_btn)
        mode_frame.setLayout(mode_layout)
        
        # Main area
        self.text_area = QTextEdit()
        self.text_area.setStyleSheet("background:#0f1117; padding:15px; border-radius:10px; font-size:14px;")
        self.text_area.setPlaceholderText("Copy text to translate or upload a document...")
        
        # Controls
        control_layout = QHBoxLayout()
        
        self.lang_box = QComboBox()
        self.lang_box.addItems(["english", "hindi", "spanish", "french", "german", "chinese"])
        self.lang_box.setStyleSheet("background:#2e3040; padding:8px; border-radius:8px;")
        
        self.upload_btn = QPushButton("📄 Upload PDF")
        self.upload_btn.clicked.connect(self.upload_pdf)
        self.upload_btn.setStyleSheet("background:#1976d2; padding:10px; border-radius:8px; font-weight:600;")
        
        self.ask_btn = QPushButton("💬 Ask Question")
        self.ask_btn.clicked.connect(self.ask_question)
        self.ask_btn.setStyleSheet("background:#9c27b0; padding:10px; border-radius:8px; font-weight:600;")
        
        control_layout.addWidget(self.lang_box)
        control_layout.addWidget(self.upload_btn)
        control_layout.addWidget(self.ask_btn)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your question...")
        self.input_field.setStyleSheet("background:#2e3040; padding:10px; border-radius:8px;")
        self.input_field.returnPressed.connect(self.ask_question)
        
        layout.addWidget(header)
        layout.addWidget(mode_frame)
        layout.addWidget(self.text_area, 1)
        layout.addLayout(control_layout)
        layout.addWidget(self.input_field)
        
        self.setLayout(layout)

    def open_web(self):
        webbrowser.open("http://localhost:5000")

    def toggle_translate(self, checked):
        if checked:
            self.clipboard_timer.start(500)
            self.text_area.append("\n✅ Auto-translate enabled")
        else:
            self.clipboard_timer.stop()
            self.text_area.append("\n⏸️  Auto-translate paused")

    def check_clipboard(self):
        if not HAS_PYPERCLIP:
            return
        try:
            text = pyperclip.paste().strip()
            if text and text != self.last_clip:
                self.last_clip = text
                self.translate_text(text)
        except:
            pass

    def translate_text(self, text):
        if not HAS_TRANSLATOR:
            self.text_area.append(f"\n📋 Copied: {text[:100]}...")
            return
        
        target = self.lang_box.currentText()
        try:
            translated = GoogleTranslator(source='auto', target=target).translate(text)
            self.text_area.append(f"\n🌍 {target.upper()}: {translated}")
        except Exception as e:
            self.text_area.append(f"\n⚠️  Translation error: {e}")

    def upload_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PDF", "", "PDF Files (*.pdf)")
        if file_path:
            self.text_area.append(f"\n⏳ Uploading {os.path.basename(file_path)}...")
            self.uploader = UploadThread(file_path)
            self.uploader.progress.connect(lambda msg: self.text_area.append(f"\n{msg}"))
            self.uploader.extracted_text.connect(self.on_pdf_loaded)
            self.uploader.start()

    def on_pdf_loaded(self, text):
        self.pdf_text = text
        self.text_area.append("\n✅ PDF loaded! Ask questions about it.")

    def ask_question(self):
        question = self.input_field.text().strip()
        if not question:
            return
        
        self.input_field.clear()
        self.text_area.append(f"\n\n<b>You:</b> {question}")
        
        context = self.pdf_text if self.pdf_text else self.last_clip
        language = self.lang_box.currentText()
        
        try:
            answer = get_ollama_response(question, context, language)
            if answer:
                self.text_area.append(f"<b>AI:</b> {answer}")
            else:
                self.text_area.append("⚠️  Failed to get response")
        except Exception as e:
            self.text_area.append(f"❌ Error: {e}")

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    print("\n" + "="*60)
    print("🚀 UNIFIED AI VOICE ASSISTANT")
    print("="*60)
    
    # Start Flask in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Start PyQt5 GUI
    app_qt = QApplication(sys.argv)
    app_qt.setQuitOnLastWindowClosed(False)
    
    # Create floating button
    floating = FloatingButton()
    floating.show()
    
    # Create main window
    window = AIAssistant()
    floating.clicked.connect(lambda: window.show() if window.isHidden() else window.hide())
    
    # System tray
    tray_icon = QSystemTrayIcon(app_qt)
    tray_menu = QMenu()
    
    show_action = QAction("Show Assistant", app_qt)
    show_action.triggered.connect(window.show)
    tray_menu.addAction(show_action)
    
    web_action = QAction("Open Web Interface", app_qt)
    web_action.triggered.connect(lambda: webbrowser.open("http://localhost:5000"))
    tray_menu.addAction(web_action)
    
    quit_action = QAction("Quit", app_qt)
    quit_action.triggered.connect(app_qt.quit)
    tray_menu.addAction(quit_action)
    
    tray_icon.setContextMenu(tray_menu)
    tray_icon.show()
    
    print("\n✅ Application started successfully!")
    print("🌐 Web interface: http://localhost:5000")
    print("🔵 Look for floating button on screen")
    print("🖱️  System tray icon available")
    print("="*60 + "\n")
    
    sys.exit(app_qt.exec_())

if __name__ == "__main__":
    main()