#!/usr/bin/env python3
"""
swipe_ai_assistant.py
Single-file PyQt5 app implementing:
- Floating gradient swipe button (left/right)
- Translator/clipboard QA popup (left)
- Document (PDF) upload + QA popup (right)
- S3 upload + duplicate detection + PDF text extraction (PyPDF2)
- Ollama integration for Q&A with formatted, point-wise answers

Requirements:
- PyQt5
- boto3
- python-dotenv
- requests
- PyPDF2
- deep_translator (optional, for translation)
- fuzzywuzzy (optional, for similarity checks)
"""

import sys
import os
import json
import traceback
from functools import partial
from pathlib import Path
from time import time

from dotenv import load_dotenv
load_dotenv()

# --- Config from .env ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")            # bucket for original PDFs
AWS_EXTRACT_BUCKET = os.getenv("AWS_EXTRACT_BUCKET")      # bucket for extracted text
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Fallbacks and sanity
if not AWS_BUCKET_NAME:
    print("Warning: AWS_BUCKET_NAME not set in .env — S3 upload will fail until configured.")

# --- Imports for functionality ---
import boto3
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QTextEdit, QComboBox,
    QVBoxLayout, QPushButton, QHBoxLayout, QCheckBox,
    QLineEdit, QMenu, QFileDialog, QLabel, QFrame
)
from PyQt5.QtCore import Qt, QTimer, QPoint, QThread, pyqtSignal, QPropertyAnimation, QRect
from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QLinearGradient, QBrush

# Optional libs
try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except Exception:
    HAS_TRANSLATOR = False

try:
    from fuzzywuzzy import fuzz
    HAS_FUZZY = True
except Exception:
    HAS_FUZZY = False

from PyPDF2 import PdfReader

# ----------------- Utility functions -----------------
def format_ollama_answer(raw_text: str) -> str:
    """
    Make assistant answers more instructive:
    - Break into numbered bullet points where possible
    - Add Example and Reference blocks if present keywords found
    """
    if not raw_text:
        return "⚠️ No response from model."

    text = raw_text.strip()

    # Try to split on common delimiters into lines
    lines = []
    if "\n\n" in text:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    else:
        parts = [p.strip() for p in text.split("\n") if p.strip()]

    # If many short lines, use bullets; else try to split sentences.
    if len(parts) >= 3:
        lines = parts
    else:
        # naive sentence split
        import re
        sentences = re.split(r'(?<=[.!?]) +', text)
        lines = [s.strip() for s in sentences if s.strip()]

    # Build numbered points
    numbered = []
    for i, ln in enumerate(lines, start=1):
        numbered.append(f"{i}. {ln}")

    out = "\n\n".join(numbered)

    # Add Example / Reference cues if present words
    if "example" in text.lower() or "e.g." in text.lower():
        out += "\n\n📘 Example:\n" + "Refer to the example given above."

    # Basic "source" hint
    out += "\n\n🔎 Source: Answer generated using the uploaded text and Ollama model."

    return out

# ----------------- S3 Upload Thread -----------------
class UploadThread(QThread):
    progress = pyqtSignal(str)
    extracted_text_signal = pyqtSignal(str, str)  # (extracted_text, s3_key)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = str(file_path)
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )

    def run(self):
        try:
            filename = os.path.basename(self.file_path)
            # Check duplicate: list objects keys in bucket
            already = False
            try:
                resp = self.s3.list_objects_v2(Bucket=AWS_BUCKET_NAME, Prefix=filename)
                if 'Contents' in resp:
                    for obj in resp['Contents']:
                        if obj['Key'] == filename:
                            already = True
                            break
            except Exception:
                # silent fallback; still attempt upload
                pass

            if already:
                self.progress.emit(f"⚠️ File '{filename}' already exists in {AWS_BUCKET_NAME}. Skipping upload.")
            else:
                self.s3.upload_file(self.file_path, AWS_BUCKET_NAME, filename)
                self.progress.emit(f"✅ Uploaded '{filename}' → s3://{AWS_BUCKET_NAME}/{filename}")

            # Extract text from PDF
            text = ""
            try:
                reader = PdfReader(self.file_path)
                for p in reader.pages:
                    # page.extract_text() may return None
                    page_text = p.extract_text() or ""
                    text += page_text + "\n\n"
            except Exception as e:
                self.progress.emit(f"⚠️ PDF extraction failed: {e}")
                text = ""

            # Store extracted .txt in extract bucket
            if text and AWS_EXTRACT_BUCKET:
                key = f"{Path(filename).stem}.txt"
                try:
                    self.s3.put_object(Bucket=AWS_EXTRACT_BUCKET, Key=key, Body=text.encode("utf-8"))
                    self.progress.emit(f"✅ Extracted text stored → s3://{AWS_EXTRACT_BUCKET}/{key}")
                    self.extracted_text_signal.emit(text, key)
                except Exception as e:
                    self.progress.emit(f"⚠️ Failed to store extracted text: {e}")
                    self.extracted_text_signal.emit(text, "")  # still provide extracted text
            else:
                # send text back even if not stored
                self.extracted_text_signal.emit(text, "")

        except Exception as e:
            tb = traceback.format_exc()
            self.progress.emit(f"⚠️ UploadThread error: {e}\n{tb}")

# ----------------- Floating Swipe Button -----------------
class FloatingSwipeButton(QWidget):
    openTranslator = pyqtSignal()
    openDocQA = pyqtSignal()

    def __init__(self, parent=None, diameter=70):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.diameter = diameter
        self.setFixedSize(diameter, diameter)
        self.icon_text = "AI"
        self.old_pos = None
        self.dragging = False
        self.start_pos = None
        self.animation = None
        self.setup_ui()
        self.show()

    def setup_ui(self):
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setToolTip("Swipe left for Translator · Swipe right for Document Q&A · Click to toggle")
        # Starting position bottom-right-ish
        screen_geo = QApplication.primaryScreen().availableGeometry()
        start_x = max(50, screen_geo.width() - 150)
        start_y = max(50, screen_geo.height() - 200)
        self.move(start_x, start_y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Gradient fill
        grad = QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0.0, QColor(106, 17, 203))  # purple
        grad.setColorAt(1.0, QColor(33, 150, 243))  # blue
        brush = QBrush(grad)
        painter.setBrush(brush)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, self.diameter, self.diameter)

        # Draw arrow hints
        painter.setPen(QPen(QColor(255,255,255,180), 2))
        # left hint
        painter.drawLine(self.width()*0.1, self.height()*0.5, self.width()*0.35, self.height()*0.5)
        painter.drawLine(self.width()*0.15, self.height()*0.45, self.width()*0.1, self.height()*0.5)
        painter.drawLine(self.width()*0.15, self.height()*0.55, self.width()*0.1, self.height()*0.5)
        # right hint
        painter.drawLine(self.width()*0.9, self.height()*0.5, self.width()*0.65, self.height()*0.5)
        painter.drawLine(self.width()*0.85, self.height()*0.45, self.width()*0.9, self.height()*0.5)
        painter.drawLine(self.width()*0.85, self.height()*0.55, self.width()*0.9, self.height()*0.5)

        # Icon text
        painter.setPen(QPen(QColor(255,255,255)))
        font = QFont("Arial", int(self.diameter/3), QFont.Bold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, self.icon_text)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.old_pos = event.globalPos()
            self.start_pos = event.globalPos()
            self.dragging = True

    def mouseMoveEvent(self, event):
        if not self.dragging or self.old_pos is None:
            return
        delta = event.globalPos() - self.old_pos
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.old_pos = event.globalPos()

    def mouseReleaseEvent(self, event):
        if not self.dragging:
            return
        self.dragging = False
        end_pos = event.globalPos()
        dx = end_pos.x() - self.start_pos.x()
        # threshold
        threshold = 80
        if dx < -threshold:
            # swipe left
            self._animate_feedback(direction="left")
            self.openTranslator.emit()
        elif dx > threshold:
            # swipe right
            self._animate_feedback(direction="right")
            self.openDocQA.emit()
        else:
            # small drag -> click = toggle translator for convenience
            self._animate_feedback(direction="tap")
            self.openTranslator.emit()

        self.old_pos = None
        self.start_pos = None

    def _animate_feedback(self, direction="tap"):
        # simple scale animation
        start_rect = self.geometry()
        if direction == "left":
            end_rect = QRect(self.x()-30, self.y(), self.width(), self.height())
        elif direction == "right":
            end_rect = QRect(self.x()+30, self.y(), self.width(), self.height())
        else:
            end_rect = QRect(self.x(), self.y(), self.width(), self.height())

        anim = QPropertyAnimation(self, b"geometry")
        anim.setDuration(180)
        anim.setStartValue(start_rect)
        anim.setEndValue(end_rect)
        anim.setEasingCurve(Qt.EaseOutCubic)
        anim.start()
        # ensure object stays referenced until finished
        self.animation = anim

# ----------------- Translator / Clipboard Popup -----------------
class TranslatorPopup(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setStyleSheet("background:#0f1117; color:white; border:2px solid #6a11cb; border-radius:12px;")
        self.setFixedSize(420, 420)
        self.old_pos = None
        self.last_clip = ""
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_clipboard)
        self.timer.start(600)

    def init_ui(self):
        # Header buttons
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(26, 26)
        close_btn.setStyleSheet("background:#ff5252; color:white; border:none; border-radius:6px;")
        close_btn.clicked.connect(self.hide)

        self.mode_label = QLabel("Translator Mode")
        self.mode_label.setStyleSheet("color:#cbe6ff; font-weight:600;")

        header = QHBoxLayout()
        header.addWidget(self.mode_label)
        header.addStretch()
        header.addWidget(close_btn)

        # Language select
        self.lang_box = QComboBox()
        self.lang_box.addItems(["english","hindi","spanish","french","german","chinese","arabic"])
        self.lang_box.setStyleSheet("background:#6a11cb; color:white; padding:6px; border-radius:6px;")

        # Ollama toggle
        self.ollama_checkbox = QCheckBox("Use Ollama for Q&A")
        self.ollama_checkbox.setStyleSheet("color:white;")
        self.ollama_checkbox.stateChanged.connect(self.on_ollama_toggle)

        # Text display
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet("background:#12121b; color:white; border-radius:6px; padding:6px;")

        # Input for Ollama
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Ask about the copied text...")
        self.input_box.setStyleSheet("background:#21222b; color:white; padding:6px; border-radius:6px;")
        self.input_box.hide()
        self.send_btn = QPushButton("Send")
        self.send_btn.setStyleSheet("background:#2e7d32; color:white; border:none; padding:6px; border-radius:6px;")
        self.send_btn.clicked.connect(self.ask_ollama)
        self.send_btn.hide()

        bottom = QHBoxLayout()
        bottom.addWidget(self.input_box)
        bottom.addWidget(self.send_btn)

        # Layout
        layout = QVBoxLayout()
        layout.addLayout(header)
        layout.addWidget(self.lang_box)
        layout.addWidget(self.ollama_checkbox)
        layout.addWidget(self.text_area)
        layout.addLayout(bottom)
        layout.setContentsMargins(10,10,10,10)
        self.setLayout(layout)

    def on_ollama_toggle(self):
        if self.ollama_checkbox.isChecked():
            self.input_box.show()
            self.send_btn.show()
            self.mode_label.setText("Ollama Q&A Mode")
            self.text_area.append("💬 Ollama enabled. Ask questions about the copied text.")
        else:
            self.input_box.hide()
            self.send_btn.hide()
            self.mode_label.setText("Translator Mode")
            self.text_area.append("📝 Translator enabled. Copied text will be translated automatically.")

    def check_clipboard(self):
        try:
            import pyperclip
            text = pyperclip.paste().strip()
        except Exception:
            return
        if not text:
            return
        if text == self.last_clip:
            return
        self.last_clip = text
        # if ollama mode off -> translate
        if not self.ollama_checkbox.isChecked():
            self.translate_and_display(text)
        else:
            # display the copied text
            display_text = f"📋 Copied Text:\n{text}"
            self.text_area.setText(display_text)

    def translate_and_display(self, text):
        target = self.lang_box.currentText()
        if HAS_TRANSLATOR:
            try:
                translated = GoogleTranslator(source='auto', target=target).translate(text)
                self.text_area.setText(translated)
            except Exception as e:
                self.text_area.setText(f"⚠️ Translation error: {e}\n\nOriginal:\n{text}")
        else:
            # fallback: show original and tell user
            self.text_area.setText(f"⚠️ Translator library not installed.\n\nOriginal:\n{text}")

    def ask_ollama(self):
        msg = self.input_box.text().strip()
        if not msg:
            return
        selected_text = self.last_clip
        if not selected_text:
            self.text_area.append("⚠️ No copied text to ask about. Copy some text and try again.")
            return

        # Optional fuzzy matching warning
        if HAS_FUZZY:
            score = fuzz.token_set_ratio(msg, selected_text)
            if score < 50:
                self.text_area.append("⚠️ Your question may not align closely with the copied text — answer will still use the copied content.")

        # Compose system prompt
        target_lang = self.lang_box.currentText()
        system_prompt = (
            f"You are an AI assistant. Answer the user's question using ONLY the copied text below. "
            f"Always respond in {target_lang}.\n\nCopied text:\n{selected_text}"
        )

        self.text_area.append(f"<b><span style='color:#00e676'>You:</span></b> {msg}")
        self.input_box.clear()

        try:
            resp = requests.post(
                f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": msg}
                    ],
                    "stream": False
                },
                timeout=20
            )
            # Ollama may return plain text or JSON depending on configuration
            assistant_reply = ""
            try:
                # Try JSON
                data = resp.json()
                # common Ollama style: {'id':..., 'object':'chat.completion', 'message': {'role':..,'content': '...'}}
                if isinstance(data, dict):
                    # try nested keys
                    if "message" in data and isinstance(data["message"], dict):
                        assistant_reply = data["message"].get("content", "")
                    elif "choices" in data and len(data["choices"])>0:
                        assistant_reply = data["choices"][0].get("message", {}).get("content","")
                    else:
                        assistant_reply = json.dumps(data)
                else:
                    assistant_reply = str(data)
            except Exception:
                # fallback to plain text
                assistant_reply = resp.text

            formatted = format_ollama_answer(assistant_reply)
            self.text_area.append(f"<b><span style='color:#4fc3f7'>Ollama:</span></b>\n{formatted}")

        except Exception as e:
            self.text_area.append(f"⚠️ Ollama request failed: {e}")

# ----------------- Document Q&A Popup -----------------
class DocumentQAPopup(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setStyleSheet("background:#081018; color:white; border:2px solid #4caf50; border-radius:12px;")
        self.setFixedSize(520, 520)
        self.current_pdf_key = None         # s3 filename
        self.pdf_extracted_text = ""        # cached extracted text
        self.init_ui()

    def init_ui(self):
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(26,26)
        close_btn.setStyleSheet("background:#ff5252; color:white; border:none; border-radius:6px;")
        close_btn.clicked.connect(self.hide)

        header_label = QLabel("Document Q&A")
        header_label.setStyleSheet("color:#c2ffd8; font-weight:700;")

        header = QHBoxLayout()
        header.addWidget(header_label)
        header.addStretch()
        header.addWidget(close_btn)

        # Upload + status
        self.upload_btn = QPushButton("Upload PDF")
        self.upload_btn.setStyleSheet("background:#1976d2; color:white; padding:8px; border-radius:8px;")
        self.upload_btn.clicked.connect(self.select_file)

        self.status_label = QLabel("No document loaded.")
        self.status_label.setStyleSheet("color:#bfe3c2;")

        upload_row = QHBoxLayout()
        upload_row.addWidget(self.upload_btn)
        upload_row.addWidget(self.status_label)

        # Question area
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Ask a question about the uploaded PDF...")
        self.question_input.setStyleSheet("background:#0f1b24; color:white; padding:8px; border-radius:8px;")
        self.ask_btn = QPushButton("Ask Ollama")
        self.ask_btn.setStyleSheet("background:#00c853; color:white; padding:8px; border-radius:8px;")
        self.ask_btn.clicked.connect(self.ask_question)
        self.ask_btn.setEnabled(False)

        qrow = QHBoxLayout()
        qrow.addWidget(self.question_input)
        qrow.addWidget(self.ask_btn)

        # Answer area
        self.answer_area = QTextEdit()
        self.answer_area.setReadOnly(True)
        self.answer_area.setStyleSheet("background:#071018; color:white; border-radius:8px; padding:8px;")

        # Layout
        layout = QVBoxLayout()
        layout.addLayout(header)
        layout.addLayout(upload_row)
        layout.addLayout(qrow)
        layout.addWidget(self.answer_area)
        layout.setContentsMargins(10,10,10,10)
        self.setLayout(layout)

    def select_file(self):
        file_tuple = QFileDialog.getOpenFileName(self, "Select PDF file", "", "PDF Files (*.pdf);;All Files (*)")
        file_path = file_tuple[0] if file_tuple else None
        if not file_path or not os.path.isfile(file_path):
            self.status_label.setText("⚠️ No valid file selected.")
            return
        # start upload thread
        self.status_label.setText(f"⏳ Uploading {os.path.basename(file_path)} ...")
        self.upload_btn.setEnabled(False)
        self.uploader = UploadThread(file_path)
        self.uploader.progress.connect(self.on_upload_progress)
        self.uploader.extracted_text_signal.connect(self.on_extracted_text)
        self.uploader.start()

    def on_upload_progress(self, msg):
        # append or set status
        self.status_label.setText(msg)
        # If upload succeeded, enable asking on next callback when text arrives
        if "Uploaded" in msg or "already exists" in msg:
            self.ask_btn.setEnabled(True)

    def on_extracted_text(self, text, s3_key):
        self.pdf_extracted_text = text or ""
        if s3_key:
            self.current_pdf_key = s3_key
        self.status_label.setText("✅ PDF loaded for Q&A.")
        if self.pdf_extracted_text:
            sample = self.pdf_extracted_text[:500].strip()
            self.answer_area.append("📄 Extracted text preview:\n" + (sample if sample else "—"))
        else:
            self.answer_area.append("⚠️ No extracted text available (PDF might be scanned or extraction failed).")
        self.upload_btn.setEnabled(True)
        self.ask_btn.setEnabled(True)

    def ask_question(self):
        q = self.question_input.text().strip()
        if not q:
            return
        if not self.pdf_extracted_text:
            self.answer_area.append("⚠️ No PDF text available. Upload a PDF with selectable text or improve extraction.")
            return

        # Compose system prompt
        system_prompt = (
            "You are an expert assistant. Answer the user's question using ONLY the uploaded PDF text below. "
            "Provide the answer point-wise, include at least one short example if relevant, and add a 'Reference' line "
            "at the end mentioning that the answer is derived from the uploaded PDF.\n\n"
            f"PDF Text:\n{self.pdf_extracted_text}"
        )

        self.answer_area.append(f"<b><span style='color:#00e676'>You:</span></b> {q}")
        self.question_input.clear()

        try:
            resp = requests.post(
                f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": q}
                    ],
                    "stream": False
                },
                timeout=30
            )
            assistant_reply = ""
            try:
                data = resp.json()
                if isinstance(data, dict):
                    if "message" in data and isinstance(data["message"], dict):
                        assistant_reply = data["message"].get("content", "")
                    elif "choices" in data and len(data["choices"])>0:
                        assistant_reply = data["choices"][0].get("message", {}).get("content","")
                    else:
                        assistant_reply = json.dumps(data)
                else:
                    assistant_reply = str(data)
            except Exception:
                assistant_reply = resp.text

            formatted = format_ollama_answer(assistant_reply)
            # append formatted with small styling
            self.answer_area.append(f"<b><span style='color:#81d4fa'>Ollama:</span></b>\n{formatted}\n\n🔗 Reference: Based on uploaded PDF file.")
        except Exception as e:
            self.answer_area.append(f"⚠️ Ollama request failed: {e}")

# ----------------- Main Application -----------------
def main():
    app = QApplication(sys.argv)

    # create popups
    translator = TranslatorPopup()
    docqa = DocumentQAPopup()

    # position popups near center-right
    screen_geo = QApplication.primaryScreen().availableGeometry()
    translator.move(screen_geo.width() - translator.width() - 160, 120)
    docqa.move(screen_geo.width() - docqa.width() - 160, 120)

    # floating button
    button = FloatingSwipeButton()
    button.openTranslator.connect(lambda: translator.show())
    button.openDocQA.connect(lambda: docqa.show())

    # show once at startup
    translator.hide()
    docqa.hide()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
