# Kokoro-ONNX Text-to-Speech (TTS) GUI
<img width="1235" height="797" alt="image" src="https://github.com/user-attachments/assets/d61deebc-dc7c-48bf-b684-ec8fef195933" />

**Local, offline English text-to-speech synthesizer** powered by **[Kokoro-ONNX](https://github.com/thewh1teagle/kokoro-onnx)**.  
This desktop application converts typed or pasted English text into natural-sounding speech using a lightweight ONNX-based neural TTS model—**no internet required after initial setup**.

> **Note**: Designed for clear, sentence-by-sentence spoken output with real-time line highlighting. Ideal for language learners, accessibility, or content preview.

---

## ✨ Features

- **Fully offline TTS** using the `kokoro-v1.0.onnx` model (~200 MB)
- **Multi-voice support** (e.g., `af_heart` and others included in `voices-v1.0.bin`)
- **Adjustable speaking speed** (0.5× to 2.0×)
- **Sentence-aware processing**: input text is auto-split at `. ! ?` for natural phrasing
- **Real-time line highlighting**: current sentence turns **blue bold** during playback
- **Graceful stop & skip**: interrupt synthesis at any time
- **Auto-download of models** on first launch (if missing)
- **Clean white-on-black UI** with explicit color enforcement (white background, black text)
- **Usage logging**: `kokoro.log` tracks number of sentences synthesized per session
- **Batch sentence playback** with per-line audio queuing and PyAudio streaming

---

## 🧠 Model Details

- Based on **Kokoro-ONNX** (English-only, female voices)
- **ONNX runtime**, CPU-only, optimized for low latency
- Model files:
  - `kokoro-v1.0.onnx` (~180 MB)
  - `voices-v1.0.bin` (~20 MB)
- Files cached in `~/.cache/kokoro/`
- First run downloads models automatically (requires internet)

---

## 📦 Requirements

- Python 3.8+
- Required packages:
  ```bash
  pip install kokoro-onnx soundfile numpy PySide6 pyaudio requests
  ```
- ~250 MB free disk space (for models + cache)
- Windows, Linux, or macOS (PyAudio must be functional)

> **Note**: On some Linux systems, you may need to install `portaudio19-dev` before `pyaudio`.

---

## 🚀 Quick Start

1. **Clone or download** the project.
2. **Install dependencies**:
   ```bash
   pip install kokoro-onnx soundfile numpy PySide6 pyaudio requests
   ```
3. **Run the app**:
   ```bash
   python kokoro.py
   ```
4. Wait for **voice loading** (first launch only — models will download automatically).
5. **Type or paste English text**, choose a voice and speed, then click **Synthesize**.

> Audio plays immediately via your speakers. No files are saved—pure real-time TTS.

---

## 📁 Logging

- A rotating log file `kokoro.log` is created in the app directory.
- Format:  
  `2025-12-11 03:45:22 PM. Sentences: 7`
- Logs **number of non-empty sentences** processed per session.
- Max size: 1 MB, with 5 backups.

---

## 🛠️ UI Controls

| Element | Function |
|--------|--------|
| **Text input box** | Enter multi-line English text |
| **Clear text** | Reset input area |
| **Voice dropdown** | Choose from available voices (loaded from model) |
| **Speed slider** | Adjust playback rate (0.5–2.0×) |
| **Synthesize** | Start TTS playback with real-time highlighting |
| **Download models** | Manual trigger if auto-download fails |

---

## ⚠️ Notes

- **Only English** is supported (other languages may produce garbled audio).
- Input is **automatically reformatted** into one sentence per line for optimal synthesis.
- If synthesis stalls, use **Synthesize** button again to **stop & restart**.
- The UI **explicitly enforces white background and black text** via `.setStyleSheet()` to ensure readability.
- Closing the window **automatically stops** any ongoing playback.

---

## 📜 License

- **Kokoro-ONNX model**: [MIT License](https://github.com/thewh1teagle/kokoro-onnx)
- **PySide6**: LGPL/GPL
- **This application**: for personal/research use

> Built with ❤️ using Python, Kokoro-ONNX, and PyAudio.  
> Version: `TTS Kokoro-ONNX V301025`

---
