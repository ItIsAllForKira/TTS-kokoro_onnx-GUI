import webbrowser
import platform
import sys
import os
import requests
import subprocess
import threading
import queue
import time
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QTextEdit, QLabel, QLineEdit,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox, QSpinBox,
    QComboBox, QDoubleSpinBox, QProgressBar
)
from PySide6.QtCore import QTimer, Qt, Signal, QCoreApplication, QThread, QEvent, QObject
from PySide6.QtGui import QTextCharFormat, QFont, QColor, QTextCursor, QBrush
import datetime
import logging
from logging.handlers import RotatingFileHandler
import soundfile as sf
import numpy as np
import pyaudio

def setup_logging(log_file_path="kokoro.log"):
    logger = logging.getLogger("SynthesisLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(log_file_path, maxBytes=1*1024*1024, backupCount=5)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s. Sentences: %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

class WorkerSignals(QObject):
    status_signal = Signal(str)
    finished = Signal()
    available_voices_signal = Signal(list)
    highlight_line_signal = Signal(int)
    clear_highlight_signal = Signal()

class TTSWorker(QThread):
    def __init__(self, text, voice, speed, lang, logger, signals, line_count):
        super().__init__()
        self.text = text
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self.logger = logger
        self.signals = signals
        self.line_count = line_count
        self.kokoro_model = None
        self.kokoro_model_path = os.path.expanduser("~/.cache/kokoro/kokoro-v1.0.onnx")
        self.kokoro_voices_path = os.path.expanduser("~/.cache/kokoro/voices-v1.0.bin")
        self.kokoro_voice = self.voice
        self.kokoro_speed = self.speed
        self.kokoro_lang = self.lang
        self.tts_task_queue = queue.Queue(maxsize=2)
        self.audio_playback_queue = queue.Queue(maxsize=5)
        self.tts_stop_event = threading.Event()
        self.tts_skip_event = threading.Event()
        self._threads_stopped = False
        self.actual_available_voices = []

    def run(self):
        synthesis_thread_started = False
        playback_thread_started = False
        try:
            import kokoro_onnx
            if self.kokoro_model is None:
                self.kokoro_model = kokoro_onnx.Kokoro(self.kokoro_model_path, self.kokoro_voices_path)
                try:
                    self.actual_available_voices = self.kokoro_model.get_voices()
                    self.logger.debug(f"Actual available voices loaded from model: {self.actual_available_voices}")
                except Exception as e:
                    self.logger.error(f"Could not get voices from model: {e}")
                    self.actual_available_voices = ["af_heart"]

            lines = self.text.split('\n')
            self.tts_stop_event.clear()
            self.tts_skip_event.clear()
            self._threads_stopped = False
            self.tts_synthesis_thread = threading.Thread(target=self.tts_synthesis_worker, daemon=True)
            self.tts_synthesis_thread.start()
            synthesis_thread_started = True
            self.tts_playback_thread = threading.Thread(target=self.tts_playback_worker, daemon=True)
            self.tts_playback_thread.start()
            playback_thread_started = True

            self.split_and_queue_text_parts(lines)

            self.tts_task_queue.join()
            if synthesis_thread_started and self.tts_synthesis_thread:
                 self.logger.debug("TTSWorker: Waiting for Synthesis Thread to finish naturally...")
                 self.tts_synthesis_thread.join()
                 self.logger.debug("TTSWorker: Synthesis Thread finished naturally.")
            if playback_thread_started and self.tts_playback_thread:
                 self.logger.debug("TTSWorker: Waiting for Playback Thread to finish naturally...")
                 self.tts_playback_thread.join()
                 self.logger.debug("TTSWorker: Playback Thread finished naturally.")
        except Exception as e:
            print(f"Critical TTS Worker Error: {e}")
            self.logger.exception(f"Critical TTS Worker Error: {e}")
        finally:
            self.stop_tts_threads()
            self.signals.clear_highlight_signal.emit()
            non_empty_lines_count = len([line for line in self.text.split('\n') if line.strip()])
            self.logger.info(str(non_empty_lines_count))
            self.signals.status_signal.emit("Ready")
            self.signals.finished.emit()

    def split_and_queue_text_parts(self, lines):
        self.tts_skip_event.clear()
        self.logger.debug("TTS Skip event cleared for new text.")
        for i, part in enumerate(lines):
            if part.strip():
                task_tuple = ("speak", part.strip(), i)
                self.tts_task_queue.put(task_tuple, block=True)
                self.logger.debug(f"Queued TTS part ({i+1}/{len(lines)}): {part.strip()[:30]}...")

    def tts_synthesis_worker(self):
        self.logger.debug("TTS Synthesis Thread Started.")
        while not self.tts_stop_event.is_set():
            try:
                try:
                    task_type, text, line_index = self.tts_task_queue.get_nowait()
                except queue.Empty:
                    time.sleep(0.01)
                    continue

                if task_type == "speak":
                    if self.tts_skip_event.is_set():
                        self.logger.debug("TTS Synthesis Thread: Skip requested, discarding task.")
                        self.tts_task_queue.task_done()
                        continue

                    if self.kokoro_model:
                        try:
                            effective_voice = self.kokoro_voice
                            if effective_voice not in self.actual_available_voices:
                                self.logger.warning(f"Selected voice '{effective_voice}' not found in available voices {self.actual_available_voices}. Attempting fallback.")
                                if "af_heart" in self.actual_available_voices:
                                    effective_voice = "af_heart"
                                    self.logger.info("Switching to fallback voice 'af_heart'.")
                                elif self.actual_available_voices:
                                    effective_voice = self.actual_available_voices[0]
                                    self.logger.info(f"Switching to first available voice '{effective_voice}'.")
                                else:
                                    self.logger.error("No voices available to synthesize speech!")
                                    self.tts_task_queue.task_done()
                                    continue

                            samples, sample_rate = self.kokoro_model.create(
                                text,
                                voice=effective_voice,
                                speed=self.kokoro_speed,
                                lang=self.kokoro_lang
                            )
                            self.logger.debug(f"TTS Synthesis Thread: Audio created. Length: {len(samples)}, Sample Rate: {sample_rate}")

                            if self.tts_stop_event.is_set():
                                self.logger.debug("TTS Synthesis Thread: Stop requested during synthesis. Discarding audio.")
                                self.tts_task_queue.task_done()
                                break

                            if self.tts_skip_event.is_set():
                                self.logger.debug("TTS Synthesis Thread: Skip requested after synthesis. Discarding audio for playback.")
                                self.tts_task_queue.task_done()
                                continue

                            if not isinstance(samples, np.ndarray):
                                samples = np.array(samples, dtype=np.float32)
                            if samples.dtype != np.float32:
                                samples = samples.astype(np.float32)

                            if not self.tts_stop_event.is_set() and not self.tts_skip_event.is_set():
                                self.audio_playback_queue.put(("play", samples, sample_rate, line_index))
                            else:
                               self.logger.debug("TTS Synthesis Thread: Stop or Skip requested after synthesis. Discarding audio for playback.")
                        except Exception as e:
                            self.logger.exception(f"TTS Synthesis Thread Error: {e}")
                    else:
                        self.logger.error("TTS Synthesis Thread: Kokoro model not initialized.")

                self.tts_task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.exception(f"Unexpected error in TTS Synthesis Thread: {e}")

        self.logger.debug("TTS Synthesis Thread Finished.")

    def tts_playback_worker(self):
        self.logger.debug("TTS Playback Thread Started.")
        while not self.tts_stop_event.is_set():
            try:
                try:
                    task_type, samples, sample_rate, line_index = self.audio_playback_queue.get_nowait()
                except queue.Empty:
                    if self.tts_stop_event.is_set():
                        self.logger.debug("TTS Playback Thread: Stop requested while waiting for queue item. Exiting.")
                        break
                    time.sleep(0.01)
                    continue

                if task_type == "play":
                    if self.tts_skip_event.is_set():
                         self.logger.debug("TTS Playback Thread: Skip requested, discarding audio.")
                         self.audio_playback_queue.task_done()
                         continue

                    try:
                        print(f"[PLAYBACK] Starting for line {line_index + 1}")
                        self.signals.highlight_line_signal.emit(line_index)
                    except Exception as e:
                        self.logger.warning(f"Could not highlight line {line_index} during playback: {e}")

                    p = None
                    stream = None
                    try:
                        p = pyaudio.PyAudio()
                        stream = p.open(
                            format=pyaudio.paFloat32,
                            channels=1,
                            rate=sample_rate,
                            output=True
                        )
                        self.logger.debug("TTS Playback Thread: Starting playback via PyAudio...")
                        if self.tts_stop_event.is_set():
                            self.logger.debug("TTS Playback Thread: Stop requested before write. Skipping write.")
                            self.audio_playback_queue.task_done()
                            break

                        if len(samples) == 0:
                            self.logger.warning(f"TTS Playback Thread: Received empty audio data for line {line_index + 1}. Skipping playback.")
                            print(f"[WARNING] Skipping playback for empty audio on line {line_index + 1}")
                        else:
                            stream.write(samples.tobytes())
                            self.logger.debug("TTS Playback Thread: Playback via PyAudio finished.")
                            print(f"[PLAYBACK] Finished for line {line_index + 1}")

                    except Exception as e:
                        self.logger.exception(f"TTS Playback Thread Error during playback/write for line {line_index + 1}: {e}")
                        print(f"[ERROR] TTS Playback Error for line {line_index + 1}: {e}")
                    finally:
                        if stream:
                            try:
                                stream.stop_stream()
                                stream.close()
                            except Exception as e:
                                self.logger.warning(f"Error closing stream for line {line_index + 1}: {e}")
                        if p:
                            try:
                                p.terminate()
                            except Exception as e:
                                self.logger.warning(f"Error terminating PyAudio for line {line_index + 1}: {e}")

                    if self.tts_stop_event.is_set():
                         self.logger.debug("TTS Playback Thread: Stop requested after playback. Exiting.")
                         self.audio_playback_queue.task_done()
                         break

                self.audio_playback_queue.task_done()
            except queue.Empty:
                if self.tts_stop_event.is_set():
                    self.logger.debug("TTS Playback Thread: Stop requested while waiting for queue item. Exiting.")
                    break
                continue
            except Exception as e:
                self.logger.exception(f"Unexpected error in TTS Playback Thread: {e}")
                print(f"[ERROR] Unexpected TTS Playback Thread Error: {e}")
                continue

        self.logger.debug("TTS Playback Thread Finished.")

    def stop_tts_threads(self):
        if self._threads_stopped:
            self.logger.debug("TTS threads already stopped.")
            return
        self.logger.debug("Stopping TTS threads...")
        self.tts_stop_event.set()
        self.tts_skip_event.set()
        self._threads_stopped = True
        self.logger.debug("Clearing TTS task and audio queues...")
        try:
            while True:
                self.tts_task_queue.get_nowait()
                self.tts_task_queue.task_done()
        except queue.Empty:
            pass
        try:
            while True:
                self.audio_playback_queue.get_nowait()
                self.audio_playback_queue.task_done()
        except queue.Empty:
            pass
        if self.tts_synthesis_thread and self.tts_synthesis_thread.is_alive():
            self.logger.debug("Waiting for TTS Synthesis Thread to finish...")
            self.tts_synthesis_thread.join(timeout=1.0)
            if self.tts_synthesis_thread.is_alive():
                self.logger.debug("TTS Synthesis Thread did not finish in time (likely blocked in kokoro.create or waiting for queue item).")
        if self.tts_playback_thread and self.tts_playback_thread.is_alive():
            self.logger.debug("Waiting for TTS Playback Thread to finish...")
            self.tts_playback_thread.join(timeout=1.0)
            if self.tts_playback_thread.is_alive():
                self.logger.debug("TTS Playback Thread did not finish in time (likely blocked in pyaudio.Stream.write or waiting for queue item).")
        self.logger.debug("TTS threads stop process completed.")

    def stop(self):
        self.stop_tts_threads()

class TextToSpeechGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TTS Kokoro-ONNX V301025")
        self.setGeometry(100, 100, 800, 600)
        self.logger = setup_logging("kokoro.log")
        self.cache_dir = Path.home() / ".cache" / "kokoro"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if getattr(sys, 'frozen', False):
            self.script_dir = Path(sys.executable).parent
        else:
            self.script_dir = Path(__file__).parent

        self.model_path = self.cache_dir / "kokoro-v1.0.onnx"
        self.voices_path = self.cache_dir / "voices-v1.0.bin"
        self.model_files = {
            self.model_path: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
            self.voices_path: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
        }

        self.current_directory = os.getcwd()
        self.setup_ui()
        self.check_and_download_models()
        self.audio_file_path = None
        self.available_voices = []
        self.tts_worker = None
        self.tts_signals = WorkerSignals()
        self.tts_signals.status_signal.connect(self.status_label.setText)
        self.tts_signals.finished.connect(self.on_tts_finished)
        self.tts_signals.highlight_line_signal.connect(self.highlight_line)
        self.tts_signals.clear_highlight_signal.connect(self.clear_highlight)
        self.synthesis_active = False
        self.original_text_color = self.text_edit.textColor()
        self.original_text_bg_color = self.text_edit.palette().window().color()
        self.previous_highlighted_line = -1

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("Kokoro-ONNX Text-to-Speech in English")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        text_input_layout = QVBoxLayout()
        text_input_layout.addWidget(QLabel("Enter text to be read aloud:"))
        self.text_edit = QTextEdit()
        self.text_edit.setMinimumHeight(300)
        self.text_edit.setPlaceholderText("Enter text here...\nCan enter multi-line text")
        # --- ВНЕДРЕНИЕ СТИЛЯ ДЛЯ ФИКСАЦИИ ЦВЕТА ---
        self.text_edit.setStyleSheet("QTextEdit { background-color: white; color: black; }")
        # --- КОНЕЦ ВНЕДРЕНИЯ ---
        text_input_layout.addWidget(self.text_edit)
        main_layout.addLayout(text_input_layout, 1)

        self.clear_button = QPushButton("Clear text")
        self.clear_button.clicked.connect(self.clear_text)
        main_layout.addWidget(self.clear_button)

        settings_layout = QHBoxLayout()
        voice_layout = QVBoxLayout()
        voice_layout.addWidget(QLabel("Voice:"))
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(["Loading voices..."])
        self.voice_combo.setCurrentText("Loading voices...")

        self.voice_combo.setEnabled(False)
        voice_layout.addWidget(self.voice_combo)
        settings_layout.addLayout(voice_layout)

        speed_layout = QVBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_spinbox = QDoubleSpinBox()
        self.speed_spinbox.setRange(0.5, 2.0)
        self.speed_spinbox.setSingleStep(0.1)
        self.speed_spinbox.setValue(0.75)
        speed_layout.addWidget(self.speed_spinbox)
        settings_layout.addLayout(speed_layout)

        main_layout.addLayout(settings_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        button_layout = QHBoxLayout()
        self.download_button = QPushButton("Download models")
        self.download_button.clicked.connect(self.download_models)
        button_layout.addWidget(self.download_button)

        self.synthesize_button = QPushButton("Synthesize")
        self.synthesize_button.clicked.connect(self.start_synthesis)
        self.synthesize_button.setStyleSheet("background-color: #4CAF50; color: white;")
        button_layout.addWidget(self.synthesize_button)

        main_layout.addLayout(button_layout)

        self.file_status_label = QLabel("")
        main_layout.addWidget(self.file_status_label)

        self.download_status_label = QLabel("")
        main_layout.addWidget(self.download_status_label)

    def check_and_download_models(self):
        missing = [str(f) for f in self.model_files if not f.exists()]
        if missing:
            self.download_status_label.setText(f"Need to download: {', '.join(missing)}")
            self.download_button.setEnabled(True)
        else:
            self.download_status_label.setText("All models downloaded")
            self.download_button.setEnabled(False)
            self.status_label.setText("Models ready for use")

            self.load_voices_from_model()

    def load_voices_from_model(self):
        try:
            import kokoro_onnx
            model = kokoro_onnx.Kokoro(str(self.model_path), str(self.voices_path))
            voices = model.get_voices()
            
            self.voice_combo.blockSignals(True)
            self.voice_combo.clear()
            if voices:
                 self.voice_combo.addItems(voices)
                 default_voice = "af_heart" if "af_heart" in voices else voices[0]
                 self.voice_combo.setCurrentText(default_voice)
            else:
                 self.voice_combo.addItems(["af_heart"])
                 self.voice_combo.setCurrentText("af_heart")
                 self.logger.warning("Model returned an empty list of voices. Using fallback.")
            self.voice_combo.setEnabled(True)
            self.voice_combo.blockSignals(False)
        except Exception as e:
            error_msg = f"Failed to load voices from model: {e}"
            self.logger.error(error_msg)
            print(error_msg)
            self.voice_combo.blockSignals(True)
            self.voice_combo.clear()
            self.voice_combo.addItems(["af_heart"])
            self.voice_combo.setCurrentText("af_heart")
            self.voice_combo.setEnabled(True)
            self.voice_combo.blockSignals(False)


    def download_models(self):
        self.download_button.setEnabled(False)
        self.status_label.setText("Starting download...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        first_file_path = next(iter(self.model_files.keys()))
        first_file_url = self.model_files[first_file_path]
        self.download_file(first_file_path, first_file_url)

    def download_file(self, filename, url):
        self.current_download = DownloadWorker(url, str(filename))
        self.current_download.download_progress.connect(self.update_download_progress)
        self.current_download.download_complete.connect(self.on_download_complete)
        self.current_download.download_error.connect(self.on_download_error)
        self.current_download.start()

    def update_download_progress(self, progress):
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"Downloading... {progress}%")

    def on_download_complete(self, filename):
        self.status_label.setText(f"Downloaded file: {filename}")
        self.progress_bar.setValue(100)
        all_downloaded = all(f.exists() for f in self.model_files.keys())
        if all_downloaded:
            self.download_status_label.setText("All models downloaded")
            self.download_button.setEnabled(False)
            self.status_label.setText("Models ready for use")
            self.progress_bar.setVisible(False)
            self.load_voices_from_model()
        else:
            remaining_files = [f for f in self.model_files.keys() if not f.exists()]
            if remaining_files:
                next_file = remaining_files[0]
                next_url = self.model_files[next_file]
                self.download_file(next_file, next_url)

    def on_download_error(self, error_message):
        self.status_label.setText(f"Download error: {error_message}")
        self.progress_bar.setVisible(False)
        self.download_button.setEnabled(True)
        print(f"Download error: {error_message}")
        QMessageBox.critical(self, "Error", f"Failed to download file: {error_message}")

    def set_all_text_black(self):
        try:
            if self.text_edit.document().isEmpty():
                self.logger.debug("Text edit is empty, nothing to color.")
                return
            cursor = QTextCursor(self.text_edit.document())
            cursor.select(QTextCursor.Document)
            fmt = QTextCharFormat()
            # --- ИНТЕГРАЦИЯ: Установка цвета фона ---
            fmt.setBackground(QBrush(QColor(255, 255, 255))) # Установить фон на белый
            # --- ИНТЕГРАЦИЯ: Установка цвета текста ---
            fmt.setForeground(QBrush(QColor(0, 0, 0))) # Установить текст на чёрный
            # ---
            fmt.setFontWeight(QFont.Weight.Normal)
            cursor.mergeCharFormat(fmt)
            cursor.clearSelection()
            self.text_edit.setTextCursor(cursor)
            self.logger.debug("All text color explicitly set to black and background to white at start.")
            print("[INFO] All text color set to black and background to white at start.")
        except Exception as e:
            error_msg = f"Failed to set text color and background: {e}"
            self.logger.error(error_msg)
            print(f"[ERROR] {error_msg}")

    def clear_text(self):
        self.text_edit.clear()
        self.clear_highlight()

    def split_text_into_sentences(self, text):
        """Разделяет текст на предложения по знакам препинания ., !, ? и возвращает список предложений."""
        import re
        parts = re.split(r'([.!?])', text)
        # parts теперь чередуется: [текст, знак, текст, знак, ...]
        sentences = []
        for i in range(0, len(parts) - 1, 2): # Идём с шагом 2: текст, затем знак
            sentence_text = parts[i].strip()
            punctuation = parts[i + 1] if i + 1 < len(parts) else ''
            if sentence_text: # Добавляем только непустые предложения
                full_sentence = sentence_text + punctuation
                sentences.append(full_sentence)
        # Обработка случая, если текст не заканчивается на [.!?], но после последнего предложения есть текст в parts[-1]
        if len(parts) % 2 == 1 and parts[-1].strip():
            last_part = parts[-1].strip()
            if last_part: # Если последняя часть не только пробелы
                pass # Игнорируем хвост без [.!?]
        return sentences

    def start_synthesis(self):
        if self.synthesis_active:
            self.stop_synthesis()
        original_text = self.text_edit.toPlainText().strip()
        if not original_text:
            QMessageBox.warning(self, "Warning", "Please enter text to synthesize")
            return

        # --- НОВЫЙ БЛОК: Разделение текста на предложения ---
        sentences = self.split_text_into_sentences(original_text)
        if sentences:
            # Соединяем предложения с переносом строки
            formatted_text = '\n'.join(sentences)
            # Обновляем текст в поле ввода
            self.text_edit.setPlainText(formatted_text)
            # Получаем обновлённый текст для синтеза
            text_to_synthesize = formatted_text
        else:
            # Если не найдено ни одного предложения, используем оригинальный текст
            text_to_synthesize = original_text
        # --- КОНЕЦ НОВОГО БЛОКА ---

        self.set_all_text_black() # Если нужно сбросить цвета после форматирования
        if not all(os.path.exists(f) for f in self.model_files.keys()):
            QMessageBox.warning(self, "Error", "Models need to be downloaded before synthesis")
            return
        selected_voice = self.voice_combo.currentText()
        if selected_voice in ["Loading voices...", "No voices available", ""]:
            QMessageBox.warning(self, "Error", "Please select a voice")
            return
        self.synthesize_button.setText("Synthesizing...")
        self.synthesis_active = True
        speed = self.speed_spinbox.value()
        lang = "en-us"
        # line_count = len(self.text_edit.toPlainText().split('\n')) # Старый подсчёт
        line_count = len(text_to_synthesize.split('\n')) # Подсчёт по новому тексту
        self.tts_worker = TTSWorker(text_to_synthesize, selected_voice, speed, lang, self.logger, self.tts_signals, line_count)
        self.tts_worker.start()
        self.status_label.setText("Synthesizing and playing...")
        # non_empty_lines = [line for line in original_text.split('\n') if line.strip()] # Старый список
        non_empty_lines = [line for line in text_to_synthesize.split('\n') if line.strip()] # Новый список
        print(f"[PROCESSING] Sentences sent for processing: {len(non_empty_lines)}")
        for idx, line in enumerate(non_empty_lines):
            print(f"  [{idx + 1}] {line}")

    def stop_synthesis(self):
        if self.tts_worker:
            self.tts_worker.stop()
            self.tts_worker.wait()
            self.tts_worker = None
        self.synthesis_active = False
        self.synthesize_button.setText("Synthesize")
        self.status_label.setText("Ready")
        self.clear_highlight()

    def on_tts_finished(self):
        self.synthesis_active = False
        self.synthesize_button.setText("Synthesize")
        text = self.text_edit.toPlainText()
        non_empty_lines_count = len([line for line in text.split('\n') if line.strip()])
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
        print(f"{timestamp}. Sentences: {non_empty_lines_count}")

    def highlight_line(self, line_index):
        block_count = self.text_edit.document().blockCount()
        if 0 <= line_index < block_count:
            block = self.text_edit.document().findBlockByNumber(line_index)
            if block.isValid():
                if 0 <= self.previous_highlighted_line < block_count:
                    prev_block = self.text_edit.document().findBlockByNumber(self.previous_highlighted_line)
                    if prev_block.isValid():
                        prev_cursor = QTextCursor(prev_block)
                        prev_fmt = QTextCharFormat()
                        prev_fmt.clearBackground()
                        prev_fmt.clearForeground()
                        prev_fmt.setFontWeight(QFont.Weight.Normal)
                        prev_cursor.select(QTextCursor.LineUnderCursor)
                        prev_cursor.mergeCharFormat(prev_fmt)

                cursor = QTextCursor(block)
                fmt = QTextCharFormat()
                fmt.setForeground(QBrush(QColor(0, 0, 255)))
                fmt.setFontWeight(QFont.Weight.Bold)
                cursor.select(QTextCursor.LineUnderCursor)
                cursor.mergeCharFormat(fmt)
                cursor.movePosition(QTextCursor.StartOfBlock)
                self.text_edit.setTextCursor(cursor)
                self.text_edit.ensureCursorVisible()
                self.previous_highlighted_line = line_index
                print(f"[HIGHLIGHT] Line {line_index + 1} highlighted.")

    def clear_highlight(self):
        if 0 <= self.previous_highlighted_line < self.text_edit.document().blockCount():
            prev_block = self.text_edit.document().findBlockByNumber(self.previous_highlighted_line)
            if prev_block.isValid():
                prev_cursor = QTextCursor(prev_block)
                prev_fmt = QTextCharFormat()
                prev_fmt.clearBackground()
                prev_fmt.clearForeground()
                prev_fmt.setFontWeight(QFont.Weight.Normal)
                prev_cursor.select(QTextCursor.LineUnderCursor)
                prev_cursor.mergeCharFormat(prev_fmt)
        self.previous_highlighted_line = -1

    def closeEvent(self, event):
        self.stop_synthesis()
        event.accept()

class DownloadWorker(QThread):
    download_progress = Signal(int)
    download_complete = Signal(str)
    download_error = Signal(str)

    def __init__(self, url, filename):
        super().__init__()
        self.url = url
        self.filename = filename

    def run(self):
        try:
            response = requests.get(self.url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            with open(self.filename, 'wb') as file:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = int((downloaded / total_size) * 100)
                            self.download_progress.emit(progress)
            self.download_complete.emit(self.filename)
        except Exception as e:
            self.download_error.emit(str(e))

def main():
    required_packages = ['kokoro_onnx', 'soundfile', 'numpy', 'PySide6', 'pyaudio']
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        error_msg = f"Required packages not found:\nNo module named '{missing_packages[0]}'\n"
        error_msg += "Install them with command:\npip install kokoro-onnx soundfile numpy PySide6 pyaudio"
        print(error_msg)
        print("\nTo install dependencies run:")
        print("pip install kokoro-onnx soundfile numpy PySide6 pyaudio")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = TextToSpeechGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()