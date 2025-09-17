# securehub_tts_app.py
import os
import io
import time
import tempfile
import threading
import queue
import traceback
import warnings
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# ML libs (user must install)
try:
    import torch
    import torchaudio
    import whisper
    import whisperx
    from pyannote.audio import Pipeline
    from pydub import AudioSegment
    from transformers import pipeline as transformers_pipeline
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import soundfile as sf
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception as e:
    # If required libs are missing, fail fast.
    raise SystemExit(e)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Optional ffmpeg folder for Windows
FFMPEG_BIN = r"C:\ffmpeg\bin"
if os.path.exists(FFMPEG_BIN):
    os.environ["PATH"] += os.pathsep + FFMPEG_BIN

# HuggingFace token if needed for pyannote / private models
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("") or "we need to add our token"

# Device selection
def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), 0, "CUDA"
    else:
        return torch.device("cpu"), -1, "CPU"


TORCH_DEVICE, TRANSFORMERS_DEVICE, DEVICE_NAME = detect_device()

# ------------------ WER implementation ------------------
def wer(ref: str, hyp: str):
    r = ref.strip().lower().split()
    h = hyp.strip().lower().split()
    n = len(r)
    m = len(h)
    if n == 0:
        return float("inf"), (0, 0, 0), 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] and r[i - 1] == h[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            I += 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1
            i -= 1
        else:
            break
    return dp[n][m] / float(n), (S, D, I), n


# ------------------ Translation model map & fallbacks ------------------
PAIRWISE_TRANSLATION_MODELS = {
    "fa->en": "Helsinki-NLP/opus-mt-mul-en",  # fallback multilingual
    "he->en": "Helsinki-NLP/opus-mt-he-en",
    "mul->en": "Helsinki-NLP/opus-mt-mul-en",
    "en->he": "Helsinki-NLP/opus-mt-en-he",
    "en->fa": "Helsinki-NLP/opus-mt-en-fa",
}

# TTS model mapping (attempts)
TTS_MODELS = {"en": "hexgrad/Kokoro-82M", "he": "facebook/mms-tts-heb", "fa": "facebook/mms-tts-fas"}

# ------------------ Helper: clean transcript ------------------
import re


def clean_transcript_text(aligned_lines):
    """
    aligned_lines: list of dicts with 'text' and optionally 'speaker' and 'timestamp'
    Returns: cleaned string (lowercase optional) without timestamps or speaker tags
    """
    pieces = []
    for seg in aligned_lines:
        text = seg.get("text", "")
        # remove timestamps like [00:12] or 00:12 or 00:12:34 etc.
        text = re.sub(r"\[?\b\d{1,2}:\d{2}(?::\d{2})?\b\]?", "", text)
        # remove speaker labels like SPEAKER_0: , Speaker 1: etc.
        text = re.sub(r"^[A-Za-z0-9_\- ]{1,30}:\s*", "", text)
        # collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pieces.append(text)
    cleaned = "\n".join(pieces)
    return cleaned


# ------------------ Main App ------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SecureHub- Offline Trilingual AI Translator System")
        self.geometry("1200x820")

        # state
        self.audio_path = None
        self.audio_wav_bytes = None
        self.models = {}
        self.diarization = None
        self.transcription_aligned = None
        self.aligned_lines = []
        self.detected_lang = None
        self.target_langs = []
        self.q = queue.Queue()
        self.transcript_clean_path = os.path.join(os.getcwd(), "transcript_clean.txt")

        self._build_ui()
        self._log(f"App started. Device: {DEVICE_NAME}")

        self.after(100, self._process_queue)

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        # top bar controls
        tk.Button(top, text="Upload Audio", bg="#2b8cff", fg="white", command=self.upload_audio).pack(side="left")
        ttk.Label(top, text=" Whisper model:").pack(side="left", padx=(12, 2))
        self.whisper_choice = ttk.Combobox(top, values=["tiny", "base", "small", "medium", "large"], state="readonly", width=8)
        self.whisper_choice.set("small")
        self.whisper_choice.pack(side="left")
        ttk.Label(top, text=" Backend:").pack(side="left", padx=(12, 2))
        self.backend_choice = ttk.Combobox(top, values=["WhisperX", "Sharif-wav2vec2", "ivrit-ai/whisper-large-v3-turbo-ct2"], state="readonly", width=28)
        self.backend_choice.set("WhisperX")
        self.backend_choice.pack(side="left")
        ttk.Label(top, text=" Diarization:").pack(side="left", padx=(12, 2))
        self.diar_backend = tk.StringVar(value="pyannote")
        ttk.Radiobutton(top, text="pyannote", variable=self.diar_backend, value="pyannote").pack(side="left")
        ttk.Radiobutton(top, text="none", variable=self.diar_backend, value="none").pack(side="left")

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=8, pady=8)
        main.columnconfigure(1, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 8))

        # Button style
        btn_style = {"width": 28, "font": ("Segoe UI", 9)}

        # Smaller Preprocess/Diarize buttons
        self.btn_preprocess = tk.Button(left, text="Preprocess", bg="#1abc9c", fg="white", command=self.start_preprocess, state="disabled", **btn_style)
        self.btn_preprocess.pack(pady=4)
        self.btn_diarize = tk.Button(left, text="Diarize", bg="#e67e22", fg="white", command=self.start_diarize, state="disabled", **btn_style)
        self.btn_diarize.pack(pady=4)

        # Rest of buttons
        self.btn_detect_lang = tk.Button(left, text="Detect Language", bg="#9b59b6", fg="white", command=self.start_detect_language, state="disabled", **btn_style)
        self.btn_detect_lang.pack(pady=6)
        self.btn_transcribe = tk.Button(left, text="Transcribe (WhisperX)", bg="#3498db", fg="white", command=self.start_transcribe, state="disabled", **btn_style)
        self.btn_transcribe.pack(pady=6)
        self.btn_translate = tk.Button(left, text="Translate", bg="#6c5ce7", fg="white", command=self.start_translate, state="disabled", **btn_style)
        self.btn_translate.pack(pady=6)
        self.btn_sentiment = tk.Button(left, text="Sentiment", bg="#e84393", fg="white", command=self.start_sentiment, state="disabled", **btn_style)
        self.btn_sentiment.pack(pady=6)
        self.btn_summarize = tk.Button(left, text="Summarize", bg="#00b894", fg="white", command=self.start_summarize, state="disabled", **btn_style)
        self.btn_summarize.pack(pady=6)
        self.btn_tts = tk.Button(left, text="TTS from Translations", bg="#ffb84d", fg="black", command=self.start_tts, state="disabled", **btn_style)
        self.btn_tts.pack(pady=6)

        # Reduced-height process log
        ttk.Label(left, text="Log").pack(anchor="w", pady=(8, 0))
        self.log_box = ScrolledText(left, height=12, state="disabled", bg="#111", fg="#d8efff")
        self.log_box.pack(fill="both", expand=False, pady=(6, 0))

        # Tabs without color styling, with spacing
        style = ttk.Style()
        style.configure("TNotebook.Tab", padding=[12, 6])  # horizontal & vertical padding

        self.tabs = ttk.Notebook(main, style="TNotebook")
        self.tabs.grid(row=0, column=1, sticky="nsew")
        self.tab_widgets = {}
        tab_config = [
            "Preprocess",
            "Diarization",
            "Language",
            "Transcript",
            "Translation",
            "Sentiment",
            "Summary",
            "TTS"
        ]

        for name in tab_config:
            frame = ttk.Frame(self.tabs)
            self.tabs.add(frame, text=name)

            if name == "Translation":
                container = ttk.Frame(frame)
                container.pack(fill="both", expand=True)
                self.translation_container = container
                self.translation_texts = {}
            elif name == "TTS":
                container = ttk.Frame(frame)
                container.pack(fill="both", expand=True)
                txt = ScrolledText(container, wrap="word", state="disabled")
                txt.pack(fill="both", expand=True, padx=6, pady=6)
                self.tab_widgets[name] = txt
            else:
                txt = ScrolledText(frame, wrap="word", state="disabled")
                txt.pack(fill="both", expand=True, padx=6, pady=6)
                self.tab_widgets[name] = txt

        self.status_var = tk.StringVar(value=f"Device: {DEVICE_NAME}")
        ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w").pack(side="bottom", fill="x")
    # ----- logging & queue handling -----
    def _log(self, msg, level="INFO"):
        ts = time.strftime("%H:%M:%S")
        s = f"[{ts}] [{level}] {msg}\n"
        try:
            self.log_box.config(state="normal")
            self.log_box.insert("end", s)
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        except Exception:
            print(s.strip())
        self.status_var.set(msg)

    def _process_queue(self):
        try:
            fn, kwargs = self.q.get_nowait()
            try:
                fn(**kwargs)
            except Exception:
                self._log("UI update error:\n" + traceback.format_exc(), "ERROR")
        except queue.Empty:
            pass
        self.after(100, self._process_queue)

    def _enqueue(self, fn, **kwargs):
        self.q.put((fn, kwargs))

    # ----- UI actions -----
    def upload_audio(self):
        fp = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a")])
        if not fp:
            return
        self.audio_path = fp
        self._log(f"Selected audio: {os.path.basename(fp)}")
        self.btn_preprocess.config(state="normal")
        # reset state
        self.audio_wav_bytes = None
        self.diarization = None
        self.transcription_aligned = None
        self.aligned_lines = []
        self.detected_lang = None
        self.target_langs = []
        for k in self.tab_widgets:
            self._enqueue(self._write_tab, tab=k, text="")

    def start_preprocess(self):
        # keep your original logic intact - calling the same worker
        self.btn_preprocess.config(state="disabled")
        threading.Thread(target=self._preprocess_worker, daemon=True).start()

    def start_diarize(self):
        self.btn_diarize.config(state="disabled")
        threading.Thread(target=self._diarize_worker, daemon=True).start()

    def start_detect_language(self):
        self.btn_detect_lang.config(state="disabled")
        threading.Thread(target=self._language_worker, daemon=True).start()

    def start_transcribe(self):
        self.btn_transcribe.config(state="disabled")
        threading.Thread(target=self._transcribe_worker, daemon=True).start()

    def start_translate(self):
        self.btn_translate.config(state="disabled")
        threading.Thread(target=self._translate_worker, daemon=True).start()

    def start_sentiment(self):
        self.btn_sentiment.config(state="disabled")
        threading.Thread(target=self._sentiment_worker, daemon=True).start()

    def start_summarize(self):
        self.btn_summarize.config(state="disabled")
        threading.Thread(target=self._summarize_worker, daemon=True).start()

    def start_tts(self):
        self.btn_tts.config(state="disabled")
        threading.Thread(target=self._tts_worker, daemon=True).start()

    # ----- workers (use your existing logic exactly) -----
    def _preprocess_worker(self):
        # kept your original logic as-is
        try:
            if not self.audio_path:
                self._enqueue(self._log, msg="No audio selected", level="ERROR")
                self._enqueue(lambda: self.btn_preprocess.config(state="normal"))
                return
            self._enqueue(self._log, msg="Preprocessing -> mono 16k WAV", level="INFO")
            audio = AudioSegment.from_file(self.audio_path).set_channels(1).set_frame_rate(16000)
            buf = io.BytesIO()
            audio.export(buf, format="wav")
            self.audio_wav_bytes = buf.getvalue()
            self._enqueue(self._write_tab, tab="Preprocess", text="Converted to mono 16k WAV")
            self._enqueue(self._log, msg="Preprocess complete", level="SUCCESS")
            self._enqueue(lambda: self.btn_diarize.config(state="normal"))
            self._enqueue(lambda: self.btn_detect_lang.config(state="normal"))
            self._enqueue(lambda: self.btn_transcribe.config(state="normal"))
        except Exception as e:
            self._enqueue(self._log, msg=f"Preprocess error: {e}", level="ERROR")
            self._enqueue(lambda: self.btn_preprocess.config(state="normal"))

    def _diarize_worker(self):
        # kept your original logic as-is
        try:
            if not self.audio_wav_bytes:
                self._enqueue(self._log, msg="Run preprocess first", level="ERROR")
                self._enqueue(lambda: self.btn_diarize.config(state="normal"))
                return
            if self.diar_backend.get() != "pyannote":
                self._enqueue(self._write_tab, tab="Diarization", text="Diarization skipped")
                self._enqueue(self._log, msg="Diarization skipped", level="INFO")
                return
            if not HF_TOKEN:
                self._enqueue(self._write_tab, tab="Diarization", text="Hugging Face token not set - cannot run pyannote")
                self._enqueue(self._log, msg="HF token missing; pyannote requires token", level="WARNING")
                return

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(self.audio_wav_bytes)
                    tmp_path = tmp.name
                if "pyannote_pipe" not in self.models:
                    self._enqueue(self._log, msg="Loading pyannote diarization...", level="INFO")
                    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
                    try:
                        pipe.to(TORCH_DEVICE)
                    except:
                        pass
                    self.models["pyannote_pipe"] = pipe
                    self._enqueue(self._log, msg="pyannote loaded", level="SUCCESS")
                pipe = self.models["pyannote_pipe"]
                self._enqueue(self._log, msg="Running diarization (may take time)...", level="INFO")
                diar = pipe(tmp_path)
                self.diarization = diar
                labels = sorted(list(set(diar.labels())))
                text = f"Speakers detected: {len(labels)}\nLabels: {', '.join(labels)}\n\nTimeline:\n"
                for turn, _, label in diar.itertracks(yield_label=True):
                    text += f"[{turn.start:.2f}s - {turn.end:.2f}s] {label}\n"
                self._enqueue(self._write_tab, tab="Diarization", text=text)
                self._enqueue(self._log, msg="Diarization complete", level="SUCCESS")
            finally:
                try:
                    os.remove(tmp_path)
                except:
                    pass
        except Exception as e:
            self._enqueue(self._log, msg=f"Diarization error: {e}", level="ERROR")
            self._enqueue(self._log, msg=traceback.format_exc(), level="ERROR")
        finally:
            self._enqueue(lambda: self.btn_diarize.config(state="normal"))

    def _language_worker(self):
        # kept your original logic as-is
        try:
            if not self.audio_wav_bytes:
                self._enqueue(self._log, msg="Preprocess first", level="ERROR")
                self._enqueue(lambda: self.btn_detect_lang.config(state="normal"))
                return
            if "whisper_lang" not in self.models:
                self._enqueue(self._log, msg="Loading Whisper tiny for language detect...", level="INFO")
                self.models["whisper_lang"] = whisper.load_model("tiny", device="cpu")
                self._enqueue(self._log, msg="Whisper tiny loaded", level="SUCCESS")
            model = self.models["whisper_lang"]
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(self.audio_wav_bytes)
                    tmp_path = tmp.name
                self._enqueue(self._log, msg="Detecting language...", level="INFO")
                res = model.transcribe(tmp_path)
                lang = res.get("language", "unknown")
                self.detected_lang = lang
                self._enqueue(self._write_tab, tab="Language", text=f"Detected language code: {lang}")
                self._enqueue(self._log, msg=f"Language detected: {lang}", level="SUCCESS")
                # choose target langs
                self._enqueue(self._set_targets_based_on_lang, lang=lang)
            finally:
                try:
                    os.remove(tmp_path)
                except:
                    pass
        except Exception as e:
            self._enqueue(self._log, msg=f"Language detect error: {e}", level="ERROR")
            self._enqueue(self._log, msg=traceback.format_exc(), level="ERROR")
        finally:
            self._enqueue(lambda: self.btn_detect_lang.config(state="normal"))

    def _set_targets_based_on_lang(self, lang):
        lang = (lang or "").lower()
        lang_map = {"en": ["he", "fa"], "he": ["en", "fa"], "fa": ["en", "he"]}
        self.target_langs = lang_map.get(lang[:2], ["en"])
        self._enqueue(self._create_translation_areas)

    def _create_translation_areas(self):
        for w in self.translation_container.winfo_children():
            w.destroy()
        self.translation_texts = {}
        for t in self.target_langs:
            lf = ttk.Labelframe(self.translation_container, text=f"Target: {t}", padding=6)
            lf.pack(fill="both", expand=True, pady=6, padx=6)
            txt = ScrolledText(lf, wrap="word", height=8, state="normal")
            txt.pack(fill="both", expand=True)
            txt.insert("1.0", f"Translation area for {t} - waiting...")
            txt.config(state="disabled")
            self.translation_texts[t] = txt

    def _transcribe_worker(self):
        # kept your existing transcription logic as-is
        try:
            import tempfile
            import numpy as np
            import torch
            import soundfile as sf
            import torchaudio

            if not self.audio_wav_bytes:
                self._enqueue(self._log, msg="Preprocess first", level="ERROR")
                self._enqueue(lambda: self.btn_transcribe.config(state="normal"))
                return

            # Save temp wav
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(self.audio_wav_bytes)
                tmp_path_full = tmp.name

            # Load full audio
            audio_input, sr = sf.read(tmp_path_full)
            if sr != 16000:
                waveform = torch.tensor(audio_input, dtype=torch.float32)
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                audio_input = resampler(waveform).squeeze().numpy()
                sr = 16000

            # Chunking setup
            chunk_duration_s = 20
            samples_per_chunk = chunk_duration_s * sr
            num_chunks = int(np.ceil(len(audio_input) / samples_per_chunk))

            all_segments = []

            # Load diarization model if needed
            diarize_enabled = self.diar_backend.get() == "pyannote"
            if diarize_enabled and "pyannote_pipe" not in self.models:
                self._enqueue(self._log, msg="Loading pyannote diarization...", level="INFO")
                from pyannote.audio import Pipeline

                pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
                try:
                    pipe.to(TORCH_DEVICE)
                except:
                    pass
                self.models["pyannote_pipe"] = pipe
                self._enqueue(self._log, msg="pyannote loaded", level="SUCCESS")

            # Load backend model
            backend = getattr(self, "backend_choice", None)
            backend_value = backend.get() if backend else "WhisperX"

            if backend_value == "Sharif-wav2vec2":
                from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

                key_proc = "Sharif_proc"
                key_model = "Sharif_model"
                if key_proc not in self.models or key_model not in self.models:
                    self._enqueue(self._log, msg="Loading SLPL/Sharif-wav2vec2 model...", level="INFO")
                    self.models[key_proc] = Wav2Vec2Processor.from_pretrained("SLPL/Sharif-wav2vec2")
                    self.models[key_model] = Wav2Vec2ForCTC.from_pretrained("SLPL/Sharif-wav2vec2")
                    try:
                        self.models[key_model].to(TORCH_DEVICE)
                    except:
                        pass
                    self._enqueue(self._log, msg="SLPL/Sharif-wav2vec2 loaded", level="SUCCESS")
                processor = self.models[key_proc]
                model = self.models[key_model]
            elif backend_value == "ivrit-ai/whisper-large-v3-turbo-ct2":
                try:
                    from faster_whisper import WhisperModel
                except ImportError:
                    self._enqueue(self._log, msg="faster_whisper not installed. Please pip install faster_whisper", level="ERROR")
                    self._enqueue(lambda: self.btn_transcribe.config(state="normal"))
                    return

                self._enqueue(self._log, msg="Loading faster-whisper model ivrit-ai/whisper-large-v3-turbo-ct2...", level="INFO")
                model = WhisperModel("ivrit-ai/whisper-large-v3-turbo-ct2", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16" if torch.cuda.is_available() else "int8")
                self._enqueue(self._log, msg="Model loaded", level="SUCCESS")

                segments, _ = model.transcribe(tmp_path_full, language="he")
                for seg in segments:
                    all_segments.append({"timestamp": seg.start, "speaker": "SPEAKER_0", "text": seg.text.strip()})

                # Save and display merged output
                self.aligned_lines = sorted(all_segments, key=lambda x: x["timestamp"])
                display = "\n".join(f"[{int(l['timestamp']//60):02d}:{int(l['timestamp']%60):02d}] {l['speaker']}: {l['text']}" for l in self.aligned_lines)
                self._enqueue(self._write_tab, tab="Transcript", text=display)

                cleaned = clean_transcript_text(self.aligned_lines)
                with open(self.transcript_clean_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)

                self._enqueue(self._log, msg="Transcription with faster-whisper complete", level="SUCCESS")

                self._enqueue(lambda: self.btn_translate.config(state="normal"))
                self._enqueue(lambda: self.btn_sentiment.config(state="normal"))
                self._enqueue(lambda: self.btn_summarize.config(state="normal"))
                self._enqueue(lambda: self.btn_tts.config(state="normal"))

                # Added return here to exit the try block after successful faster-whisper transcription
                return

            elif backend_value == "WhisperX":
                import whisper, whisperx

                model_size = self.whisper_choice.get() or "small"
                key = f"whisper_asr_{model_size}"
                if key not in self.models:
                    self._enqueue(self._log, msg=f"Loading Whisper {model_size}...", level="INFO")
                    self.models[key] = whisper.load_model(model_size, device=TORCH_DEVICE)
                    self._enqueue(self._log, msg="Whisper loaded", level="SUCCESS")
                asr = self.models[key]

            # Process each chunk
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * samples_per_chunk
                chunk_end = min((chunk_idx + 1) * samples_per_chunk, len(audio_input))
                chunk_audio = audio_input[chunk_start:chunk_end]
                chunk_offset_sec = chunk_start / sr

                # Save chunk to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpc:
                    sf.write(tmpc.name, chunk_audio, sr)
                    chunk_path = tmpc.name

                # Diarization for chunk
                diar_result = None
                if diarize_enabled:
                    pipe = self.models["pyannote_pipe"]
                    diar = pipe(chunk_path)
                    diar_result = []
                    for turn, _, label in diar.itertracks(yield_label=True):
                        diar_result.append({"start": turn.start + chunk_offset_sec, "end": turn.end + chunk_offset_sec, "speaker": label})

                # Transcription for chunk
                if backend_value == "Sharif-wav2vec2":
                    inputs = processor(chunk_audio, sampling_rate=sr, return_tensors="pt", padding=True)
                    if isinstance(inputs, dict):
                        input_values = inputs.get("input_values")
                    else:
                        input_values = getattr(inputs, "input_values", None)
                    if input_values is None:
                        raise RuntimeError("Processor did not return 'input_values'")
                    if not isinstance(input_values, torch.Tensor):
                        input_values = torch.tensor(input_values, dtype=torch.float32)
                    input_values = input_values.to(TORCH_DEVICE)
                    with torch.no_grad():
                        logits = model(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1).cpu().numpy()
                    text = processor.batch_decode(predicted_ids)[0]
                    all_segments.append({"timestamp": chunk_offset_sec, "speaker": diar_result[0]["speaker"] if diar_result else "SPEAKER_0", "text": text})

                elif backend_value == "WhisperX":
                    import whisperx

                    result = asr.transcribe(chunk_path, word_timestamps=True)
                    lang = result.get("language", "en")
                    align_key = f"whisperx_align_{lang}"
                    if align_key not in self.models:
                        self._enqueue(self._log, msg="Loading WhisperX align model...", level="INFO")
                        align_model, align_meta = whisperx.load_align_model(language_code=lang, device="cpu")
                        self.models[align_key] = (align_model, align_meta)
                        self._enqueue(self._log, msg="WhisperX align loaded", level="SUCCESS")
                    align_model, align_meta = self.models[align_key]
                    audio_np = whisperx.load_audio(chunk_path, sr=16000)
                    self._enqueue(self._log, msg="Aligning with WhisperX...", level="INFO")
                    aligned = whisperx.align(result["segments"], align_model, align_meta, audio_np, device="cpu", return_char_alignments=False)
                    if diar_result:
                        try:
                            assigned = whisperx.assign_word_speakers(diar_result, aligned)
                            aligned = assigned
                        except Exception:
                            pass
                    for seg in aligned.get("segments", []):
                        ts = seg.get("start", 0) + chunk_offset_sec
                        speaker = seg.get("speaker", "SPEAKER")
                        text = seg.get("text", "").strip()
                        all_segments.append({"timestamp": ts, "speaker": speaker, "text": text})

            # Save and display merged output
            self.aligned_lines = sorted(all_segments, key=lambda x: x["timestamp"])
            display = "\n".join(f"[{int(l['timestamp']//60):02d}:{int(l['timestamp']%60):02d}] {l['speaker']}: {l['text']}" for l in self.aligned_lines)
            self._enqueue(self._write_tab, tab="Transcript", text=display)
            cleaned = clean_transcript_text(self.aligned_lines)
            with open(self.transcript_clean_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            self._enqueue(self._log, msg="Chunked transcription complete", level="SUCCESS")

            self._enqueue(lambda: self.btn_translate.config(state="normal"))
            self._enqueue(lambda: self.btn_sentiment.config(state="normal"))
            self._enqueue(lambda: self.btn_summarize.config(state="normal"))
            self._enqueue(lambda: self.btn_tts.config(state="normal"))
        except Exception as e:
            self._enqueue(self._log, msg=f"Transcription error: {e}", level="ERROR")
            import traceback
            self._enqueue(self._log, msg=traceback.format_exc(), level="ERROR")
            self._enqueue(lambda: self.btn_transcribe.config(state="normal"))

    # ----- translation & summarization & sentiment & WER kept as-is (if present in your original file) -----
    # For brevity in this snippet, we assume your previous implementations for:
    #   _nllb_translate, _translate_worker, _write_translation_tab, compute_wer,
    #   _perform_summarization, _sentiment_worker, _summarize_worker
    # are present exactly as in your original file. If not, include them here unchanged.

    # We'll include the previously provided implementations for those functions:
    def _nllb_translate(self, text, src_iso, tgt_iso):
        lang_map = {"en": "eng_Latn", "he": "heb_Hebr", "fa": "pes_Arab"}
        supported_pairs = {("he", "en"), ("en", "he"), ("fa", "en"), ("en", "fa"), ("he", "fa"), ("fa", "he")}
        if (src_iso, tgt_iso) not in supported_pairs:
            self._log(f"NLLB does not support this pair: {src_iso}->{tgt_iso}", level="WARNING")
            return None
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            if "nllb_tokenizer" not in self.models or "nllb_model" not in self.models:
                self._log("Loading facebook/nllb-200-1.3B model...", level="INFO")
                self.models["nllb_tokenizer"] = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B")
                self.models["nllb_model"] = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-1.3B", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(TORCH_DEVICE)
                self._log("NLLB-200-1.3B loaded successfully", level="SUCCESS")
            tokenizer = self.models["nllb_tokenizer"]
            model = self.models["nllb_model"]
            src_lang_code = lang_map[src_iso]
            tgt_lang_code = lang_map[tgt_iso]
            max_chars = 800
            chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
            translated_chunks = []
            for idx, chunk in enumerate(chunks, start=1):
                self._log(f"NLLB translating chunk {idx}/{len(chunks)}...", level="INFO")
                inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(TORCH_DEVICE)
                outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code], max_length=1024)
                translated_chunks.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
            return "\n".join(translated_chunks)
        except Exception as e:
            self._log(f"NLLB translation error: {e}", level="ERROR")
            import traceback
            self._log(traceback.format_exc(), level="ERROR")
            return None

    def _translate_worker(self):
        # Keep your previously supplied translate logic intact (unchanged)
        try:
            raw = "\n".join(l["text"] for l in self.aligned_lines) if self.aligned_lines else ""
            if not raw:
                self._enqueue(self._log, msg="No transcript to translate", level="ERROR")
                self._enqueue(lambda: self.btn_translate.config(state="normal"))
                return

            if not hasattr(self, "translation_texts") or not self.translation_texts:
                self._enqueue(self._set_targets_based_on_lang, lang=self.detected_lang or "en")
                time.sleep(0.1)

            src = (self.detected_lang or "auto").lower()[:2]
            targets = self.target_langs or ["en"]

            for tgt in targets:
                tgt_code = tgt[:2]
                self._enqueue(self._log, msg=f"Attempting translation from {src} to {tgt_code}", level="INFO")

                translated_text = None

                # 1. Try NLLB first for supported pairs
                nllb_result = self._nllb_translate(raw, src, tgt_code)
                if nllb_result:
                    translated_text = nllb_result
                    self._enqueue(self._log, msg=f"NLLB translation successful {src}->{tgt_code}", level="SUCCESS")
                else:
                    self._enqueue(self._log, msg=f"NLLB not used or failed for {src}->{tgt_code}, falling back to Helsinki.", level="WARNING")

                # 2. If NLLB failed, run existing Helsinki logic
                if not translated_text:
                    candidates = [PAIRWISE_TRANSLATION_MODELS.get(f"{src}->{tgt_code}")]
                    if tgt_code == "en":
                        candidates.append(PAIRWISE_TRANSLATION_MODELS.get("mul->en"))
                    candidates = [c for c in candidates if c]

                    for cand_model_name in candidates:
                        try:
                            task_name = f"translation_{src}_to_{tgt_code}"
                            cache_key = f"translator::{cand_model_name}::{task_name}"
                            translator = self.models.get(cache_key)
                            if not translator:
                                try:
                                    translator = transformers_pipeline(task_name, model=cand_model_name, device=TRANSFORMERS_DEVICE)
                                except Exception:
                                    translator = transformers_pipeline("translation", model=cand_model_name, device=TRANSFORMERS_DEVICE)
                                if translator:
                                    self.models[cache_key] = translator
                            if translator:
                                pieces = []
                                max_chars = 700
                                chunks = [raw[i : i + max_chars] for i in range(0, len(raw), max_chars)]
                                for chunk in chunks:
                                    out = translator(chunk, max_length=1024)
                                    if isinstance(out, list) and out:
                                        pieces.append(out[0].get("translation_text", ""))
                                    elif isinstance(out, dict):
                                        pieces.append(out.get("translation_text", ""))
                                    else:
                                        pieces.append(str(out))
                                translated_text = "\n".join(pieces)
                                break
                        except Exception as e:
                            self._enqueue(self._log, msg=f"Helsinki translation error: {e}", level="ERROR")

                if not translated_text:
                    translated_text = "(translation failed or unavailable)"

                self._enqueue(self._write_translation_tab, lang=tgt, text=translated_text)
                self._enqueue(self._log, msg=f"Translation attempt to {tgt_code} complete.", level="SUCCESS" if translated_text != "(translation failed or unavailable)" else "WARNING")

            self._enqueue(lambda: self.btn_tts.config(state="normal"))
        except Exception as e:
            self._enqueue(self._log, msg=f"Overall Translation worker error: {e}", level="ERROR")
            import traceback
            self._enqueue(self._log, msg=traceback.format_exc(), level="ERROR")
        finally:
            self._enqueue(lambda: self.btn_translate.config(state="normal"))

    def _write_translation_tab(self, lang, text):
        # Assuming self.translation_texts is a dict mapping lang codes to ScrolledText widgets
        widget = self.translation_texts.get(lang)
        if widget:
            widget.config(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", text)
            widget.config(state="disabled")

    def _write_tab(self, tab, text):
        widget = self.tab_widgets.get(tab)
        if widget:
            widget.config(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", text)
            widget.config(state="disabled")


    # ----- summarization helpers (kept as-is) -----
    def _perform_summarization(self, text):
        if "summarizer_model" not in self.models or "summarizer_tokenizer" not in self.models:
            self._enqueue(self._log, msg="Summarization model not loaded.", level="ERROR")
            return "(Summarization model not loaded)"

        tokenizer = self.models["summarizer_tokenizer"]
        model = self.models["summarizer_model"]

        input_text = "summarize: " + text
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

        summary_ids = model.generate(inputs["input_ids"], max_length=500, min_length=100, num_beams=4, early_stopping=True)

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary

    # ----- sentiment & summarization workers kept as-is (if present earlier) -----
    def _sentiment_worker(self):
        try:
            from transformers import pipeline as transformers_pipeline

            if not self.detected_lang:
                self._enqueue(self._log, msg="No detected language for sentiment analysis.", level="ERROR")
                self._enqueue(self._write_tab, tab="Sentiment", text="No detected language.")
                return

            lang = self.detected_lang.lower()[:2]
            self._enqueue(self._log, msg=f"Starting sentiment analysis for language: {lang}", level="INFO")

            model_map = {
                "en": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "he": "avichr/hebEMO_trust",
                "fa": "HooshvareLab/bert-base-parsbert-uncased",
            }
            model_name = model_map.get(lang)
            if not model_name:
                self._enqueue(self._log, msg=f"No sentiment model configured for language: {lang}", level="ERROR")
                self._enqueue(self._write_tab, tab="Sentiment", text=f"No model for {lang}")
                return

            if lang == "en":
                text_to_analyze = clean_transcript_text(self.aligned_lines)
            else:
                if hasattr(self, "translation_texts") and "en" in self.translation_texts:
                    self.translation_texts["en"].config(state="normal")
                    text_to_analyze = self.translation_texts["en"].get("1.0", "end").strip()
                    self.translation_texts["en"].config(state="disabled")
                else:
                    text_to_analyze = clean_transcript_text(self.aligned_lines)

            if not text_to_analyze.strip():
                self._enqueue(self._log, msg="No text available for sentiment analysis.", level="ERROR")
                self._enqueue(self._write_tab, tab="Sentiment", text="No text available.")
                return

            cache_key = f"sentiment::{model_name}"
            sentiment_pipe = self.models.get(cache_key)
            if not sentiment_pipe:
                self._enqueue(self._log, msg=f"Loading sentiment model: {model_name}", level="INFO")
                sentiment_pipe = transformers_pipeline("sentiment-analysis", model=model_name, device=TRANSFORMERS_DEVICE)
                self.models[cache_key] = sentiment_pipe

            max_chars = 500
            chunks = [text_to_analyze[i : i + max_chars] for i in range(0, len(text_to_analyze), max_chars)]
            results = []
            for idx, chunk in enumerate(chunks):
                res = sentiment_pipe(chunk)
                results.append(f"Chunk {idx+1}: {res}")

            final_output = "\n".join(map(str, results))
            self._enqueue(self._write_tab, tab="Sentiment", text=final_output)
            self._enqueue(self._log, msg="Sentiment analysis complete.", level="SUCCESS")

        except Exception as e:
            self._enqueue(self._log, msg=f"Sentiment analysis error: {e}", level="ERROR")
            self._enqueue(self._log, msg=traceback.format_exc(), level="ERROR")
        finally:
            self._enqueue(lambda: self.btn_sentiment.config(state="normal"))

    def _summarize_worker(self):
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            if "summarizer_model" not in self.models or "summarizer_tokenizer" not in self.models:
                self._enqueue(self._log, msg="Loading summarization model csebuetnlp/mT5_multilingual_XLSum...", level="INFO")
                try:
                    tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
                    model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
                    self.models["summarizer_tokenizer"] = tokenizer
                    self.models["summarizer_model"] = model
                    self._enqueue(self._log, msg="Summarization model loaded successfully", level="SUCCESS")
                except Exception as e:
                    self._enqueue(self._log, msg=f"Failed to load summarization model: {e}", level="ERROR")
                    self._enqueue(self._log, msg=traceback.format_exc(), level="ERROR")
                    self._enqueue(self._write_tab, tab="Summary", text=f"Error loading summarization model: {e}")
                    self._enqueue(lambda: self.btn_summarize.config(state="normal"))
                    return  # Exit worker if loading fails

            cleaned_transcript = clean_transcript_text(self.aligned_lines)
            if not cleaned_transcript:
                self._enqueue(self._log, msg="No transcript available for summarization. Run transcription first.", level="ERROR")
                self._enqueue(self._write_tab, tab="Summary", text="No transcript available for summarization.")
                return

            native_summary = None
            if self.detected_lang and self.detected_lang.lower() != "unknown":
                self._enqueue(self._log, msg=f"Starting native language ({self.detected_lang}) summarization...", level="INFO")
                try:
                    native_summary = self._perform_summarization(cleaned_transcript)
                    self._enqueue(self._log, msg=f"Native language ({self.detected_lang}) summarization complete.", level="SUCCESS")
                except Exception as e:
                    self._enqueue(self._log, msg=f"Native language summarization error: {e}", level="ERROR")
                    self._enqueue(self._log, msg=traceback.format_exc(), level="ERROR")
                    native_summary = f"(Error during native language summarization: {e})"
            else:
                self._enqueue(self._log, msg="No or unknown language detected. Skipping native language summarization.", level="WARNING")
                native_summary = "(Native language summarization skipped - language unknown)"

            english_summary = None
            english_text_for_summary = None

            if self.detected_lang and self.detected_lang.lower().startswith("en"):
                self._enqueue(self._log, msg="Detected language is English. Using transcript for English summary.", level="INFO")
                english_text_for_summary = cleaned_transcript
            else:
                if hasattr(self, "translation_texts") and "en" in self.translation_texts and isinstance(self.translation_texts["en"], ScrolledText):
                    self.translation_texts["en"].config(state="normal")
                    english_text_for_summary = self.translation_texts["en"].get("1.0", "end").strip()
                    self.translation_texts["en"].config(state="disabled")
                    if english_text_for_summary and english_text_for_summary != "Translation area for en - waiting...":
                        self._enqueue(self._log, msg="Using existing English translation for English summary.", level="INFO")
                    else:
                        english_text_for_summary = None

                if not english_text_for_summary:
                    self._enqueue(self._log, msg="No usable existing English translation found. Attempting translation for English summary...", level="INFO")
                    try:
                        from transformers import pipeline as transformers_pipeline

                        src = (self.detected_lang or "auto").lower()[:2]
                        tgt_code = "en"
                        cand_model_name = PAIRWISE_TRANSLATION_MODELS.get(f"{src}->{tgt_code}") or PAIRWISE_TRANSLATION_MODELS.get("mul->en")
                        if cand_model_name:
                            translate_key = f"translator::{cand_model_name}::translation_{src}_to_{tgt_code}"
                            translator = self.models.get(translate_key)
                            if not translator:
                                self._enqueue(self._log, msg=f"Loading translator {cand_model_name} for {src}->{tgt_code} for summary translation...", level="INFO")
                                try:
                                    translator = transformers_pipeline(f"translation_{src}_to_{tgt_code}", model=cand_model_name, device=TRANSFORMERS_DEVICE)
                                except Exception:
                                    translator = transformers_pipeline("translation", model=cand_model_name, device=TRANSFORMERS_DEVICE)
                                if translator:
                                    self.models[translate_key] = translator
                                else:
                                    self._enqueue(self._log, msg=f"Failed to load translator {cand_model_name} for summary translation.", level="ERROR")

                            if translator:
                                self._enqueue(self._log, msg=f"Translating for English summary with {cand_model_name}...", level="INFO")
                                try:
                                    chunk_size = min(len(cleaned_transcript), 2000)
                                    out = translator(cleaned_transcript[:chunk_size], max_length=512)
                                    if isinstance(out, list) and out:
                                        english_text_for_summary = out[0].get("translation_text", "")
                                    elif isinstance(out, dict):
                                        english_text_for_summary = out.get("translation_text", "")
                                    else:
                                        english_text_for_summary = str(out)
                                    self._enqueue(self._log, msg="Translation for English summary complete.", level="SUCCESS")
                                except Exception as e_trans:
                                    self._enqueue(self._log, msg=f"Error during translation for English summary: {e_trans}", level="ERROR")
                                    self._enqueue(self._log, msg=traceback.format_exc(), level="ERROR")
                                    english_text_for_summary = ""
                            else:
                                self._enqueue(self._log, msg=f"Translator for {cand_model_name} not available for summary translation.", level="WARNING")
                                english_text_for_summary = ""
                        else:
                            self._enqueue(self._log, msg=f"No translation model configured for {src}->en for summary translation.", level="WARNING")
                            english_text_for_summary = ""
                    except Exception as e_overall_trans:
                        self._enqueue(self._log, msg=f"Overall error during translation attempt for English summary: {e_overall_trans}", level="ERROR")
                        self._enqueue(self._log, msg=traceback.format_exc(), level="ERROR")
                        english_text_for_summary = ""

            if english_text_for_summary:
                self._enqueue(self._log, msg="Starting English summarization...", level="INFO")
                try:
                    english_summary = self._perform_summarization(english_text_for_summary)
                    self._enqueue(self._log, msg="English summarization complete.", level="SUCCESS")
                except Exception as e:
                    self._enqueue(self._log, msg=f"English summarization error: {e}", level="ERROR")
                    self._enqueue(self._log, msg=traceback.format_exc(), level="ERROR")
                    english_summary = f"(Error during English summarization: {e})"
            else:
                self._enqueue(self._log, msg="No English text available for summarization.", level="WARNING")
                english_summary = "(English summarization skipped - no English text)"

            combined_summary = ""
            if native_summary:
                native_lang_label = self.detected_lang or "Native Language"
                combined_summary += f"--- {native_lang_label.upper()} SUMMARY ---\n{native_summary}\n\n"

            if english_summary:
                combined_summary += "--- ENGLISH SUMMARY ---\n"
                combined_summary += english_summary

            if not combined_summary:
                combined_summary = "No summaries could be generated."

            self._enqueue(self._write_tab, tab="Summary", text=combined_summary)
            self._enqueue(self._log, msg="Summaries displayed in Summary tab.", level="INFO")

        except Exception as e:
            self._enqueue(self._log, msg=f"Summarization worker error: {e}", level="ERROR")
            self._enqueue(self._log, msg=traceback.format_exc(), level="ERROR")
        finally:
            self._enqueue(lambda: self.btn_summarize.config(state="normal"))

    # ----- TTS worker (fully implemented) -----
    def _tts_worker(self):
        """
        TTS worker:
         - Uses TTS_MODELS mapping for 'en', 'he', 'fa'
         - Gathers text from translation tabs (if available) or transcript
         - Loads model/tokenizer, generates waveform, saves wav to output/tts/
         - Attempts to autoplay using simpleaudio or pydub/winsound fallback
        """
        try:
            out_dir = os.path.join(os.getcwd(), "output", "tts")
            os.makedirs(out_dir, exist_ok=True)

            # Determine source texts for each language. Prefer translations if present.
            lang_text_map = {}
            for lang in ("he", "fa", "en"):
                txt = ""
                # Prefer translation tab content if present
                try:
                    if hasattr(self, "translation_texts") and lang in self.translation_texts:
                        widget = self.translation_texts[lang]
                        widget.config(state="normal")
                        txt = widget.get("1.0", "end").strip()
                        widget.config(state="disabled")
                        if txt and txt.startswith("Translation area for"):
                            txt = ""
                except Exception:
                    txt = ""
                # if no translation, fallback to cleaned transcript
                if not txt:
                    cleaned = clean_transcript_text(self.aligned_lines)
                    txt = cleaned
                lang_text_map[lang] = txt

            # If nothing available at all, log and return
            if not any(lang_text_map.values()):
                self._enqueue(self._log, msg="No text available for TTS (no translations or transcripts).", level="ERROR")
                self._enqueue(lambda: self.btn_tts.config(state="normal"))
                return

            # Helper: attempt playback using several libraries
            def try_playback(wav_path):
                # try simpleaudio
                try:
                    import simpleaudio as sa

                    wave_obj = sa.WaveObject.from_wave_file(wav_path)
                    play_obj = wave_obj.play()
                    # do not block; allow playback to run
                    return True
                except Exception:
                    pass
                # try pydub playback
                try:
                    from pydub import AudioSegment
                    from pydub.playback import play

                    seg = AudioSegment.from_file(wav_path, format="wav")
                    threading.Thread(target=play, args=(seg,), daemon=True).start()
                    return True
                except Exception:
                    pass
                # windows winsound
                try:
                    if os.name == "nt":
                        import winsound

                        winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                        return True
                except Exception:
                    pass
                return False

            # For each requested language, load corresponding TTS model and generate audio
            for lang, model_id in TTS_MODELS.items():
                text = lang_text_map.get(lang, "") or ""
                if not text.strip():
                    self._enqueue(self._log, msg=f"No text available for TTS in '{lang}', skipping.", level="WARNING")
                    continue

                self._enqueue(self._log, msg=f"TTS: Preparing to generate for '{lang}' using {model_id}", level="INFO")

                # Load/cache tokenizer & model
                tkn_key = f"tts_tokenizer::{model_id}"
                model_key = f"tts_model::{model_id}"
                try:
                    tokenizer = self.models.get(tkn_key)
                    model = self.models.get(model_key)
                    if not tokenizer or not model:
                        # Try multiple possible model classes for MMS-TTS style models
                        self._enqueue(self._log, msg=f"Loading TTS model {model_id} ... (may take time)", level="INFO")
                        try:
                            # preferred import
                            from transformers import VitsModel, AutoTokenizer

                            tokenizer = AutoTokenizer.from_pretrained(model_id)
                            model = VitsModel.from_pretrained(model_id).to(TORCH_DEVICE)
                        except Exception:
                            # fallback: try VitsForConditionalGeneration or AutoModel
                            try:
                                from transformers import AutoTokenizer, AutoModel

                                tokenizer = AutoTokenizer.from_pretrained(model_id)
                                model = AutoModel.from_pretrained(model_id).to(TORCH_DEVICE)
                            except Exception:
                                # final fallback: try pipeline TTS (may use inference API)
                                tokenizer = None
                                model = None
                                self._enqueue(self._log, msg=f"Couldn't load model classes directly for {model_id}; will try pipeline fallback.", level="WARNING")

                        if tokenizer and model:
                            self.models[tkn_key] = tokenizer
                            self.models[model_key] = model
                            self._enqueue(self._log, msg=f"TTS model {model_id} loaded and cached.", level="SUCCESS")
                except Exception as e:
                    self._enqueue(self._log, msg=f"Error loading TTS model {model_id}: {e}", level="ERROR")
                    self._enqueue(self._log, msg=traceback.format_exc(), level="ERROR")
                    tokenizer = None
                    model = None

                # If we failed to get a model instance, try pipeline fallback
                pipeline_tts = None
                if (not model) or (not tokenizer):
                    try:
                        # This may call the HF inference locally or via API if not cached
                        self._enqueue(self._log, msg=f"Loading text-to-speech pipeline for {model_id} as fallback...", level="INFO")
                        pipeline_tts = transformers_pipeline("text-to-speech", model=model_id, device=TRANSFORMERS_DEVICE)
                        self.models[f"tts_pipeline::{model_id}"] = pipeline_tts
                        self._enqueue(self._log, msg=f"TTS pipeline loaded for {model_id}", level="SUCCESS")
                    except Exception as e:
                        self._enqueue(self._log, msg=f"Failed to load TTS pipeline for {model_id}: {e}", level="ERROR")
                        self._enqueue(self._log, msg=traceback.format_exc(), level="ERROR")
                        pipeline_tts = None

                # Prepare filename
                ts = int(time.time())
                safe_name = f"tts_{lang}_{ts}.wav"
                wav_path = os.path.join(out_dir, safe_name)

                # Generation branch: pipeline available
                if pipeline_tts:
                    try:
                        self._enqueue(self._log, msg=f"Generating TTS via pipeline for {lang}...", level="INFO")
                        # pipeline returns dict or Audio object depending on implementation
                        res = pipeline_tts(text)
                        # Some pipelines return a dict with "wav" bytes or "array"
                        if isinstance(res, dict):
                            audio_bytes = None
                            # try multiple keys
                            for k in ("wav", "audio", "array", "waveform"):
                                if k in res:
                                    audio_bytes = res[k]
                                    break
                            if audio_bytes is None:
                                # maybe the pipeline returned bytes directly
                                audio_bytes = res.get("content") if "content" in res else None
                            if isinstance(audio_bytes, (bytes, bytearray)):
                                with open(wav_path, "wb") as f:
                                    f.write(audio_bytes)
                                self._enqueue(self._log, msg=f"TTS saved: {wav_path}", level="SUCCESS")
                                try_playback(wav_path)
                                self._enqueue(self._write_tab, tab="TTS", text=f"{lang}: {wav_path}\n",)
                            elif isinstance(audio_bytes, list) or isinstance(audio_bytes, tuple):
                                # if array, try writing with soundfile
                                import numpy as np
                                arr = np.array(audio_bytes)
                                sf.write(wav_path, arr, samplerate=22050)
                                self._enqueue(self._log, msg=f"TTS saved (array): {wav_path}", level="SUCCESS")
                                try_playback(wav_path)
                                self._enqueue(self._write_tab, tab="TTS", text=f"{lang}: {wav_path}\n",)
                            else:
                                # attempt to handle res as bytes-like
                                try:
                                    with open(wav_path, "wb") as f:
                                        f.write(res)
                                    self._enqueue(self._log, msg=f"TTS saved (raw): {wav_path}", level="SUCCESS")
                                    try_playback(wav_path)
                                    self._enqueue(self._write_tab, tab="TTS", text=f"{lang}: {wav_path}\n",)
                                except Exception:
                                    self._enqueue(self._log, msg=f"Unknown pipeline response for {model_id}: {type(res)}", level="ERROR")
                        elif isinstance(res, (bytes, bytearray)):
                            with open(wav_path, "wb") as f:
                                f.write(res)
                            self._enqueue(self._log, msg=f"TTS saved: {wav_path}", level="SUCCESS")
                            try_playback(wav_path)
                            self._enqueue(self._write_tab, tab="TTS", text=f"{lang}: {wav_path}\n",)
                        else:
                            # unknown pipeline response type
                            self._enqueue(self._log, msg=f"Unexpected pipeline response type: {type(res)}", level="ERROR")
                    except Exception as e:
                        self._enqueue(self._log, msg=f"TTS pipeline generation error for {model_id}: {e}", level="ERROR")
                        self._enqueue(self._log, msg=traceback.format_exc(), level="ERROR")
                    continue  # move to next language

                # Generation branch: model & tokenizer available (attempt VITS style)
                if tokenizer and model:
                    try:
                        self._enqueue(self._log, msg=f"Tokenizing input for {model_id}...", level="INFO")
                        inputs = tokenizer(text, return_tensors="pt", padding=True)
                        # Move tensors to the right device if present
                        inputs = {k: v.to(TORCH_DEVICE) for k, v in inputs.items() if hasattr(v, "to")}
                        self._enqueue(self._log, msg=f"Running model for {model_id}...", level="INFO")
                        # Many VITS-style models return a dict with 'waveform' or .waveform attribute
                        with torch.no_grad():
                            try:
                                out = model.generate(**inputs)
                                # if generate returns tensors, try to handle
                                if isinstance(out, torch.Tensor):
                                    audio_tensor = out
                                elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                                    audio_tensor = out[0]
                                elif isinstance(out, dict) and "waveform" in out:
                                    audio_tensor = out["waveform"]
                                elif hasattr(out, "waveform"):
                                    audio_tensor = getattr(out, "waveform")
                                else:
                                    audio_tensor = None
                            except Exception:
                                # fallback to calling model(**inputs)
                                result = model(**inputs)
                                if isinstance(result, dict) and "waveform" in result:
                                    audio_tensor = result["waveform"]
                                elif hasattr(result, "waveform"):
                                    audio_tensor = getattr(result, "waveform")
                                else:
                                    # try result as tensor
                                    audio_tensor = None

                        if audio_tensor is None:
                            # can't find waveform — log and skip
                            self._enqueue(self._log, msg=f"Model did not return waveform for {model_id}.", level="ERROR")
                            continue

                        # Move to CPU and convert to numpy
                        audio_np = audio_tensor.squeeze().cpu().numpy()

                        # Determine sample rate
                        sr = getattr(model.config, "sampling_rate", None) or getattr(model.config, "sample_rate", None) or 22050

                        # Save wav
                        sf.write(wav_path, audio_np, samplerate=sr)
                        self._enqueue(self._log, msg=f"TTS saved: {wav_path}", level="SUCCESS")
                        # attempt playback
                        try_playback(wav_path)
                        self._enqueue(self._write_tab, tab="TTS", text=f"{lang}: {wav_path}\n",)
                    except Exception as e:
                        self._enqueue(self._log, msg=f"TTS generation error for {model_id}: {e}", level="ERROR")
                        self._enqueue(self._log, msg=traceback.format_exc(), level="ERROR")
                        continue
                else:
                    self._enqueue(self._log, msg=f"No model or pipeline available for {model_id}, skipped.", level="WARNING")

            self._enqueue(self._log, msg="TTS worker complete.", level="SUCCESS")
        except Exception as e:
            self._enqueue(self._log, msg=f"TTS worker error: {e}", level="ERROR")
            self._enqueue(self._log, msg=traceback.format_exc(), level="ERROR")
        finally:
            self._enqueue(lambda: self.btn_tts.config(state="normal"))

# Run app
if __name__ == "__main__":
    app = App()
    app.mainloop()