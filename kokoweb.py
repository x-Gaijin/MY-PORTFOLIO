from faster_whisper import WhisperModel
import httpx
import asyncio
import time
import threading
import os
import re
import torch
import tempfile
import wave
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import websockets
from websockets.server import serve
import pyaudio
import signal
import sys
import glob
import gc

# -----------------------------
# KOKORO IMPORT
# -----------------------------
from kokoro import KPipeline

# -----------------------------
# LANGCHAIN IMPORTS
# -----------------------------
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# -----------------------------
# DYNAMIC CONFIG
# -----------------------------
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

# DYNAMIC PATHS
BASE_DIR        = r"C:\Users\tristhan\Desktop\AI\ai"
SCHOOL_DOCS_DIR = os.path.join(BASE_DIR, "Documents")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db")

# HuggingFace Kokoro voices path (confirmed working)
HF_VOICES_DIR = r"C:\Users\tristhan\.cache\huggingface\hub\models--hexgrad--Kokoro-82M\snapshots\f3ff3571791e39611d31c381e3a41a3af07b4987\voices"

# Document sections
DOCUMENT_SECTIONS = {
    'ACADEMIC_CALENDAR': 'Academic Calendar',
    'CENAR':             'CENAR',
    'CENAR_FACULTY':     'CENAR Faculty',
    'ENROLLMENT':        'Enrollment',
    'EXAM_DATE':         'Exam Date',
    'GENERAL_INFO':      'General Info',
}

# WebSocket settings
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8888

# Audio settings
SAMPLE_RATE = 24000
FORMAT      = pyaudio.paInt16
CHANNELS    = 1

# Fixed volume gain — increase if too soft, decrease if distorted (max safe ~2.5)
VOLUME_GAIN = 1.0

MAX_RESPONSE_TOKENS = 250

# ============================================================
# KOKORO CHUNKING SETTINGS
# ============================================================
PHRASE_WORD_TARGET    = 10      # words per phrase chunk (8-12 = best balance)
CROSSFADE_SAMPLES     = 480     # 20ms crossfade at 24kHz for smooth joins
TTS_THREAD_WORKERS    = 2       # thread pool for CPU inference
ESP32_SAFE_SLICE_SIZE = 4096    # 4KB WebSocket slices

# ============================================================
# DEBUGGING SETTINGS
# ============================================================
DEBUG_MODE      = True
DEBUG_GPU       = True
DEBUG_AUDIO     = True
DEBUG_WEBSOCKET = True
DEBUG_TTS       = True


class DebugLogger:
    @staticmethod
    def gpu(msg, force=False):
        if DEBUG_GPU or force:
            print(f"[GPU_DEBUG] {msg}")

    @staticmethod
    def audio(msg, force=False):
        if DEBUG_AUDIO or force:
            print(f"[AUDIO_DEBUG] {msg}")

    @staticmethod
    def ws(msg, force=False):
        if DEBUG_WEBSOCKET or force:
            print(f"[WS_DEBUG] {msg}")

    @staticmethod
    def tts(msg, force=False):
        if DEBUG_TTS or force:
            print(f"[TTS_DEBUG] {msg}")

    @staticmethod
    def error(msg):
        print(f"[ERROR] ❌ {msg}")

    @staticmethod
    def warning(msg):
        print(f"[WARNING] ⚠️ {msg}")


debug = DebugLogger()

# -----------------------------
# GRACEFUL SHUTDOWN
# -----------------------------
def setup_graceful_shutdown():
    def _handler(signum, frame):
        print(f"\n📡 Signal {signum} — shutting down...")
        global_state.is_shutting_down = True
        global_state.is_recording     = False
        print("✅ Graceful shutdown initiated")
        os._exit(0)

    try:
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT,  _handler)
        print("✅ Signal handlers registered")
    except Exception:
        print("⚠️ Signal handlers unavailable on this platform")


# -----------------------------
# GLOBAL STATE
# -----------------------------
class GlobalState:
    def __init__(self):
        self.is_recording               = False
        self.audio_buffer               = bytearray()
        self.websocket_clients          = set()
        self.current_websocket          = None
        self.is_shutting_down           = False
        self.active_tasks               = set()
        self.max_buffer_size            = 2_000_000
        self.buffer_warning_threshold   = 1_500_000
        self.rag_system                 = None
        self.restart_flag_file          = os.path.join(BASE_DIR, "chroma_db_restart.flag")
        self.is_playing_audio           = False

    async def cleanup(self):
        self.is_shutting_down = True
        self.is_recording     = False
        for task in self.active_tasks:
            if not task.done():
                task.cancel()
        for ws in self.websocket_clients.copy():
            try:
                await ws.close()
            except Exception:
                pass
        await close_lm_client()
        if os.path.exists(self.restart_flag_file):
            try:
                os.remove(self.restart_flag_file)
            except Exception:
                pass


global_state  = GlobalState()
tts_executor  = ThreadPoolExecutor(max_workers=TTS_THREAD_WORKERS)


# ============================================================
# KOKORO TTS ENGINE - GPU ACCELERATED WITH VOICE BLENDING
# ============================================================
class KokoroTTSEngine:
    def __init__(self):
        self.pipeline     = None
        self.is_ready     = False
        self.blended_voice = None

        # ── Voice blend config ──────────────────────────────────────────
        # Community favourite: af_heart (warm) + af_bella (crisp) 60/40
        # Change weights or swap voices to taste.
        self.voice_blend = {
            'af_heart': 0.6,   # 60% Heart — warm, smooth
            'af_bella': 0.4,   # 40% Bella — crisp, clear
        }
        # ────────────────────────────────────────────────────────────────

        self.speed       = 0.92
        self.sample_rate = 24000
        self._lock       = threading.Lock()
        self.voice_cache = {}   # cache loaded voice tensors

        # GPU DETECTION
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[KOKORO] 🚀 GPU Available: {torch.cuda.is_available()}")
        if self.device == 'cuda':
            print(f"[KOKORO] 🎮 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[KOKORO] 💻 GPU not available, using CPU")

        self.init_thread = threading.Thread(target=self._initialize_model, daemon=True)
        self.init_thread.start()

    # ----------------------------------------------------------
    # Voice list (use in voice_blend dict above):
    # af_heart    American female, warm & natural  ← default
    # af_bella    American female, crisp
    # af_nova     American female, confident
    # af_sarah    American female, soft
    # af_sky      American female, upbeat
    # am_adam     American male, neutral
    # am_michael  American male, deep
    # am_echo     American male, smooth
    # am_eric     American male, casual
    # bf_emma     British female, proper
    # bf_isabella British female, elegant
    # bm_george   British male, classic
    # bm_lewis    British male, younger
    # ----------------------------------------------------------

    def _get_voice_tensor(self, voice_name: str):
        """Load voice .pt file from HuggingFace cache or local voices/ folder."""
        if voice_name in self.voice_cache:
            return self.voice_cache[voice_name]

        script_dir = os.path.dirname(os.path.abspath(__file__))

        candidates = [
            # Confirmed HuggingFace cache location
            os.path.join(HF_VOICES_DIR, f"{voice_name}.pt"),
            # Also scan all snapshots dynamically (survives model updates)
            *self._hf_snapshot_candidates(voice_name),
            # Local voices/ folder next to script
            os.path.join(script_dir, "voices", f"{voice_name}.pt"),
            os.path.join(script_dir, f"{voice_name}.pt"),
            os.path.join(os.getcwd(), "voices", f"{voice_name}.pt"),
        ]

        voice_path = next((p for p in candidates if os.path.isfile(p)), None)

        if voice_path is None:
            print(f"[KOKORO] ⚠️ Voice file not found for: {voice_name}")
            print(f"[KOKORO]    Looked in: {candidates[0]}")
            print(f"[KOKORO]    Download: https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices")
            return None

        try:
            tensor = torch.load(voice_path, map_location=self.device, weights_only=True)
            self.voice_cache[voice_name] = tensor
            print(f"[KOKORO] ✅ Loaded: {voice_path}  shape={tensor.shape}")
            return tensor
        except Exception as e:
            print(f"[KOKORO] ⚠️ Could not load {voice_path}: {e}")
            return None

    def _hf_snapshot_candidates(self, voice_name):
        """Dynamically list all HuggingFace snapshot dirs for robustness."""
        base = os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub",
            "models--hexgrad--Kokoro-82M", "snapshots"
        )
        if not os.path.isdir(base):
            return []
        return [
            os.path.join(base, snap, "voices", f"{voice_name}.pt")
            for snap in os.listdir(base)
        ]

    def _create_blended_voice(self):
        """Plain weighted average of voice tensors — no normalization."""
        blended      = None
        total_weight = sum(self.voice_blend.values())

        for voice_name, weight in self.voice_blend.items():
            tensor = self._get_voice_tensor(voice_name)
            if tensor is None:
                continue
            norm_weight = weight / total_weight
            if blended is None:
                blended = tensor * norm_weight
            else:
                # Handle any shape mismatch
                min_len = min(blended.shape[-1], tensor.shape[-1])
                blended = blended[..., :min_len] + tensor[..., :min_len] * norm_weight
            print(f"[KOKORO] 🎯 Added {voice_name} weight={norm_weight:.2f}")

        if blended is None:
            print("[KOKORO] ⚠️ Blend failed — falling back to af_heart")
            return 'af_heart'

        print(f"[KOKORO] ✅ Blended voice tensor shape={blended.shape}")
        return blended

    def _initialize_model(self):
        try:
            print(f"[KOKORO] 🚀 Loading Kokoro TTS on {self.device.upper()}...")

            self.pipeline = KPipeline(
                lang_code='a',
                device=self.device
            )

            print("[KOKORO] 🔥 Loading & blending voices...")
            self.blended_voice = self._create_blended_voice()

            # ── Monkey-patch load_voice to accept a pre-blended tensor ──
            # Without this, KPipeline calls voice.split() treating it as a
            # string name → crashes with TypeError on a Tensor.
            _original_load_voice = self.pipeline.load_voice
            def _patched_load_voice(voice):
                if isinstance(voice, torch.Tensor):
                    return voice
                return _original_load_voice(voice)
            self.pipeline.load_voice = _patched_load_voice
            print("[KOKORO] 🔧 Patched pipeline.load_voice for tensor support")

            if isinstance(self.blended_voice, str):
                print(f"[KOKORO] ⚠️ Blend failed — warming up fallback '{self.blended_voice}'...")
            else:
                print("[KOKORO] ✅ Blend created — warming up...")

            # Single warmup pass
            for _, _, audio in self.pipeline(
                "Hello, warming up now.",
                voice=self.blended_voice,
                speed=self.speed,
            ):
                if isinstance(audio, torch.Tensor):
                    audio.cpu()

            self.is_ready = True
            blend_label = (
                "FALLBACK:" + self.blended_voice
                if isinstance(self.blended_voice, str)
                else "+".join(f"{v}@{w}" for v, w in self.voice_blend.items())
            )
            print(f"[KOKORO] ✅ Ready! Voice: {blend_label} on {self.device.upper()}!")

        except Exception as e:
            print(f"[KOKORO] ❌ Init error: {e}")
            import traceback
            traceback.print_exc()

    def wait_until_ready(self):
        self.init_thread.join(timeout=60)
        return self.is_ready

    def set_voice_blend(self, blend_dict):
        """Dynamically change voice blend at runtime."""
        self.voice_blend = blend_dict
        self.voice_cache = {}
        self.blended_voice = self._create_blended_voice()
        print(f"[KOKORO] 🎯 New blend: {blend_dict}")

    def _split_into_phrases(self, text: str) -> list:
        """Split on sentence boundaries + word-count limit for GPU efficiency."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        phrases = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            words = sentence.split()
            if len(words) <= PHRASE_WORD_TARGET:
                phrases.append(sentence)
            else:
                # Split long sentences at comma/semicolon boundaries
                parts = re.split(r'(?<=[,;])\s+', sentence)
                current = ""
                for part in parts:
                    candidate = (current + " " + part).strip() if current else part
                    if len(candidate.split()) >= PHRASE_WORD_TARGET:
                        if current:
                            phrases.append(current.strip())
                        phrases.append(part.strip())
                        current = ""
                    else:
                        current = candidate
                if current.strip():
                    phrases.append(current.strip())
        return phrases if phrases else [text]

    def _stream_to_queue(self, text: str, queue, loop):
        """
        Split into short phrases first (GPU efficiency),
        then stream each phrase chunk-by-chunk as Kokoro generates.
        Sends None sentinel when all phrases are done.
        """
        try:
            phrases = self._split_into_phrases(text)
            for phrase in phrases:
                with self._lock:
                    for gs, ps, audio in self.pipeline(
                        phrase,
                        voice=self.blended_voice,
                        speed=self.speed,
                    ):
                        if audio is not None and len(audio) > 0:
                            if isinstance(audio, torch.Tensor):
                                audio = audio.cpu().numpy()
                            asyncio.run_coroutine_threadsafe(
                                queue.put(audio.astype(np.float32)), loop)
        except Exception as e:
            print(f"[KOKORO] ❌ Stream error: {e}")
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    async def inference_stream_async(self, text: str):
        """
        Short phrases → Kokoro → stream chunks immediately.
        GPU processes short bursts instead of long sustained load.
        """
        if not self.is_ready:
            return

        print(f"[KOKORO] 📝 Streaming: '{text[:60]}'")
        loop  = asyncio.get_event_loop()
        queue = asyncio.Queue()

        loop.run_in_executor(tts_executor, self._stream_to_queue, text, queue, loop)

        while True:
            if global_state.is_shutting_down:
                break
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk


tts_model = KokoroTTSEngine()


# ============================================================
# WEBSOCKET MONITOR
# ============================================================
class WebSocketMonitor:
    def __init__(self):
        self.sent_bytes   = 0
        self.sent_packets = 0
        self.send_times   = deque(maxlen=100)
        self.last_send_time = None

    def log_send(self, bytes_sent, slice_num, total_slices):
        now = time.time()
        if self.last_send_time:
            interval = now - self.last_send_time
            self.send_times.append(interval)
            if interval > 0.1:
                debug.ws(f"Large gap: {interval*1000:.1f}ms")
        self.last_send_time  = now
        self.sent_bytes     += bytes_sent
        self.sent_packets   += 1
        debug.ws(f"📤 Slice {slice_num}/{total_slices} | {bytes_sent}B | "
                 f"Total:{self.sent_bytes/1024:.1f}KB")

    def get_stats(self):
        return {
            'sent_bytes':   self.sent_bytes,
            'sent_packets': self.sent_packets,
            'avg_interval': (sum(self.send_times) / len(self.send_times)) if self.send_times else 0,
            'max_interval': max(self.send_times) if self.send_times else 0,
        }

    def reset(self):
        self.sent_bytes     = 0
        self.sent_packets   = 0
        self.send_times.clear()
        self.last_send_time = None


# ============================================================
# WEB AUDIO STREAMER
# ============================================================
class WebAudioStreamer:
    def __init__(self):
        self.is_streaming           = False
        self.current_session_id     = None
        self.stream_lock            = asyncio.Lock()
        self.text_queue             = asyncio.Queue()
        self.processing_task        = None
        self.current_playback_task  = None
        self.pending_confirmation   = None
        self.session_complete       = asyncio.Event()

        self.total_chunks_sent  = 0
        self.failed_chunks      = 0
        self.ws_monitor         = WebSocketMonitor()

        self.confirmation_timeout = 45.0

        print("[STREAMER] 🎯 Kokoro WebAudio Streamer initialized")
        print(f"[STREAMER] Phrase={PHRASE_WORD_TARGET}w | "
              f"Crossfade={CROSSFADE_SAMPLES}smp | "
              f"Volume gain={VOLUME_GAIN}x")

    async def add_text_to_stream(self, text, is_voice=True):
        if not text or not text.strip():
            return
        if self.text_queue.qsize() >= 5:
            debug.warning("Queue full — dropping oldest item")
            try:
                self.text_queue.get_nowait()
                self.text_queue.task_done()
            except Exception:
                pass
        await self.text_queue.put((text, is_voice))
        debug.tts(f"📥 Queued: '{text[:60]}' (q={self.text_queue.qsize()})")

        if not self.processing_task or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._process_stream_queue())
            global_state.active_tasks.add(self.processing_task)
            self.processing_task.add_done_callback(
                lambda t: global_state.active_tasks.discard(t))

    async def _pregenerate_audio(self, text: str) -> list:
        """
        Pre-generate TTS audio for a sentence into a list of numpy arrays.
        Called in the background WHILE the previous sentence is playing,
        so there is zero wait when it is time to send.
        """
        chunks = []
        try:
            async for audio in tts_model.inference_stream_async(text):
                if global_state.is_shutting_down:
                    break
                if audio is not None and len(audio) > 0:
                    rms = np.sqrt(np.mean(audio ** 2))
                    if rms >= 0.005:
                        chunks.append(audio)
        except Exception as e:
            debug.error(f"Pre-generate error: {e}")
        return chunks

    async def _process_stream_queue(self):
        """
        Key idea: pre-generate sentence N+1 audio WHILE sentence N is playing.
        When browser sends AUDIO_FINISHED, the next audio is already ready
        so it fires instantly with no gap.
        """
        print("[STREAMER] 🚀 Queue processor started")

        next_text       = None
        next_is_voice   = True
        next_audio_task = None  # background pre-generation task

        while not global_state.is_shutting_down:
            try:
                # ── Get current sentence (or use pre-fetched one) ──────────
                if next_text is not None:
                    text, is_voice = next_text, next_is_voice
                    next_text = None
                else:
                    text, is_voice = await asyncio.wait_for(
                        self.text_queue.get(), timeout=1.0)

                if not text:
                    self.text_queue.task_done()
                    continue

                # ── Pre-generate audio for this sentence ───────────────────
                # (runs immediately — on GPU it's fast ~0.4s)
                if next_audio_task is None:
                    pre_audio = await self._pregenerate_audio(text)
                else:
                    # Already pre-generated in the background
                    pre_audio = await next_audio_task
                    next_audio_task = None

                # ── Peek at next sentence and start pre-generating NOW ─────
                # This runs in background while we wait for AUDIO_FINISHED
                try:
                    next_text, next_is_voice = self.text_queue.get_nowait()
                    debug.tts(f"🔮 Pre-generating next: '{next_text[:40]}'")
                    next_audio_task = asyncio.create_task(
                        self._pregenerate_audio(next_text))
                except asyncio.QueueEmpty:
                    next_text = None

                # ── Open session and send pre-generated audio ──────────────
                self.pending_confirmation = asyncio.Future()

                if not await self._start_streaming_session():
                    debug.error("Could not open session")
                    if next_audio_task:
                        next_audio_task.cancel()
                    break

                try:
                    t_start    = time.time()
                    first_sent = False
                    debug.tts(f"🔊 Sending: '{text[:60]}'")

                    for audio in pre_audio:
                        if global_state.is_shutting_down or not self.is_streaming:
                            break
                        await self._send_audio(audio)
                        self.total_chunks_sent += 1
                        if not first_sent:
                            print(f"[STREAMER] ⚡ First audio in {time.time()-t_start:.3f}s")
                            first_sent = True

                    debug.tts(f"✅ {len(pre_audio)} chunks sent")
                    await self._end_streaming_session()

                except Exception as e:
                    debug.error(f"Send error: {e}")
                finally:
                    self.session_complete.set()

                # ── Wait for AUDIO_FINISHED while watching for next sentence ──
                # Problem: sentence 2 often arrives in queue AFTER get_nowait()
                # was already called, so next_audio_task = None.
                # Fix: poll the queue every 50ms while waiting — the moment
                # sentence 2 text arrives, immediately start pre-generating it
                # on GPU so it's ready by the time AUDIO_FINISHED comes back.
                if self.pending_confirmation and not self.pending_confirmation.done():
                    debug.tts("⏳ Waiting for AUDIO_FINISHED…")
                    deadline = time.time() + self.confirmation_timeout
                    while (not self.pending_confirmation.done()
                           and time.time() < deadline
                           and not global_state.is_shutting_down):

                        # If next sentence just arrived and we haven't started
                        # pre-generating yet, kick it off right now
                        if next_audio_task is None and next_text is None:
                            try:
                                next_text, next_is_voice = self.text_queue.get_nowait()
                                debug.tts(f"🔮 Sentence arrived — pre-generating NOW: '{next_text[:40]}'")
                                next_audio_task = asyncio.create_task(
                                    self._pregenerate_audio(next_text))
                            except asyncio.QueueEmpty:
                                pass

                        try:
                            await asyncio.wait_for(
                                asyncio.shield(self.pending_confirmation),
                                timeout=0.05)
                        except asyncio.TimeoutError:
                            continue  # keep polling
                        except Exception:
                            break

                    if not self.pending_confirmation.done():
                        debug.warning(f"⚠️ No AUDIO_FINISHED — advancing anyway")
                    else:
                        debug.tts("✅ Browser confirmed — next audio ready, firing instantly")

                self.pending_confirmation = None
                self.session_complete.clear()
                self.text_queue.task_done()

            except asyncio.TimeoutError:
                if global_state.is_shutting_down:
                    break
                # Queue empty and nothing pre-fetched — we are done
                if next_audio_task:
                    next_audio_task.cancel()
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                debug.error(f"Queue error: {e}")
                await asyncio.sleep(0.1)

        print("[STREAMER] ✅ Queue processor done")

    async def _start_streaming_session(self):
        async with self.stream_lock:
            if self.is_streaming or not global_state.current_websocket:
                return False
            try:
                self.current_session_id = f"stream_{int(time.time() * 1000)}"
                await global_state.current_websocket.send(
                    f"AUDIO_START:{self.current_session_id}")
                await asyncio.sleep(0.05)

                self.is_streaming             = True
                global_state.is_playing_audio = True
                self.stream_start_time        = time.time()
                self.ws_monitor.reset()
                self.total_chunks_sent        = 0
                self.failed_chunks            = 0

                print(f"[STREAMER] 🎯 AUDIO_START — {self.current_session_id}")
                return True
            except websockets.exceptions.ConnectionClosed:
                debug.error("WebSocket closed at start")
                self.is_streaming             = False
                global_state.is_playing_audio = False
                return False

    async def _end_streaming_session(self):
        async with self.stream_lock:
            if (self.is_streaming
                    and global_state.current_websocket
                    and not global_state.is_shutting_down):
                try:
                    await global_state.current_websocket.send("AUDIO_END")
                    print("[STREAMER] 🎯 AUDIO_END sent")
                except Exception:
                    pass
            self.is_streaming             = False
            global_state.is_playing_audio = False
            self.current_session_id       = None

    async def _stream_text(self, text, is_voice=True):
        if not global_state.current_websocket:
            debug.tts("❌ No WebSocket")
            if self.pending_confirmation and not self.pending_confirmation.done():
                self.pending_confirmation.set_result(False)
            return

        if not await self._start_streaming_session():
            debug.tts("❌ Could not start streaming session")
            if self.pending_confirmation and not self.pending_confirmation.done():
                self.pending_confirmation.set_result(False)
            return

        try:
            debug.tts(f"🔊 TTS: '{text[:60]}'")
            chunk_count  = 0
            silent_count = 0
            first_sent   = False
            t_start      = time.time()

            async for audio in tts_model.inference_stream_async(text):
                if global_state.is_shutting_down or not self.is_streaming:
                    break
                if audio is None or len(audio) == 0:
                    continue

                rms = np.sqrt(np.mean(audio ** 2))
                if rms < 0.005:
                    silent_count += 1
                    continue

                success = await self._send_audio(audio)
                if success:
                    chunk_count            += 1
                    self.total_chunks_sent += 1
                    if not first_sent:
                        print(f"[STREAMER] ⚡ First audio in {time.time()-t_start:.3f}s")
                        first_sent = True

            debug.tts(f"✅ {chunk_count} chunks sent, {silent_count} silent skipped")
            await self._end_streaming_session()

        except Exception as e:
            debug.error(f"Stream error: {e}")
            import traceback
            traceback.print_exc()
            if self.pending_confirmation and not self.pending_confirmation.done():
                self.pending_confirmation.set_result(False)
        finally:
            async with self.stream_lock:
                self.is_streaming             = False
                global_state.is_playing_audio = False
                self.current_session_id       = None

    async def _send_audio(self, audio: np.ndarray) -> bool:
        """
        Web-aligned audio send:
        1. Fixed gain boost (no dynamic processing = no crackling)
        2. JSON header so browser knows sample rate + byte length
        3. Complete phrase as ONE binary message (no slicing)
        """
        try:
            if not global_state.current_websocket or not self.is_streaming:
                return False

            # Fixed gain — no RMS normalize, no tanh, no dynamic processing
            # Those cause crackling. Just amplify by a fixed multiplier.
            audio_boosted = np.clip(
                audio.astype(np.float32) * VOLUME_GAIN, -1.0, 1.0)
            audio_int16   = (audio_boosted * 32767).astype(np.int16)
            audio_bytes   = audio_int16.tobytes()

            if len(audio_bytes) == 0:
                debug.warning("Empty audio bytes — skipping")
                return False

            # JSON header — browser pre-allocates AudioBuffer correctly
            header = json.dumps({
                "type":        "audio_chunk",
                "sample_rate": SAMPLE_RATE,
                "channels":    1,
                "encoding":    "pcm16",
                "bytes":       len(audio_bytes),
                "duration_ms": round(len(audio_int16) / SAMPLE_RATE * 1000, 1),
            })
            await global_state.current_websocket.send(header)

            # One complete phrase per message = perfectly aligned
            await global_state.current_websocket.send(audio_bytes)

            self.ws_monitor.log_send(len(audio_bytes), self.total_chunks_sent + 1, "?")
            debug.ws(f"📤 Phrase sent: {len(audio_bytes)//2} samples "
                     f"({len(audio_bytes)/1024:.1f}KB)")

            return True

        except websockets.exceptions.ConnectionClosed:
            debug.error("WebSocket closed during send")
            self.is_streaming             = False
            global_state.is_playing_audio = False
        except Exception as e:
            debug.error(f"Send error: {e}")
            self.failed_chunks += 1

        return False

    async def confirm_playback_complete(self, session_id=None):
        expected = getattr(self, '_expected_confirmation_id', None)
        if session_id and expected and session_id != expected:
            debug.warning(f"⚠️ Stale confirmation for '{session_id}' (want '{expected}')")
            return
        debug.tts(f"✅ Browser confirmed (session={session_id or 'untagged'})")
        if self.pending_confirmation and not self.pending_confirmation.done():
            self.pending_confirmation.set_result(True)
        self._expected_confirmation_id = None


audio_streamer = WebAudioStreamer()


# ============================================================
# SIMPLE WORD FIXER
# ============================================================
class SimpleWordFixer:
    def __init__(self):
        self.common_fixes = [
            (r"(?i)\bdont\b",  "don't"),
            (r"(?i)\bcant\b",  "can't"),
            (r"(?i)\bwont\b",  "won't"),
            (r"(?i)\bim\b",    "I'm"),
            (r"(?i)\byoure\b", "you're"),
            (r"(?i)\btheyre\b","they're"),
            (r"(?i)\bwere\b",  "we're"),
            (r"(?i)\bthats\b", "that's"),
            (r"(?i)\bwhats\b", "what's"),
            (r"(?i)\bwheres\b","where's"),
            (r"(?i)\bhowre\b", "how're"),
            (r"(?i)\bwhys\b",  "why's"),
            (r"(?i)\bwhore\b", "who're"),
            (r"(?i)\bits\b",   "it's"),
            (r"(?i)\bshes\b",  "she's"),
            (r"(?i)\bhes\b",   "he's"),

            (r"(?i)\btristan\b",   "Tristhan"),
            (r"(?i)\btristen\b",   "Tristhan"),
            (r"(?i)\btristin\b",   "Tristhan"),
            (r"(?i)\btristian\b",  "Tristhan"),
            (r"(?i)\btrystan\b",   "Tristhan"),
            (r"(?i)\btrysten\b",   "Tristhan"),
            (r"(?i)\btrystin\b",   "Tristhan"),
            (r"(?i)\btriztan\b",   "Tristhan"),
            (r"(?i)\btrizten\b",   "Tristhan"),
            (r"(?i)\btriztin\b",   "Tristhan"),
            (r"(?i)\btristanh\b",  "Tristhan"),
            (r"(?i)\btristam\b",   "Tristhan"),

            (r"(?i)\bcabera\b",   "Cabrera"),
            (r"(?i)\bcabrira\b",  "Cabrera"),
            (r"(?i)\bcabrerra\b", "Cabrera"),
            (r"(?i)\bcabrerah\b", "Cabrera"),
            (r"(?i)\bcabrella\b", "Cabrera"),
            (r"(?i)\bcabrara\b",  "Cabrera"),

            (r"(?i)\btristan cabrera\b",  "Tristhan D. Cabrera"),
            (r"(?i)\btristen cabrera\b",  "Tristhan D. Cabrera"),
            (r"(?i)\btristin cabrera\b",  "Tristhan D. Cabrera"),
            (r"(?i)\btrystan cabrera\b",  "Tristhan D. Cabrera"),
        ]

    def fix_text(self, text: str) -> str:
        if not text or not text.strip():
            return text
        for pattern, replacement in self.common_fixes:
            text = re.sub(pattern, replacement, text)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"([.,!?;:])(\w)", r"\1 \2", text)
        text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
        text = re.sub(r"(\d)(?!(st|nd|rd|th))([a-zA-Z])", r"\1 \3", text)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text


word_fixer = SimpleWordFixer()


# ============================================================
# DYNAMIC LANGCHAIN RAG SYSTEM
# ============================================================
class DynamicLangChainRAGSystem:
    def __init__(self, base_dir, school_docs_dir, persist_directory):
        self.base_dir           = base_dir
        self.school_docs_dir    = school_docs_dir
        self.persist_directory  = persist_directory
        self.vectorstore        = None
        self.retriever          = None
        self.memory             = None
        self.is_initialized     = False
        self.word_fixer         = word_fixer
        self.restart_flag_file  = os.path.join(base_dir, "chroma_db_restart.flag")

    def check_for_restart(self):
        if os.path.exists(self.restart_flag_file):
            print("[RAG] 🔄 Restart flag detected! Reinitializing...")
            try:
                os.remove(self.restart_flag_file)
            except Exception:
                pass
            return True
        return False

    def initialize(self):
        print("[LANGCHAIN] 🚀 Initializing DYNAMIC RAG system...")
        print(f"[LANGCHAIN] 📁 Docs: {self.school_docs_dir}")
        print(f"[LANGCHAIN] 📁 ChromaDB: {self.persist_directory}")

        if self.check_for_restart():
            import shutil
            if os.path.exists(self.persist_directory):
                try:
                    shutil.rmtree(self.persist_directory)
                    print("[LANGCHAIN] ✅ Deleted old ChromaDB")
                except Exception as e:
                    print(f"[LANGCHAIN] ⚠️ Could not delete ChromaDB: {e}")

        model_kwargs  = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings    = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        print(f"[LANGCHAIN] ✅ Embeddings on {model_kwargs['device']}")

        if os.path.exists(self.persist_directory):
            try:
                print("[LANGCHAIN] 🔄 Loading existing vector DB...")
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=embeddings,
                )
                print("[LANGCHAIN] ✅ Loaded existing vector DB")
            except Exception as e:
                print(f"[LANGCHAIN] ❌ {e} — rebuilding")
                if not self._index_all_documents(embeddings):
                    return False
        else:
            if not self._index_all_documents(embeddings):
                return False

        if self.vectorstore:
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7},
            )
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history", k=3,
                return_messages=True, output_key='answer',
            )
            print("[LANGCHAIN] ✅ Retriever + memory ready")
            self.is_initialized = True
            return True
        return False

    def _get_all_pdf_files(self):
        pdf_files = []
        for section in DOCUMENT_SECTIONS.keys():
            section_dir = os.path.join(self.school_docs_dir, section)
            if os.path.exists(section_dir):
                pdfs = glob.glob(os.path.join(section_dir, "*.pdf"))
                pdf_files.extend(pdfs)
                print(f"[LANGCHAIN] 📂 {section}: {len(pdfs)} PDF(s)")
        print(f"[LANGCHAIN] 📚 Total: {len(pdf_files)} PDFs")
        return pdf_files

    def _index_all_documents(self, embeddings):
        try:
            pdf_files     = self._get_all_pdf_files()
            all_documents = []

            if pdf_files:
                for pdf_path in pdf_files:
                    try:
                        loader = PyPDFLoader(pdf_path)
                        docs   = loader.load()
                        for doc in docs:
                            parts = pdf_path.split(os.sep)
                            if len(parts) >= 2:
                                doc.metadata['section'] = parts[-2]
                            doc.metadata['source'] = os.path.basename(pdf_path)
                        all_documents.extend(docs)
                        print(f"[LANGCHAIN] ✅ {len(docs)} pages from {os.path.basename(pdf_path)}")
                    except Exception as e:
                        print(f"[LANGCHAIN] ⚠️ Failed {pdf_path}: {e}")

            if all_documents:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
                )
                chunks = splitter.split_documents(all_documents)
                print(f"[LANGCHAIN] 🔪 {len(chunks)} chunks")
                self.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=self.persist_directory,
                )
            else:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=embeddings,
                )

            print("[LANGCHAIN] ✅ Documents indexed!")
            return True
        except Exception as e:
            print(f"[LANGCHAIN] ❌ Indexing error: {e}")
            return False

    def reinitialize_from_scratch(self):
        self.vectorstore    = None
        self.retriever      = None
        self.is_initialized = False
        return self.initialize()

    async def smart_search(self, query):
        if not self.retriever:
            return [], []
        try:
            fixed = self.word_fixer.fix_text(query)
            if fixed != query:
                print(f"[LANGCHAIN] 🔧 Query fixed: '{query}' → '{fixed}'")
                query = fixed
            docs      = await self.retriever.ainvoke(query)
            documents = [d.page_content for d in docs]
            metadatas = [{'page': d.metadata.get('page', 'N/A'),
                          'source': d.metadata.get('source', 'N/A'),
                          'section': d.metadata.get('section', 'N/A')} for d in docs]
            print(f"[LANGCHAIN] 🔍 {len(documents)} chunks for '{query}'")
            return documents, metadatas
        except Exception as e:
            print(f"[LANGCHAIN] ❌ Search error: {e}")
            return [], []

    async def get_contextual_prompt(self, query, max_chunks=4):
        fixed = self.word_fixer.fix_text(query)
        if fixed != query:
            query = fixed
        relevant_docs, _ = await self.smart_search(query)
        if not relevant_docs:
            return (f"Question: {query}\n\n"
                    "Please answer based on general knowledge. "
                    "No specific information was found in the database.")
        context = "\n\n".join(relevant_docs[:max_chunks])
        return f"""Here is information about your CREATOR, Tristhan D. Cabrera (NOT about you):

{context}

Question: {query}

IMPORTANT INSTRUCTIONS:
1. You are GAIJIN, the AI assistant. You are NOT Tristhan.
2. Refer to Tristhan in THIRD PERSON (e.g., "My creator Tristhan...")
3. Never say "I built..." or "I worked on..." for Tristhan's projects
4. If asked who YOU are: you are Gaijin, an AI made BY Tristhan
5. Keep your answer concise and conversational

Answer naturally:"""


rag_system = DynamicLangChainRAGSystem(
    base_dir=BASE_DIR,
    school_docs_dir=SCHOOL_DOCS_DIR,
    persist_directory=PERSIST_DIRECTORY,
)


# ============================================================
# HTTP CLIENT
# ============================================================
lm_client = None


async def initialize_lm_client():
    global lm_client
    if lm_client is None:
        print("[LM_CLIENT] 🚀 Initializing HTTP client...")
        lm_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0),
            transport=httpx.AsyncHTTPTransport(
                retries=1,
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=3),
            ),
        )
        print("[LM_CLIENT] ✅ Ready")


async def close_lm_client():
    global lm_client
    if lm_client:
        await lm_client.aclose()
        lm_client = None


# ============================================================
# WHISPER STT
# ============================================================
class WhisperSTT:
    def __init__(self):
        try:
            if torch.cuda.is_available():
                print("[STT] Loading Whisper on CUDA...")
                self.whisper_model = WhisperModel("base", device="cuda",
                                                   compute_type="float16")
                print("[STT] ✅ Whisper ready on CUDA!")
            else:
                self.whisper_model = WhisperModel("base", device="cpu",
                                                   compute_type="int8")
                print("[STT] ✅ Whisper ready on CPU!")
        except Exception as e:
            print(f"[STT] ❌ CUDA error: {e} — falling back to CPU")
            self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

    async def process_audio_buffer(self, audio_buffer):
        if not audio_buffer or len(audio_buffer) < 1000:
            return ""
        print(f"[STT] 🔊 Processing {len(audio_buffer)} bytes...")
        t = time.time()
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
            try:
                with wave.open(tmp, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(audio_buffer)
                segments, _ = self.whisper_model.transcribe(
                    tmp, language="en",
                    beam_size=1, best_of=1,
                    without_timestamps=True, patience=1,
                    vad_filter=True,
                    vad_parameters=dict(threshold=0.5,
                                        min_speech_duration_ms=250,
                                        min_silence_duration_ms=100),
                )
                text = " ".join(s.text for s in segments).strip()
            finally:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass

            print(f"[STT] ✅ {time.time()-t:.3f}s → '{text}'")
            return text
        except Exception as e:
            print(f"[STT] ❌ Error: {e}")
            return ""


whisper_stt = WhisperSTT()


# ============================================================
# PIPELINE
# ============================================================
class Pipeline:
    def __init__(self):
        self.word_fixer = word_fixer

    @staticmethod
    def _format_digits_for_tts(text: str) -> str:
        _ORDINAL_WORDS = {
            1: 'first',   2: 'second',  3: 'third',   4: 'fourth',
            5: 'fifth',   6: 'sixth',   7: 'seventh',  8: 'eighth',
            9: 'ninth',  10: 'tenth',  11: 'eleventh', 12: 'twelfth',
            13: 'thirteenth', 14: 'fourteenth', 15: 'fifteenth',
            16: 'sixteenth',  17: 'seventeenth', 18: 'eighteenth',
            19: 'nineteenth', 20: 'twentieth',
            21: 'twenty first',  22: 'twenty second', 23: 'twenty third',
            24: 'twenty fourth', 25: 'twenty fifth',  26: 'twenty sixth',
            27: 'twenty seventh',28: 'twenty eighth', 29: 'twenty ninth',
            30: 'thirtieth',
        }
        text = re.sub(
            r'([A-Za-z]*)(\d+)(st|nd|rd|th)\b',
            lambda m: (m.group(1) + ' ' if m.group(1) else '')
                      + (_ORDINAL_WORDS.get(int(m.group(2)), str(m.group(2)))),
            text, flags=re.IGNORECASE)
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)

        def _url_to_speech(m):
            url = re.sub(r'^https?://', '', m.group(0))
            url = re.sub(r'^www\.', '', url)
            url = url.replace('/', ' slash ').replace('.', ' dot ').replace('-', ' ')
            return re.sub(r'\s+', ' ', url).strip()
        text = re.sub(r'https?://[^\s,;)\'\"]+|www\.[^\s,;)\'\"]+', _url_to_speech, text)

        def _email_to_speech(m):
            return f"{m.group(1)} at {m.group(2).replace('.', ' dot ')}"
        text = re.sub(r'([A-Za-z0-9._%+\-]+)@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})',
                      _email_to_speech, text)

        text = re.sub(r'\d{5,}', lambda m: ' '.join(list(m.group(0))), text)
        return text

    def _clean_text_for_tts(self, text):
        if not text or not text.strip():
            return ""
        text = self.word_fixer.fix_text(text)
        text = re.sub(r'###\s*(User|AI|System|Jarvis|Gaijin):?', '', text)
        text = re.sub(r'^(User|AI|System|Jarvis|Gaijin):?\s*', '', text)
        text = self._format_digits_for_tts(text)
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            print(f"[PIPELINE] 📝 TTS-ready: '{text[:80]}'")
        return text

    async def stream_text(self, text, is_voice=True):
        if global_state.is_shutting_down or not text or not text.strip():
            return
        if any(m in text for m in ['###', 'User:', 'AI:', 'System:', 'Jarvis:', 'Gaijin:']):
            return
        cleaned = self._clean_text_for_tts(text)
        if cleaned:
            await audio_streamer.add_text_to_stream(cleaned, is_voice)


pipeline = Pipeline()


# ============================================================
# LM STUDIO
# ============================================================
async def stream_from_lm_studio_enhanced(prompt, text_callback=None, is_voice=True):
    global lm_client
    if global_state.is_shutting_down:
        return "System shutting down"
    if lm_client is None:
        await initialize_lm_client()

    lm_start = time.time()

    if is_voice:
        system_content = """You are Gaijin, an AI voice assistant.

IDENTITY:
- Your name is Gaijin
- You are an AI assistant, NOT a human
- You are NOT Tristhan D. Cabrera
- Tristhan D. Cabrera is your CREATOR — he is a Computer Engineering student from Lipa City, Batangas
- Always refer to Tristhan in THIRD PERSON (e.g., "My creator Tristhan..." or "Tristhan built...")
- His projects, achievements, and experiences belong to HIM, not you
- Never say "I built", "I developed", or "I worked on" when referring to Tristhan's work

RESPONSE RULES:
- Keep responses VERY SHORT (1 to 3 sentences max)
- Write for TEXT TO SPEECH
- No markdown, bullet points, or numbered lists
- No special characters
- Be conversational and friendly"""

    else:
        system_content = """You are Gaijin, a text-based AI assistant.

IDENTITY:
- Your name is Gaijin
- You are an AI assistant, NOT a human
- You are NOT Tristhan D. Cabrera
- Tristhan D. Cabrera is your CREATOR — always refer to him in third person
- His projects and achievements belong to HIM, not you
- Never claim ownership of Tristhan's work or experiences

RESPONSE RULES:
- Keep answers brief and direct
- Be helpful and conversational"""

    messages = [{"role": "system", "content": system_content}]

    fixed_prompt = word_fixer.fix_text(prompt)
    if fixed_prompt != prompt:
        print(f"[WORD_FIXER] '{prompt}' → '{fixed_prompt}'")
        prompt = fixed_prompt

    enhanced_prompt = await rag_system.get_contextual_prompt(prompt)
    messages.append({"role": "user", "content": enhanced_prompt})

    payload = {
        "model": "meta-llama-3.1-8b-instruct",
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": MAX_RESPONSE_TOKENS,
        "stream": True,
        "stop": ["###", "User:", "AI:", "System:"],
    }

    full_response   = ""
    sentence_buffer = ""

    try:
        async with lm_client.stream(
            "POST", LM_STUDIO_URL, json=payload,
            headers={'Content-Type': 'application/json'},
        ) as response:
            if response.status_code != 200:
                return f"Error: {response.status_code}"

            first_token = False
            token_count = 0
            t_stream    = time.time()

            async for line in response.aiter_lines():
                if global_state.is_shutting_down:
                    break
                if not line or not line.startswith('data: '):
                    continue
                if line == 'data: [DONE]':
                    break
                try:
                    data = json.loads(line[6:])
                    if 'choices' not in data or not data['choices']:
                        continue
                    token = data['choices'][0].get('delta', {}).get('content', '')
                    for marker in ['###', 'User:', 'AI:', 'System:', 'Jarvis:', 'Gaijin:']:
                        token = token.replace(marker, '')
                    if not token.strip():
                        continue

                    token_count += 1
                    if token_count > MAX_RESPONSE_TOKENS:
                        break
                    if not first_token:
                        print(f"[LM] ⚡ First token: {time.time()-t_stream:.3f}s")
                        first_token = True

                    full_response   += token
                    sentence_buffer += token

                    if is_voice and re.search(r'[.!?]\s*$', sentence_buffer):
                        if text_callback and sentence_buffer.strip():
                            await text_callback(sentence_buffer, is_voice=True)
                            sentence_buffer = ""

                except (json.JSONDecodeError, Exception):
                    continue

        if is_voice and sentence_buffer.strip() and text_callback \
                and not global_state.is_shutting_down:
            await text_callback(sentence_buffer, is_voice=True)

        print(f"[LM] ✅ {token_count} tokens in {time.time()-lm_start:.3f}s")

        clean = re.sub(r'###\s*(User|AI|System|Jarvis|Gaijin):?', '', full_response)
        clean = re.sub(r'^(User|AI|System|Jarvis|Gaijin):?\s*', '', clean).strip()

        if global_state.current_websocket and not global_state.is_shutting_down and clean:
            clean_display = word_fixer.fix_text(clean)
            await global_state.current_websocket.send(f"AI_RESPONSE: {clean_display}")

        return clean

    except Exception as e:
        print(f"[LM] ❌ Error: {e}")
        return f"Error: {e}"


# ============================================================
# WEBSOCKET SERVER
# ============================================================
async def websocket_handler(websocket, path):
    if global_state.is_shutting_down:
        return

    print(f"🔌 Connected: {websocket.remote_address}")
    global_state.websocket_clients.add(websocket)
    global_state.current_websocket = websocket

    try:
        pong = await websocket.ping()
        await pong
        print("✅ Ping OK")
    except Exception:
        print("⚠️ Ping failed")

    try:
        await websocket.send("CONNECTED: Python backend ready")

        async for message in websocket:
            if global_state.is_shutting_down:
                break

            if isinstance(message, bytes):
                if global_state.is_recording:
                    size = len(global_state.audio_buffer)
                    if size >= global_state.buffer_warning_threshold:
                        print(f"⚠️ Buffer warning: {size}/{global_state.max_buffer_size}")
                    if size < global_state.max_buffer_size:
                        global_state.audio_buffer.extend(message)
                    else:
                        global_state.is_recording = False
                        await websocket.send("BUFFER_FULL")

            elif isinstance(message, str):
                print(f"📨 {message[:100]}")

                if message.startswith("AUDIO_FINISHED"):
                    parts      = message.split(":", 1)
                    session_id = parts[1].strip() if len(parts) > 1 else None
                    print(f"✅ Browser audio finished (session={session_id or 'untagged'})")
                    await audio_streamer.confirm_playback_complete(session_id)

                elif message == "START_RECORDING":
                    global_state.is_recording = True
                    global_state.audio_buffer = bytearray()
                    print("🎤 Recording STARTED")
                    await websocket.send("RECORDING_STARTED")

                elif message == "STOP_RECORDING":
                    global_state.is_recording = False
                    buf_size = len(global_state.audio_buffer)
                    print(f"🛑 Stopped — {buf_size} bytes")
                    await websocket.send("PROCESSING_AUDIO")

                    if global_state.audio_buffer and buf_size > 1000:
                        user_input = await whisper_stt.process_audio_buffer(
                            global_state.audio_buffer)
                        if user_input and user_input.strip():
                            orig       = user_input
                            user_input = word_fixer.fix_text(user_input)
                            if orig != user_input:
                                print(f"[WORD_FIXER] '{orig}' → '{user_input}'")
                            print(f"👤 Voice: {user_input}")
                            await websocket.send(f"TRANSCRIBED: {user_input}")
                            t0 = time.time()
                            await stream_from_lm_studio_enhanced(
                                user_input,
                                lambda txt, is_voice=True: pipeline.stream_text(txt, is_voice=True),
                                is_voice=True,
                            )
                            print(f"[TIMING] 🚀 End-to-end: {time.time()-t0:.3f}s")
                            print("=" * 50)
                        else:
                            await websocket.send("NO_SPEECH_DETECTED")
                    else:
                        await websocket.send("NO_AUDIO_RECEIVED")

                    global_state.audio_buffer = bytearray()

                elif message.startswith("{"):
                    try:
                        data = json.loads(message)
                        if (data.get('type') == 'message'
                                and data.get('input_type') == 'text'):
                            text = data.get('content', '').strip()
                            if text:
                                print(f"📝 Text input: {text}")
                                await websocket.send(f"TRANSCRIBED: {text}")
                                t0 = time.time()
                                await stream_from_lm_studio_enhanced(text, None, is_voice=False)
                                print(f"[TIMING] 🚀 End-to-end: {time.time()-t0:.3f}s")
                                print("=" * 50)
                    except json.JSONDecodeError:
                        print(f"⚠️ Invalid JSON: {message[:50]}")

                elif message == "PING":
                    await websocket.send("PONG")

                elif message == "GET_DEBUG":
                    blend_info = {v: w for v, w in tts_model.voice_blend.items()}
                    stats = {
                        'tts_voice':     blend_info,
                        'ws':            audio_streamer.ws_monitor.get_stats(),
                        'chunks_sent':   audio_streamer.total_chunks_sent,
                        'chunks_failed': audio_streamer.failed_chunks,
                        'volume_gain':   VOLUME_GAIN,
                    }
                    await websocket.send(f"DEBUG: {json.dumps(stats)}")

    except websockets.exceptions.ConnectionClosed as e:
        print(f"🔌 Disconnected: {websocket.remote_address} — {e}")
    except Exception as e:
        print(f"🔌 WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        global_state.websocket_clients.discard(websocket)
        if global_state.current_websocket is websocket:
            global_state.current_websocket = None


async def start_websocket_server():
    print(f"🌐 WebSocket on {WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    async with serve(websocket_handler, WEBSOCKET_HOST, WEBSOCKET_PORT,
                     ping_interval=120, ping_timeout=115, close_timeout=30):
        print("✅ WebSocket server running!")
        await asyncio.Future()


# ============================================================
# MAIN
# ============================================================
async def main_async():
    print("🤖 Gaijin AI — RAG + Kokoro TTS")
    print("=" * 60)
    blend_str = " + ".join(f"{v}@{int(w*100)}%" for v, w in tts_model.voice_blend.items())
    print(f"   TTS      : Kokoro [{blend_str}] on GPU")
    print(f"   VOLUME   : {VOLUME_GAIN}x fixed gain (no dynamic processing)")
    print(f"   CHUNKING : {PHRASE_WORD_TARGET} words/phrase + {CROSSFADE_SAMPLES}-sample crossfade")
    print(f"   PROTOCOL : header+binary per phrase (web-aligned)")
    print(f"   STT      : Whisper base (GPU)")
    print(f"   RAG      : LangChain + ChromaDB")
    print(f"   DOCS     : {SCHOOL_DOCS_DIR}")
    print("=" * 60)

    setup_graceful_shutdown()

    try:
        global_state.rag_system = rag_system
        if not rag_system.initialize():
            print("[LANGCHAIN] ❌ RAG init failed — exiting")
            return
        print("[LANGCHAIN] ✅ RAG ready!")

        await initialize_lm_client()

        print("⏳ Waiting for Kokoro to warm up...")
        if tts_model.wait_until_ready():
            print("✅ All systems ready!")
        else:
            print("⚠️  Kokoro warmup timed out — will retry on first use")

        print(f"\n🔗 Upload docs to: {SCHOOL_DOCS_DIR}")
        print("=" * 60)
        await start_websocket_server()

    except asyncio.CancelledError:
        print("🔴 Main cancelled")
    except Exception as e:
        print(f"❌ Main error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await global_state.cleanup()


def main():
    exit_code = 0
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n👋 Interrupted")
    except Exception as e:
        print(f"❌ Fatal: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        tts_executor.shutdown(wait=False)
        print("🏁 Application terminated")

        restart_flag = os.path.join(BASE_DIR, "chroma_db_restart.flag")
        if os.path.exists(restart_flag):
            print("\n🔄 RESTART REQUESTED BY FLASK ADMIN!")
            try:
                import shutil
                if os.path.exists(PERSIST_DIRECTORY):
                    shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
            except Exception:
                pass
            sys.exit(99)

        sys.exit(exit_code)


if __name__ == "__main__":
    main()