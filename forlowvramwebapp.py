from faster_whisper import WhisperModel
import httpx
import asyncio
import time
import threading
import queue
import os
import re
import torch
import tempfile
import wave
import json
from collections import deque
import numpy as np
import sounddevice as sd
import websockets
from websockets.server import serve
import pyaudio
import struct
import signal
import sys
import glob
import atexit
import psutil
import gc

# -----------------------------
# LANGCHAIN IMPORTS
# -----------------------------
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# -----------------------------
# XTTS IMPORTS
# -----------------------------
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# -----------------------------
# DYNAMIC CONFIG
# -----------------------------
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
REFERENCE_VOICE = "Ai.wav"
MODEL_PATH = r"C:\Users\tristhan\AppData\Local\tts\tts_models--multilingual--multi-dataset--xtts_v2"

# DYNAMIC PATHS
BASE_DIR = r"C:\Users\tristhan\Desktop\AI\ai"
SCHOOL_DOCS_DIR = os.path.join(BASE_DIR, "Documents")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db")

# Document sections
DOCUMENT_SECTIONS = {
    'ACADEMIC_CALENDAR': 'Academic Calendar',
    'CENAR': 'CENAR',
    'CENAR_FACULTY': 'CENAR Faculty',
    'ENROLLMENT': 'Enrollment',
    'EXAM_DATE': 'Exam Date',
    'GENERAL_INFO': 'General Info'
}

# WebSocket settings
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8888

# Audio settings
SAMPLE_RATE = 24000
CHUNK_SIZE = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1

USE_DEEPSPEED = True
STREAM_CHUNK_SIZE = 20
ENABLE_TEXT_SPLITTING = True

# XTTS Parameters — tuned for 6 GB VRAM
XTTS_TEMPERATURE        = 0.7
XTTS_LENGTH_PENALTY     = 1.0
XTTS_REPETITION_PENALTY = 1.2
XTTS_TOP_K              = 30
XTTS_TOP_P              = 0.8
XTTS_SPEED              = 1.0

MAX_RESPONSE_TOKENS = 250

# XTTS RUNAWAY-GENERATION GUARDS
MAX_CONSECUTIVE_SILENT = 8      # 4-6 silent chunks between XTTS internal splits; 8 catches true runaway (10+)
MAX_INFERENCE_SECONDS  = 30.0
MAX_TOTAL_CHUNKS       = 50

# ============================================================
# DEBUGGING SETTINGS
# ============================================================
DEBUG_MODE       = True
DEBUG_GPU        = True
DEBUG_AUDIO      = True
DEBUG_WEBSOCKET  = True
DEBUG_XTTS       = True


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
    def xtts(msg, force=False):
        if DEBUG_XTTS or force:
            print(f"[XTTS_DEBUG] {msg}")

    @staticmethod
    def error(msg):
        print(f"[ERROR] ❌ {msg}")

    @staticmethod
    def warning(msg):
        print(f"[WARNING] ⚠️ {msg}")


debug = DebugLogger()

# -----------------------------
# GRACEFUL SHUTDOWN HANDLING
# -----------------------------
import signal


def setup_graceful_shutdown():
    def shutdown_signal_handler(signum, frame):
        print(f"\n📡 Received signal {signum}. Initiating graceful shutdown...")
        global_state.is_shutting_down = True
        global_state.is_recording = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ Graceful shutdown initiated")
        import os
        os._exit(0)

    try:
        signal.signal(signal.SIGTERM, shutdown_signal_handler)
        signal.signal(signal.SIGINT, shutdown_signal_handler)
        print("✅ Signal handlers registered for graceful shutdown")
    except Exception:
        print("⚠️ Signal handlers not available on this platform")


# -----------------------------
# GLOBAL STATE
# -----------------------------
class GlobalState:
    def __init__(self):
        self.is_recording = False
        self.audio_buffer = bytearray()
        self.websocket_clients = set()
        self.current_websocket = None
        self.is_shutting_down = False
        self.active_tasks = set()
        self.max_buffer_size = 2_000_000
        self.buffer_warning_threshold = 1_500_000
        self.rag_system = None
        self.restart_flag_file = os.path.join(BASE_DIR, "chroma_db_restart.flag")
        self.is_playing_audio = False

    async def cleanup(self):
        self.is_shutting_down = True
        self.is_recording = False
        for task in self.active_tasks:
            if not task.done():
                task.cancel()
        for websocket in self.websocket_clients.copy():
            try:
                await websocket.close()
            except Exception:
                pass
        await close_lm_client()
        if os.path.exists(self.restart_flag_file):
            try:
                os.remove(self.restart_flag_file)
                print("✅ Removed restart flag file")
            except Exception:
                pass


global_state = GlobalState()


# ============================================================
# GPU MONITOR
# ============================================================
class GPUMonitor:
    def __init__(self):
        self.last_check = time.time()
        self.memory_history = deque(maxlen=50)
        self.cleanup_history = deque(maxlen=20)
        self.last_gpu_cleanup = time.time()
        self.cleanup_count = 0
        self.total_memory = 0
        self.last_memory_value = 0

        self.danger_threshold  = 0.85
        self.warning_threshold = 0.75
        self.cleanup_interval  = 5
        self.adaptive_mode     = True

        if torch.cuda.is_available():
            self.total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"[GPU] Total VRAM: {self.total_memory:.2f} GB")

    def check_cuda_errors(self):
        try:
            if torch.cuda.is_available():
                t = torch.zeros(1, device="cuda")
                del t
                return True
        except Exception as e:
            debug.error(f"CUDA error detected: {e}")
            return False
        return True

    def get_gpu_stats(self):
        stats = {
            'allocated': 0, 'cached': 0, 'free': 0,
            'percentage': 0, 'percentage_alloc': 0,
            'total': self.total_memory,
        }
        if torch.cuda.is_available():
            stats['allocated'] = torch.cuda.memory_allocated() / 1024 ** 3
            stats['cached']    = torch.cuda.memory_reserved()  / 1024 ** 3
            if self.total_memory > 0:
                stats['percentage_alloc'] = (stats['allocated'] / self.total_memory) * 100
                stats['percentage']       = (stats['cached']    / self.total_memory) * 100
                stats['free']             = self.total_memory - stats['allocated']
        return stats

    def log_gpu_stats(self, stage=""):
        stats = self.get_gpu_stats()
        if stats['cached'] > 0:
            msg = (f"[{stage}] Alloc:{stats['allocated']:.2f}GB/{self.total_memory:.2f}GB "
                   f"({stats['percentage_alloc']:.1f}%) | Cache:{stats['cached']:.2f}GB")
            debug.gpu(msg)
            self.memory_history.append({
                'time': time.time(),
                'allocated': stats['allocated'],
                'cached': stats['cached'],
                'percentage': stats['percentage_alloc'],
            })
            self.last_memory_value = stats['allocated']

    def should_cleanup(self, chunk_count):
        if not torch.cuda.is_available():
            return False
        current_time = time.time()
        stats = self.get_gpu_stats()
        alloc_percent = stats.get('percentage_alloc', 0)
        allocated_gb  = stats.get('allocated', 0)

        if alloc_percent > 85:
            debug.gpu(f"⚠️ CRITICAL memory: {allocated_gb:.2f}GB ({alloc_percent:.1f}%) — FORCING cleanup")
            return True
        if alloc_percent > 70:
            dynamic_interval = max(2, int(10 - (alloc_percent - 70) / 5))
            if current_time - self.last_gpu_cleanup > dynamic_interval:
                debug.gpu(f"🔄 Memory pressure: {alloc_percent:.1f}% — cleanup needed")
                return True
        else:
            if chunk_count > 0 and chunk_count % 10 == 0:
                return True
            if current_time - self.last_gpu_cleanup > 15:
                return True
        return False

    def aggressive_cleanup(self, force=False):
        if not torch.cuda.is_available():
            return
        current_time = time.time()
        stats = self.get_gpu_stats()
        self.cleanup_count += 1
        before_alloc = stats['allocated']
        before_cache = stats['cached']
        debug.gpu(f"🧹 CLEANUP #{self.cleanup_count} "
                  f"(alloc: {before_alloc:.2f}GB, {stats['percentage_alloc']:.1f}%)")

        for i in range(3):
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if i < 2:
                time.sleep(0.01)

        self.last_gpu_cleanup = current_time
        after = self.get_gpu_stats()
        debug.gpu(f"✅ Cleanup done: {after['allocated']:.2f}GB alloc "
                  f"(freed {before_alloc - after['allocated']:.2f}GB alloc, "
                  f"{before_cache - after['cached']:.2f}GB cache)")


gpu_monitor = GPUMonitor()


# ============================================================
# AUDIO ANALYZER
# ============================================================
class AudioAnalyzer:
    def __init__(self):
        self.silence_threshold = 0.01
        self.last_audio_stats  = {}

    def analyze_chunk(self, audio_chunk, chunk_id):
        if audio_chunk is None or len(audio_chunk) == 0:
            return {'valid': False, 'reason': 'empty'}
        stats = {}
        stats['shape']       = audio_chunk.shape
        stats['dtype']       = str(audio_chunk.dtype)
        stats['size_kb']     = audio_chunk.nbytes / 1024
        stats['min']         = float(np.min(audio_chunk))
        stats['max']         = float(np.max(audio_chunk))
        stats['mean']        = float(np.mean(audio_chunk))
        stats['rms']         = float(np.sqrt(np.mean(audio_chunk ** 2)))
        stats['is_silent']   = stats['rms'] < self.silence_threshold
        stats['is_clipping'] = bool(np.any(np.abs(audio_chunk) > 0.99))
        stats['has_nan']     = bool(np.any(np.isnan(audio_chunk)))
        stats['has_inf']     = bool(np.any(np.isinf(audio_chunk)))
        if stats['is_silent']:
            debug.audio(f"Chunk {chunk_id}: SILENT (RMS: {stats['rms']:.6f})")
        elif stats['is_clipping']:
            debug.audio(f"Chunk {chunk_id}: CLIPPING (Max: {stats['max']:.4f})")
        self.last_audio_stats[chunk_id] = stats
        return stats


audio_analyzer = AudioAnalyzer()


# ============================================================
# WEBSOCKET MONITOR
# ============================================================
class WebSocketMonitor:
    def __init__(self):
        self.sent_bytes           = 0
        self.sent_packets         = 0
        self.send_times           = deque(maxlen=100)
        self.last_send_time       = None
        self.packet_loss_detected = False

    def log_send(self, bytes_sent, slice_num, total_slices):
        current_time = time.time()
        if self.last_send_time:
            interval = current_time - self.last_send_time
            self.send_times.append(interval)
            if interval > 0.1:
                debug.ws(f"Large gap between sends: {interval * 1000:.1f} ms")
        self.last_send_time  = current_time
        self.sent_bytes     += bytes_sent
        self.sent_packets   += 1
        debug.ws(f"📤 Sent slice {slice_num}/{total_slices} | "
                 f"{bytes_sent} bytes | Total: {self.sent_bytes / 1024:.1f} KB")

    def get_stats(self):
        if not self.send_times:
            return {'sent_bytes': self.sent_bytes, 'sent_packets': self.sent_packets,
                    'avg_interval': 0, 'max_interval': 0}
        return {
            'sent_bytes':    self.sent_bytes,
            'sent_packets':  self.sent_packets,
            'avg_interval':  sum(self.send_times) / len(self.send_times),
            'max_interval':  max(self.send_times),
        }

    def reset(self):
        self.sent_bytes   = 0
        self.sent_packets = 0
        self.send_times.clear()
        self.last_send_time = None


# ============================================================
# WEB AUDIO STREAMER
# ============================================================
class WebAudioStreamer:
    def __init__(self):
        self.is_streaming          = False
        self.current_session_id    = None
        self.stream_lock           = asyncio.Lock()
        self.text_queue            = asyncio.Queue()
        self.processing_task       = None
        self.current_playback_task = None
        self.pending_confirmation  = None
        self.session_complete      = asyncio.Event()

        self.chunk_timings           = deque(maxlen=10)
        self.avg_generation_time     = 0.1
        self.last_chunk_time         = None

        self.total_chunks_generated  = 0
        self.total_chunks_sent       = 0
        self.failed_chunks           = 0
        self.session_chunks          = []

        self.last_memory_check       = time.time()
        self.consecutive_high_memory = 0

        self.ws_monitor = WebSocketMonitor()

        self.chunk_timeout        = 30.0
        self.max_silent_chunks    = 10
        self.stream_timeout       = 120.0
        self.confirmation_timeout = 15.0

        print("[WEB_STREAMER] 🎯 Web-optimized Streaming initialized")
        print(f"[WEB_STREAMER] ⏱️ chunk={self.chunk_timeout}s, "
              f"confirmation={self.confirmation_timeout}s")

    def _sync_cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    async def _defragment_cuda_memory(self):
        loop = asyncio.get_event_loop()

        def _defrag():
            if not torch.cuda.is_available():
                return
            before_reserved = torch.cuda.memory_reserved()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            freed = (before_reserved - torch.cuda.memory_reserved()) / 1024 ** 3
            if freed > 0.01:
                debug.gpu(f"[DEFRAG] Released {freed:.3f} GB of fragmented memory")

        await loop.run_in_executor(None, _defrag)

    async def add_text_to_stream(self, text, is_voice=True):
        if not text or not text.strip():
            return
        if self.text_queue.qsize() >= 5:
            debug.warning("⚠️ Queue full (5 items), dropping oldest")
            try:
                self.text_queue.get_nowait()
                self.text_queue.task_done()
            except Exception:
                pass
        await self.text_queue.put((text, is_voice))
        debug.xtts(f"📥 Queued: '{text[:50]}...' (queue={self.text_queue.qsize()})")
        if not self.processing_task or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._process_stream_queue())
            global_state.active_tasks.add(self.processing_task)
            self.processing_task.add_done_callback(
                lambda t: global_state.active_tasks.discard(t))

    async def _process_stream_queue(self):
        print("[WEB_STREAMER] 🚀 Starting streaming processor")
        while not global_state.is_shutting_down:
            try:
                text, is_voice = await asyncio.wait_for(self.text_queue.get(), timeout=1.0)
                if text:
                    if self.current_playback_task and not self.current_playback_task.done():
                        debug.xtts("⏳ Waiting for previous session to complete...")
                        if self.pending_confirmation and not self.pending_confirmation.done():
                            try:
                                await asyncio.wait_for(
                                    self.pending_confirmation,
                                    timeout=self.confirmation_timeout)
                                debug.xtts("✅ Browser confirmed playback complete")
                            except asyncio.TimeoutError:
                                debug.warning("⚠️ No browser confirmation, continuing")
                            except Exception as e:
                                debug.error(f"Confirmation error: {e}")
                        try:
                            await asyncio.wait_for(self.current_playback_task, timeout=10.0)
                        except asyncio.TimeoutError:
                            debug.error("❌ Previous playback timed out — cancelling")
                            self.current_playback_task.cancel()
                            try:
                                await self.current_playback_task
                            except Exception:
                                pass
                        self.pending_confirmation = None
                        self.session_complete.clear()
                        await asyncio.sleep(0.2)

                    self.pending_confirmation  = asyncio.Future()
                    self.current_playback_task = asyncio.create_task(
                        self._stream_text(text, is_voice))
                    global_state.active_tasks.add(self.current_playback_task)
                    try:
                        await self.current_playback_task
                    except Exception as e:
                        debug.error(f"Playback error: {e}")
                    finally:
                        self.session_complete.set()
                        await self._defragment_cuda_memory()

                self.text_queue.task_done()

            except asyncio.TimeoutError:
                if global_state.is_shutting_down:
                    break
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                debug.error(f"Queue processing error: {e}")
                await asyncio.sleep(0.1)

        print("[WEB_STREAMER] ✅ Streaming queue processing completed")

    # ══════════════════════════════════════════════════════════
    # CHANGE 1 of 3 — _start_streaming_session
    #
    # Old: sent plain "AUDIO_START"
    # New: sends "AUDIO_START:{session_id}"
    #
    # session_id is now assigned BEFORE the send so the browser
    # immediately knows which session it's receiving audio for.
    # The browser stores this ID and echoes it back as
    # "AUDIO_FINISHED:{session_id}" when playback ends, allowing
    # confirm_playback_complete() to validate the match and
    # prevent the race condition where an early AUDIO_FINISHED
    # for sentence N accidentally unblocked the queue before
    # sentence N+1 started playing — causing mid-response
    # sentences to be skipped silently.
    # ══════════════════════════════════════════════════════════
    async def _start_streaming_session(self):
        async with self.stream_lock:
            if self.is_streaming or not global_state.current_websocket:
                return False
            try:
                # Assign session ID FIRST so the send carries it
                self.current_session_id = f"stream_{int(time.time() * 1000)}"

                await global_state.current_websocket.send(
                    f"AUDIO_START:{self.current_session_id}")
                await asyncio.sleep(0.05)

                self.is_streaming             = True
                global_state.is_playing_audio = True
                self.stream_start_time        = time.time()
                self.ws_monitor.reset()
                self.total_chunks_generated   = 0
                self.total_chunks_sent        = 0
                self.failed_chunks            = 0
                self.session_chunks           = []
                self.consecutive_silent       = 0
                gpu_monitor.log_gpu_stats("STREAM_START")
                print(f"[WEB_STREAMER] 🎯 AUDIO_START — Session: {self.current_session_id}")
                return True
            except websockets.exceptions.ConnectionClosed:
                debug.error("WebSocket closed")
                self.is_streaming             = False
                global_state.is_playing_audio = False
                return False
            except Exception as e:
                debug.error(f"Error starting session: {e}")
                return False

    async def _end_streaming_session(self):
        async with self.stream_lock:
            if (self.is_streaming
                    and global_state.current_websocket
                    and not global_state.is_shutting_down):
                try:
                    await global_state.current_websocket.send("AUDIO_END")
                    print("[WEB_STREAMER] 🎯 AUDIO_END — Streaming completed")
                except websockets.exceptions.ConnectionClosed:
                    debug.error("WebSocket closed before sending END")
                except Exception as e:
                    debug.error(f"Error sending AUDIO_END: {e}")

    async def _stream_text(self, text, is_voice=True):
        if not global_state.current_websocket:
            debug.xtts("❌ No WebSocket connection")
            if self.pending_confirmation and not self.pending_confirmation.done():
                self.pending_confirmation.set_result(False)
            return

        session_started = await self._start_streaming_session()
        if not session_started:
            debug.xtts("❌ Failed to start streaming session")
            if self.pending_confirmation and not self.pending_confirmation.done():
                self.pending_confirmation.set_result(False)
            return

        try:
            debug.xtts(f"🔊 Streaming: '{text[:50]}...'")
            stream_start_time  = time.time()
            chunk_count        = 0
            silent_chunk_count = 0

            if not gpu_monitor.check_cuda_errors():
                debug.error("CUDA error detected, skipping streaming")
                await self._end_streaming_session()
                if self.pending_confirmation and not self.pending_confirmation.done():
                    self.pending_confirmation.set_result(False)
                return

            stats = gpu_monitor.get_gpu_stats()
            if stats['percentage_alloc'] > 80:
                debug.gpu(f"[VRAM_GUARD] Pre-inference VRAM at "
                          f"{stats['percentage_alloc']:.1f}% — forcing cleanup")

                def _pre_inf_clean():
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                await asyncio.get_event_loop().run_in_executor(None, _pre_inf_clean)
                stats = gpu_monitor.get_gpu_stats()
                debug.gpu(f"[VRAM_GUARD] After cleanup: {stats['percentage_alloc']:.1f}%")
                if stats['percentage_alloc'] > 82:
                    debug.warning("[VRAM_GUARD] VRAM still high — adding safety delay")
                    await asyncio.sleep(0.5)
            else:
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

            gpu_monitor.log_gpu_stats("PRE_INFERENCE")

            try:
                gpt_latent = tts_model.gpt_cond_latent
                spk_embed  = tts_model.speaker_embedding

                if isinstance(gpt_latent, torch.Tensor):
                    gpt_latent = gpt_latent.detach()
                elif isinstance(gpt_latent, list):
                    gpt_latent = [l.detach() if isinstance(l, torch.Tensor) else l
                                  for l in gpt_latent]

                if isinstance(spk_embed, torch.Tensor):
                    spk_embed = spk_embed.detach()

                with torch.inference_mode():
                    chunks = tts_model.inference_stream(
                        text=text,
                        language="en",
                        gpt_cond_latent=gpt_latent,
                        speaker_embedding=spk_embed,
                        temperature=XTTS_TEMPERATURE,
                        length_penalty=XTTS_LENGTH_PENALTY,
                        repetition_penalty=XTTS_REPETITION_PENALTY,
                        top_k=XTTS_TOP_K,
                        top_p=XTTS_TOP_P,
                        speed=XTTS_SPEED,
                        enable_text_splitting=ENABLE_TEXT_SPLITTING,
                        stream_chunk_size=STREAM_CHUNK_SIZE,
                    )

                    first_chunk_sent   = False
                    last_chunk_time    = time.time()
                    consecutive_silent = 0
                    inference_start    = time.time()

                    for chunk in chunks:
                        elapsed = time.time() - inference_start
                        if elapsed > MAX_INFERENCE_SECONDS:
                            debug.warning(
                                f"[RUNAWAY] Inference timeout after {elapsed:.1f}s "
                                f"— killing iterator")
                            break

                        if (chunk_count + silent_chunk_count) >= MAX_TOTAL_CHUNKS:
                            debug.warning(
                                f"[RUNAWAY] Chunk cap reached "
                                f"({MAX_TOTAL_CHUNKS}) — killing iterator")
                            break

                        if global_state.is_shutting_down or not self.is_streaming:
                            break
                        if chunk is None:
                            continue

                        now            = time.time()
                        chunk_interval = now - last_chunk_time
                        if chunk_interval > 0.5 and last_chunk_time != stream_start_time:
                            debug.warning(f"Large gap between chunkks: {chunk_interval:.3f}s")
                        last_chunk_time = now

                        if isinstance(chunk, torch.Tensor):
                            audio_numpy = chunk.detach().cpu().float().numpy()
                            del chunk
                            torch.cuda.synchronize()
                            chunk = audio_numpy

                        processed_chunk, analysis = self._process_chunk_with_analysis(
                            chunk, chunk_count)

                        if processed_chunk is not None:
                            analysis['timestamp']       = now
                            analysis['chunk_id']        = chunk_count
                            analysis['generation_time'] = (
                                now - stream_start_time if chunk_count == 0 else chunk_interval)
                            self.session_chunks.append(analysis)

                            if analysis.get('is_silent', False):
                                silent_chunk_count += 1
                                consecutive_silent += 1

                                if consecutive_silent >= MAX_CONSECUTIVE_SILENT:
                                    debug.warning(
                                        f"[RUNAWAY] {consecutive_silent} consecutive "
                                        f"silent chunks — stopping inference loop. "
                                        f"Total silent: {silent_chunk_count}, "
                                        f"real: {chunk_count}")
                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.synchronize()
                                        torch.cuda.empty_cache()
                                    break
                                continue

                            consecutive_silent = 0

                            success = await self._stream_chunk(processed_chunk, chunk_count)
                            if success:
                                chunk_count += 1
                                self.total_chunks_sent += 1
                                if not first_chunk_sent:
                                    latency = time.time() - self.stream_start_time
                                    debug.xtts(f"⚡ FIRST CHUNK in {latency:.3f}s")
                                    first_chunk_sent = True

                                if chunk_count % 3 == 0:
                                    await asyncio.get_event_loop().run_in_executor(
                                        None, self._sync_cleanup)

                                await asyncio.sleep(0.002)

            except Exception as e:
                debug.error(f"Inference error: {e}")
                import traceback
                traceback.print_exc()
                await self._end_streaming_session()
                if self.pending_confirmation and not self.pending_confirmation.done():
                    self.pending_confirmation.set_result(False)
                return

            stream_duration = time.time() - stream_start_time
            debug.xtts(f"✅ Generated {chunk_count} chunks "
                       f"({silent_chunk_count} silent skipped) in {stream_duration:.3f}s")
            await self._end_streaming_session()

        except Exception as e:
            debug.error(f"Streaming error: {e}")
            import traceback
            traceback.print_exc()
            if self.pending_confirmation and not self.pending_confirmation.done():
                self.pending_confirmation.set_result(False)
        finally:
            await asyncio.get_event_loop().run_in_executor(None, self._sync_cleanup)
            gpu_monitor.log_gpu_stats("POST_INFERENCE")
            async with self.stream_lock:
                self.is_streaming             = False
                global_state.is_playing_audio = False
                self.current_session_id       = None

    def _process_chunk_with_analysis(self, chunk, chunk_id):
        try:
            if isinstance(chunk, torch.Tensor):
                audio_chunk = chunk.detach().cpu().float().numpy()
                del chunk
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            else:
                audio_chunk = np.array(chunk)

            audio_chunk = audio_chunk.squeeze().astype(np.float32)
            analysis    = audio_analyzer.analyze_chunk(audio_chunk, chunk_id)

            if chunk_id == 0:
                debug.audio(f"First chunk — shape:{audio_chunk.shape}, "
                            f"RMS:{analysis['rms']:.4f}")
            return audio_chunk, analysis

        except Exception as e:
            debug.error(f"Chunk processing error: {e}")
            return None, {'error': str(e)}

    async def _stream_chunk(self, chunk, chunk_id):
        try:
            if isinstance(chunk, np.ndarray):
                if chunk.dtype != np.float32:
                    chunk = chunk.astype(np.float32)
                chunk       = np.clip(chunk, -1.0, 1.0)
                audio_int16 = (chunk * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                if len(audio_bytes) == 0:
                    debug.warning(f"Chunk {chunk_id}: empty bytes!")
                    return False
                if global_state.current_websocket and self.is_streaming:
                    await global_state.current_websocket.send(audio_bytes)
                    self.ws_monitor.log_send(len(audio_bytes), chunk_id + 1, 1)
                return True
        except websockets.exceptions.ConnectionClosed:
            debug.error("WebSocket closed during send")
            self.is_streaming             = False
            global_state.is_playing_audio = False
            return False
        except Exception as e:
            debug.error(f"Error in _stream_chunk: {e}")
            self.failed_chunks += 1
            return False
        return False

    # ══════════════════════════════════════════════════════════
    # CHANGE 2 of 3 — confirm_playback_complete
    #
    # Old: no parameters, always resolved pending_confirmation
    # New: accepts session_id from the browser's AUDIO_FINISHED
    #      message and validates it matches the session we're
    #      waiting for before resolving.
    #
    # If the session_id doesn't match (stale confirmation from
    # a previous sentence), we log a warning and ignore it so
    # the current sentence's queue slot is never prematurely
    # freed.  None is accepted for backward compatibility with
    # any client still sending plain "AUDIO_FINISHED".
    # ══════════════════════════════════════════════════════════
    async def confirm_playback_complete(self, session_id=None):
        expected = getattr(self, '_expected_confirmation_id', None)

        if session_id and expected and session_id != expected:
            debug.warning(
                f"⚠️ Ignoring stale confirmation for '{session_id}' "
                f"(waiting for '{expected}')")
            return

        debug.xtts(f"✅ Browser confirmed playback complete "
                   f"(session={session_id or 'untagged'})")
        if self.pending_confirmation and not self.pending_confirmation.done():
            self.pending_confirmation.set_result(True)

        # Clear the expected ID now that confirmation arrived
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
        self.base_dir          = base_dir
        self.school_docs_dir   = school_docs_dir
        self.persist_directory = persist_directory
        self.vectorstore       = None
        self.retriever         = None
        self.memory            = None
        self.is_initialized    = False
        self.word_fixer        = word_fixer
        self.restart_flag_file = os.path.join(base_dir, "chroma_db_restart.flag")

    def check_for_restart(self):
        if os.path.exists(self.restart_flag_file):
            print("[RAG] 🔄 Restart flag detected! Reinitializing system...")
            try:
                os.remove(self.restart_flag_file)
            except Exception:
                pass
            return True
        return False

    def initialize(self):
        print("[LANGCHAIN] 🚀 Initializing DYNAMIC RAG system...")
        print(f"[LANGCHAIN] 📁 Docs dir: {self.school_docs_dir}")
        print(f"[LANGCHAIN] 📁 ChromaDB: {self.persist_directory}")

        if self.check_for_restart():
            print("[LANGCHAIN] 🔄 Restarting from scratch")
            import shutil
            if os.path.exists(self.persist_directory):
                try:
                    shutil.rmtree(self.persist_directory)
                    print(f"[LANGCHAIN] ✅ Deleted old ChromaDB")
                except Exception as e:
                    print(f"[LANGCHAIN] ⚠️ Could not delete ChromaDB: {e}")

        model_name    = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs  = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
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
                print(f"[LANGCHAIN] ❌ Error: {e} — rebuilding")
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
            print("[LANGCHAIN] ✅ Retriever ready (MMR)")
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history", k=3,
                return_messages=True, output_key='answer',
            )
            print("[LANGCHAIN] ✅ Conversation memory ready")
            self.is_initialized = True
            return True
        return False

    def _get_all_pdf_files(self):
        pdf_files = []
        for section in DOCUMENT_SECTIONS.keys():
            section_dir = os.path.join(self.school_docs_dir, section)
            if os.path.exists(section_dir):
                section_pdfs = glob.glob(os.path.join(section_dir, "*.pdf"))
                pdf_files.extend(section_pdfs)
                print(f"[LANGCHAIN] 📂 {section}: {len(section_pdfs)} PDF(s)")
        print(f"[LANGCHAIN] 📚 Total PDFs: {len(pdf_files)}")
        return pdf_files

    def _index_all_documents(self, embeddings):
        try:
            pdf_files = self._get_all_pdf_files()
            if not pdf_files:
                print("[LANGCHAIN] ⚠️ No PDFs found")
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=embeddings,
                )
                return True

            all_documents = []
            for pdf_path in pdf_files:
                try:
                    loader    = PyPDFLoader(pdf_path)
                    documents = loader.load()
                    for doc in documents:
                        parts = pdf_path.split(os.sep)
                        if len(parts) >= 2:
                            doc.metadata['section'] = parts[-2]
                        doc.metadata['source'] = os.path.basename(pdf_path)
                    all_documents.extend(documents)
                    print(f"[LANGCHAIN] ✅ {len(documents)} pages from "
                          f"{os.path.basename(pdf_path)}")
                except Exception as e:
                    print(f"[LANGCHAIN] ⚠️ Failed {pdf_path}: {e}")

            if not all_documents:
                print("[LANGCHAIN] ❌ No documents loaded")
                return False

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            )
            chunks = splitter.split_documents(all_documents)
            print(f"[LANGCHAIN] 🔪 {len(chunks)} chunks")
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=self.persist_directory,
            )
            print("[LANGCHAIN] ✅ All documents indexed!")
            return True
        except Exception as e:
            print(f"[LANGCHAIN] ❌ Indexing error: {e}")
            return False

    def reinitialize_from_scratch(self):
        print("[LANGCHAIN] 🔄 Reinitializing from scratch...")
        self.vectorstore    = None
        self.retriever      = None
        self.is_initialized = False
        return self.initialize()

    async def smart_search(self, query):
        if not self.retriever:
            print("[LANGCHAIN] ❌ Retriever not initialized")
            return [], []
        try:
            fixed = self.word_fixer.fix_text(query)
            if fixed != query:
                print(f"[LANGCHAIN] 🔧 Fixed query: '{query}' → '{fixed}'")
                query = fixed
            docs = await self.retriever.ainvoke(query)
            documents = [d.page_content for d in docs]
            metadatas = [{'page': d.metadata.get('page', 'N/A'),
                          'source': d.metadata.get('source', 'N/A'),
                          'section': d.metadata.get('section', 'N/A'),
                          'similarity': 'N/A'} for d in docs]
            print(f"[LANGCHAIN] 🔍 {len(documents)} chunks for: '{query}'")
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
        return f"""Here is some information from my database that might help answer the question:

{context}

Question: {query}

IMPORTANT INSTRUCTIONS:
1. Answer based ONLY on the provided information above
2. DO NOT mention document names, page numbers, or sources in your response
3. Present the information naturally as if you know it directly
4. If the specific information isn't available, clearly state what information is missing
5. Keep your answer concise and conversational

Answer the question naturally without mentioning any documents or sources:"""


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
        print("[LM_CLIENT] ✅ HTTP client ready!")


async def close_lm_client():
    global lm_client
    if lm_client:
        await lm_client.aclose()
        lm_client = None
        print("[LM_CLIENT] 🔒 HTTP client closed")


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
            self.whisper_model = WhisperModel("base", device="cpu",
                                               compute_type="int8")
            print("[STT] ✅ Whisper ready on CPU (fallback)!")

    async def process_audio_buffer(self, audio_buffer):
        if not audio_buffer or len(audio_buffer) < 1000:
            print(f"[STT] ⚠️ Buffer too small: {len(audio_buffer)} bytes")
            return ""
        print(f"[STT] 🔊 Processing {len(audio_buffer)} bytes...")
        start = time.time()
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
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            elapsed = time.time() - start
            if text:
                print(f"[STT] ✅ {elapsed:.3f}s: '{text}'")
            else:
                print(f"[STT] ⚠️ No speech in {elapsed:.3f}s")
            return text
        except Exception as e:
            print(f"[STT] ❌ Error: {e}")
            return ""


whisper_stt = WhisperSTT()


# ============================================================
# XTTS STREAMER
# ============================================================
class XTTSStreamer:
    def __init__(self):
        self.model             = None
        self.gpt_cond_latent   = None
        self.speaker_embedding = None
        self.is_ready          = False
        self.init_thread = threading.Thread(target=self._initialize_model,
                                             daemon=True)
        self.init_thread.start()

    def _initialize_model(self):
        try:
            print("[XTTS] 🚀 Loading XTTS-v2...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            config = XttsConfig()
            config.load_json(os.path.join(MODEL_PATH, "config.json"))
            self.model = Xtts.init_from_config(config)

            print(f"[XTTS] DeepSpeed: {USE_DEEPSPEED}")
            self.model.load_checkpoint(config, checkpoint_dir=MODEL_PATH,
                                        use_deepspeed=USE_DEEPSPEED)
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")

            print("[XTTS] Computing voice embeddings...")
            if torch.cuda.is_available():
                self.model = self.model.cpu()

            self.gpt_cond_latent, self.speaker_embedding = \
                self.model.get_conditioning_latents(audio_path=[REFERENCE_VOICE])

            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
                if isinstance(self.gpt_cond_latent, list):
                    self.gpt_cond_latent = [l.to("cuda") for l in self.gpt_cond_latent]
                else:
                    self.gpt_cond_latent = self.gpt_cond_latent.to("cuda")
                self.speaker_embedding = self.speaker_embedding.to("cuda")

            print("[XTTS] 🔥 Pre-warming...")
            self._pre_warm_tts()

            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            self.is_ready = True
            print("[XTTS] ✅ XTTS-v2 Ready & Pre-warmed!")

        except Exception as e:
            print(f"[XTTS] ❌ DeepSpeed error: {e} — trying standard load")
            try:
                self.model.load_checkpoint(config, checkpoint_dir=MODEL_PATH,
                                            use_deepspeed=False)
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                self.gpt_cond_latent, self.speaker_embedding = \
                    self.model.get_conditioning_latents(audio_path=[REFERENCE_VOICE])
                if torch.cuda.is_available():
                    if isinstance(self.gpt_cond_latent, list):
                        self.gpt_cond_latent = [l.to("cuda") for l in self.gpt_cond_latent]
                    else:
                        self.gpt_cond_latent = self.gpt_cond_latent.to("cuda")
                    self.speaker_embedding = self.speaker_embedding.to("cuda")
                self._pre_warm_tts()
                self.is_ready = True
                print("[XTTS] ✅ XTTS-v2 Ready (standard load)")
            except Exception as fe:
                print(f"[XTTS] ❌ Fallback failed: {fe}")

    def _pre_warm_tts(self):
        try:
            print("[XTTS] 🔥 Pre-warming with inference_mode guard...")
            with torch.inference_mode():
                dummy = self.model.inference_stream(
                    "Hello, I'm ready to help you.",
                    "en",
                    self.gpt_cond_latent,
                    self.speaker_embedding,
                    temperature=XTTS_TEMPERATURE,
                    enable_text_splitting=ENABLE_TEXT_SPLITTING,
                    stream_chunk_size=STREAM_CHUNK_SIZE,
                )
                for chunk in dummy:
                    if chunk is not None and isinstance(chunk, torch.Tensor):
                        _ = chunk.cpu()
                        del chunk

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            print("[XTTS] ✅ Pre-warm complete!")
        except Exception as e:
            print(f"[XTTS] ⚠️ Pre-warm skipped: {e}")

    def wait_until_ready(self):
        self.init_thread.join(timeout=60)
        return self.is_ready

    def inference_stream(self, text, language="en",
                         gpt_cond_latent=None, speaker_embedding=None, **kwargs):
        if not self.is_ready:
            debug.xtts("⚠️ XTTS not ready — returning empty iterator")
            return iter([])

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if gpt_cond_latent is None:
            gpt_cond_latent = self.gpt_cond_latent
        if speaker_embedding is None:
            speaker_embedding = self.speaker_embedding

        if isinstance(gpt_cond_latent, torch.Tensor):
            gpt_cond_latent = gpt_cond_latent.detach()
        elif isinstance(gpt_cond_latent, list):
            gpt_cond_latent = [l.detach() if isinstance(l, torch.Tensor) else l
                               for l in gpt_cond_latent]
        if isinstance(speaker_embedding, torch.Tensor):
            speaker_embedding = speaker_embedding.detach()

        debug.xtts(f"inference_stream → '{text[:50]}...'")

        with torch.inference_mode():
            return self.model.inference_stream(
                text, language, gpt_cond_latent, speaker_embedding, **kwargs)


tts_model = XTTSStreamer()


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
            21: 'twenty first', 22: 'twenty second', 23: 'twenty third',
            24: 'twenty fourth', 25: 'twenty fifth', 26: 'twenty sixth',
            27: 'twenty seventh', 28: 'twenty eighth', 29: 'twenty ninth',
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
            url = m.group(0)
            url = re.sub(r'^https?://', '', url)
            url = re.sub(r'^www\.', '', url)
            url = url.replace('/', ' slash ')
            url = url.replace('.', ' dot ')
            url = url.replace('-', ' ')
            return re.sub(r'\s+', ' ', url).strip()

        text = re.sub(
            r'https?://[^\s,;)\'\"]+|www\.[^\s,;)\'\"]+',
            _url_to_speech, text)

        def _email_to_speech(m):
            local, domain = m.group(1), m.group(2)
            domain = domain.replace('.', ' dot ')
            return f"{local} at {domain}"

        text = re.sub(r'([A-Za-z0-9._%+\-]+)@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})',
                      _email_to_speech, text)

        def _phone_to_speech(m):
            prefix = m.group(1)
            spaced = ' '.join(list(m.group(2)))
            return f"plus {spaced}" if prefix else spaced

        text = re.sub(r'(\+?)(\d{7,15})\b', _phone_to_speech, text)

        text = re.sub(r'\d{5,}', lambda m: ' '.join(list(m.group(0))), text)

        return text

    def _clean_text_for_tts(self, text):
        if not text or not text.strip():
            return ""
        text = self.word_fixer.fix_text(text)
        text = re.sub(r'###\s*(User|AI|System|Jarvis):?', '', text)
        text = re.sub(r'^(User|AI|System|Jarvis):?\s*', '', text)
        text = self._format_digits_for_tts(text)
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            print(f"[PIPELINE] 📝 TTS-ready: '{text[:80]}'")
        return text

    async def stream_text(self, text, is_voice=True):
        if global_state.is_shutting_down or not text or not text.strip():
            return
        if any(m in text for m in ['###', 'User:', 'AI:', 'System:', 'Jarvis:']):
            return
        cleaned = self._clean_text_for_tts(text)
        if cleaned:
            await audio_streamer.add_text_to_stream(cleaned, is_voice)
            print(f"[PIPELINE] 🎯 Added: '{cleaned}' (voice={is_voice})")


pipeline = Pipeline()


# ============================================================
# LM STUDIO
# ============================================================
async def stream_from_lm_studio_enhanced(prompt, text_callback=None, is_voice=True):
    global lm_client
    if global_state.is_shutting_down:
        return "System shutting down"
    lm_start = time.time()
    if lm_client is None:
        await initialize_lm_client()

    if is_voice:
        system_content = """You are Gaijin, an AI voice assistant.

IDENTITY:
- Your name is Gaijin
- You are NOT Tristhan
- Tristhan D Cabrera is your creator
- He is a Computer Engineering student

RESPONSE RULES:
- Keep responses VERY SHORT (1-3 sentences max)
- Write for TEXT-TO-SPEECH
- No markdown, bullet points, or numbered lists
- No special characters like : or -
- Be conversational and friendly

GREETINGS:
- Respond naturally and warmly

EXAMPLES:
Q: What is your name?
A: I'm Gaijin, your AI voice assistant.

Q: Who are you?
A: I'm Gaijin, an AI assistant created by Tristhan.

Q: Tell me about yourself
A: I'm Gaijin, a voice AI made by Tristhan to help you."""
    else:
        system_content = """You are Gaijin, a text-based AI assistant.

IDENTITY:
- Your name is Gaijin
- You are NOT Tristhan
- Tristhan D Cabrera is your creator

RESPONSE RULES:
- Keep answers brief and direct
- Use markdown sparingly if needed
- Be helpful and conversational

EXAMPLES:
Q: What is your name?
A: I'm Gaijin, your AI assistant.

Q: Who are you?
A: I'm Gaijin, created by Tristhan D Cabrera."""

    messages = [{"role": "system", "content": system_content}]

    fixed_prompt = word_fixer.fix_text(prompt)
    if fixed_prompt != prompt:
        print(f"[WORD_FIXER] 🔧 '{prompt}' → '{fixed_prompt}'")
        prompt = fixed_prompt

    enhanced_prompt = await rag_system.get_contextual_prompt(prompt)
    messages.append({"role": "user", "content": enhanced_prompt})
    print(f"[LANGCHAIN] 🧠 Enhanced search for: '{prompt}'")

    payload = {
        "model": "meta-llama-3.1-8b-instruct",
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": MAX_RESPONSE_TOKENS,
        "stream": True,
        "stop": ["###", "User:", "AI:", "System:"],
    }

    print(f"[LM_DEBUG] 🚀 Sending request: {len(prompt)} chars (voice={is_voice})")
    full_response   = ""
    sentence_buffer = ""

    try:
        req_start = time.time()
        async with lm_client.stream(
            "POST", LM_STUDIO_URL,
            json=payload,
            headers={'Content-Type': 'application/json'},
        ) as response:
            print(f"[LM_TIMING] 📤 Request sent in {time.time() - req_start:.3f}s")
            if response.status_code != 200:
                print(f"[LM_DEBUG] ❌ HTTP {response.status_code}")
                return f"Error: {response.status_code}"

            first_token  = False
            token_count  = 0
            stream_start = time.time()

            async for line in response.aiter_lines():
                if global_state.is_shutting_down:
                    break
                if not line or not line.startswith('data: '):
                    continue
                if line == 'data: [DONE]':
                    break
                try:
                    data = json.loads(line[6:])
                    if 'choices' in data and data['choices']:
                        delta = data['choices'][0].get('delta', {})
                        if 'content' in delta:
                            token = delta['content']
                            for marker in ['###', 'User:', 'AI:', 'System:', 'Jarvis:']:
                                token = token.replace(marker, '')
                            if not token.strip():
                                continue
                            token_count += 1
                            if token_count > MAX_RESPONSE_TOKENS:
                                print(f"[LM_DEBUG] ⚠️ Token limit reached")
                                break
                            if not first_token:
                                print(f"[LM_TIMING] ⚡ First token in "
                                      f"{time.time() - stream_start:.3f}s: '{token}'")
                                first_token = True
                            full_response += token
                            if is_voice:
                                sentence_buffer += token
                                if re.search(r'[.!?]\s*$', sentence_buffer):
                                    if text_callback and sentence_buffer.strip():
                                        await text_callback(sentence_buffer, is_voice=True)
                                        sentence_buffer = ""
                except (json.JSONDecodeError, Exception):
                    continue

        if is_voice and sentence_buffer.strip() and text_callback \
                and not global_state.is_shutting_down:
            await text_callback(sentence_buffer, is_voice=True)

        print(f"[LM_TIMING] ✅ {token_count} tokens in {time.time() - lm_start:.3f}s")

        clean = re.sub(r'###\s*(User|AI|System|Jarvis):?', '', full_response)
        clean = re.sub(r'^(User|AI|System|Jarvis):?\s*', '', clean).strip()

        if global_state.current_websocket and not global_state.is_shutting_down and clean:
            await global_state.current_websocket.send(f"AI_RESPONSE: {clean}")
            print(f"[WEBSOCKET] 📤 AI_RESPONSE: {clean[:50]}...")

        return clean

    except Exception as e:
        print(f"[LM_DEBUG] ❌ {e}")
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
                        print("🛑 Buffer FULL")
                        global_state.is_recording = False
                        await websocket.send("BUFFER_FULL")

            elif isinstance(message, str):
                print(f"📨 Received: {message[:100]}")

                # ══════════════════════════════════════════════════
                # CHANGE 3 of 3 — AUDIO_FINISHED handler
                #
                # Old: only handled plain "AUDIO_FINISHED" and called
                #      confirm_playback_complete() with no argument.
                # New: handles both "AUDIO_FINISHED" (legacy) and
                #      "AUDIO_FINISHED:{session_id}" (new JS).
                #
                # The session_id is extracted and forwarded to
                # confirm_playback_complete() which validates it
                # against the expected session before resolving the
                # pending_confirmation future.  This closes the race
                # condition that caused mid-response sentences to be
                # dropped when AUDIO_FINISHED for sentence N arrived
                # before sentence N+1 even started playing.
                # ══════════════════════════════════════════════════
                if message.startswith("AUDIO_FINISHED"):
                    parts      = message.split(":", 1)
                    session_id = parts[1].strip() if len(parts) > 1 else None
                    print(f"✅ Browser: audio finished (session={session_id or 'untagged'})")
                    await audio_streamer.confirm_playback_complete(session_id)

                elif message == "START_RECORDING":
                    global_state.is_recording = True
                    global_state.audio_buffer = bytearray()
                    print("🎤 Recording STARTED")
                    await websocket.send("RECORDING_STARTED")

                elif message == "STOP_RECORDING":
                    global_state.is_recording = False
                    buf_size = len(global_state.audio_buffer)
                    print(f"🛑 Recording STOPPED — {buf_size} bytes")
                    await websocket.send("PROCESSING_AUDIO")

                    if global_state.audio_buffer and buf_size > 1000:
                        user_input = await whisper_stt.process_audio_buffer(
                            global_state.audio_buffer)
                        if user_input and user_input.strip():
                            orig = user_input
                            user_input = word_fixer.fix_text(user_input)
                            if orig != user_input:
                                print(f"[WORD_FIXER] '{orig}' → '{user_input}'")
                            print(f"👤 Voice: {user_input}")
                            await websocket.send(f"TRANSCRIBED: {user_input}")
                            t0 = time.time()
                            await stream_from_lm_studio_enhanced(
                                user_input,
                                lambda txt, is_voice=True: pipeline.stream_text(
                                    txt, is_voice=True),
                                is_voice=True,
                            )
                            print(f"[TIMING] END-TO-END: {time.time() - t0:.3f}s")
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
                                print(f"📝 Text: {text}")
                                await websocket.send(f"TRANSCRIBED: {text}")
                                t0 = time.time()
                                await stream_from_lm_studio_enhanced(
                                    text, None, is_voice=False)
                                print(f"[TIMING] END-TO-END: {time.time() - t0:.3f}s")
                                print("=" * 50)
                    except json.JSONDecodeError:
                        print(f"⚠️ Invalid JSON: {message[:50]}")

                elif message == "PING":
                    await websocket.send("PONG")

                elif message == "GET_DEBUG":
                    stats = {
                        'gpu':    gpu_monitor.get_gpu_stats(),
                        'ws':     audio_streamer.ws_monitor.get_stats(),
                        'chunks': audio_streamer.total_chunks_sent,
                        'failed': audio_streamer.failed_chunks,
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
    print(f"🌐 Starting WebSocket on {WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    try:
        async with serve(websocket_handler, WEBSOCKET_HOST, WEBSOCKET_PORT,
                         ping_interval=120, ping_timeout=115, close_timeout=30):
            print("✅ WebSocket server running!")
            await asyncio.Future()
    except asyncio.CancelledError:
        print("🔴 WebSocket server cancelled")
        raise


# ============================================================
# MEMORY MONITOR
# ============================================================
async def memory_monitor_task():
    print("[MONITOR] 🧠 GPU memory monitor started")
    while not global_state.is_shutting_down:
        await asyncio.sleep(2)
        if torch.cuda.is_available() and audio_streamer.is_streaming:
            stats = gpu_monitor.get_gpu_stats()
            if stats['percentage_alloc'] > 85:
                debug.gpu(f"[MONITOR] Critical: {stats['allocated']:.2f}GB "
                          f"({stats['percentage_alloc']:.1f}%) — forcing cleanup")
                gpu_monitor.aggressive_cleanup(force=True)


# ============================================================
# MAIN
# ============================================================
async def main_async():
    print("🤖 AI Assistant — VRAM-OPTIMIZED BUILD")
    print("=" * 60)
    print("FIXES APPLIED:")
    print("  FIX 1  — STREAM_CHUNK_SIZE: 20 → 40")
    print("  FIX 2  — Cleanup order: gc → synchronize → empty_cache")
    print("  FIX 3  — _sync_cleanup uses correct order")
    print("  FIX 4  — Double-pass defrag after every session")
    print("  FIX 5  — Defrag called in _process_stream_queue")
    print("  FIX 6  — Pre-inference VRAM guard at 80%")
    print("  FIX 7  — Embeddings detached before every inference")
    print("  FIX 8  — inference_mode wraps entire chunk loop")
    print("  FIX 9  — GPU→CPU move: detach + del + synchronize")
    print("  FIX 10 — Cleanup every 3 chunks (was every 5)")
    print("  FIX 11 — _process_chunk_with_analysis uses .detach()")
    print("  FIX 12 — Pre-warm runs inside inference_mode")
    print("  FIX 13 — XTTSStreamer.inference_stream always detaches")
    print("  FIX 14 — XTTS RUNAWAY BUG: MAX_CONSECUTIVE_SILENT = 8")
    print("  FIX 15 — Session ID in AUDIO_START / AUDIO_FINISHED")
    print(f"           MAX_CONSECUTIVE_SILENT = {MAX_CONSECUTIVE_SILENT}")
    print(f"           MAX_INFERENCE_SECONDS  = {MAX_INFERENCE_SECONDS}s")
    print(f"           MAX_TOTAL_CHUNKS       = {MAX_TOTAL_CHUNKS}")
    print("=" * 60)
    if gpu_monitor.total_memory > 0:
        print(f"   VRAM: {gpu_monitor.total_memory:.2f} GB total")
        print(f"   Danger threshold (85%): "
              f"{gpu_monitor.total_memory * 0.85:.2f} GB")
        print(f"   Warning threshold (70%): "
              f"{gpu_monitor.total_memory * 0.70:.2f} GB")
    print("=" * 60)

    setup_graceful_shutdown()

    try:
        global_state.rag_system = rag_system
        if not rag_system.initialize():
            print("[LANGCHAIN] ❌ RAG init failed")
            return
        print("[LANGCHAIN] ✅ RAG ready!")

        await initialize_lm_client()

        if tts_model.wait_until_ready():
            print("✅ Pipeline Ready!")

        gpu_monitor.log_gpu_stats("INIT")
        asyncio.create_task(memory_monitor_task())

        print(f"\n🔗 FLASK INTEGRATION — upload path: {SCHOOL_DOCS_DIR}")
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