/**
 * ChatPanel — Floating AI assistant with glassmorphism UI.
 * Connects directly to the Python backend WebSocket (Whisper STT + Kokoro TTS).
 *
 * Protocol:
 *   → "START_RECORDING"                       begin mic capture
 *   → raw PCM Int16 audio bytes               streamed while recording
 *   → "STOP_RECORDING"                        stop, backend processes
 *   ← "CONNECTED: ..."                        server ready
 *   ← "RECORDING_STARTED"                     ack
 *   ← "PROCESSING_AUDIO"                      STT in progress
 *   ← "TRANSCRIBED: {text}"                   user's speech as text
 *   ← "AI_RESPONSE: {text}"                   AI reply text
 *   ← "AUDIO_START:{session_id}"              TTS session begins
 *   ← JSON header  { type:"audio_chunk", sample_rate, channels, encoding, bytes, duration_ms }
 *   ← binary PCM16 (exactly `bytes` bytes)    one complete crossfaded phrase
 *   ← ... (header+binary pairs repeat for each phrase)
 *   ← "AUDIO_END"                             all phrases sent
 *   → "AUDIO_FINISHED:{session_id}"           browser confirms last phrase played
 *   ← "NO_SPEECH_DETECTED"
 *
 * Audio playback model (Kokoro):
 *   Each phrase arrives as a matched JSON header + binary pair.
 *   Phrases are SCHEDULED onto a single AudioContext timeline so playback is
 *   perfectly gapless — no collect-then-play, no misaligned fragments.
 */

// ═══════════════════════════════════════════════════════
//  CONFIG
// ═══════════════════════════════════════════════════════
// In production (Vercel), VITE_WS_URL is set via the dashboard or
// .env.production → wss://driving-dominant-bobcat.ngrok-free.app
//
// In local dev (npm run dev), VITE_WS_URL is undefined, so we fall
// back to the same-host /ws proxy defined in vite.config.js
// (proxy: '/ws' → ws://localhost:8888) — no separate ngrok tunnel needed.
//
// The ngrok-skip-browser-warning query param bypasses ngrok's interstitial
// warning page so direct WebSocket connections are never blocked.
const _WS_BASE = import.meta.env.VITE_WS_URL
  ?? `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws`;
const WS_URL   = _WS_BASE.includes('?')
  ? _WS_BASE + '&ngrok-skip-browser-warning=true'
  : _WS_BASE + '?ngrok-skip-browser-warning=true';
const SAMPLE_RATE = 24000;   // must match Python backend
const CHANNELS    = 1;
const PHRASE_GAP  = 0.005;   // 5ms silence between phrases (gapless feel)
// ═══════════════════════════════════════════════════════

export function initChatPanel() {
  const root = document.getElementById('chat-root');
  if (!root) return;

  // ─── Inject HTML ─────────────────────────────────────
  root.innerHTML = `
    <!-- Floating Toggle Button -->
    <button id="chat-toggle"
      class="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-2xl bg-neon-cyan/10 border border-neon-cyan/30
             flex items-center justify-center text-neon-cyan text-xl
             shadow-neon animate-glow-pulse hover:scale-110 transition-transform duration-300 group"
      aria-label="Open AI Assistant">
      <i class='bx bx-bot'></i>
      <span class="absolute bottom-full right-0 mb-3 px-3 py-1.5 rounded-lg bg-deep-space border border-white/10
                   text-xs text-white/70 whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none">
        AI Assistant
      </span>
    </button>

    <!-- Chat Panel -->
    <div id="chat-panel"
      class="fixed bottom-24 right-6 z-50 w-[380px] max-w-[calc(100vw-2rem)] h-[520px]
             glass-panel flex flex-col overflow-hidden
             opacity-0 scale-95 pointer-events-none translate-y-4
             transition-all duration-300 ease-out">

      <!-- Header -->
      <div class="flex items-center justify-between px-5 py-4 border-b border-white/[0.06]">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 rounded-xl bg-neon-cyan/10 flex items-center justify-center text-neon-cyan">
            <i class='bx bx-bot text-lg'></i>
          </div>
          <div>
            <h4 class="text-sm font-semibold text-white">AI Assistant</h4>
            <div id="chat-status" class="status-badge status-offline mt-0.5">
              <span id="chat-status-dot" class="w-1.5 h-1.5 rounded-full bg-red-400"></span>
              <span id="chat-status-label">Offline</span>
            </div>
          </div>
        </div>
        <button id="chat-close"
          class="w-8 h-8 rounded-lg flex items-center justify-center text-white/40 hover:text-white hover:bg-white/[0.06] transition-all"
          aria-label="Close chat">
          <i class='bx bx-x text-xl'></i>
        </button>
      </div>

      <!-- Messages -->
      <div id="chat-messages" class="flex-1 overflow-y-auto px-5 py-4 space-y-4 scroll-smooth">
        <div class="chat-bubble-ai">
          <p class="text-sm">Hello! 👋 I'm Tristhan's AI assistant. Type a message or tap the mic to talk.</p>
        </div>
      </div>

      <!-- Typing / Processing indicator -->
      <div id="chat-typing" class="hidden px-5 pb-2">
        <div class="chat-bubble-ai inline-flex items-center gap-1.5 !py-3 !px-4">
          <span class="w-2 h-2 rounded-full bg-white/40 animate-typing-dot"></span>
          <span class="w-2 h-2 rounded-full bg-white/40 animate-typing-dot" style="animation-delay: 0.2s"></span>
          <span class="w-2 h-2 rounded-full bg-white/40 animate-typing-dot" style="animation-delay: 0.4s"></span>
        </div>
      </div>

      <!-- Audio Playing Indicator -->
      <div id="chat-audio-playing" class="hidden px-5 pb-2">
        <div class="flex items-center gap-2 text-neon-cyan/70 text-xs">
          <span class="w-2 h-2 rounded-full bg-neon-cyan animate-pulse"></span>
          <span>AI is speaking...</span>
        </div>
      </div>

      <!-- Input Area -->
      <div class="px-4 py-3 border-t border-white/[0.06]">
        <div class="flex items-center gap-2">
          <!-- Mic Button -->
          <button id="chat-mic"
            class="w-10 h-10 rounded-xl bg-white/[0.04] border border-white/10 text-white/50
                   flex items-center justify-center transition-all duration-200
                   hover:text-neon-cyan hover:border-neon-cyan/30 hover:bg-neon-cyan/5"
            aria-label="Voice input" title="Click to speak">
            <i class='bx bx-microphone text-lg'></i>
          </button>

          <!-- Text Input -->
          <input id="chat-input" type="text" placeholder="Type a message…"
            class="flex-1 bg-white/[0.04] border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white
                   placeholder-white/30 outline-none focus:border-neon-cyan/30 transition-colors duration-200" />

          <!-- Send Button -->
          <button id="chat-send" disabled
            class="w-10 h-10 rounded-xl bg-neon-cyan/10 border border-neon-cyan/20 text-neon-cyan
                   flex items-center justify-center transition-all duration-200
                   disabled:opacity-30 disabled:cursor-not-allowed
                   enabled:hover:bg-neon-cyan/20 enabled:hover:shadow-neon"
            aria-label="Send">
            <i class='bx bx-send text-sm'></i>
          </button>
        </div>

        <!-- Recording indicator bar -->
        <div id="chat-recording" class="hidden flex items-center gap-2 mt-2 px-1">
          <span class="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
          <span id="chat-recording-label" class="text-[11px] text-red-400 font-mono">Listening… click mic to stop</span>
        </div>
      </div>
    </div>
  `;

  // ─── DOM Refs ─────────────────────────────────────────
  const toggle         = document.getElementById('chat-toggle');
  const panel          = document.getElementById('chat-panel');
  const closeBtn       = document.getElementById('chat-close');
  const messagesEl     = document.getElementById('chat-messages');
  const input          = document.getElementById('chat-input');
  const sendBtn        = document.getElementById('chat-send');
  const micBtn         = document.getElementById('chat-mic');
  const typingEl       = document.getElementById('chat-typing');
  const audioPlayingEl = document.getElementById('chat-audio-playing');
  const recordEl       = document.getElementById('chat-recording');
  const recordLbl      = document.getElementById('chat-recording-label');
  const statusEl       = document.getElementById('chat-status');
  const statusDot      = document.getElementById('chat-status-dot');
  const statusLbl      = document.getElementById('chat-status-label');

  let isOpen = false;

  // ═══════════════════════════════════════════════════════
  //  WEBSOCKET STATE
  // ═══════════════════════════════════════════════════════
  /** @type {WebSocket|null} */
  let ws = null;
  let reconnectTimeout  = null;
  let reconnectAttempts = 0;
  const MAX_RECONNECT   = 10;

  // ═══════════════════════════════════════════════════════
  //  RECORDING STATE
  // ═══════════════════════════════════════════════════════
  let isRecording  = false;
  /** @type {MediaStream|null}          */ let micStream  = null;
  /** @type {AudioContext|null}         */ let recCtx     = null;
  /** @type {ScriptProcessorNode|null}  */ let scriptNode = null;

  // ═══════════════════════════════════════════════════════
  //  KOKORO AUDIO PLAYBACK STATE
  //
  //  The backend now sends one COMPLETE crossfaded phrase per
  //  header+binary pair, NOT raw sliced fragments.
  //
  //  We schedule each phrase directly onto a Web Audio timeline
  //  so playback is gapless from the first phrase onward.
  //  There is no "collect everything then play" step — phrases
  //  start playing as soon as they arrive, overlapping with
  //  the Python backend still generating the next phrase.
  // ═══════════════════════════════════════════════════════
  /** @type {AudioContext|null} */
  let playCtx         = null;
  let nextStartTime   = 0;        // when the next phrase should start on the timeline
  let sessionId       = null;     // current AUDIO_START session ID
  let pendingHeader   = null;     // JSON header waiting for its binary pair
  let isSessionActive = false;    // true between AUDIO_START and last phrase end
  let phrasesScheduled = 0;       // phrases queued this session
  let phrasesEnded     = 0;       // phrases that have fired onended
  let sessionEndSignalled = false;// true once AUDIO_END received
  let finishedTimeout = null;     // safety fallback timer

  // ═══════════════════════════════════════════════════════
  //  STATUS HELPERS
  // ═══════════════════════════════════════════════════════
  function setStatus(type) {
    const map = {
      connected:    { label: 'Connected',    cls: 'status-connected',    dot: 'bg-emerald-400' },
      reconnecting: { label: 'Reconnecting', cls: 'status-reconnecting', dot: 'bg-amber-400'   },
      offline:      { label: 'Offline',      cls: 'status-offline',      dot: 'bg-red-400'     },
    };
    const s = map[type] || map.offline;
    statusEl.className    = `status-badge ${s.cls} mt-0.5`;
    statusDot.className   = `w-1.5 h-1.5 rounded-full ${s.dot}`;
    statusLbl.textContent = s.label;
  }

  // ═══════════════════════════════════════════════════════
  //  WEBSOCKET
  // ═══════════════════════════════════════════════════════
  function connectWS() {
    if (ws && ws.readyState === WebSocket.OPEN) return;
    try {
      ws = new WebSocket(WS_URL);
      ws.binaryType = 'arraybuffer';   // ← required for binary PCM frames
    } catch (err) {
      console.warn('[WS] Failed to create WebSocket:', err);
      setStatus('offline');
      return;
    }

    ws.onopen = () => {
      console.log('[WS] ✅ Connected');
      reconnectAttempts = 0;
      setStatus('connected');
    };

    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        handleText(event.data);
      } else {
        handleBinary(event.data);   // ArrayBuffer — PCM audio phrase
      }
    };

    ws.onerror = () => { /* handled by onclose */ };

    ws.onclose = () => {
      setStatus('offline');
      if (isOpen && reconnectAttempts < MAX_RECONNECT) {
        setStatus('reconnecting');
        const delay = Math.min(1000 * 2 ** reconnectAttempts, 30000);
        reconnectAttempts++;
        reconnectTimeout = setTimeout(connectWS, delay);
      }
    };
  }

  function wsSend(data) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(data);
  }

  // ═══════════════════════════════════════════════════════
  //  TEXT MESSAGE HANDLER
  // ═══════════════════════════════════════════════════════
  function handleText(msg) {
    console.log('[WS] ←', msg.length > 120 ? msg.substring(0, 120) + '…' : msg);

    // ── Connection ──────────────────────────────────────
    if (msg.startsWith('CONNECTED:')) {
      setStatus('connected');
      return;
    }

    // ── Recording ───────────────────────────────────────
    if (msg === 'RECORDING_STARTED') {
      recordLbl.textContent = 'Listening… click mic to stop';
      return;
    }
    if (msg === 'PROCESSING_AUDIO') {
      recordLbl.textContent = 'Processing…';
      showTyping();
      return;
    }

    // ── Transcript / Response ───────────────────────────
    if (msg.startsWith('TRANSCRIBED:')) {
      const text = msg.slice('TRANSCRIBED:'.length).trim();
      if (text) replaceLastUserMessage(text);
      return;
    }
    if (msg.startsWith('AI_RESPONSE:')) {
      const text = msg.slice('AI_RESPONSE:'.length).trim();
      hideTyping();
      if (text) addMessage(text, 'ai');
      return;
    }

    // ── AUDIO_START — new session ────────────────────────
    if (msg.startsWith('AUDIO_START')) {
      const colonIdx = msg.indexOf(':');
      sessionId = colonIdx !== -1 ? msg.slice(colonIdx + 1).trim() : null;

      // Reset session state
      pendingHeader        = null;
      phrasesScheduled     = 0;
      phrasesEnded         = 0;
      sessionEndSignalled  = false;
      isSessionActive      = true;
      if (finishedTimeout) { clearTimeout(finishedTimeout); finishedTimeout = null; }

      // Initialise / re-use AudioContext
      _ensurePlaybackCtx();
      // New session starts NOW on the timeline
      nextStartTime = playCtx.currentTime;

      showAudioPlaying();
      console.log(`[TTS] 🎵 Session start — id: ${sessionId || '(none)'}`);
      return;
    }

    // ── JSON audio_chunk header ──────────────────────────
    // Python sends: { type:"audio_chunk", sample_rate, channels, encoding, bytes, duration_ms }
    // followed immediately by a binary message of exactly `bytes` bytes.
    if (msg.charAt(0) === '{') {
      try {
        const obj = JSON.parse(msg);
        if (obj.type === 'audio_chunk') {
          pendingHeader = obj;
          // The binary message arrives next via handleBinary()
        }
      } catch (_) {
        console.warn('[WS] Unknown JSON:', msg.substring(0, 80));
      }
      return;
    }

    // ── AUDIO_END — all phrases have been sent ───────────
    if (msg === 'AUDIO_END') {
      sessionEndSignalled = true;
      console.log(`[TTS] 🏁 AUDIO_END — ${phrasesScheduled} phrases scheduled`);
      // If all phrases already ended before this message arrived, finish now.
      _checkSessionComplete();
      return;
    }

    // ── Misc ────────────────────────────────────────────
    if (msg === 'NO_SPEECH_DETECTED') {
      hideTyping(); hideRecording();
      addMessage("I didn't catch that. Try again?", 'ai');
      return;
    }
    if (msg === 'NO_AUDIO_RECEIVED') { hideTyping(); hideRecording(); return; }
    if (msg === 'BUFFER_FULL')  { stopRecording(); addMessage('⚠️ Recording buffer full.', 'ai'); return; }
    if (msg === 'PONG')         { return; }
  }

  // ═══════════════════════════════════════════════════════
  //  BINARY HANDLER — one complete PCM16 phrase
  // ═══════════════════════════════════════════════════════
  function handleBinary(arrayBuffer) {
    if (!pendingHeader) {
      console.warn('[TTS] ⚠️ Binary received without header — dropped');
      return;
    }
    if (!isSessionActive) {
      console.warn('[TTS] ⚠️ Binary received outside session — dropped');
      pendingHeader = null;
      return;
    }

    const header  = pendingHeader;
    pendingHeader = null;

    _schedulePhraseForPlayback(arrayBuffer, header);
  }

  // ═══════════════════════════════════════════════════════
  //  PLAYBACK CORE — schedule a single phrase
  // ═══════════════════════════════════════════════════════

  /**
   * MOBILE FIX: iOS/Android Chrome blocks AudioContext until a user gesture.
   * Call this inside every tap/click handler so the context is already
   * running by the time audio data arrives from the server.
   */
  function _unlockAudio() {
    try {
      if (!playCtx || playCtx.state === 'closed') {
        playCtx       = new (window.AudioContext || window.webkitAudioContext)();
        nextStartTime = 0;
        console.log('[TTS] 🔓 AudioContext created via user gesture');
      }
      if (playCtx.state === 'suspended') {
        playCtx.resume().then(() => {
          console.log('[TTS] ▶ AudioContext resumed');
        }).catch(() => {});
      }
    } catch (e) {
      console.warn('[TTS] _unlockAudio failed:', e);
    }
  }

  function _ensurePlaybackCtx() {
    if (!playCtx || playCtx.state === 'closed') {
      playCtx       = new (window.AudioContext || window.webkitAudioContext)();
      nextStartTime = 0;
    }
    if (playCtx.state === 'suspended') {
      playCtx.resume().catch(() => {});
    }
  }

  function _schedulePhraseForPlayback(arrayBuffer, header) {
    _ensurePlaybackCtx();

    const sr         = header.sample_rate || SAMPLE_RATE;
    const pcm16      = new Int16Array(arrayBuffer);
    const numSamples = pcm16.length;

    // Decode signed int16 → float32
    const float32 = new Float32Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
      float32[i] = pcm16[i] / 32768.0;
    }

    // Build AudioBuffer — use playCtx.sampleRate so mobile browsers
    // don't reject mismatched rates. Resample if backend SR differs.
    let finalFloat32 = float32;
    const ctxSR = playCtx.sampleRate;
    if (sr !== ctxSR) {
      const ratio     = ctxSR / sr;
      const newLen    = Math.round(numSamples * ratio);
      finalFloat32    = new Float32Array(newLen);
      for (let i = 0; i < newLen; i++) {
        finalFloat32[i] = float32[Math.min(Math.floor(i / ratio), numSamples - 1)];
      }
    }
    const audioBuf = playCtx.createBuffer(1, finalFloat32.length, ctxSR);
    audioBuf.getChannelData(0).set(finalFloat32);

    // Schedule: start at nextStartTime (≥ now)
    const startAt = Math.max(playCtx.currentTime + 0.005, nextStartTime);
    nextStartTime = startAt + audioBuf.duration + PHRASE_GAP;

    const source = playCtx.createBufferSource();
    source.buffer = audioBuf;
    source.connect(playCtx.destination);

    const phraseIndex = phrasesScheduled;
    phrasesScheduled++;

    source.onended = () => {
      phrasesEnded++;
      console.log(`[TTS] ✅ Phrase ${phraseIndex + 1} ended (${phrasesEnded}/${phrasesScheduled} done)`);
      _checkSessionComplete();
    };

    source.start(startAt);

    const startsInMs = Math.max(0, (startAt - playCtx.currentTime) * 1000);
    console.log(`[TTS] ▶ Phrase ${phraseIndex + 1}: ${numSamples} samples, `
              + `${audioBuf.duration.toFixed(2)}s, starts in ${startsInMs.toFixed(0)}ms`);
  }

  /**
   * Check whether all scheduled phrases have finished AND the backend has
   * signalled AUDIO_END. When both are true, send AUDIO_FINISHED.
   */
  function _checkSessionComplete() {
    if (!isSessionActive) return;
    if (!sessionEndSignalled) return;           // still receiving phrases
    if (phrasesEnded < phrasesScheduled) return; // still playing

    _finishSession(sessionId);
  }

  function _finishSession(sid) {
    if (!isSessionActive) return;
    isSessionActive = false;
    hideAudioPlaying();

    if (finishedTimeout) { clearTimeout(finishedTimeout); finishedTimeout = null; }

    const msg = sid ? `AUDIO_FINISHED:${sid}` : 'AUDIO_FINISHED';
    console.log(`[TTS] 🔔 ${msg}`);
    wsSend(msg);
  }

  /** Safety fallback: if onended never fires (tab hidden, browser bug) */
  function _armSafetyTimeout(totalDurationMs) {
    if (finishedTimeout) clearTimeout(finishedTimeout);
    finishedTimeout = setTimeout(() => {
      if (isSessionActive) {
        console.warn('[TTS] ⏰ Safety timeout — forcing session finish');
        _finishSession(sessionId);
      }
    }, totalDurationMs + 3000);
  }

  // ═══════════════════════════════════════════════════════
  //  PANEL OPEN / CLOSE
  // ═══════════════════════════════════════════════════════
  function openPanel() {
    isOpen = true;
    _unlockAudio(); // mobile: create AudioContext during user gesture
    panel.classList.remove('opacity-0', 'scale-95', 'pointer-events-none', 'translate-y-4');
    panel.classList.add('opacity-100', 'scale-100', 'translate-y-0');
    toggle.classList.add('hidden');
    input.focus();
    connectWS();
  }

  function closePanel() {
    isOpen = false;
    panel.classList.add('opacity-0', 'scale-95', 'pointer-events-none', 'translate-y-4');
    panel.classList.remove('opacity-100', 'scale-100', 'translate-y-0');
    toggle.classList.remove('hidden');

    if (isRecording) stopRecording();

    // Tear down audio
    if (playCtx) { playCtx.close(); playCtx = null; }
    if (finishedTimeout) { clearTimeout(finishedTimeout); finishedTimeout = null; }
    isSessionActive     = false;
    pendingHeader       = null;
    phrasesScheduled    = 0;
    phrasesEnded        = 0;
    sessionEndSignalled = false;
    hideAudioPlaying();
  }

  toggle.addEventListener('click', openPanel);
  closeBtn.addEventListener('click', closePanel);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && isOpen) closePanel(); });

  // ═══════════════════════════════════════════════════════
  //  TEXT SEND
  // ═══════════════════════════════════════════════════════
  function sendTextMessage() {
    _unlockAudio(); // mobile: keep AudioContext alive
    const text = input.value.trim();
    if (!text) return;
    addMessage(text, 'user');
    input.value      = '';
    sendBtn.disabled = true;
    showTyping();
    wsSend(JSON.stringify({ type: 'message', input_type: 'text', content: text }));
  }

  sendBtn.addEventListener('click', sendTextMessage);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendTextMessage(); }
  });
  input.addEventListener('input', () => { sendBtn.disabled = !input.value.trim(); });

  // ═══════════════════════════════════════════════════════
  //  MIC RECORDING
  // ═══════════════════════════════════════════════════════
  micBtn.addEventListener('click', () => { _unlockAudio(); isRecording ? stopRecording() : startRecording(); });

  async function startRecording() {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate:         SAMPLE_RATE,
          channelCount:       CHANNELS,
          echoCancellation:   true,
          noiseSuppression:   true,
          autoGainControl:    true,
        },
      });

      recCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
      const src = recCtx.createMediaStreamSource(micStream);

      scriptNode = recCtx.createScriptProcessor(4096, 1, 1);
      scriptNode.onaudioprocess = (e) => {
        if (!isRecording) return;
        const f32  = e.inputBuffer.getChannelData(0);
        const i16  = new Int16Array(f32.length);
        for (let i = 0; i < f32.length; i++) {
          const s = Math.max(-1, Math.min(1, f32[i]));
          i16[i]  = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        wsSend(i16.buffer);
      };

      src.connect(scriptNode);
      scriptNode.connect(recCtx.destination);

      isRecording = true;
      wsSend('START_RECORDING');

      micBtn.classList.add('!text-red-400', '!border-red-500/40', '!bg-red-500/10', 'animate-pulse');
      micBtn.querySelector('i').className = 'bx bx-stop text-lg';
      recordEl.classList.remove('hidden');
      recordLbl.textContent = 'Listening… click mic to stop';
      addMessage('🎤 Listening…', 'user');

    } catch (err) {
      console.error('[Mic] Error:', err);
      addMessage('⚠️ Microphone access denied. Please allow mic permissions.', 'ai');
    }
  }

  function stopRecording() {
    if (!isRecording) return;
    isRecording = false;
    wsSend('STOP_RECORDING');

    if (scriptNode) { scriptNode.disconnect(); scriptNode = null; }
    if (recCtx && recCtx.state !== 'closed') { recCtx.close(); recCtx = null; }
    if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }

    micBtn.classList.remove('!text-red-400', '!border-red-500/40', '!bg-red-500/10', 'animate-pulse');
    micBtn.querySelector('i').className = 'bx bx-microphone text-lg';
    hideRecording();
  }

  // ═══════════════════════════════════════════════════════
  //  UI HELPERS
  // ═══════════════════════════════════════════════════════
  function showAudioPlaying()  { audioPlayingEl?.classList.remove('hidden'); scrollToBottom(); }
  function hideAudioPlaying()  { audioPlayingEl?.classList.add('hidden'); }
  function showTyping()        { typingEl.classList.remove('hidden'); scrollToBottom(); }
  function hideTyping()        { typingEl.classList.add('hidden'); }
  function hideRecording()     { recordEl.classList.add('hidden'); }

  function addMessage(text, sender) {
    const div     = document.createElement('div');
    div.className = sender === 'user'
      ? 'chat-bubble-user animate-slide-up'
      : 'chat-bubble-ai  animate-slide-up';
    div.innerHTML = `<p class="text-sm">${escapeHtml(text)}</p>`;
    messagesEl.appendChild(div);
    scrollToBottom();
  }

  function replaceLastUserMessage(text) {
    const bubbles = messagesEl.querySelectorAll('.chat-bubble-user');
    const last    = bubbles[bubbles.length - 1];
    if (last) {
      const prefix  = last.innerHTML.includes('🎤') ? '🎤 ' : '';
      last.innerHTML = `<p class="text-sm">${prefix}${escapeHtml(text)}</p>`;
    }
  }

  function scrollToBottom() {
    requestAnimationFrame(() => { messagesEl.scrollTop = messagesEl.scrollHeight; });
  }

  function escapeHtml(str) {
    const el = document.createElement('span');
    el.textContent = str;
    return el.innerHTML;
  }

  // Inject micro-styles
  const style = document.createElement('style');
  style.textContent = `
    .chat-bubble-user.voice-message::before {
      content: '🎤';
      margin-right: 4px;
      font-size: 12px;
      opacity: 0.7;
    }
  `;
  document.head.appendChild(style);
}