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

// --- Original text/TTS WebSocket (kokoweb.py) ---
const _WS_BASE = import.meta.env.VITE_WS_URL
  ?? `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws`;
const WS_URL = _WS_BASE.includes('?')
  ? _WS_BASE + '&ngrok-skip-browser-warning=true'
  : _WS_BASE + '?ngrok-skip-browser-warning=true';

// --- VAD real-time voice WebSocket (code/server.py) ---
// In dev: /vad-ws proxy → ws://localhost:8000/ws  (defined in vite.config.js)
// In prod: set VITE_VAD_WS_URL to the deployed VAD backend URL
const _VAD_BASE = import.meta.env.VITE_VAD_WS_URL
  ?? `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/vad-ws`;
const VAD_WS_URL = _VAD_BASE.includes('?')
  ? _VAD_BASE + '&ngrok-skip-browser-warning=true'
  : _VAD_BASE + '?ngrok-skip-browser-warning=true';

const SAMPLE_RATE = 24000;   // must match Python backend
const CHANNELS = 1;
const PHRASE_GAP = 0.005;   // 5ms silence between phrases (gapless feel)

// --- VAD audio batching (from code/static/app.js) ---
const BATCH_SAMPLES = 2048;          // samples per batch to send
const HEADER_BYTES = 8;             // 4-byte timestamp_ms + 4-byte flags
const FRAME_BYTES = BATCH_SAMPLES * 2;  // Int16 = 2 bytes per sample
const MESSAGE_BYTES = HEADER_BYTES + FRAME_BYTES;

// ═══════════════════════════════════════════════════════


export function initChatPanel() {
  const root = document.getElementById('chat-root');
  if (!root) return;

  // ─── Inject HTML (UNCHANGED — same UI layout) ─────────
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
          <!-- Mic Button (VAD toggle) -->
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

        <div id="chat-recording" class="hidden flex items-center gap-2 mt-2 px-1">
          <span class="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
          <span id="chat-recording-label" class="text-[11px] text-red-400 font-mono">Listening…</span>
        </div>
      </div>
    </div>
  `;

  // ─── DOM Refs ─────────────────────────────────────────
  const toggle = document.getElementById('chat-toggle');
  const panel = document.getElementById('chat-panel');
  const closeBtn = document.getElementById('chat-close');
  const messagesEl = document.getElementById('chat-messages');
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send');
  const micBtn = document.getElementById('chat-mic');
  const typingEl = document.getElementById('chat-typing');
  const audioPlayingEl = document.getElementById('chat-audio-playing');
  const recordEl = document.getElementById('chat-recording');
  const recordLbl = document.getElementById('chat-recording-label');
  const statusEl = document.getElementById('chat-status');
  const statusDot = document.getElementById('chat-status-dot');
  const statusLbl = document.getElementById('chat-status-label');

  let isOpen = false;

  // ═══════════════════════════════════════════════════════
  //  ORIGINAL TEXT WEBSOCKET STATE (kokoweb.py)
  // ═══════════════════════════════════════════════════════
  /** @type {WebSocket|null} */
  let ws = null;
  let reconnectTimeout = null;
  let reconnectAttempts = 0;
  const MAX_RECONNECT = 10;

  // ═══════════════════════════════════════════════════════
  //  VAD VOICE WEBSOCKET STATE (code/server.py)
  // ═══════════════════════════════════════════════════════
  /** @type {WebSocket|null} */
  let vadWs = null;
  let vadReconnectTimeout = null;
  let vadReconnectAttempts = 0;
  const VAD_MAX_RECONNECT = 10;

  // ═══════════════════════════════════════════════════════
  //  VAD RECORDING STATE
  // ═══════════════════════════════════════════════════════
  let isVADActive = false;                 // Is the VAD mic currently on?
  /** @type {AudioContext|null} */
  let vadAudioContext = null;
  /** @type {MediaStream|null} */
  let vadMediaStream = null;
  /** @type {AudioWorkletNode|null} */
  let vadMicWorkletNode = null;
  /** @type {AudioWorkletNode|null} */
  let vadTtsWorkletNode = null;

  // ═══════════════════════════════════════════════════════
  //  VAD TTS PLAYBACK STATE
  // ═══════════════════════════════════════════════════════
  let isTTSPlaying = false;    // Is TTS audio currently playing through worklet?
  let ignoreIncomingTTS = false;   // Ignore TTS chunks after stop_tts until next session

  // ═══════════════════════════════════════════════════════
  //  VAD AUDIO BATCHING (from code/static/app.js)
  //
  //  PCM samples are accumulated in batches of BATCH_SAMPLES.
  //  Each batch is sent as a binary WebSocket message with an
  //  8-byte header: [uint32 timestamp_ms | uint32 flags].
  //  Flag bit 0 = isTTSPlaying (so server-side VAD knows to
  //  discount echo from TTS audio).
  // ═══════════════════════════════════════════════════════
  const bufferPool = [];
  let batchBuffer = null;
  let batchView = null;
  let batchInt16 = null;
  let batchOffset = 0;

  /** Allocate or recycle a batch buffer */
  function initBatch() {
    if (!batchBuffer) {
      batchBuffer = bufferPool.pop() || new ArrayBuffer(MESSAGE_BYTES);
      batchView = new DataView(batchBuffer);
      batchInt16 = new Int16Array(batchBuffer, HEADER_BYTES);
      batchOffset = 0;
    }
  }

  /** Write the 8-byte header and send the batch to the VAD backend */
  function flushBatch() {
    const ts = Date.now() & 0xFFFFFFFF;
    batchView.setUint32(0, ts, false);                     // big-endian timestamp_ms
    const flags = isTTSPlaying ? 1 : 0;
    batchView.setUint32(4, flags, false);                  // big-endian flags

    if (vadWs && vadWs.readyState === WebSocket.OPEN) {
      vadWs.send(batchBuffer);
    }

    bufferPool.push(batchBuffer);
    batchBuffer = null;
  }

  /** Flush any partially filled batch (zero-pad remainder) */
  function flushRemainder() {
    if (batchOffset > 0) {
      for (let i = batchOffset; i < BATCH_SAMPLES; i++) {
        batchInt16[i] = 0;
      }
      flushBatch();
    }
  }

  // ═══════════════════════════════════════════════════════
  //  VAD LIVE TEXT STATE
  //
  //  Partial transcriptions and partial assistant answers
  //  are shown as "typing" bubbles that update in real-time.
  //  When the final version arrives, the typing bubble is
  //  replaced with a permanent chat message.
  // ═══════════════════════════════════════════════════════
  /** @type {HTMLElement|null} — the current "typing" user bubble from VAD */
  let vadUserTypingBubble = null;
  /** @type {HTMLElement|null} — the current "typing" assistant bubble from VAD */
  let vadAssistantTypingBubble = null;
  /** Dedup guard — blocks duplicate final_assistant_answer messages */
  let lastFinalAnswerText = null;

  // ═══════════════════════════════════════════════════════
  //  KOKORO AUDIO PLAYBACK STATE (original text mode TTS)
  // ═══════════════════════════════════════════════════════
  /** @type {AudioContext|null} */
  let playCtx = null;
  let nextStartTime = 0;
  let sessionId = null;
  let pendingHeader = null;
  let isSessionActive = false;
  let phrasesScheduled = 0;
  let phrasesEnded = 0;
  let sessionEndSignalled = false;
  let finishedTimeout = null;

  // ═══════════════════════════════════════════════════════
  //  STATUS HELPERS
  // ═══════════════════════════════════════════════════════
  function setStatus(type) {
    const map = {
      connected: { label: 'Connected', cls: 'status-connected', dot: 'bg-emerald-400' },
      reconnecting: { label: 'Reconnecting', cls: 'status-reconnecting', dot: 'bg-amber-400' },
      offline: { label: 'Offline', cls: 'status-offline', dot: 'bg-red-400' },
    };
    const s = map[type] || map.offline;
    statusEl.className = `status-badge ${s.cls} mt-0.5`;
    statusDot.className = `w-1.5 h-1.5 rounded-full ${s.dot}`;
    statusLbl.textContent = s.label;
  }

  // ═══════════════════════════════════════════════════════
  //  ORIGINAL TEXT WEBSOCKET (kokoweb.py)
  // ═══════════════════════════════════════════════════════
  function connectWS() {
    if (ws && ws.readyState === WebSocket.OPEN) return;
    try {
      ws = new WebSocket(WS_URL);
      ws.binaryType = 'arraybuffer';
    } catch (err) {
      console.warn('[WS] Failed to create WebSocket:', err);
      setStatus('offline');
      return;
    }

    ws.onopen = () => {
      console.log('[WS] ✅ Connected to text backend');
      reconnectAttempts = 0;
      setStatus('connected');
    };

    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        handleTextWSMessage(event.data);
      } else {
        handleTextWSBinary(event.data);
      }
    };

    ws.onerror = () => { /* handled by onclose */ };

    ws.onclose = () => {
      // Only change status if VAD is also not connected
      const vadConnected = vadWs && vadWs.readyState === WebSocket.OPEN;
      if (!vadConnected) {
        setStatus('offline');
      }
      if (isOpen && reconnectAttempts < MAX_RECONNECT) {
        // Silently retry — don't show "reconnecting" if VAD is working
        if (!vadConnected) {
          setStatus('reconnecting');
        }
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
  //  TEXT WS MESSAGE HANDLER (kokoweb.py protocol)
  //  — unchanged from original ChatPanel
  // ═══════════════════════════════════════════════════════
  function handleTextWSMessage(msg) {
    console.log('[WS] ←', msg.length > 120 ? msg.substring(0, 120) + '…' : msg);

    if (msg.startsWith('CONNECTED:')) { setStatus('connected'); return; }
    if (msg === 'RECORDING_STARTED') { return; }
    if (msg === 'PROCESSING_AUDIO') { showTyping(); return; }

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

    // Kokoro TTS session handling
    if (msg.startsWith('AUDIO_START')) {
      const colonIdx = msg.indexOf(':');
      sessionId = colonIdx !== -1 ? msg.slice(colonIdx + 1).trim() : null;
      pendingHeader = null;
      phrasesScheduled = 0;
      phrasesEnded = 0;
      sessionEndSignalled = false;
      isSessionActive = true;
      if (finishedTimeout) { clearTimeout(finishedTimeout); finishedTimeout = null; }
      _ensurePlaybackCtx();
      nextStartTime = playCtx.currentTime;
      showAudioPlaying();
      return;
    }
    if (msg.charAt(0) === '{') {
      try {
        const obj = JSON.parse(msg);
        if (obj.type === 'audio_chunk') { pendingHeader = obj; }
      } catch (_) { }
      return;
    }
    if (msg === 'AUDIO_END') {
      sessionEndSignalled = true;
      _checkSessionComplete();
      return;
    }
    if (msg === 'NO_SPEECH_DETECTED') {
      hideTyping();
      addMessage("I didn't catch that. Try again?", 'ai');
      return;
    }
    if (msg === 'NO_AUDIO_RECEIVED') { hideTyping(); return; }
    if (msg === 'BUFFER_FULL') { addMessage('⚠️ Buffer full.', 'ai'); return; }
  }

  // ═══════════════════════════════════════════════════════
  //  TEXT WS BINARY HANDLER (Kokoro TTS — original)
  // ═══════════════════════════════════════════════════════
  function handleTextWSBinary(arrayBuffer) {
    if (!pendingHeader || !isSessionActive) {
      pendingHeader = null;
      return;
    }
    const header = pendingHeader;
    pendingHeader = null;
    _schedulePhraseForPlayback(arrayBuffer, header);
  }

  // ═══════════════════════════════════════════════════════
  //  KOKORO PLAYBACK (unchanged from original)
  // ═══════════════════════════════════════════════════════
  function _unlockAudio() {
    try {
      if (!playCtx || playCtx.state === 'closed') {
        playCtx = new (window.AudioContext || window.webkitAudioContext)();
        nextStartTime = 0;
      }
      if (playCtx.state === 'suspended') {
        playCtx.resume().catch(() => { });
      }
    } catch (e) {
      console.warn('[TTS] _unlockAudio failed:', e);
    }
  }

  function _ensurePlaybackCtx() {
    if (!playCtx || playCtx.state === 'closed') {
      playCtx = new (window.AudioContext || window.webkitAudioContext)();
      nextStartTime = 0;
    }
    if (playCtx.state === 'suspended') playCtx.resume().catch(() => { });
  }

  function _schedulePhraseForPlayback(arrayBuffer, header) {
    _ensurePlaybackCtx();
    const sr = header.sample_rate || SAMPLE_RATE;
    const pcm16 = new Int16Array(arrayBuffer);
    const numSamples = pcm16.length;
    const float32 = new Float32Array(numSamples);
    for (let i = 0; i < numSamples; i++) float32[i] = pcm16[i] / 32768.0;

    let finalFloat32 = float32;
    const ctxSR = playCtx.sampleRate;
    if (sr !== ctxSR) {
      const ratio = ctxSR / sr;
      const newLen = Math.round(numSamples * ratio);
      finalFloat32 = new Float32Array(newLen);
      for (let i = 0; i < newLen; i++) {
        finalFloat32[i] = float32[Math.min(Math.floor(i / ratio), numSamples - 1)];
      }
    }
    const audioBuf = playCtx.createBuffer(1, finalFloat32.length, ctxSR);
    audioBuf.getChannelData(0).set(finalFloat32);

    const startAt = Math.max(playCtx.currentTime + 0.005, nextStartTime);
    nextStartTime = startAt + audioBuf.duration + PHRASE_GAP;

    const source = playCtx.createBufferSource();
    source.buffer = audioBuf;
    source.connect(playCtx.destination);
    phrasesScheduled++;
    source.onended = () => { phrasesEnded++; _checkSessionComplete(); };
    source.start(startAt);
  }

  function _checkSessionComplete() {
    if (!isSessionActive || !sessionEndSignalled) return;
    if (phrasesEnded < phrasesScheduled) return;
    _finishSession(sessionId);
  }

  function _finishSession(sid) {
    if (!isSessionActive) return;
    isSessionActive = false;
    hideAudioPlaying();
    if (finishedTimeout) { clearTimeout(finishedTimeout); finishedTimeout = null; }
    const msg = sid ? `AUDIO_FINISHED:${sid}` : 'AUDIO_FINISHED';
    wsSend(msg);
  }

  // ═══════════════════════════════════════════════════════
  //  VAD WEBSOCKET (code/server.py)
  //
  //  This WebSocket carries the real-time voice pipeline:
  //  - Outgoing: binary PCM audio batches with 8-byte header
  //  - Incoming: JSON messages for partial/final transcripts,
  //    partial/final assistant answers, and TTS audio chunks
  // ═══════════════════════════════════════════════════════
  function connectVADWebSocket() {
    if (vadWs && vadWs.readyState === WebSocket.OPEN) return;
    try {
      vadWs = new WebSocket(VAD_WS_URL);
    } catch (err) {
      console.warn('[VAD-WS] Failed to create WebSocket:', err);
      return;
    }

    vadWs.onopen = () => {
      console.log('[VAD-WS] ✅ Connected to VAD backend');
      vadReconnectAttempts = 0;
      setStatus('connected');
    };

    vadWs.onmessage = (evt) => {
      if (typeof evt.data === 'string') {
        try {
          const msg = JSON.parse(evt.data);
          handleVADMessage(msg);
        } catch (e) {
          console.error('[VAD-WS] Error parsing message:', e);
        }
      }
    };

    vadWs.onerror = () => { /* handled by onclose */ };

    vadWs.onclose = () => {
      console.log('[VAD-WS] Connection closed');
      if (isVADActive && vadReconnectAttempts < VAD_MAX_RECONNECT) {
        const delay = Math.min(1000 * 2 ** vadReconnectAttempts, 30000);
        vadReconnectAttempts++;
        vadReconnectTimeout = setTimeout(connectVADWebSocket, delay);
      }
    };
  }

  function vadWsSend(data) {
    if (vadWs && vadWs.readyState === WebSocket.OPEN) vadWs.send(data);
  }

  // ═══════════════════════════════════════════════════════
  //  VAD MESSAGE HANDLER
  //
  //  Handles the JSON protocol from code/server.py:
  //  - Partial user speech → live typing bubble
  //  - Final user speech → permanent user message
  //  - Partial AI answer → live streaming AI response
  //  - Final AI answer → permanent AI message
  //  - TTS chunks → audio playback via worklet
  //  - Interruptions → clear TTS, reset state
  // ═══════════════════════════════════════════════════════
  function handleVADMessage({ type, content }) {

    // ── Partial user transcription (live speech-to-text) ──
    if (type === 'partial_user_request') {
      const text = content?.trim();
      if (text) {
        if (!vadUserTypingBubble) {
          // Create a new typing bubble
          vadUserTypingBubble = document.createElement('div');
          vadUserTypingBubble.className = 'chat-bubble-user animate-slide-up';
          vadUserTypingBubble.setAttribute('data-vad-typing', 'user');
          messagesEl.appendChild(vadUserTypingBubble);
        }
        vadUserTypingBubble.innerHTML = `<p class="text-sm">${escapeHtml(text)}<span style="opacity:.5;margin-left:4px;"></span></p>`;
        scrollToBottom();
      }
      return;
    }

    // ── Final user transcription ─────────────────────────
    if (type === 'final_user_request') {
      // Remove the typing bubble and add a permanent message
      // Only add the message if there was a VAD typing bubble
      // (text-sent messages are already shown by sendTextMessage)
      if (vadUserTypingBubble) {
        vadUserTypingBubble.remove();
        vadUserTypingBubble = null;
        if (content?.trim()) {
          addMessage(content.trim(), 'user');
        }
      }
      return;
    }

    // ── Partial assistant answer (streaming AI response) ──
    //  Hide typing dots immediately, show/update ONE streaming bubble.
    if (type === 'partial_assistant_answer') {
      const text = content?.trim();
      if (!text) return;

      // Hide typing dots — streaming text replaces them
      hideTyping();

      // Create the streaming bubble once, update it on each partial
      if (!vadAssistantTypingBubble) {
        vadAssistantTypingBubble = document.createElement('div');
        vadAssistantTypingBubble.className = 'chat-bubble-ai animate-slide-up';
        messagesEl.appendChild(vadAssistantTypingBubble);
      }
      vadAssistantTypingBubble.innerHTML = `<p class="text-sm">${escapeHtml(text)}</p>`;
      scrollToBottom();
      return;
    }

    // ── Final assistant answer ───────────────────────────
    //  Finalize the streaming bubble. If no streaming bubble exists
    //  (short answer with no partials), create a fresh message.
    if (type === 'final_assistant_answer') {
      hideTyping();
      const finalText = content?.trim();

      // Dedup: skip if this exact final was already processed
      if (finalText && finalText === lastFinalAnswerText) return;
      lastFinalAnswerText = finalText || null;

      if (vadAssistantTypingBubble) {
        // Just update the text in the existing bubble — no new elements
        if (finalText) {
          vadAssistantTypingBubble.innerHTML = `<p class="text-sm">${escapeHtml(finalText)}</p>`;
        }
        vadAssistantTypingBubble = null; // Release reference (keep element in DOM)
      } else if (finalText) {
        addMessage(finalText, 'ai');
      }
      return;
    }

    // ── TTS audio chunk (base64-encoded Int16 PCM) ───────
    //  Decoded and fed to the TTS playback AudioWorklet
    //  for immediate low-latency playback.
    if (type === 'tts_chunk') {
      if (ignoreIncomingTTS) return;
      const int16Data = base64ToInt16Array(content);
      if (vadTtsWorkletNode) {
        vadTtsWorkletNode.port.postMessage(int16Data);
        // Only show "AI is speaking" when TTS worklet is actually playing
        showAudioPlaying();
      }
      return;
    }

    // ── TTS interruption (barge-in) ──────────────────────
    //  Server detected user speech while TTS was playing.
    //  Clear the TTS buffer and re-enable incoming chunks.
    if (type === 'tts_interruption') {
      if (vadTtsWorkletNode) {
        vadTtsWorkletNode.port.postMessage({ type: 'clear' });
      }
      isTTSPlaying = false;
      ignoreIncomingTTS = false;
      hideAudioPlaying();
      // Don't clear vadAssistantTypingBubble here — let final_assistant_answer handle it
      return;
    }

    // ── Stop TTS (server requests playback stop) ─────────
    //  Happens when user speech is detected during TTS.
    //  Clears the buffer and blocks new chunks until next session.
    if (type === 'stop_tts') {
      if (vadTtsWorkletNode) {
        vadTtsWorkletNode.port.postMessage({ type: 'clear' });
      }
      isTTSPlaying = false;
      ignoreIncomingTTS = true;
      hideAudioPlaying();
      console.log('[VAD] TTS stopped — user interrupted');
      vadWsSend(JSON.stringify({ type: 'tts_stop' }));
      return;
    }
  }

  /**
   * Decode a base64 string into an Int16Array.
   * Used for TTS audio chunks from the VAD backend.
   */
  function base64ToInt16Array(b64) {
    const raw = atob(b64);
    const buf = new ArrayBuffer(raw.length);
    const view = new Uint8Array(buf);
    for (let i = 0; i < raw.length; i++) {
      view[i] = raw.charCodeAt(i);
    }
    return new Int16Array(buf);
  }

  // ═══════════════════════════════════════════════════════
  //  VAD MIC CAPTURE (AudioWorklet)
  //
  //  Uses pcmWorkletProcessor.js (in public/) to capture
  //  mic audio as Int16 PCM in a worklet thread. Samples are
  //  batched into BATCH_SAMPLES-sized chunks with an 8-byte
  //  binary header and sent to the VAD backend.
  // ═══════════════════════════════════════════════════════
  async function startVADCapture() {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: { ideal: SAMPLE_RATE },
          channelCount: CHANNELS,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      vadMediaStream = stream;

      // Create AudioContext for VAD pipeline
      // IMPORTANT: Use DEFAULT sample rate (typically 48kHz), NOT 24kHz!
      // The server's UpsampleOverlap module upsamples TTS audio from 24kHz→48kHz
      // before sending, so TTS playback must run at 48kHz (default).
      // The server's AudioInputProcessor handles resampling mic input internally.
      vadAudioContext = new (window.AudioContext || window.webkitAudioContext)();

      // Load the PCM capture worklet processor
      await vadAudioContext.audioWorklet.addModule('/pcmWorkletProcessor.js');
      vadMicWorkletNode = new AudioWorkletNode(vadAudioContext, 'pcm-worklet-processor');

      // ── Mic data handler ──
      // Each message from the worklet is a raw Int16 ArrayBuffer.
      // We accumulate samples into batches and send them with headers.
      vadMicWorkletNode.port.onmessage = ({ data }) => {
        const incoming = new Int16Array(data);
        let read = 0;
        while (read < incoming.length) {
          initBatch();
          const toCopy = Math.min(
            incoming.length - read,
            BATCH_SAMPLES - batchOffset
          );
          batchInt16.set(incoming.subarray(read, read + toCopy), batchOffset);
          batchOffset += toCopy;
          read += toCopy;
          if (batchOffset === BATCH_SAMPLES) {
            flushBatch();
          }
        }
      };

      // Connect mic source → worklet
      const source = vadAudioContext.createMediaStreamSource(stream);
      source.connect(vadMicWorkletNode);

      console.log('[VAD] 🎙️ Mic capture started');
    } catch (err) {
      console.error('[VAD] Mic access error:', err);
      addMessage('⚠️ Microphone access denied. Please allow mic permissions.', 'ai');
      throw err;
    }
  }

  // ═══════════════════════════════════════════════════════
  //  VAD TTS PLAYBACK (AudioWorklet)
  //
  //  Uses ttsPlaybackProcessor.js (in public/) to play back
  //  TTS audio chunks through the audio output. The worklet
  //  emits ttsPlaybackStarted/ttsPlaybackStopped events to
  //  notify the VAD backend about TTS state (so VAD can
  //  discount echo from TTS audio).
  // ═══════════════════════════════════════════════════════
  async function setupVADTTSPlayback() {
    if (!vadAudioContext) return;

    await vadAudioContext.audioWorklet.addModule('/ttsPlaybackProcessor.js');
    vadTtsWorkletNode = new AudioWorkletNode(vadAudioContext, 'tts-playback-processor');

    // ── TTS state change events ──
    // Notify the VAD backend when TTS starts/stops playing so
    // the server-side VAD can discount echo from TTS audio.
    vadTtsWorkletNode.port.onmessage = (event) => {
      const { type } = event.data;
      if (type === 'ttsPlaybackStarted') {
        if (!isTTSPlaying) {
          isTTSPlaying = true;
          showAudioPlaying();
          console.log('[VAD] TTS playback started');
          vadWsSend(JSON.stringify({ type: 'tts_start' }));
        }
      } else if (type === 'ttsPlaybackStopped') {
        if (isTTSPlaying) {
          isTTSPlaying = false;
          ignoreIncomingTTS = false;
          hideAudioPlaying();
          console.log('[VAD] TTS playback stopped');
          vadWsSend(JSON.stringify({ type: 'tts_stop' }));
        }
      }
    };

    // Connect worklet to audio output
    vadTtsWorkletNode.connect(vadAudioContext.destination);
    console.log('[VAD] 🔊 TTS playback worklet ready');
  }

  // ═══════════════════════════════════════════════════════
  //  VAD CLEANUP
  // ═══════════════════════════════════════════════════════
  function cleanupVADAudio() {
    flushRemainder();

    if (vadMicWorkletNode) { vadMicWorkletNode.disconnect(); vadMicWorkletNode = null; }
    if (vadTtsWorkletNode) { vadTtsWorkletNode.disconnect(); vadTtsWorkletNode = null; }
    if (vadAudioContext && vadAudioContext.state !== 'closed') {
      vadAudioContext.close();
      vadAudioContext = null;
    }
    if (vadMediaStream) {
      vadMediaStream.getAudioTracks().forEach(track => track.stop());
      vadMediaStream = null;
    }

    // Reset batching state
    batchBuffer = null;
    batchView = null;
    batchInt16 = null;
    batchOffset = 0;

    // Reset TTS state
    isTTSPlaying = false;
    ignoreIncomingTTS = false;

    // Clear any typing bubbles
    if (vadUserTypingBubble) {
      vadUserTypingBubble.remove();
      vadUserTypingBubble = null;
    }
    if (vadAssistantTypingBubble) {
      vadAssistantTypingBubble.remove();
      vadAssistantTypingBubble = null;
    }
  }

  // ═══════════════════════════════════════════════════════
  //  MIC BUTTON — TOGGLE VAD ON/OFF
  //
  //  Click mic → microphone activates → VAD listens for speech
  //  Click again → stop VAD, release mic
  // ═══════════════════════════════════════════════════════
  async function startVAD() {
    try {
      // Connect VAD WebSocket if needed
      connectVADWebSocket();

      // Wait briefly for WS to connect
      await new Promise((resolve) => {
        if (vadWs && vadWs.readyState === WebSocket.OPEN) {
          resolve();
          return;
        }
        const origOnOpen = vadWs?.onopen;
        if (vadWs) {
          vadWs.onopen = (e) => {
            if (origOnOpen) origOnOpen(e);
            resolve();
          };
        }
        // Fallback timeout
        setTimeout(resolve, 3000);
      });

      // Start mic capture + TTS playback
      await startVADCapture();
      await setupVADTTSPlayback();

      isVADActive = true;

      // Update mic button UI — active VAD state
      micBtn.classList.add('!text-red-400', '!border-red-500/40', '!bg-red-500/10', 'animate-pulse');
      micBtn.querySelector('i').className = 'bx bx-stop text-lg';
      micBtn.title = 'Click to stop listening';
      recordEl.classList.remove('hidden');
      recordLbl.textContent = 'Listening…';

      console.log('[VAD] ✅ Voice mode activated');
    } catch (err) {
      console.error('[VAD] Failed to start:', err);
      stopVAD();
    }
  }

  function stopVAD() {
    isVADActive = false;

    // Clean up audio resources
    cleanupVADAudio();

    // Close VAD WebSocket
    if (vadReconnectTimeout) { clearTimeout(vadReconnectTimeout); vadReconnectTimeout = null; }
    if (vadWs) {
      vadWs.close();
      vadWs = null;
    }

    // Reset mic button UI
    micBtn.classList.remove('!text-red-400', '!border-red-500/40', '!bg-red-500/10', 'animate-pulse');
    micBtn.querySelector('i').className = 'bx bx-microphone text-lg';
    micBtn.title = 'Click to speak';
    recordEl.classList.add('hidden');

    hideAudioPlaying();

    console.log('[VAD] ⏹️ Voice mode deactivated');
  }

  micBtn.addEventListener('click', () => {
    _unlockAudio();
    if (isVADActive) {
      stopVAD();
    } else {
      startVAD();
    }
  });

  // ═══════════════════════════════════════════════════════
  //  PANEL OPEN / CLOSE
  // ═══════════════════════════════════════════════════════
  function openPanel() {
    isOpen = true;
    _unlockAudio();
    panel.classList.remove('opacity-0', 'scale-95', 'pointer-events-none', 'translate-y-4');
    panel.classList.add('opacity-100', 'scale-100', 'translate-y-0');
    toggle.classList.add('hidden');
    input.focus();
    // Connect to VAD backend immediately so status shows 'Connected'
    // and text messages can be sent through it
    connectVADWebSocket();
  }

  function closePanel() {
    isOpen = false;
    panel.classList.add('opacity-0', 'scale-95', 'pointer-events-none', 'translate-y-4');
    panel.classList.remove('opacity-100', 'scale-100', 'translate-y-0');
    toggle.classList.remove('hidden');

    // Stop VAD if active
    if (isVADActive) stopVAD();

    // Disconnect VAD WebSocket
    if (vadReconnectTimeout) { clearTimeout(vadReconnectTimeout); vadReconnectTimeout = null; }
    if (vadWs) { vadWs.close(); vadWs = null; }

    // Tear down Kokoro playback (original text TTS)
    if (playCtx) { playCtx.close(); playCtx = null; }
    if (finishedTimeout) { clearTimeout(finishedTimeout); finishedTimeout = null; }
    isSessionActive = false;
    pendingHeader = null;
    phrasesScheduled = 0;
    phrasesEnded = 0;
    sessionEndSignalled = false;
    hideAudioPlaying();
  }

  toggle.addEventListener('click', openPanel);
  closeBtn.addEventListener('click', closePanel);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && isOpen) closePanel(); });

  // ═══════════════════════════════════════════════════════
  //  TEXT SEND — routes through VAD backend
  // ═══════════════════════════════════════════════════════
  function sendTextMessage() {
    _unlockAudio();
    const text = input.value.trim();
    if (!text) return;
    addMessage(text, 'user');
    input.value = '';
    sendBtn.disabled = true;
    showTyping();
    // Send text through the VAD WebSocket to server.py
    vadWsSend(JSON.stringify({ type: 'text_message', content: text }));
  }

  sendBtn.addEventListener('click', sendTextMessage);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendTextMessage(); }
  });
  input.addEventListener('input', () => { sendBtn.disabled = !input.value.trim(); });

  // ═══════════════════════════════════════════════════════
  //  UI HELPERS
  // ═══════════════════════════════════════════════════════
  function showAudioPlaying() { audioPlayingEl?.classList.remove('hidden'); scrollToBottom(); }
  function hideAudioPlaying() { audioPlayingEl?.classList.add('hidden'); }
  function showTyping() { typingEl.classList.remove('hidden'); scrollToBottom(); }
  function hideTyping() { typingEl.classList.add('hidden'); }

  function addMessage(text, sender) {
    const div = document.createElement('div');
    div.className = sender === 'user'
      ? 'chat-bubble-user animate-slide-up'
      : 'chat-bubble-ai  animate-slide-up';
    div.innerHTML = `<p class="text-sm">${escapeHtml(text)}</p>`;
    messagesEl.appendChild(div);
    scrollToBottom();
  }

  function replaceLastUserMessage(text) {
    const bubbles = messagesEl.querySelectorAll('.chat-bubble-user');
    const last = bubbles[bubbles.length - 1];
    if (last) {
      const prefix = last.innerHTML.includes('🎤') ? '🎤 ' : '';
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