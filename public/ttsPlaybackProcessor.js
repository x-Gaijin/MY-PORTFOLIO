// ttsPlaybackProcessor.js — Plays back TTS audio chunks via AudioWorklet
// Ported from code/static/ttsPlaybackProcessor.js for the VAD voice pipeline
class TTSPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferQueue = [];
    this.readOffset = 0;
    this.samplesRemaining = 0;
    this.isPlaying = false;

    // Listen for incoming messages (PCM chunks or control commands)
    this.port.onmessage = (event) => {
      // Control message: clear the playback buffer (used for barge-in / interruption)
      if (event.data && typeof event.data === "object" && event.data.type === "clear") {
        this.bufferQueue = [];
        this.readOffset = 0;
        this.samplesRemaining = 0;
        this.isPlaying = false;
        return;
      }

      // Otherwise it's a PCM chunk (Int16Array) — queue it for playback
      this.bufferQueue.push(event.data);
      this.samplesRemaining += event.data.length;
    };
  }

  process(inputs, outputs) {
    const outputChannel = outputs[0][0];

    // No audio to play — output silence
    if (this.samplesRemaining === 0) {
      outputChannel.fill(0);
      if (this.isPlaying) {
        this.isPlaying = false;
        // Notify main thread that TTS playback has stopped
        this.port.postMessage({ type: 'ttsPlaybackStopped' });
      }
      return true;
    }

    // Playback just started
    if (!this.isPlaying) {
      this.isPlaying = true;
      // Notify main thread that TTS playback has started
      this.port.postMessage({ type: 'ttsPlaybackStarted' });
    }

    // Fill output buffer from queued PCM chunks
    let outIdx = 0;
    while (outIdx < outputChannel.length && this.bufferQueue.length > 0) {
      const currentBuffer = this.bufferQueue[0];
      const sampleValue = currentBuffer[this.readOffset] / 32768;
      outputChannel[outIdx++] = sampleValue;

      this.readOffset++;
      this.samplesRemaining--;

      if (this.readOffset >= currentBuffer.length) {
        this.bufferQueue.shift();
        this.readOffset = 0;
      }
    }

    // Zero-fill any remaining output samples
    while (outIdx < outputChannel.length) {
      outputChannel[outIdx++] = 0;
    }

    return true;
  }
}

registerProcessor('tts-playback-processor', TTSPlaybackProcessor);
