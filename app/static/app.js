/* Speaker Isolator — frontend logic */
"use strict";

const SAMPLE_RATE = 16000;
const CHUNK_SAMPLES = 8000; // 0.5 sec

// ── DOM refs ────────────────────────────────────────────────────────────
const $speakerId      = document.getElementById("speaker-id");
const $btnRecord      = document.getElementById("btn-record");
const $enrollFile     = document.getElementById("enroll-file");
const $recordStatus   = document.getElementById("record-status");
const $enrollResult   = document.getElementById("enroll-result");
const $btnRefresh     = document.getElementById("btn-refresh");
const $speakersList   = document.getElementById("speakers-list");
const $speakerSelect  = document.getElementById("speaker-select");
const $threshold      = document.getElementById("threshold");
const $thresholdVal   = document.getElementById("threshold-val");
const $btnStream      = document.getElementById("btn-stream");
const $streamStatus   = document.getElementById("stream-status");
const $streamStats    = document.getElementById("stream-stats");
const $streamResults  = document.getElementById("stream-results");
const $originalAudio  = document.getElementById("original-audio");
const $isolatedAudio  = document.getElementById("isolated-audio");
const $dlOriginal     = document.getElementById("download-original");
const $dlIsolated     = document.getElementById("download-isolated");

// ── State ───────────────────────────────────────────────────────────────
let enrollRecorder = null;
let enrollChunks = [];
let isEnrolling = false;

// Streaming state
let wsStream = null;
let isStreaming = false;
let micStream = null;
let audioWorklet = null;
let pcmBuffer = new Float32Array(0);       // accumulates mic samples
let originalChunks = [];   // raw PCM-16 chunks (original)
let isolatedChunks = [];   // filtered PCM-16 chunks (from server)
let chunksSent = 0;
let chunksKept = 0;
let streamStartTime = 0;

// ── Threshold slider ────────────────────────────────────────────────────
$threshold.addEventListener("input", () => {
  $thresholdVal.textContent = $threshold.value;
});

// ── Refresh speakers list ───────────────────────────────────────────────
async function refreshSpeakers() {
  try {
    const res = await fetch("/api/speakers");
    const data = await res.json();
    $speakersList.textContent = data.speakers.length ? data.speakers.join(", ") : "(אין דוברים רשומים)";
    // Also populate select
    const prev = $speakerSelect.value;
    $speakerSelect.innerHTML = '<option value="">בחר דובר...</option>';
    for (const id of data.speakers) {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = id;
      $speakerSelect.appendChild(opt);
    }
    if (prev) $speakerSelect.value = prev;
  } catch (e) {
    $speakersList.textContent = "שגיאה: " + e.message;
  }
}
$btnRefresh.addEventListener("click", refreshSpeakers);

// ═══════════════════════════════════════════════════════════════════════
//  ENROLL
// ═══════════════════════════════════════════════════════════════════════

async function startEnrollRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  enrollChunks = [];
  isEnrolling = true;
  enrollRecorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
  enrollRecorder.ondataavailable = (e) => { if (e.data.size > 0) enrollChunks.push(e.data); };
  enrollRecorder.onstop = async () => {
    stream.getTracks().forEach(t => t.stop());
    isEnrolling = false;
    const blob = new Blob(enrollChunks, { type: "audio/webm" });
    const wavBlob = await webmToWav(blob);
    $btnRecord.textContent = "🎤 הקלט";
    $recordStatus.textContent = "ממיר ומעלה...";
    await doEnroll(wavBlob);
  };
  enrollRecorder.start();
}

$btnRecord.addEventListener("click", () => {
  if (isEnrolling) { enrollRecorder.stop(); return; }
  if (!$speakerId.value.trim()) { $recordStatus.textContent = "⚠️ הכנס שם דובר"; return; }
  $btnRecord.textContent = "⏹ עצור";
  $recordStatus.textContent = "מקליט...";
  startEnrollRecording();
});

$enrollFile.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  if (!$speakerId.value.trim()) { $recordStatus.textContent = "⚠️ הכנס שם דובר"; return; }
  $recordStatus.textContent = "מעלה...";
  const wavBlob = file.name.endsWith(".wav") ? file : await audioFileToWav(file);
  await doEnroll(wavBlob);
});

async function doEnroll(wavBlob) {
  const fd = new FormData();
  fd.append("speaker_id", $speakerId.value.trim());
  fd.append("file", wavBlob, "sample.wav");
  try {
    const res = await fetch("/api/enroll", { method: "POST", body: fd });
    const data = await res.json();
    $enrollResult.textContent = data.message || JSON.stringify(data);
    $recordStatus.textContent = "";
    refreshSpeakers();
  } catch (e) {
    $enrollResult.textContent = "שגיאה: " + e.message;
    $recordStatus.textContent = "";
  }
}

// ═══════════════════════════════════════════════════════════════════════
//  REAL-TIME STREAMING
// ═══════════════════════════════════════════════════════════════════════

$btnStream.addEventListener("click", () => {
  if (isStreaming) {
    stopStreaming();
  } else {
    startStreaming();
  }
});

async function startStreaming() {
  const speakerId = $speakerSelect.value;
  if (!speakerId) { $streamStatus.textContent = "⚠️ בחר דובר"; return; }

  // Reset
  originalChunks = [];
  isolatedChunks = [];
  chunksSent = 0;
  chunksKept = 0;
  pcmBuffer = new Float32Array(0);
  $streamResults.hidden = true;
  $streamStats.innerHTML = "";

  // 1. Open WebSocket
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  wsStream = new WebSocket(`${proto}//${location.host}/ws/isolate`);
  wsStream.binaryType = "arraybuffer";

  wsStream.onopen = () => {
    wsStream.send(JSON.stringify({
      speaker_id: speakerId,
      threshold: parseFloat($threshold.value),
    }));
  };

  let expectBinary = false;

  wsStream.onmessage = (event) => {
    if (typeof event.data === "string") {
      const msg = JSON.parse(event.data);
      if (msg.status === "ready") {
        $streamStatus.textContent = "🟢 מחובר — מקליט...";
        startMic();
      } else if (msg.error) {
        $streamStatus.textContent = "❌ " + msg.error;
        stopStreaming();
      } else if (msg.type === "decision") {
        chunksSent++;
        if (msg.is_target) chunksKept++;
        const elapsed = ((Date.now() - streamStartTime) / 1000).toFixed(1);
        $streamStats.innerHTML =
          `<span>⏱ ${elapsed}s</span>` +
          `<span>📊 ${chunksKept}/${chunksSent}</span>` +
          `<span>📈 sim=${msg.similarity.toFixed(3)}</span>`;
        expectBinary = true;
      }
    } else if (expectBinary) {
      // Binary = filtered PCM-16 audio data
      isolatedChunks.push(new Uint8Array(event.data));
      expectBinary = false;
    }
  };

  wsStream.onclose = () => {
    if (isStreaming) stopStreaming();
  };
  wsStream.onerror = () => {
    $streamStatus.textContent = "❌ שגיאת חיבור WebSocket";
    stopStreaming();
  };

  isStreaming = true;
  streamStartTime = Date.now();
  $btnStream.textContent = "⏹ עצור הקלטה";
  $streamStatus.textContent = "מתחבר...";
}

async function startMic() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 48000, channelCount: 1, echoCancellation: true, noiseSuppression: false }
    });
  } catch (e) {
    $streamStatus.textContent = "❌ לא ניתן לגשת למיקרופון";
    stopStreaming();
    return;
  }

  const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });

  // ScriptProcessorNode (widely supported, simple)
  const source = audioCtx.createMediaStreamSource(micStream);
  const processor = audioCtx.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = (e) => {
    if (!isStreaming) return;
    const input = e.inputBuffer.getChannelData(0);
    // Accumulate float32 samples
    const newBuf = new Float32Array(pcmBuffer.length + input.length);
    newBuf.set(pcmBuffer);
    newBuf.set(input, pcmBuffer.length);
    pcmBuffer = newBuf;

    // Send 0.5-sec chunks (CHUNK_SAMPLES) as PCM-16
    while (pcmBuffer.length >= CHUNK_SAMPLES) {
      const chunk = pcmBuffer.slice(0, CHUNK_SAMPLES);
      pcmBuffer = pcmBuffer.slice(CHUNK_SAMPLES);

      const int16 = float32ToInt16(chunk);
      originalChunks.push(new Uint8Array(int16.buffer));

      if (wsStream && wsStream.readyState === WebSocket.OPEN) {
        wsStream.send(int16.buffer);
      }
    }
  };

  source.connect(processor);
  processor.connect(audioCtx.destination);

  // Store for cleanup
  audioWorklet = { audioCtx, source, processor };
}

function stopStreaming() {
  isStreaming = false;
  $btnStream.textContent = "🎙 התחל הקלטה";

  // Close mic
  if (audioWorklet) {
    try { audioWorklet.processor.disconnect(); } catch {}
    try { audioWorklet.source.disconnect(); } catch {}
    try { audioWorklet.audioCtx.close(); } catch {}
    audioWorklet = null;
  }
  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  }

  // Close WebSocket
  if (wsStream) {
    try { wsStream.close(); } catch {}
    wsStream = null;
  }

  pcmBuffer = new Float32Array(0);

  // Build result audio
  if (originalChunks.length > 0) {
    const origWav = pcm16ChunksToWav(originalChunks);
    const isoWav  = pcm16ChunksToWav(isolatedChunks);

    const origUrl = URL.createObjectURL(origWav);
    const isoUrl  = URL.createObjectURL(isoWav);

    $originalAudio.src = origUrl;
    $isolatedAudio.src = isoUrl;
    $dlOriginal.href = origUrl;
    $dlIsolated.href = isoUrl;

    $streamResults.hidden = false;
    const elapsed = ((Date.now() - streamStartTime) / 1000).toFixed(1);
    $streamStatus.textContent = `✅ הושלם — ${elapsed}s, ${chunksKept}/${chunksSent} קטעים שמורים`;
  } else {
    $streamStatus.textContent = "הקלטה הסתיימה (ללא נתונים)";
  }
}

// ── Helpers: PCM-16 chunks → WAV Blob ───────────────────────────────────
function pcm16ChunksToWav(chunks) {
  let totalLen = 0;
  for (const c of chunks) totalLen += c.length;
  const pcmData = new Uint8Array(totalLen);
  let offset = 0;
  for (const c of chunks) {
    pcmData.set(c, offset);
    offset += c.length;
  }
  const wavBuf = new ArrayBuffer(44 + pcmData.length);
  const view = new DataView(wavBuf);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + pcmData.length, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);           // PCM
  view.setUint16(22, 1, true);           // mono
  view.setUint32(24, SAMPLE_RATE, true); // sample rate
  view.setUint32(28, SAMPLE_RATE * 2, true); // byte rate
  view.setUint16(32, 2, true);           // block align
  view.setUint16(34, 16, true);          // bits per sample
  writeString(view, 36, "data");
  view.setUint32(40, pcmData.length, true);
  new Uint8Array(wavBuf, 44).set(pcmData);
  return new Blob([wavBuf], { type: "audio/wav" });
}

function float32ToInt16(float32) {
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return int16;
}

// ── WebM → WAV conversion ───────────────────────────────────────────────
async function webmToWav(blob) {
  const arrayBuf = await blob.arrayBuffer();
  const ctx = new OfflineAudioContext(1, 1, SAMPLE_RATE);
  const decoded = await ctx.decodeAudioData(arrayBuf);
  return audioBufferToWav(decoded);
}

async function audioFileToWav(file) {
  const arrayBuf = await file.arrayBuffer();
  const ctx = new OfflineAudioContext(1, 1, SAMPLE_RATE);
  const decoded = await ctx.decodeAudioData(arrayBuf);
  return audioBufferToWav(decoded);
}

function audioBufferToWav(buffer) {
  const numFrames = buffer.length;
  const sampleRate = buffer.sampleRate;
  let samples;
  if (sampleRate !== SAMPLE_RATE) {
    const ratio = sampleRate / SAMPLE_RATE;
    const newLen = Math.round(numFrames / ratio);
    samples = new Float32Array(newLen);
    const src = buffer.getChannelData(0);
    for (let i = 0; i < newLen; i++) {
      const srcIdx = Math.min(Math.floor(i * ratio), numFrames - 1);
      samples[i] = src[srcIdx];
    }
  } else {
    samples = buffer.getChannelData(0);
  }
  const wavBuf = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(wavBuf);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, SAMPLE_RATE, true);
  view.setUint32(28, SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([wavBuf], { type: "audio/wav" });
}

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

// ── Init ────────────────────────────────────────────────────────────────
refreshSpeakers();
