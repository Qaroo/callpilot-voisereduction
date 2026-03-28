"""Speaker Isolator — simple ECAPA-TDNN speaker filter.

Endpoints:
  POST /api/enroll   — enroll speaker voice-print from WAV upload
  GET  /api/speakers — list enrolled speakers
  GET  /docs         — API documentation / usage guide
  WS   /ws/isolate   — real-time: send PCM-16 chunks, receive filtered audio
"""

from __future__ import annotations

import asyncio
import json
import logging

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import APP_NAME, APP_VERSION, DEFAULT_THRESHOLD, VOICEPRINT_STORE
from app.schemas import EnrollResponse, SpeakersResponse
from app.speaker_service import RealTimeSpeakerFilter
from app.storage import VoiceprintStore

logger = logging.getLogger(__name__)

app = FastAPI(title=APP_NAME, version=APP_VERSION)

# ── Singletons ───────────────────────────────────────────────────────────
store = VoiceprintStore(VOICEPRINT_STORE)
engine = RealTimeSpeakerFilter()


# ── REST endpoints ───────────────────────────────────────────────────────

@app.post("/api/enroll", response_model=EnrollResponse)
async def enroll(
    speaker_id: str = Form(...),
    file: UploadFile = File(...),
):
    """רישום דובר — Enroll or update a speaker voice-print from a WAV sample."""
    wav_bytes = await file.read()
    embedding = engine.embedding_from_wav_bytes(wav_bytes)
    store.upsert(speaker_id, embedding.tolist())
    return EnrollResponse(
        speaker_id=speaker_id,
        message=f"Speaker '{speaker_id}' enrolled successfully.",
    )


@app.get("/api/speakers", response_model=SpeakersResponse)
async def list_speakers():
    """Return list of enrolled speaker IDs."""
    return SpeakersResponse(speakers=list(store.all().keys()))


# ── API Documentation page ──────────────────────────────────────────────

DOCS_HTML = """<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>API Docs — Speaker Isolator</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 2rem; line-height: 1.7; }
  .container { max-width: 900px; margin: 0 auto; }
  h1 { color: #58a6ff; margin-bottom: 0.5rem; }
  h2 { color: #79c0ff; margin: 2rem 0 0.5rem; border-bottom: 1px solid #30363d; padding-bottom: 0.3rem; }
  h3 { color: #d2a8ff; margin: 1.2rem 0 0.3rem; }
  pre { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
        padding: 1rem; overflow-x: auto; font-size: 0.9rem; margin: 0.5rem 0 1rem; }
  code { color: #f0883e; }
  .endpoint { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
              padding: 1.2rem; margin: 1rem 0; }
  .method { display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-weight: bold; font-size: 0.85rem; margin-left: 0.5rem; }
  .post { background: #238636; color: #fff; }
  .get { background: #1f6feb; color: #fff; }
  .ws { background: #8957e5; color: #fff; }
  .path { font-family: monospace; color: #f0883e; font-size: 1.05rem; }
  table { border-collapse: collapse; width: 100%; margin: 0.5rem 0; }
  th, td { border: 1px solid #30363d; padding: 0.4rem 0.8rem; text-align: right; }
  th { background: #161b22; }
  .note { background: #1c2333; border-right: 3px solid #58a6ff; padding: 0.8rem 1rem;
          margin: 1rem 0; border-radius: 4px; }
  a { color: #58a6ff; }
</style>
</head>
<body>
<div class="container">

<h1>📡 Speaker Isolator — API Docs</h1>
<p>בידוד דובר בזמן אמת באמצעות ECAPA-TDNN. שלח קטעי אודיו ב-WebSocket, קבל בחזרה אודיו מסונן.</p>

<h2>1. רישום דובר</h2>
<div class="endpoint">
  <span class="method post">POST</span>
  <span class="path">/api/enroll</span>
  <p>רושם טביעת קול (voice-print) של דובר מקובץ WAV.</p>
  <h3>פרמטרים (multipart/form-data)</h3>
  <table>
    <tr><th>שם</th><th>סוג</th><th>תיאור</th></tr>
    <tr><td><code>speaker_id</code></td><td>string</td><td>שם/מזהה הדובר</td></tr>
    <tr><td><code>file</code></td><td>WAV file</td><td>הקלטת קול (3-10 שניות מומלץ)</td></tr>
  </table>
  <h3>דוגמה — curl</h3>
  <pre>curl -X POST http://localhost:8000/api/enroll \\
  -F "speaker_id=ilay" \\
  -F "file=@sample.wav"</pre>
  <h3>דוגמה — Python</h3>
  <pre>import requests

with open("sample.wav", "rb") as f:
    resp = requests.post("http://localhost:8000/api/enroll",
        data={"speaker_id": "ilay"},
        files={"file": ("sample.wav", f, "audio/wav")})
print(resp.json())
# {"speaker_id": "ilay", "message": "Speaker 'ilay' enrolled successfully."}</pre>
  <h3>תגובה</h3>
  <pre>{"speaker_id": "ilay", "message": "Speaker 'ilay' enrolled successfully."}</pre>
</div>

<h2>2. רשימת דוברים</h2>
<div class="endpoint">
  <span class="method get">GET</span>
  <span class="path">/api/speakers</span>
  <pre>curl http://localhost:8000/api/speakers
# {"speakers": ["ilay", "dvora", "roee"]}</pre>
</div>

<h2>3. WebSocket — בידוד דובר בזמן אמת</h2>
<div class="endpoint">
  <span class="method ws">WS</span>
  <span class="path">/ws/isolate</span>

  <h3>פרוטוקול</h3>
  <ol>
    <li><strong>התחבר</strong> — <code>ws://localhost:8000/ws/isolate</code></li>
    <li><strong>שלח JSON init</strong> — <code>{"speaker_id": "ilay", "threshold": 0.25}</code></li>
    <li><strong>קבל JSON ready</strong> — <code>{"status": "ready", "speaker_id": "ilay"}</code></li>
    <li><strong>שלח binary</strong> — קטע אודיו PCM-16 mono 16kHz (מומלץ 0.5 שניות = 16,000 bytes)</li>
    <li><strong>קבל JSON</strong> — <code>{"type": "decision", "is_target": true, "similarity": 0.87}</code></li>
    <li><strong>קבל binary</strong> — אותו קטע אודיו (אם target) או שקט (אם לא)</li>
    <li>חזור לשלב 4 עבור כל קטע</li>
  </ol>

  <div class="note">
    <strong>פורמט אודיו:</strong> PCM-16 signed little-endian, mono, 16000 Hz.<br>
    כל קטע = 0.5 שניות = 8000 דגימות = 16,000 bytes.
  </div>

  <h3>דוגמה — Python</h3>
  <pre>import asyncio
import json
import wave
import websockets

async def isolate_speaker():
    uri = "ws://localhost:8000/ws/isolate"
    async with websockets.connect(uri) as ws:
        # 1. Init
        await ws.send(json.dumps({
            "speaker_id": "ilay",
            "threshold": 0.25
        }))
        ready = json.loads(await ws.recv())
        print("Server:", ready)  # {"status": "ready", ...}

        # 2. Read WAV file and send chunks
        with wave.open("recording.wav", "rb") as wf:
            assert wf.getsampwidth() == 2  # PCM-16
            assert wf.getnchannels() == 1  # mono
            assert wf.getframerate() == 16000

            chunk_size = 8000  # 0.5 sec = 8000 samples * 2 bytes
            output_frames = b""

            while True:
                pcm_data = wf.readframes(chunk_size)
                if not pcm_data:
                    break

                # Send audio chunk
                await ws.send(pcm_data)

                # Receive decision JSON
                decision = json.loads(await ws.recv())
                print(f"sim={decision['similarity']:.3f} "
                      f"{'KEEP' if decision['is_target'] else 'MUTE'}")

                # Receive filtered audio (same size)
                filtered = await ws.recv()
                output_frames += filtered

        # 3. Save output
        with wave.open("isolated.wav", "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(output_frames)

        print("Done! Saved isolated.wav")

asyncio.run(isolate_speaker())</pre>

  <h3>דוגמה — JavaScript (Browser)</h3>
  <pre>const SAMPLE_RATE = 16000;
const CHUNK_SAMPLES = 8000; // 0.5 sec

const ws = new WebSocket("ws://localhost:8000/ws/isolate");
ws.binaryType = "arraybuffer";

ws.onopen = () => {
  ws.send(JSON.stringify({
    speaker_id: "ilay",
    threshold: 0.25
  }));
};

ws.onmessage = (event) => {
  if (typeof event.data === "string") {
    const msg = JSON.parse(event.data);
    console.log("JSON:", msg);
    // {"status":"ready",...} or {"type":"decision",...}
  } else {
    // Binary = filtered PCM-16 audio chunk
    const pcm16 = new Int16Array(event.data);
    // → play via AudioContext or accumulate
  }
};

// Send audio from MediaRecorder / AudioWorklet:
// processor.port.onmessage = ({data}) => {
//   const int16 = float32ToInt16(data);
//   ws.send(int16.buffer);
// };</pre>
</div>

<h2>4. פורמט אודיו</h2>
<div class="note">
  <p>כל האודיו חייב להיות:</p>
  <ul>
    <li><strong>PCM-16</strong> signed little-endian</li>
    <li><strong>Mono</strong> (ערוץ אחד)</li>
    <li><strong>16,000 Hz</strong> sample rate</li>
  </ul>
  <p>לרישום דובר — ניתן לשלוח WAV רגיל (הממיר אוטומטי).</p>
  <p>ל-WebSocket — שלח raw PCM-16 bytes בלבד (ללא WAV header).</p>
</div>

</div>
</body>
</html>"""


@app.get("/api/docs", response_class=HTMLResponse)
async def docs_page():
    """API documentation page with usage examples."""
    return HTMLResponse(content=DOCS_HTML)


# ── WebSocket: real-time speaker isolation ───────────────────────────────

@app.websocket("/ws/isolate")
async def ws_isolate(ws: WebSocket):
    """Real-time speaker isolation via WebSocket.

    Protocol:
      1. Connect
      2. Send JSON init: {"speaker_id": ..., "threshold": 0.25}
      3. Receive JSON:   {"status": "ready", "speaker_id": ...}
      4. Send binary PCM-16 chunks (16 kHz mono, ~0.5 s = 16000 bytes)
      5. Receive JSON:   {"type": "decision", "is_target": bool, "similarity": float}
      6. Receive binary: filtered audio (original if target, silence if not)
      7. Repeat 4-6
    """
    await ws.accept()
    try:
        # 1. Init message
        init_raw = await ws.receive_text()
        init = json.loads(init_raw)
        speaker_id = init.get("speaker_id", "")
        threshold = float(init.get("threshold", DEFAULT_THRESHOLD))

        embedding_list = store.get(speaker_id)
        if embedding_list is None:
            await ws.send_json({"error": f"Speaker '{speaker_id}' not found."})
            await ws.close()
            return

        target_emb = np.array(embedding_list, dtype=np.float32)
        await ws.send_json({"status": "ready", "speaker_id": speaker_id})

        # 2. Stream loop — process each chunk via ECAPA
        while True:
            audio_bytes = await ws.receive_bytes()
            result = await asyncio.to_thread(
                engine.process_chunk, audio_bytes, target_emb, threshold,
            )

            await ws.send_json({
                "type": "decision",
                "is_target": result["is_target"],
                "similarity": result["similarity"],
            })
            await ws.send_bytes(result["audio_bytes"])

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
        try:
            await ws.close(code=1011)
        except Exception:
            pass


# ── Static files (must be last) ─────────────────────────────────────────
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
