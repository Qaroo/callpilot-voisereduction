from __future__ import annotations

"""RealTimeSpeakerFilter — ECAPA-TDNN speaker isolation (no transcription).

Architecture
────────────
1. **Enroll** — short voice sample → 192-d ECAPA-TDNN embedding (voice-print).
2. **Process chunk** — every ~0.5 s chunk of raw PCM-16 audio:
   • Convert bytes → float32 numpy.
   • ECAPA-TDNN → 192-d embedding → cosine similarity vs target.
   • similarity < threshold → MUTE (zero-fill).
   • similarity ≥ threshold → KEEP (pass through).
3. **Return PCM-16** — caller gets cleaned audio bytes immediately.
"""

import io
import time
import wave

import numpy as np
import torch
import torchaudio

# ── torchaudio compat shims (2.11+ removed legacy backend API) ──────────
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["dispatcher"]  # type: ignore[attr-defined]
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda _: None  # type: ignore[attr-defined]

from speechbrain.inference.speaker import EncoderClassifier

from app.config import MODEL_SAVEDIR, MODEL_SOURCE

# ── Constants ────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
CHUNK_SAMPLES = SAMPLE_RATE // 2          # 0.5-second chunks (8 000 samples)
FADE_SAMPLES = int(0.005 * SAMPLE_RATE)  # 5 ms crossfade   (80 samples)


class RealTimeSpeakerFilter:
    """ECAPA-TDNN speaker isolation engine. No transcription."""

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = EncoderClassifier.from_hparams(
            source=MODEL_SOURCE,
            savedir=str(MODEL_SAVEDIR),
            run_opts={"device": self.device},
        )

    # ─────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────

    def embedding_from_wav_bytes(self, wav_bytes: bytes) -> np.ndarray:
        """Compute 192-d speaker embedding from WAV bytes."""
        waveform = self._wav_bytes_to_waveform(wav_bytes)
        return self._embed(waveform)

    def process_chunk(
        self,
        audio_bytes: bytes,
        target_embedding: np.ndarray,
        threshold: float = 0.25,
    ) -> dict:
        """Process a single ~0.5 s PCM-16 chunk (16 kHz mono).

        Returns {"is_target": bool, "similarity": float, "audio_bytes": bytes}.
        - If target speaker → returns original audio_bytes.
        - Otherwise → returns silence (zeros).
        """
        t0 = time.perf_counter()
        chunk_f32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(
            np.float32,
        ) / 32768.0
        silence = np.zeros(len(chunk_f32), dtype=np.int16).tobytes()

        # Energy gate — skip embedding for silence
        if float(np.abs(chunk_f32).mean()) < 0.008:
            dt = (time.perf_counter() - t0) * 1000
            print(f"[CHUNK] silence skip  {dt:.0f}ms")
            return {"is_target": False, "similarity": 0.0, "audio_bytes": silence}

        similarity = float(self._cosine_sim(
            target_embedding, self._embed_numpy(chunk_f32),
        ))
        dt = (time.perf_counter() - t0) * 1000
        tag = "✓ KEEP" if similarity >= threshold else "✗ MUTE"
        print(f"[CHUNK] {tag}  sim={similarity:.3f}  {dt:.0f}ms")

        if similarity >= threshold:
            return {"is_target": True, "similarity": round(similarity, 4),
                    "audio_bytes": audio_bytes}
        return {"is_target": False, "similarity": round(similarity, 4),
                "audio_bytes": silence}

    # ─────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _wav_bytes_to_float32(self, wav_bytes: bytes) -> np.ndarray:
        """WAV bytes → mono float32 array at 16 kHz (stdlib wave, fastest)."""
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if sw == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 1:
            data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            data = (data - 128.0) / 128.0
        elif sw == 4:
            data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported sample width: {sw}")

        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)

        if sr != SAMPLE_RATE:
            t = torch.from_numpy(data).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, SAMPLE_RATE)
            data = t.squeeze(0).numpy()

        return data

    def _wav_bytes_to_waveform(self, wav_bytes: bytes) -> torch.Tensor:
        """WAV bytes → torch [1, N] at 16 kHz."""
        return torch.from_numpy(self._wav_bytes_to_float32(wav_bytes)).unsqueeze(0)

    def _embed(self, waveform: torch.Tensor) -> np.ndarray:
        """192-d ECAPA-TDNN embedding from [1, N] waveform."""
        signal = waveform.to(self.device)
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        with torch.inference_mode():
            emb = self.classifier.encode_batch(signal)
        return emb.squeeze().detach().cpu().numpy().astype(np.float32)

    def _embed_numpy(self, audio_f32: np.ndarray) -> np.ndarray:
        """Embedding from float32 numpy array."""
        return self._embed(torch.from_numpy(audio_f32).unsqueeze(0))

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
