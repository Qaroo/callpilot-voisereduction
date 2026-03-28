import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


class VoiceprintStore:
    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path
        self._lock = Lock()
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self._write({})

    def _read(self) -> dict:
        with self.store_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: dict) -> None:
        with self.store_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)

    def upsert(self, speaker_id: str, embedding: list[float]) -> str:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            data = self._read()
            data[speaker_id] = {
                "embedding": embedding,
                "updated_at": now,
            }
            self._write(data)
        return now

    def get(self, speaker_id: str) -> list[float] | None:
        with self._lock:
            data = self._read()
            record = data.get(speaker_id)
        if not record:
            return None
        return record.get("embedding")

    def all(self) -> dict[str, list[float]]:
        with self._lock:
            data = self._read()
        return {speaker_id: payload.get("embedding", []) for speaker_id, payload in data.items()}
