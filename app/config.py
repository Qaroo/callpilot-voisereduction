from pathlib import Path

APP_NAME = "Speaker Isolator"
APP_VERSION = "2.0.0"
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_SAVEDIR = Path("pretrained_models/spkrec")
VOICEPRINT_STORE = Path("data/voiceprints.json")
DEFAULT_THRESHOLD = 0.25
