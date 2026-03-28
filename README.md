# SpeechBrain Speaker Recognition App

A minimal full-stack app with:
- SpeechBrain-based speaker embedding extraction
- FastAPI endpoints for enroll/verify/identify/call-auth
- Browser UI for manual testing

## API Endpoints

- `GET /health`
- `POST /api/v1/speakers/enroll`
  - form-data: `speaker_id`, `audio`
- `POST /api/v1/speakers/verify`
  - form-data: `speaker_id`, `audio`, optional `threshold`
- `POST /api/v1/agent/identify`
  - form-data: `audio`, optional `top_k`
- `POST /api/v1/agent/call-auth`
  - form-data: `audio`, optional `claimed_speaker_id`, optional `threshold`

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Notes

- Use clear voice samples from the same recording environment for better accuracy.
- Browser-uploaded files are accepted as `audio/*`; WAV is generally most reliable.
- First startup downloads the SpeechBrain model to `pretrained_models/`.
