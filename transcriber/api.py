"""HTTP service: POST an audio file to /transcribe."""

import tempfile

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from transcriber.audio import AudioError, save_upload
from transcriber.config import settings

app = FastAPI(title="Transcription Pipeline", version="0.1.0")


class UploadAccepted(BaseModel):
    filename: str
    size_bytes: int


@app.exception_handler(AudioError)
def audio_error_handler(_: Request, exc: AudioError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Plain `def` (not `async def`): FastAPI runs it in a worker thread, so blocking file I/O
# and, later, CPU-bound transcription do not stall the event loop.
@app.post("/transcribe", response_model=UploadAccepted)
def transcribe(file: UploadFile = File(...)) -> UploadAccepted:
    with tempfile.TemporaryDirectory() as tmp:
        path = save_upload(file.file, file.filename, tmp, settings.max_upload_bytes)
        return UploadAccepted(filename=file.filename, size_bytes=path.stat().st_size)
