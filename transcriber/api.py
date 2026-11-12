"""HTTP service: POST an audio file to /transcribe."""

import tempfile
from dataclasses import asdict
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from transcriber.audio import AudioError, save_upload
from transcriber.config import settings
from transcriber.engines import Segment, Transcriber, TranscriptionError, get_transcriber


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the engine before accepting traffic: bad config fails the deploy, not a user
    # request, and the first caller does not pay the model-load latency.
    get_transcriber()
    yield


app = FastAPI(title="Transcription Pipeline", version="0.1.0", lifespan=lifespan)


class TranscriptionResponse(BaseModel):
    filename: str
    engine: str
    language: str | None
    duration: float
    text: str
    segments: list[Segment]


@app.exception_handler(AudioError)
@app.exception_handler(TranscriptionError)
def pipeline_error_handler(_: Request, exc: AudioError | TranscriptionError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Plain `def` (not `async def`): FastAPI runs it in a worker thread, so blocking file I/O
# and CPU-bound transcription do not stall the event loop.
@app.post("/transcribe", response_model=TranscriptionResponse)
def transcribe(
    file: UploadFile = File(...),
    transcriber: Transcriber = Depends(get_transcriber),
) -> TranscriptionResponse:
    with tempfile.TemporaryDirectory() as tmp:
        path = save_upload(file.file, file.filename, tmp, settings.max_upload_bytes)
        result = transcriber.transcribe(path)
    return TranscriptionResponse(filename=file.filename, engine=transcriber.name, **asdict(result))
