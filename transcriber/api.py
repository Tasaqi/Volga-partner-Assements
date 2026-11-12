"""HTTP service: POST an audio file to /transcribe."""

import json
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Iterator

from fastapi import Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from transcriber.audio import AudioError, AudioInfo, probe_audio, save_upload
from transcriber.config import settings
from transcriber.engines import Segment, Transcriber, Transcription, TranscriptionError, get_transcriber
from transcriber.pipeline import merge, transcribe_chunks


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
    source: AudioInfo  # what was uploaded, as detected from the bytes


@app.exception_handler(AudioError)
@app.exception_handler(TranscriptionError)
def pipeline_error_handler(_: Request, exc: AudioError | TranscriptionError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


# `async` so it runs on the event loop, not in the worker thread pool: when every worker is
# busy transcribing, the health check must still answer or the orchestrator kills the pod.
@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def _ndjson_events(
    chunks: Iterator[Transcription], header: dict, tmp: tempfile.TemporaryDirectory
) -> Iterator[str]:
    """One JSON object per line: start, segment..., then done (or error)."""
    line = lambda obj: json.dumps(obj, ensure_ascii=False) + "\n"
    try:
        yield line({"type": "start", **header})
        language, duration = None, 0.0
        for chunk in chunks:
            for segment in chunk.segments:
                yield line({"type": "segment", **asdict(segment)})
            language, duration = chunk.language, duration + chunk.duration
        yield line({"type": "done", "language": language, "duration": round(duration, 3)})
    except (AudioError, TranscriptionError) as e:
        # Headers (200) are already sent, so a mid-stream failure is reported in-band.
        yield line({"type": "error", "status": e.status_code, "detail": str(e)})
    finally:
        tmp.cleanup()  # also runs when the client disconnects (generator is closed)


# Plain `def` (not `async def`): FastAPI runs it in a worker thread, so blocking file I/O
# and CPU-bound transcription do not stall the event loop.
@app.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    responses={200: {"content": {"application/x-ndjson": {}}}},
)
def transcribe(
    file: UploadFile = File(...),
    stream: bool = Query(False, description="Stream segments as NDJSON while transcribing"),
    transcriber: Transcriber = Depends(get_transcriber),
):
    tmp = tempfile.TemporaryDirectory()
    try:
        # Validation happens before any response starts, so bad input is a normal 4xx even
        # in streaming mode.
        path = save_upload(file.file, file.filename, tmp.name, settings.max_upload_bytes)
        source = probe_audio(path)  # reject non-audio before spending model time on it
    except BaseException:
        tmp.cleanup()
        raise
    chunks = transcribe_chunks(path, transcriber, settings.chunk_seconds)
    header = {"filename": file.filename, "engine": transcriber.name, "source": asdict(source)}

    if stream:
        return StreamingResponse(_ndjson_events(chunks, header, tmp), media_type="application/x-ndjson")
    try:
        result = merge(chunks)
    finally:
        tmp.cleanup()
    return TranscriptionResponse(**header, **asdict(result))
