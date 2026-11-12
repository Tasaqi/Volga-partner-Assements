"""FastAPI service. Build it with :func:`create_app` (tests inject config + engine).

Run: ``uvicorn app.api:create_app --factory --port 8000``
"""

from __future__ import annotations

import secrets
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app import pipeline
from app.audio import AudioError, AudioInfo, AudioTooLongError, UnsupportedAudioError
from app.config import Settings
from app.engines import TranscriptionEngine, TranscriptionError, build_engine
from app.languages import normalize_language
from app.postprocess import MEDIA_TYPES, render
from app.schemas import Job, JobStatus, OutputFormat, TranscriptionResult

UPLOAD_PIECE = 1024 * 1024  # copy uploads in 1 MB pieces
MULTIPART_OVERHEAD = 1024 * 1024  # room for multipart headers around the file itself

ERROR_STATUS: dict[type[AudioError], int] = {
    UnsupportedAudioError: 415,
    AudioTooLongError: 413,
}


class JobStore:
    """Thread-safe in-memory job registry.

    shortcut: jobs live in process memory and are lost on restart; production would
    use a queue (TaskIQ/Celery + Redis), object storage for audio and Postgres for jobs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._results: dict[str, TranscriptionResult] = {}

    def create(self, filename: str, fmt: OutputFormat) -> Job:
        job = Job(job_id=uuid.uuid4().hex, filename=filename, format=fmt)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> tuple[Job, TranscriptionResult | None] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return (job.model_copy(), self._results.get(job_id)) if job else None

    def update(self, job_id: str, **fields: object) -> None:
        with self._lock:
            job = self._jobs[job_id]
            self._jobs[job_id] = job.model_copy(update=fields)

    def complete(self, job_id: str, result: TranscriptionResult) -> None:
        with self._lock:
            self._results[job_id] = result
            job = self._jobs[job_id]
            self._jobs[job_id] = job.model_copy(
                update={"status": JobStatus.completed, "progress": 1.0}
            )


class _BodyTooLarge(BaseException):
    """BaseException on purpose: FastAPI turns any Exception raised while parsing the
    body into a generic 400, and this must reach RequestGuard as a 413."""


class RequestGuard:
    """Checks the API key and upload size BEFORE the request body is read.

    FastAPI parses multipart forms before it runs dependencies, so checks written as
    dependencies only run after the whole upload is already on disk: a client without
    a key, or with a 50 GB file, could fill the disk. This ASGI middleware rejects
    early on headers and counts body bytes as they arrive (covers chunked uploads too).
    """

    def __init__(self, app: ASGIApp, api_key: str, max_body_bytes: int):
        self.app, self.api_key, self.max_body_bytes = app, api_key, max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/transcriptions"):
            return await self.app(scope, receive, send)

        headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
        given = headers.get("x-api-key", "")
        if self.api_key and not secrets.compare_digest(given.encode(), self.api_key.encode()):
            return await _reject(scope, receive, send, 401, "Invalid or missing X-API-Key")
        if int(headers.get("content-length") or 0) > self.max_body_bytes:
            return await _reject(scope, receive, send, 413, self._too_large())

        received = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self.max_body_bytes:
                raise _BodyTooLarge
            return message

        try:
            await self.app(scope, counting_receive, send)
        except _BodyTooLarge:
            await _reject(scope, receive, send, 413, self._too_large())

    def _too_large(self) -> str:
        return f"Upload exceeds {(self.max_body_bytes - MULTIPART_OVERHEAD) // (1024 * 1024)} MB"


async def _reject(scope: Scope, receive: Receive, send: Send, status: int, detail: str) -> None:
    await JSONResponse({"detail": detail}, status_code=status)(scope, receive, send)


def _save_upload(upload: UploadFile, dest_dir: Path, max_bytes: int) -> Path:
    """Stream the upload to disk in 1 MB pieces, enforcing the size limit on real bytes.

    The on-disk name ignores the client filename (no path traversal, and the
    extension is never trusted: ffprobe decides what the content is).
    """
    dest = dest_dir / "upload"
    size = 0
    with dest.open("wb") as out:
        while piece := upload.file.read(UPLOAD_PIECE):
            size += len(piece)
            if size > max_bytes:
                raise HTTPException(413, f"Upload exceeds {max_bytes // (1024 * 1024)} MB")
            out.write(piece)
    if size == 0:
        raise HTTPException(400, "Uploaded file is empty")
    return dest


def _respond(result: TranscriptionResult, fmt: OutputFormat) -> Response:
    return Response(render(result, fmt), media_type=MEDIA_TYPES[fmt])


def create_app(cfg: Settings | None = None, engine: TranscriptionEngine | None = None) -> FastAPI:
    cfg = cfg or Settings.from_env()
    engine = engine or build_engine(cfg)
    jobs = JobStore()
    executor = ThreadPoolExecutor(max_workers=cfg.job_workers, thread_name_prefix="tp-job")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Transcription Pipeline", version="1.0.0", lifespan=lifespan)
    app.add_middleware(RequestGuard, api_key=cfg.api_key,
                       max_body_bytes=cfg.max_upload_bytes + MULTIPART_OVERHEAD)

    @app.exception_handler(AudioError)
    async def audio_error(_: Request, exc: AudioError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=ERROR_STATUS.get(type(exc), 400))

    @app.exception_handler(TranscriptionError)
    async def engine_error(_: Request, exc: TranscriptionError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=exc.status_code)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "engine": engine.name,
            "model": engine.model,
            "model_loaded": engine.is_loaded(),
        }

    def run_job(job: Job, path: Path, workdir: Path, language: str | None, info: AudioInfo) -> None:
        job_id = job.job_id
        jobs.update(job_id, status=JobStatus.running)
        try:
            result = pipeline.transcribe_file(
                path, cfg, engine, language=language, info=info, filename=job.filename,
                progress=lambda p: jobs.update(job_id, progress=round(min(p, 0.99), 4)),
            )
            jobs.complete(job_id, result)
        except Exception as exc:  # noqa: BLE001 - job must record any failure
            jobs.update(job_id, status=JobStatus.failed, error=str(exc) or type(exc).__name__)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # Plain ``def``: FastAPI runs it in a worker thread, so blocking I/O and
    # subprocess calls never stall the event loop.
    @app.post(
        "/transcriptions",
        responses={202: {"description": "Long file accepted as a background job"}},
    )
    def create_transcription(
        file: UploadFile,
        language: str | None = Query(None, description='"en" or "english"; auto-detect if omitted'),
        format: OutputFormat = Query(OutputFormat.json),
    ) -> Response:
        try:
            language = normalize_language(language) if language else None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        workdir = Path(tempfile.mkdtemp(prefix="tp-upload-"))
        handed_off = False
        try:
            path = _save_upload(file, workdir, cfg.max_upload_bytes)  # 1 MB pieces; 413 / 400
            info = pipeline.inspect(path, cfg)  # ffprobe content check (415), max duration (413)
            filename = file.filename or "upload"

            if info.duration <= cfg.sync_max_duration_s:
                # The header can understate the length, so decode at most the sync limit here:
                # a file that turns out longer becomes a background job, not a long request.
                sync_cfg = replace(cfg, max_duration_s=cfg.sync_max_duration_s)
                try:
                    result = pipeline.transcribe_file(path, sync_cfg, engine, language=language,
                                                      filename=filename, info=info)
                    return _respond(result, format)
                except AudioTooLongError:
                    pass

            job = jobs.create(filename, format)
            future = executor.submit(run_job, job, path, workdir, language, info)
            handed_off = True  # the job now owns workdir and deletes it
            # A job cancelled at shutdown never runs, so its own cleanup never runs either.
            future.add_done_callback(
                lambda f: f.cancelled() and shutil.rmtree(workdir, ignore_errors=True)
            )
            location = f"/transcriptions/{job.job_id}"
            body = {"job_id": job.job_id, "status": job.status.value, "status_url": location}
            return JSONResponse(body, status_code=202, headers={"Location": location})
        finally:
            if not handed_off:
                shutil.rmtree(workdir, ignore_errors=True)

    @app.get("/transcriptions/{job_id}")
    def get_transcription(job_id: str) -> Response:
        found = jobs.get(job_id)
        if found is None:
            raise HTTPException(404, "Job not found")
        job, result = found
        if job.status is JobStatus.completed and result is not None:
            return _respond(result, job.format)
        return JSONResponse(job.model_dump(mode="json"))

    return app
