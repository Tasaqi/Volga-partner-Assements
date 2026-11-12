# Transcription Pipeline

A speech-to-text HTTP service built with FastAPI. Send it an audio or video file in any common format, of any length up to 4 hours. It returns a transcript with absolute timestamps for every segment, as JSON, SRT, VTT or plain text.

Transcription uses the **OpenAI Whisper API** as the primary engine. If an API call fails, or no `OPENAI_API_KEY` is set, the service falls back to [faster-whisper](https://github.com/SYSTRAN/faster-whisper), the open-source Whisper model running locally on CTranslate2. A failure costs only the affected chunk, not the request. No model is trained here. All decoding, probing and silence detection go through `ffmpeg` and `ffprobe`.

| | |
|---|---|
| Runtime | Python 3.11+, FastAPI, Uvicorn, Pydantic v2 |
| Speech-to-text | OpenAI Whisper API (`whisper-1`), falling back to faster-whisper (`int8`, VAD on, beam 5) |
| Audio | ffmpeg / ffprobe via `subprocess` |
| Tests | pytest + httpx. Fixtures are generated with ffmpeg `lavfi`, so the repo holds no binary files |

---

## Quick start

Requires `ffmpeg` on `PATH` (`brew install ffmpeg` / `apt-get install ffmpeg`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # settings and credentials live here; never commit it
```

### API

```bash
uvicorn app.api:create_app --factory --env-file .env --port 8000

curl -F "file=@meeting.m4a" "localhost:8000/transcriptions?language=en"
curl -F "file=@meeting.m4a" "localhost:8000/transcriptions?format=srt"

# files longer than 120 s are accepted as a background job (202 + Location header)
curl -i -F "file=@podcast.mp3" localhost:8000/transcriptions
curl localhost:8000/transcriptions/<job_id>
```

Set `OPENAI_API_KEY` in `.env` to use the OpenAI API, with local Whisper as the fallback. Without a key, everything runs locally on faster-whisper. For an offline demo with no model download, start the server with `TP_ENGINE=mock`.

Interactive docs: <http://localhost:8000/docs>

### Docker

```bash
docker build -t transcription-pipeline .                          # bakes in the "base" model
docker build --build-arg MODEL_SIZE=small -t transcription-pipeline .
docker run --env-file .env -p 8000:8000 transcription-pipeline
```

The model is downloaded while the image builds, and the container runs with `HF_HUB_OFFLINE=1`, so it starts without network access. Without that setting, every cold start checks Hugging Face Hub, which added more than a minute in testing.

### Tests

```bash
pytest -q
```

The tests use `MockEngine`, so they never download a model. They take about 5 seconds.

---

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/transcriptions` | Multipart `file` upload. Query parameters: `language` (an ISO code such as `en`, or a name such as `english`; auto-detected if omitted) and `format` (`json` \| `srt` \| `vtt` \| `txt`, default `json`). Returns **200** with the transcript, or **202** with `{job_id, status, status_url}` and a `Location` header when the audio is longer than `TP_SYNC_MAX_DURATION_S`. |
| `GET` | `/transcriptions/{job_id}` | While the job is queued or running: `{job_id, status, progress (0-1), filename, format, error, created_at}`. Once complete: the transcript in the format requested at upload. **404** if the job ID is unknown. |
| `GET` | `/health` | `{status, engine, model, model_loaded}` |

| Status | When |
|---|---|
| 400 | Empty upload |
| 401 | `TP_API_KEY` is set and the `X-API-Key` header is missing or wrong (checked before the upload is read) |
| 413 | Upload larger than `TP_MAX_UPLOAD_MB`, or audio longer than `TP_MAX_DURATION_S` |
| 415 | Not decodable audio: a corrupt file, a non-media file with an audio extension, or a video with no audio track |
| 422 | Invalid query parameter, e.g. `format=docx` or an unknown `language` |
| 500 / 502 | Every configured engine failed (500: local engine, 502: upstream API). With the default fallback, an API failure alone never causes this |

A failed job returns `status: "failed"` with an `error` message.

### Example response (`format=json`)

```json
{
  "filename": "meeting.mp3",
  "language": "en",
  "duration": 7.699,
  "text": "Welcome to the weekly product meeting. First, let's review the release timeline for the mobile app.",
  "segments": [
    {
      "id": 1,
      "start": 0.0,
      "end": 4.2,
      "text": "Welcome to the weekly product meeting.",
      "confidence": 0.7203,
      "timeframe": "00:00:00.000 --> 00:00:04.200"
    },
    {
      "id": 2,
      "start": 4.2,
      "end": 7.5,
      "text": "First, let's review the release timeline for the mobile app.",
      "confidence": 0.7203,
      "timeframe": "00:00:04.200 --> 00:00:07.500"
    }
  ],
  "metadata": {
    "engine": "faster-whisper",
    "model": "tiny",
    "chunks": 1,
    "chunks_cut_in_silence": 0,
    "fallback_chunks": 0,
    "processing_seconds": 1.681,
    "source_format": {
      "codec": "mp3",
      "container": "mp3",
      "sample_rate": 22050,
      "channels": 1
    }
  }
}
```

This is real output from the local `tiny` model on a two-sentence macOS `say` recording. `processing_seconds` includes loading the model, which happens once per process. `start` and `end` are seconds from the start of the original file, including for long files split into chunks; `timeframe` is the same range written for people to read.

---

## Pipeline

```
 upload
        │
        ▼
 ┌──────────────┐  stream to disk in 1 MB pieces, size limit (413), empty check (400)
 │ save upload  │
 └──────┬───────┘
        ▼
 ┌──────────────┐  ffprobe: real audio stream? valid duration? (415)
 │    probe     │  duration ≤ TP_MAX_DURATION_S (413)
 └──────┬───────┘  record codec / container / rate / channels
        │
        ├── duration > TP_SYNC_MAX_DURATION_S ──► 202 + job_id ──► ThreadPoolExecutor
        ▼                                                                 │
 ┌──────────────┐  ffmpeg -vn -ac 1 -ar 16000 -c:a pcm_s16le              │
 │  normalise   │  ── every later stage sees ONLY this format ──          │
 └──────┬───────┘                                                         │
        ▼                                                                 │
 ┌──────────────┐  ffmpeg silencedetect (long files only)                 │
 │ plan chunks  │  ~10 min chunks; boundary → nearest silence ±30 s       │
 └──────┬───────┘  else hard cut + 2 s overlap each side                  │
        ▼                                                                 │
 ┌──────────────┐  extract chunk ─► engine.transcribe ─► delete chunk     │
 │  transcribe  │  TP_CHUNK_WORKERS in parallel, progress after each ─────┘ (job.progress)
 └──────┬───────┘  engine = OpenAI API ──fails──► faster-whisper (same chunk)
        ▼
 ┌──────────────┐  shift to absolute time; keep segment only in the chunk
 │    merge     │  whose OWNED range holds its midpoint; sort by start
 └──────┬───────┘
        ▼
 ┌──────────────┐  whitespace, drop empty, drop hallucinations,
 │ post-process │  confidence = clamp(exp(avg_logprob)), render json/srt/vtt/txt
 └──────────────┘
```

## Modules

| File | Responsibility |
|---|---|
| [`app/config.py`](app/config.py) | `Settings` dataclass read from `TP_*` environment variables |
| [`app/schemas.py`](app/schemas.py) | Pydantic models: `Segment`, `Metadata`, `TranscriptionResult`, `Job`, `JobStatus`, `OutputFormat` |
| [`app/audio.py`](app/audio.py) | `probe`, `normalise`, `detect_silences`, `plan_chunks`, `extract`, and the input error types |
| [`app/engines.py`](app/engines.py) | The `TranscriptionEngine` protocol, `OpenAIEngine`, `FasterWhisperEngine` (model loaded through `lru_cache`), `FallbackEngine` (with a circuit breaker), and `MockEngine` |
| [`app/languages.py`](app/languages.py) | Whisper language names mapped to ISO codes (the API returns `"english"`, faster-whisper returns `"en"`) |
| [`app/pipeline.py`](app/pipeline.py) | `transcribe_file`: orchestration, parallel chunks, thread-safe progress, timestamp merge |
| [`app/postprocess.py`](app/postprocess.py) | Text cleanup, hallucination filter, confidence, SRT / VTT / TXT rendering |
| [`app/api.py`](app/api.py) | `create_app(cfg, engine)` factory, `RequestGuard` (auth and size checked before the body is read), upload streaming, in-memory `JobStore`, HTTP error mapping |
| [`tests/`](tests/) | 77 tests covering audio, chunking, post-processing, engines and fallback, the pipeline, the API, auth and configuration |

---

## Design decisions

### Different audio formats

- **The file extension is never trusted.** Uploads are saved as `upload` with no extension, and `ffprobe` decides what the content is. A file is accepted only if it has a real audio stream and a positive duration. A text file renamed `.mp3`, a corrupt file, or a video without an audio track gets a **415** with a clear message.
- **One canonical format.** Every input is converted once to **16 kHz, mono, 16-bit PCM WAV** (`-vn -ac 1 -ar 16000 -c:a pcm_s16le`). 16 kHz mono is what Whisper consumes internally. Nothing after this step handles codecs. That includes chunk extraction, which slices the WAV by frame index using the stdlib `wave` module: it is sample-exact and needs no extra ffmpeg process per chunk. It also covers handing audio to Whisper, which receives a float32 array directly instead of decoding the file a second time.
- **The source format is recorded.** `metadata.source_format` keeps the original codec, container, sample rate and channel count, which helps debug quality problems ("why is this transcript bad?" → "8 kHz phone audio").
- **Tested formats:** WAV, MP3, M4A/AAC, OGG (Opus), FLAC, and an AAC track inside MP4 video. Anything ffmpeg can decode works.

### Long audio

- **Limits.** FastAPI reads the whole multipart body before any endpoint code runs, so a size check in the endpoint would only fire after a huge upload had already filled the disk. A small ASGI middleware (`RequestGuard`) therefore checks the API key and the declared `Content-Length` before reading anything, and counts bytes as they arrive, so the limit also holds for chunked uploads that send no `Content-Length`. The exact file size is checked again while the file is saved. Over-limit requests are rejected after 0 bytes have been read. The duration limit (`TP_MAX_DURATION_S`, default 4 h) is checked with ffprobe before any decoding work starts.
- **Synchronous vs background.** Audio up to `TP_SYNC_MAX_DURATION_S` (120 s) is answered in the request. Anything longer returns **202** with a job ID and is processed on a `ThreadPoolExecutor`. `GET /transcriptions/{id}` reports `progress` after each chunk. Jobs are kept in an in-memory `JobStore`. **In production this would be TaskIQ or Celery on Redis for the queue, object storage (S3/GCS) for uploaded audio, and Postgres for job state and results**, so jobs survive restarts and workers can scale separately from the API.
- **Why chunk when Whisper already uses 30 s windows internally?**
  - *Bounded memory.* A 4-hour file is about 460 MB of PCM. Each chunk holds roughly 19 MB.
  - *Progress.* faster-whisper yields segments lazily but cannot report a reliable fraction done. Counting chunks gives a real `progress` value.
  - *Parallelism.* `TP_CHUNK_WORKERS` runs chunks concurrently: on several CPU cores today, and the same code path would spread chunks across GPU workers.
  - *Per-chunk retry.* A failure costs one 10-minute chunk, not the whole file. The chunk is also the natural unit of work for a distributed queue.
- **Cut in silence, not mid-word.** Boundaries target `TP_CHUNK_SECONDS` (600 s) and move to the midpoint of the nearest silence found by `ffmpeg silencedetect` within ±`TP_SILENCE_SEARCH_S` (30 s). A boundary never moves back far enough to shrink a chunk below half its target size. If no silence is near, the chunk is hard-cut and **both** neighbouring chunks get `TP_CHUNK_OVERLAP_S` (2 s) of extra audio.
- **Owned range vs cut range.** Each chunk has an *owned* range (its share of the timeline; together the owned ranges cover the file with no gaps or overlaps) and a *cut* range (owned plus overlap, which is the audio actually transcribed).
- **Merge by midpoint.** Segment timestamps are shifted by `cut_start` to absolute time. A segment is kept **only** by the chunk whose owned range contains its midpoint, so speech in an overlap region is never duplicated and never dropped. The merged segments are then sorted by start time.
- **Bounded disk use.** Each chunk file is deleted as soon as it has been transcribed. The normalised WAV and the upload sit in a temporary directory that is removed on success, on failure, when a job finishes, and when a queued job is cancelled at shutdown.

### Output quality

- **Decoding settings.** `vad_filter=True` skips non-speech, which is faster and reduces hallucinations on silence. `beam_size=5` follows Whisper's own accuracy setting. `condition_on_previous_text=False` stops one bad window from feeding errors into the following windows (the well-known "repetition loop").
- **Hallucination filter.** A segment is dropped when `no_speech_prob > 0.6` **and** `avg_logprob < -1.0`, which matches Whisper's own thresholds. Both conditions are required: confident speech in a quiet passage is kept.
- **Cleanup.** Whitespace is collapsed and empty segments are dropped. Segment IDs are renumbered from 1 after filtering, so SRT cue numbers have no gaps.
- **Confidence.** `clamp(exp(avg_logprob), 0, 1)` is the geometric mean of the token probabilities: easy to read, and good for flagging segments that need human review.
- **Language.** Names and codes are both accepted (`english` or `en`) and checked up front, so an unknown value is a clear 422. If no language is given, chunks are transcribed one at a time until one contains speech. Its detected language is then pinned for every remaining chunk, which then run in parallel. Without this, each chunk would detect on its own, and a noisy chunk of an English file could come out in Japanese.
- **Stop early on failure.** If a chunk fails on every engine, chunks still in the queue are cancelled. A 4-hour file does not burn hours of CPU and paid API minutes on a result that will be discarded.

### Engines and fallback

- **OpenAI first, open source as the safety net.** The hosted `whisper-1` model needs no GPU and responds quickly, so it is the primary engine. Local faster-whisper is the fallback, so an outage, a rate limit or an expired key degrades quality or speed instead of failing the request.
- **Fallback runs per chunk, not per file.** `FallbackEngine` wraps any two engines. If the primary raises `TranscriptionError`, the same chunk is retried on the fallback with the same language. On a 2-hour file with one bad API call, only that one 10-minute chunk is re-transcribed locally.
- **Retry first, then fall back.** The OpenAI SDK already retries 429, 5xx and connection errors with exponential backoff (`max_retries=2`, 120 s timeout). The fallback only starts after those retries are used up.
- **Circuit breaker, for outages only.** After a connection error, timeout, 429, 5xx or authentication failure, the primary is skipped for `TP_FALLBACK_COOLDOWN_S` (60 s), so during a real outage chunks go straight to the fallback. A 400, 413 or 422 means only that request was rejected: the request still falls back, but the breaker stays closed, so one bad request cannot switch OpenAI off for every other user.
- **Fail at startup, not mid-outage.** Every local model in the chain is loaded when the service starts, under a lock, so parallel first requests cannot each load their own multi-GB copy. A broken fallback should stop the deploy, not appear for the first time during an OpenAI incident. A missing `OPENAI_API_KEY` logs a warning and the chain runs without OpenAI, whether OpenAI was the primary or the fallback, so the service still works offline and in CI. Invalid settings (for example `TP_CHUNK_SECONDS=0`) also fail at startup with a clear message, and blank values in `.env` mean "use the default".
- **Errors from both engines look the same.** OpenAI SDK errors and local CTranslate2 `RuntimeError`s (for example, out of memory) both become `TranscriptionError`. That makes the reverse chain work too (`TP_ENGINE=faster-whisper TP_FALLBACK_ENGINE=openai`). If both engines fail, the error message names both causes.
- **Traceable output.** `metadata.engine` shows the chain (`openai+faster-whisper`) and `metadata.fallback_chunks` counts how many chunks the fallback produced. A drop in quality can then be traced to its cause.
- **Same output from both engines.** Each chunk is sent to OpenAI as lossless FLAC, about half the size of WAV: a 10-minute chunk is roughly 10 MB, well under the API's 25 MB limit. The request uses `verbose_json`, so the response includes `avg_logprob` and `no_speech_prob`, and the confidence score and hallucination filter work exactly as they do for local output. Language names are mapped to ISO codes.
- **Privacy trade-off.** The default sends audio to OpenAI. For sensitive recordings, set `TP_ENGINE=faster-whisper TP_FALLBACK_ENGINE=none` and audio never leaves the machine.

### Extensibility

The pipeline only knows the `TranscriptionEngine` protocol (`name`, `model`, `transcribe(wav_path, language)`, `is_loaded()`). Adding Deepgram or AssemblyAI takes one new class plus one branch in `build_engine`, and `FallbackEngine` can chain it with any other engine. Engines always receive a clean 16 kHz mono WAV of at most one chunk. `create_app(cfg, engine)` takes both dependencies as arguments, which is how the tests run against `MockEngine` with small chunk settings.

### Configuration

All settings are environment variables prefixed with `TP_`. [`.env.example`](.env.example) lists them all; copy it to `.env` (git- and docker-ignored) and pass it with `--env-file`.

| Variable | Default | Purpose |
|---|---|---|
| `TP_API_KEY` | *(empty)* | When set, `/transcriptions` endpoints require header `X-API-Key: <value>` (401 otherwise). `/health` stays open for probes. Empty disables auth |
| `TP_ENGINE` | `openai` | Primary engine: `openai`, `faster-whisper` or `mock` |
| `OPENAI_API_KEY` | *(empty)* | Used by the OpenAI SDK. If unset, the service runs on the fallback engine only |
| `TP_OPENAI_MODEL` | `whisper-1` | Hosted model. It must support `verbose_json` segments |
| `TP_FALLBACK_ENGINE` | `faster-whisper` | Used when the primary fails. `none` disables the fallback |
| `TP_FALLBACK_COOLDOWN_S` | `60` | How long to skip a failing primary (circuit breaker) |
| `TP_MODEL_SIZE` | `base` | `tiny`, `base`, `small`, `medium`, `large-v3`. In Docker, choose it at build time (`--build-arg MODEL_SIZE=...`): the image runs offline with only that model |
| `TP_DEVICE` | `cpu` | `cpu`, `cuda`, `auto` |
| `TP_COMPUTE_TYPE` | `int8` | CTranslate2 compute type (`float16` on GPU) |
| `TP_MAX_UPLOAD_MB` | `500` | Upload size limit (413) |
| `TP_MAX_DURATION_S` | `14400` | Audio duration limit (413) |
| `TP_SYNC_MAX_DURATION_S` | `120` | Longer audio becomes a background job (202) |
| `TP_JOB_WORKERS` | `2` | Concurrent background jobs |
| `TP_CHUNK_SECONDS` | `600` | Target chunk length |
| `TP_SILENCE_SEARCH_S` | `30` | Search window around each boundary |
| `TP_CHUNK_OVERLAP_S` | `2` | Overlap added on both sides of a hard cut |
| `TP_SILENCE_NOISE_DB` | `-35` | `silencedetect` noise floor |
| `TP_SILENCE_MIN_S` | `0.5` | Minimum silence length |
| `TP_CHUNK_WORKERS` | `1` | Chunks transcribed in parallel per file |

### Alternatives considered

| Option | Decision | Why |
|---|---|---|
| `openai-whisper` (reference implementation) | **faster-whisper** | Same model weights, up to 4x faster with less memory (the project's published benchmarks), and supports `int8` on CPU. It includes a Silero VAD filter, so a separate VAD step isn't needed. |
| Hosted API only, or local model only | **OpenAI API with a per-chunk local fallback** | The API is fast and needs no GPU, but it is an external dependency with outages and rate limits. Local Whisper costs nothing per minute and is always available. Combining them gives the API's speed with local availability. Sensitive deployments can switch to local only with one setting. |
| PyAV / pydub for decoding | **ffmpeg / ffprobe subprocess** | ffmpeg is the most complete decoder available, its CLI interface is stable, and probing, normalising and `silencedetect` all use one tool. Running it as a subprocess also keeps a crash on a malformed file out of the API process. |
| Trust the extension or MIME type | **Probe the content** | Both are client-controlled. ffprobe reading the actual streams is the only reliable check. |
| Fixed-size chunks | **Silence-snapped chunks, with overlap as fallback** | Fixed cuts split words, and Whisper then garbles both halves. Snapping costs one fast `silencedetect` pass. The overlap plus midpoint ownership keeps hard cuts correct too. |
| De-duplicate overlaps by text similarity | **Midpoint ownership** | Deterministic and O(n). Text matching is fuzzy and can drop real repeated speech ("no, no"). |
| Always asynchronous | **Synchronous up to 120 s, background job beyond** | Most short clips finish in a few seconds, and a single request is simpler for clients. Long files must not hold an HTTP connection open for minutes. |
| Streaming segments over HTTP (NDJSON or SSE) | **Polling jobs** | Polling survives client disconnects and proxies, and maps directly onto a real queue later. Streaming could be added for live captions. |
| Processes for chunk parallelism | **Threads** | CTranslate2 releases the GIL during inference and ffmpeg runs in its own process, so threads get real parallelism while sharing one loaded model in memory. |

### What I'd do next for production

1. **Queue and storage.** Replace the in-memory job store with TaskIQ/Celery on Redis, keep uploads and results in object storage (S3/GCS) with presigned upload URLs so large files bypass the API, and store job state in Postgres. Add retries per chunk and idempotency keys.
2. **GPU workers with `large-v3`** (`float16`, batched inference), autoscaled on queue depth. Keep the CPU `base` model as a cheap tier.
3. **Speaker diarization** (pyannote) merged onto segments, giving `speaker` labels.
4. **Word-level timestamps** (`word_timestamps=True`) for subtitle editing and search.
5. **Quality tracking.** WER on a held-out, domain-specific labelled set in CI, run on every model or setting change, plus confidence distributions in production dashboards.
6. **Authentication and rate limits.** Per-tenant API keys or OAuth (today: one shared `TP_API_KEY`), per-tenant quotas on minutes of audio, request size limits at the gateway.
7. **Retention policy.** Delete audio after N days, encrypt at rest, offer PII redaction, and log access to transcripts.
8. **Operations.** Structured logs, Prometheus metrics (real-time factor, queue latency, failure rate), tracing, and graceful shutdown that hands unfinished jobs back to the queue.
