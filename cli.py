"""Command-line interface.

    python cli.py input.mp3 --format srt -o out.srt --model base --language en
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.audio import AudioError
from app.config import Settings
from app.engines import TranscriptionError, build_engine
from app.pipeline import transcribe_file
from app.postprocess import render
from app.schemas import OutputFormat


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe an audio (or video) file.")
    parser.add_argument("input", type=Path, help="audio/video file in any ffmpeg-readable format")
    parser.add_argument("--format", choices=[f.value for f in OutputFormat], default="json")
    parser.add_argument("-o", "--output", type=Path, help="write here instead of stdout")
    parser.add_argument("--model", help="Whisper model size (default: TP_MODEL_SIZE or base)")
    parser.add_argument("--language", help='ISO code, e.g. "en"; auto-detect if omitted')
    parser.add_argument("--engine", choices=["faster-whisper", "openai", "mock"], help="default: TP_ENGINE")
    args = parser.parse_args(argv)

    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")

    overrides = {k: v for k, v in {"model_size": args.model, "engine": args.engine}.items() if v}
    cfg = Settings.from_env(**overrides)

    def show_progress(p: float) -> None:
        print(f"\rprogress: {p:6.1%}", end="", file=sys.stderr, flush=True)

    try:
        result = transcribe_file(
            args.input, cfg, build_engine(cfg), language=args.language, progress=show_progress
        )
    except AudioError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2
    except TranscriptionError as exc:
        print(f"\ntranscription failed: {exc}", file=sys.stderr)
        return 1
    print(file=sys.stderr)

    body = render(result, OutputFormat(args.format))
    if args.output:
        args.output.write_text(body, encoding="utf-8")
        print(f"wrote {args.output} ({len(result.segments)} segments)", file=sys.stderr)
    else:
        print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
