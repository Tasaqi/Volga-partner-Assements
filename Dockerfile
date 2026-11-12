FROM python:3.12-slim

# ffmpeg/ffprobe do all decoding, probing and silence detection.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model into the image so containers start without network access.
ARG MODEL_SIZE=base
# Fixed cache path: the model is downloaded as root but read by the non-root user.
ENV TP_MODEL_SIZE=${MODEL_SIZE} HF_HOME=/opt/hf-cache
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${MODEL_SIZE}', device='cpu', compute_type='int8')" \
    && chmod -R a+rX "$HF_HOME"
# Model is local now; skip slow Hugging Face Hub lookups at runtime.
ENV HF_HUB_OFFLINE=1

COPY app ./app
COPY cli.py .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
