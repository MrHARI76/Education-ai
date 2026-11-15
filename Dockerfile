# NexusEd AI app image: Python + FastAPI + local ML models (runs offline).
FROM python:3.11-slim

# Don't write .pyc files; print logs immediately; keep Hugging Face models inside the image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5

WORKDIR /app

# 1) CPU-only PyTorch first (the default build includes ~2 GB of GPU libraries we don't need).
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

# 2) The rest of the dependencies. Copying requirements.txt alone first lets Docker
#    reuse this layer when only the source code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) Download both models during the build so the container needs no internet at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# From now on, never try to reach the Hugging Face website.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 4) App code.
COPY app ./app
COPY frontend ./frontend

# 5) Run as a normal (non-root) user for safety.
RUN useradd --create-home appuser && mkdir -p /app/data && chown -R appuser /app /opt/hf-cache
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
