FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/home/trustdesk/.cache/huggingface

RUN useradd --create-home --uid 1000 trustdesk
WORKDIR /app

# CPU-only PyTorch. The default build bundles CUDA and adds several gigabytes.
# Everything else comes from PyPI at the versions frozen in constraints.txt.
COPY requirements.txt constraints.txt ./
RUN pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
 && pip install -r requirements.txt -c constraints.txt \
 && pip check

COPY --chown=trustdesk:trustdesk . .
RUN chown trustdesk:trustdesk /app
USER trustdesk

# Download the embedding model and seed the database at build time, so the
# container starts offline with fresh demo data every time it's recreated.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
 && python -m scripts.seed --reset
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
