FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 WEB_CONCURRENCY=4
WORKDIR /srv

# WITH_SEMANTIC=1 installs the MiniLM router (fastembed/ONNX) and bakes the model into the image so start-up
# never needs the internet:   docker build --build-arg WITH_SEMANTIC=1 .
ARG WITH_SEMANTIC=0
ENV SEMANTIC_CACHE_DIR=/opt/models

# the glob lets the build succeed when requirements-semantic.txt is absent (it is only needed for WITH_SEMANTIC=1)
COPY requirements.txt requirements-semantic.tx[t] ./
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$WITH_SEMANTIC" = "1" ]; then pip install --no-cache-dir -r requirements-semantic.txt; fi

COPY app ./app
COPY data ./data
COPY static ./static
COPY scripts ./scripts
RUN if [ "$WITH_SEMANTIC" = "1" ]; then python -m scripts.download_semantic_model; fi
# Optional: `cd frontend && npm ci && npm run build` first to serve the React UI from here.
COPY frontend/dist ./frontend/dist

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

# WEB_CONCURRENCY sets the worker count. Access logs off: they cost measurable time per request.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--http", "httptools", "--no-access-log", "--proxy-headers", "--forwarded-allow-ips", "*"]
