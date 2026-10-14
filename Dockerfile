
# Stage 1: Build the React frontend
FROM node:20-alpine AS frontend-build

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# Stage 2: Build the Python backend
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WEB_CONCURRENCY=4

WORKDIR /srv

ARG WITH_SEMANTIC=0
ENV SEMANTIC_CACHE_DIR=/opt/models

COPY requirements.txt requirements-semantic.tx[t] ./

RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$WITH_SEMANTIC" = "1" ]; then \
        pip install --no-cache-dir -r requirements-semantic.txt; \
    fi

COPY app ./app
COPY data ./data
COPY static ./static
COPY scripts ./scripts

# Copy the frontend built in Stage 1
COPY --from=frontend-build /frontend/dist ./frontend/dist

RUN if [ "$WITH_SEMANTIC" = "1" ]; then \
    python -m scripts.download_semantic_model; \
    fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--http", "httptools", "--no-access-log", "--proxy-headers", "--forwarded-allow-ips", "*"]
