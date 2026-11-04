FROM python:3.15-slim-bookworm@sha256:4e03260adeeebafccd084f3584e6b84823aba5651cfb53ef8a078ed441ddf9cd AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FI_DATA_DIR=/data/runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-docker.lock /app/
RUN pip install --requirement requirements-docker.lock

COPY pyproject.toml README.md LICENSE /app/
COPY configs /app/configs
COPY src /app/src
RUN pip install --no-deps .

RUN useradd --create-home --uid 10001 football \
    && mkdir -p /data/runtime \
    && chown -R football:football /data /app
USER football

EXPOSE 8080
HEALTHCHECK --interval=20s --timeout=5s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/v1/health', timeout=3)"
CMD ["football-intelligence", "serve", "--host", "0.0.0.0", "--port", "8080", "--data-dir", "/data/runtime"]
