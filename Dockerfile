# syntax=docker/dockerfile:1.7
#
# API image. Multi-stage: uv installs dependencies straight from uv.lock in the
# builder; the runtime stage carries only the virtualenv and the source.
#
#   docker build -t graphrag-lab .

FROM python:3.12-slim-bookworm AS builder

RUN pip install --no-cache-dir uv==0.11.32

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies first (lockfile + manifest only) so code edits don't bust this layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm AS runtime

RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LLM_CACHE_DIR=/app/.cache/llm

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
RUN mkdir -p /app/.cache/llm /home/app/.cache/huggingface \
    && chown -R app:app /app/.cache /home/app/.cache

USER app

EXPOSE 8000

CMD ["uvicorn", "graphrag_lab.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
