# ---------- build stage ----------
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- runtime stage ----------
FROM python:3.11-slim

LABEL maintainer="StellaSentinal Team"

# Install the pre-built packages from the builder stage
COPY --from=builder /install /usr/local

# Create a non-root user
RUN groupadd --gid 1000 stella \
    && useradd --uid 1000 --gid stella --shell /bin/bash --create-home stella

WORKDIR /app

# Copy project files (respects .dockerignore)
COPY --chown=stella:stella . .

# Ensure the workspace env var points to the container working directory
ENV STELLA_WORKSPACE=/app \
    PYTHONUNBUFFERED=1

EXPOSE 8000

USER stella

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
