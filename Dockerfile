FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first (cached layer): the lockfile pins what was tested locally.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/chroma /app/data/uploads \
    && chown -R appuser:appuser /app

# Copy only what runs (data/, .env, tests excluded via .dockerignore).
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser frontend ./frontend

USER appuser

# 8000 = FastAPI, 8501 = Streamlit. compose picks the command per service.
EXPOSE 8000 8501

# --host 0.0.0.0 is REQUIRED in a container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]