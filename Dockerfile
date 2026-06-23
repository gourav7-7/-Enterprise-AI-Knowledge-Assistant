# Slim Python base keeps the image small.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first (cached layer) — only re-runs when requirements.txt changes.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy only what runs (data/, .env, tests excluded via .dockerignore).
COPY app ./app
COPY frontend ./frontend

# 8000 = FastAPI, 8501 = Streamlit. compose picks the command per service.
EXPOSE 8000 8501

# Default command = the API. --host 0.0.0.0 is REQUIRED in a container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]