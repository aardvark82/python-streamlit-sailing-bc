FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HELPER_DATA_DIR=/data \
    PORT=5111

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY VERSION ./VERSION

RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 5111

# Single worker (APScheduler is in-process — multi-worker would duplicate
# the cron). --threads 8 lets us serve concurrent requests so one slow
# endpoint can't freeze the whole UI.
CMD ["gunicorn", "-w", "1", "--threads", "8", "-b", "0.0.0.0:5111", "--timeout", "60", "backend.app:app"]
