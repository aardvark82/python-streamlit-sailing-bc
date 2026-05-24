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

# Single gunicorn worker — APScheduler is in-process, multi-worker would
# duplicate the cron and write multiple times per slot.
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5111", "--timeout", "60", "backend.app:app"]
