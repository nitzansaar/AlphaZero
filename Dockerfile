FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5001 \
    GUNICORN_TIMEOUT=180 \
    GUNICORN_MAX_REQUESTS=200 \
    GUNICORN_MAX_REQUESTS_JITTER=50 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0" \
    && pip install --no-cache-dir -r requirements.txt

COPY tictactoe/ ./tictactoe/

WORKDIR /app/tictactoe

EXPOSE 5001

CMD ["sh", "-c", "gunicorn --workers 1 --timeout ${GUNICORN_TIMEOUT} --max-requests ${GUNICORN_MAX_REQUESTS} --max-requests-jitter ${GUNICORN_MAX_REQUESTS_JITTER} --bind 0.0.0.0:${PORT} play_human_vs_bot_flask:app"]
