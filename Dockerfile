FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5001 \
    GUNICORN_TIMEOUT=180

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0" \
    && pip install --no-cache-dir -r requirements.txt

COPY tictactoe/ ./tictactoe/

WORKDIR /app/tictactoe

EXPOSE 5001

CMD ["sh", "-c", "gunicorn --workers 1 --timeout ${GUNICORN_TIMEOUT} --bind 0.0.0.0:${PORT} play_human_vs_bot_flask:app"]
