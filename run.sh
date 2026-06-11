#!/usr/bin/env bash
# run.sh — запуск пайплайна (Linux / macOS)
set -e

VENV_DIR=".venv"
REQUIREMENTS="fastapi uvicorn python-multipart pymupdf numpy opencv-python-headless"

# Создаём venv если нет
if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] Создание виртуального окружения..."
  python3 -m venv "$VENV_DIR"
fi

# Активируем
source "$VENV_DIR/bin/activate"

# Устанавливаем зависимости если нужно
pip install --quiet $REQUIREMENTS

# Запуск API по умолчанию.
# Примеры:
#   ./run.sh
#   ./run.sh serve --host 127.0.0.1 --port 8000
#   ./run.sh analyze --pdf input/drawing.pdf --fallback-mm-per-px 2.5
python pipeline.py "$@"
