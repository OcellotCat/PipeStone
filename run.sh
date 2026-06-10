#!/usr/bin/env bash
# run.sh — запуск пайплайна (Linux / macOS)
set -e

VENV_DIR=".venv"
REQUIREMENTS="pdf2image requests pyyaml reportlab"

# Создаём venv если нет
if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] Создание виртуального окружения..."
  python3 -m venv "$VENV_DIR"
fi

# Активируем
source "$VENV_DIR/bin/activate"

# Устанавливаем зависимости если нужно
pip install --quiet $REQUIREMENTS

# Запуск (можно передать аргументы: ./run.sh --pdf other.pdf)
python pipeline.py "$@"
