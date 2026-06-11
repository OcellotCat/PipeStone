@echo off
REM run.bat — запуск пайплайна (Windows)
setlocal

set VENV_DIR=.venv
set REQUIREMENTS=fastapi uvicorn python-multipart pymupdf numpy opencv-python-headless

REM Создаём venv если нет
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [setup] Создание виртуального окружения...
    python -m venv %VENV_DIR%
)

REM Активируем
call %VENV_DIR%\Scripts\activate.bat

REM Устанавливаем зависимости
pip install --quiet %REQUIREMENTS%

REM Запуск API по умолчанию.
REM Примеры:
REM   run.bat
REM   run.bat serve --host 127.0.0.1 --port 8000
REM   run.bat analyze --pdf input/drawing.pdf --fallback-mm-per-px 2.5
python pipeline.py %*
