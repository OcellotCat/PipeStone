@echo off
REM run.bat — запуск пайплайна (Windows)
setlocal

set VENV_DIR=.venv
set REQUIREMENTS=pdf2image requests pyyaml reportlab

REM Создаём venv если нет
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [setup] Создание виртуального окружения...
    python -m venv %VENV_DIR%
)

REM Активируем
call %VENV_DIR%\Scripts\activate.bat

REM Устанавливаем зависимости
pip install --quiet %REQUIREMENTS%

REM Запуск (можно передать аргументы: run.bat --pdf other.pdf)
python pipeline.py %*
