# Парсер вакансий hh.ru через Chrome

Скрипт собирает вакансии по URL поисковой выдачи в CSV и JSON. Пароль, email и cookies в коде не сохраняются. Авторизация остаётся в профиле Chrome.

## Установка

Из корня проекта в PowerShell:

```powershell
python -m venv .\hh_parser\.venv
.\hh_parser\.venv\Scripts\Activate.ps1
pip install -r .\hh_parser\requirements.txt
```

Используется установленный Google Chrome, поэтому отдельная загрузка Chromium не требуется.

## Обычный запуск

```powershell
python .\hh_parser\hh_parser.py --pages 3
```

При первом запуске откроется отдельный профиль Chrome. Войдите в hh.ru по своему email вручную. Скрипт дождётся входа, соберёт выдачу и сохранит сессию для следующих запусков в `hh_parser/.chrome-profile/`.

После первого входа можно запускать без окна:

```powershell
python .\hh_parser\hh_parser.py --pages 5 --headless
```

Результаты появятся в `hh_parser/output/`.

## Подключение к уже запущенному Chrome

Chrome должен быть запущен с DevTools-портом. Готовый запускатель сам найдёт Chrome, откроет выдачу и переиспользует сохранённую сессию:

```powershell
.\hh_parser\start_chrome_debug.ps1
```

В открывшемся Chrome войдите в hh.ru, затем в другом PowerShell запустите:

```powershell
python .\hh_parser\hh_parser.py --cdp-url http://127.0.0.1:9222 --pages 3
```

Скрипт не закрывает Chrome при работе через `--cdp-url`.

## Другой запрос

```powershell
python .\hh_parser\hh_parser.py --url "https://spb.hh.ru/search/vacancy?text=Python&area=2" --pages 5
```

Не запускайте сбор слишком часто: оставляйте паузу между страницами и учитывайте правила hh.ru. Если появилась CAPTCHA, решите её вручную в окне Chrome и повторите запуск.
