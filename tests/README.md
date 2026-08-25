# Тесты подсчёта площади

Тесты запускают `color_mask_hatch.py` как отдельный CLI-процесс, считывают строку
`Total area: ... m^2` и сравнивают полученную суммарную площадь с ожидаемой.
Для baseline используется допуск ±10%, для масштабированных изображений — ±12%.

## Требования

- Запускайте команды из корня проекта `PipeStone`.
- Установите зависимости проекта.
- Tesseract должен быть доступен в `PATH`.
- В Tesseract должны быть установлены языки `rus` и `eng`.

Проверить языки Tesseract:

```powershell
tesseract --list-langs
```

Тестовые изображения и образец штриховки находятся в `tests/tests_data`.

## Запуск всех тестов

```powershell
python -m unittest discover -s tests -v
```

Будут выполнены три теста:

| Тест                 | Изображение  | Ожидаемая площадь | Допустимый диапазон |
| ------------------------ | ----------------------- | --------------------------------: | ------------------------------------: |
| `test_baseline_area`   | `test.jpg`            |                          144 м² |                     129,6–158,4 м² |
| `test_tiled_up_area`   | `test_tiled_up.jpg`   |                           72 м² |                     63,36–80,64 м² |
| `test_tiled_down_area` | `test_tiled_down.jpg` |                           72 м² |                     63,36–80,64 м² |

## Запуск отдельного теста

Исходное изображение:

```powershell
python -m unittest tests.test_color_mask_hatch_area.ColorMaskHatchAreaTest.test_baseline_area -v
```

Изображение `test_tiled_up.jpg`:

```powershell
python -m unittest tests.test_color_mask_hatch_area.ColorMaskHatchAreaTest.test_tiled_up_area -v
```

Изображение `test_tiled_down.jpg`:

```powershell
python -m unittest tests.test_color_mask_hatch_area.ColorMaskHatchAreaTest.test_tiled_down_area -v
```

## Особенности

- Каждый тест выполняет полный поиск штриховки, OCR и подсчёт площади, поэтому прогон может занять несколько минут.
- Выходные изображения и JSON-файлы создаются во временном каталоге и удаляются после завершения теста.
- При ошибке тест выводит stdout и stderr запущенного CLI-процесса.
- Таймаут обработки одного изображения — 15 минут.
