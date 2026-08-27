# Регрессионные тесты подсчёта площади

В проекте есть два набора end-to-end тестов:

- `test_pdf_area_regression.py` запускает полный pipeline для PDF: поиск легенды,
  OCR и подсчёт площади;
- `test_color_mask_hatch_area.py` запускает `color_mask_hatch.py` как отдельный
  CLI-процесс для тестовых JPG.

Для PDF используется допуск ±10%. Расчёт PDF выполняется при 400 DPI и с четырьмя
OCR-worker, чтобы результаты соответствовали эталонным значениям.

## Требования

- Запускайте команды из корня проекта `PipeStone`.
- Установите зависимости проекта.
- Tesseract должен быть доступен в `PATH`.
- В Tesseract должны быть установлены языки `rus` и `eng`.

Проверить языки Tesseract:

```powershell
tesseract --list-langs
```

PDF для регрессионных тестов должны находиться в каталоге `tests/pdfs`:

```text
tests/pdfs/
├── test_blue.pdf
├── merged_test.pdf
└── first_full_test.pdf
```

Тестовые JPG и образец штриховки находятся в `tests/tests_data`.

## Запуск PDF-регрессий

Запуск всех трёх PDF-тестов:

```powershell
.\.env\Scripts\python.exe -m unittest tests.test_pdf_area_regression -v
```

| Тест                                          | PDF                                | Ожидаемая площадь | Допустимый диапазон |
| ------------------------------------------------- | ---------------------------------- | --------------------------------: | ------------------------------------: |
| `test_test_blue_pdf_area_is_about_10_5_m2`      | `tests/pdfs/test_blue.pdf`       |                         10,5 м² |                      9,45–11,55 м² |
| `test_merged_test_pdf_area_is_about_288_m2`     | `tests/pdfs/merged_test.pdf`     |                          288 м² |                     259,2–316,8 м² |
| `test_first_full_test_pdf_area_is_about_144_m2` | `tests/pdfs/first_full_test.pdf` |                          144 м² |                     129,6–158,4 м² |

Запуск отдельных PDF-тестов:

```powershell
.\.env\Scripts\python.exe -m unittest tests.test_pdf_area_regression.PdfAreaRegressionTests.test_test_blue_pdf_area_is_about_10_5_m2 -v
```

```powershell
.\.env\Scripts\python.exe -m unittest tests.test_pdf_area_regression.PdfAreaRegressionTests.test_merged_test_pdf_area_is_about_288_m2 -v
```

```powershell
.\.env\Scripts\python.exe -m unittest tests.test_pdf_area_regression.PdfAreaRegressionTests.test_first_full_test_pdf_area_is_about_144_m2 -v
```

## Запуск всех тестов проекта

```powershell
.\.env\Scripts\python.exe -m unittest discover -s tests -v
```

Эта команда также запускает длительные PDF-регрессии.

## JPG-регрессии

| Тест                 | Изображение  | Ожидаемая площадь | Допустимый диапазон |
| ------------------------ | ----------------------- | --------------------------------: | ------------------------------------: |
| `test_baseline_area`   | `test.jpg`            |                          144 м² |                     129,6–158,4 м² |
| `test_tiled_up_area`   | `test_tiled_up.jpg`   |                           72 м² |                     63,36–80,64 м² |
| `test_tiled_down_area` | `test_tiled_down.jpg` |                           72 м² |                     63,36–80,64 м² |

Запуск всех JPG-тестов:

```powershell
.\.env\Scripts\python.exe -m unittest tests.test_color_mask_hatch_area -v
```

### Запуск отдельного JPG-теста

Исходное изображение:

```powershell
.\.env\Scripts\python.exe -m unittest tests.test_color_mask_hatch_area.ColorMaskHatchAreaTest.test_baseline_area -v
```

Изображение `test_tiled_up.jpg`:

```powershell
.\.env\Scripts\python.exe -m unittest tests.test_color_mask_hatch_area.ColorMaskHatchAreaTest.test_tiled_up_area -v
```

Изображение `test_tiled_down.jpg`:

```powershell
.\.env\Scripts\python.exe -m unittest tests.test_color_mask_hatch_area.ColorMaskHatchAreaTest.test_tiled_down_area -v
```

## Особенности

- Каждый PDF-тест выполняет полный поиск легенды, OCR и подсчёт площади. Полный
  PDF-набор может выполняться 10–15 минут в зависимости от CPU.
- Выходные изображения и JSON-файлы создаются во временном каталоге и удаляются после завершения теста.
- PDF-тест завершится понятной ошибкой `Reference PDF is missing`, если нужного
  файла нет в `tests/pdfs`.
- При ошибке JPG-тест выводит stdout и stderr запущенного CLI-процесса.
- Таймаут обработки одного JPG — 15 минут.
