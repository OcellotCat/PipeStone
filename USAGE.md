# Запуск PipeStone

Команды выполняются из корня проекта.

## Найти страницы со штриховкой

```powershell
.\.env\Scripts\python.exe pipeline.py --pdf test.pdf
```

Результат — массив номеров страниц, например `[3, 5, 8]`.

## Получить полный JSON

```powershell
.\.env\Scripts\python.exe pipeline.py --pdf test.pdf --json
```

Основные поля:

- `legend_pages` — страницы с легендой;
- `hatch_pages` — страницы с найденной штриховкой;
- `pattern_images` — сохранённые изображения образцов штриховки.

## Найти штриховку и посчитать площадь

```powershell
.\.env\Scripts\python.exe pipeline.py --pdf test.pdf --calculate-area --json
```

Результат находится в `area_calculation`:

- `pages` — площадь отдельно для каждой страницы;
- `total_area_m2` — общая площадь в квадратных метрах;
- `pattern_image` — использованный trimmed patch;
- `elements_image` — изображение найденных элементов.

Для подсчёта в `color_mask_hatch.process_images()` передаются RGB-массивы
страниц из `hatch_pattern_box_pages` и сохранённый `pattern_image`.

## Указать папку результатов

```powershell
.\.env\Scripts\python.exe pipeline.py --pdf test.pdf --calculate-area --output-dir output --json
```

Файлы сохраняются в `output/runs/<идентификатор_запуска>/`.
Лог выполнения сохраняется рядом в файле `pipeline.log`; путь к нему также
возвращается в JSON-поле `log_file`. Исходный PDF копируется в эту же папку
без изменения имени; его путь возвращается в JSON-поле `source_pdf`.

## Обработать отдельное изображение

```powershell
.\.env\Scripts\python.exe pipeline.py --image drawing.png --json
```

## Вызвать color_mask_hatch из Python

```python
from pathlib import Path

from color_mask_hatch import process_images, read_rgb

images = [read_rgb(Path("page_001.png")), read_rgb(Path("page_002.png"))]
patch = read_rgb(Path("output/runs/<run_id>/pattern_results/page_001_legend_pattern_trimmed.png"))

results = process_images(images, patch, calculate_area=True)
print([result["total_area_m2"] for result in results])
```

Для системного Python вместо `.\.env\Scripts\python.exe` можно использовать `python`.

## Веб-интерфейс

```powershell
.\.env\Scripts\python.exe -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

После запуска откройте `http://127.0.0.1:8000`. Страница проверяет PDF,
после загрузки автоматически выполняет только поиск легенды и показывает
найденные записи в прокручиваемом списке. Затем можно запустить этапы расчёта
и получить группы материалов и общую площадь. Одновременно выполняется только один расчёт; состояние сервера
«Свободен / Идёт расчёт» обновляется автоматически каждые 30 секунд.
Состояние текущей задачи привязано к `HttpOnly` cookie-сессии и автоматически
восстанавливается после перезагрузки страницы. Сессия хранится в памяти
работающего сервера и не переживает его перезапуск.
Подробный прогресс доступен только владельцу задачи. Остальные посетители видят
общий статус сервера без идентификатора и результатов чужой задачи. Владелец
может оставить операцию выполняться в фоне либо остановить поиск легенды или
расчёт соответствующей кнопкой.
