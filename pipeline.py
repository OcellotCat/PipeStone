"""
pipeline.py — КМ1 Stone Quotation Pipeline
Использование: python pipeline.py [--config config.yaml] [--pdf path/to/drawing.pdf]
"""
import argparse, base64, datetime, io, os, sys
import requests, yaml
from pathlib import Path
from pdf2image import convert_from_path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
# ─── Конфиг ──────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
# ─── Курс ЦБ РФ ──────────────────────────────────────────────────────────────
def get_cbr_rate() -> float:
    try:
        today = datetime.date.today().strftime("%d/%m/%Y")
        url = f"https://www.cbr.ru/scripts/XML_daily.asp?date_req={today}"
        r = requests.get(url, timeout=10)
        from xml.etree import ElementTree as ET
        root = ET.fromstring(r.content)
        for v in root.findall("Valute"):
            if v.find("CharCode").text == "USD":
                rate = float(v.find("Value").text.replace(",", "."))
                nominal = int(v.find("Nominal").text)
                return rate / nominal
    except Exception as e:
        print(f"[warn] Не удалось получить курс ЦБ: {e}. Используется 80.0")
    return 80.0
# ─── PDF → PNG ────────────────────────────────────────────────────────────────
def pdf_to_images(pdf_path: str, temp_dir: str, dpi: int) -> list:
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=dpi)
    paths = []
    for i, img in enumerate(images):
        p = os.path.join(temp_dir, f"page_{i+1:03d}.png")
        img.save(p, "PNG")
        paths.append(p)
    print(f"[ok] PDF → {len(paths)} страниц в {temp_dir}")
    return paths
# ─── Вызов модели ─────────────────────────────────────────────────────────────
def call_model(image_path: str, endpoint: str, timeout: int) -> dict:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {"image_base64": b64, "filename": os.path.basename(image_path)}
    r = requests.post(endpoint, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()
def collect_marks(image_paths: list, cfg: dict) -> list:
    endpoint = cfg["model"]["endpoint"]
    timeout = cfg["model"]["timeout"]
    aggregated = {}
    for i, img_path in enumerate(image_paths):
        print(f"  страница {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        try:
            result = call_model(img_path, endpoint, timeout)
            for item in result.get("marks", []):
                key = item["mark"]
                if key not in aggregated:
                    aggregated[key] = item.copy()
                else:
                    aggregated[key]["count"] += item["count"]
        except Exception as e:
            print(f"  [warn] Ошибка на странице {i+1}: {e}")
    return list(aggregated.values())
# ─── Расчёт площадей ──────────────────────────────────────────────────────────
def calc_areas(marks: list) -> list:
    rows = []
    for m in marks:
        area = round(m["width_mm"] / 1000 * m["height_mm"] / 1000 * m["count"], 4)
        rows.append({**m, "area_m2": area})
    return rows
# ─── Генерация КП PDF ─────────────────────────────────────────────────────────
def generate_kp(rows: list, cfg: dict, rate: float, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mat = cfg["materials"][0]
    qcfg = cfg["quotation"]
    date_str = datetime.date.today().strftime("%d.%m.%Y")
    fname = os.path.join(output_dir, f"KP_KM1_{date_str.replace('.','_')}.pdf")
    doc = SimpleDocTemplate(fname, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=14, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Normal"], fontSize=10,
                         textColor=colors.HexColor("#555555"), spaceAfter=10)
    normal = styles["Normal"]
    story = []
    story.append(Paragraph(qcfg["title"], h1))
    story.append(Paragraph(f"Объект: {qcfg['object']}", h2))
    story.append(Paragraph(
        f"Дата: {date_str} &nbsp;&nbsp; Курс ЦБ: {rate:.2f} руб/USD &nbsp;&nbsp; "
        f"Поставщик: {mat['supplier']}", h2))
    story.append(Spacer(1, 6*mm))
    total_area = round(sum(r["area_m2"] for r in rows), 4)
    total_usd = round(total_area * mat["price_usd_per_m2"], 2)
    total_rub = round(total_usd * rate, 2)
    # Таблица позиций
    header = ["Марка", "Ш×В (мм)", "Кол-во", "Площадь м²", "Цена USD/м²",
              "Сумма USD", "Сумма руб"]
    data = [header]
    for r in rows:
        area = r["area_m2"]
        usd = round(area * mat["price_usd_per_m2"], 2)
        rub = round(usd * rate, 2)
        data.append([
            r["mark"],
            f"{r['width_mm']}×{r['height_mm']}",
            str(r["count"]),
            f"{area:.4f}",
            f"{mat['price_usd_per_m2']:.2f}",
            f"{usd:,.2f}",
            f"{rub:,.0f}",
        ])
    data.append(["ИТОГО", "", "", f"{total_area:.4f}", "",
                 f"{total_usd:,.2f}", f"{total_rub:,.0f}"])
    col_w = [28*mm, 28*mm, 18*mm, 24*mm, 24*mm, 26*mm, 28*mm]
    t = Table(data, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#2C2C2A")),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTSIZE",    (0,0), (-1,0),  8),
        ("FONTSIZE",    (0,1), (-1,-1), 8),
        ("BACKGROUND",  (0,-1),(-1,-1), colors.HexColor("#F1EFE8")),
        ("FONTNAME",    (0,-1),(-1,-1), "Helvetica-Bold"),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#B4B2A9")),
        ("ROWBACKGROUNDS",(0,1),(-1,-2),[colors.white, colors.HexColor("#F9F8F5")]),
        ("ALIGN",       (2,0), (-1,-1), "RIGHT"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 8*mm))
    # Условия
    conditions = [
        f"Материал: {mat['name']}, толщина {mat['thickness_mm']} мм",
        f"В стоимость включено: {mat['includes']}",
        f"НДС: {mat['vat_pct']}% (включён в цену)",
        f"Условия оплаты: {mat['payment_terms']}",
        f"Срок поставки: {mat['lead_time']}",
        f"Оплата в рублях по курсу ЦБ на день оплаты",
    ]
    for line in conditions:
        story.append(Paragraph(f"• {line}", normal))
    doc.build(story)
    print(f"[ok] КП сохранён: {fname}")
    return fname
# ─── Точка входа ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="КМ1 Stone Quotation Pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pdf", default=None, help="переопределить путь к PDF")
    args = parser.parse_args()
    cfg = load_config(args.config)
    pdf_path = args.pdf or cfg["paths"]["input_pdf"]
    if not os.path.exists(pdf_path):
        sys.exit(f"[error] PDF не найден: {pdf_path}")
    print(f"[1/4] Рендер PDF: {pdf_path}")
    images = pdf_to_images(pdf_path, cfg["paths"]["temp_dir"], cfg["render"]["dpi"])
    print(f"[2/4] Распознавание марок ({len(images)} стр.)...")
    marks = collect_marks(images, cfg)
    if not marks:
        sys.exit("[error] Модель не вернула ни одной марки. Проверьте endpoint.")
    print(f"[3/4] Расчёт площадей ({len(marks)} марок)...")
    rows = calc_areas(marks)
    total = sum(r["area_m2"] for r in rows)
    print(f"  Итого: {total:.4f} м²")
    print("[4/4] Получение курса ЦБ и генерация КП...")
    rate = get_cbr_rate()
    print(f"  Курс USD/RUB: {rate:.4f}")
    out = generate_kp(rows, cfg, rate, cfg["paths"]["output_dir"])
    print(f"\n✓ Готово → {out}")
if __name__ == "__main__":
    main()
