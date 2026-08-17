#!/usr/bin/env python3
"""Parse hh.ru vacancy search results using an authenticated Chrome session."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

if TYPE_CHECKING:
    from playwright.async_api import Page


DEFAULT_URL = (
    "https://spb.hh.ru/search/vacancy?"
    "resume=5cef9fa3ff080d093b0039ed1f456156794c54&"
    "hhtmFromLabel=tab_byResume&hhtmFrom=main"
)

CARD_SELECTOR = '[data-qa="vacancy-serp__vacancy"], [data-qa="serp-item"]'


@dataclass(frozen=True)
class Vacancy:
    id: str
    title: str
    company: str
    salary: str
    location: str
    experience: str
    employment: str
    work_format: str
    description: str
    requirements: str
    published: str
    url: str
    source_page: int


def clean(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def vacancy_id(url: str) -> str:
    match = re.search(r"/vacancy/(\d+)", url)
    return match.group(1) if match else ""


def page_url(url: str, page_number: int) -> str:
    """HH uses a zero-based `page` query parameter."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    query["page"] = [str(page_number - 1)]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


async def wait_for_login(page: Page, url: str, timeout_minutes: int) -> None:
    await page.goto(url, wait_until="domcontentloaded", timeout=90_000)
    await page.wait_for_timeout(2_000)

    if await page.locator(CARD_SELECTOR).count():
        return

    login_markers = page.locator(
        '[data-qa="login"], [data-qa="account-login"], '
        'input[type="email"], input[name="username"]'
    )
    current_url = page.url.lower()
    if "login" not in current_url and not await login_markers.count():
        return

    print(
        "HH.ru просит авторизацию. Войдите по email в открытом окне Chrome; "
        f"ожидаю до {timeout_minutes} мин...",
        flush=True,
    )
    deadline = asyncio.get_running_loop().time() + timeout_minutes * 60
    while asyncio.get_running_loop().time() < deadline:
        if page.is_closed():
            raise RuntimeError("Окно Chrome было закрыто до завершения входа.")
        if "login" not in page.url.lower():
            await page.goto(url, wait_until="domcontentloaded", timeout=90_000)
            await page.wait_for_timeout(2_000)
            if await page.locator(CARD_SELECTOR).count():
                return
        await page.wait_for_timeout(2_000)

    raise TimeoutError("Время ожидания входа в hh.ru истекло.")


async def extract_page(page: Page, source_page: int) -> list[Vacancy]:
    await page.wait_for_timeout(1_500)
    cards = page.locator(CARD_SELECTOR)
    count = await cards.count()

    if count == 0:
        # Layout fallback: find vacancy links and use their closest visual card.
        links = page.locator('a[href*="/vacancy/"]')
        link_count = await links.count()
        if link_count == 0:
            return []
        cards = links
        count = link_count

    rows: list[Vacancy] = []
    seen: set[str] = set()
    for index in range(count):
        item = cards.nth(index)
        data: dict[str, Any] = await item.evaluate(
            """(node) => {
                const qa = (root, names) => {
                    for (const name of names) {
                        const el = root.querySelector(`[data-qa="${name}"]`);
                        if (el?.textContent?.trim()) return el.textContent.trim();
                    }
                    return '';
                };
                const isLink = node.matches?.('a[href*="/vacancy/"]');
                const root = isLink
                    ? (node.closest('[data-qa="vacancy-serp__vacancy"], [data-qa="serp-item"], article') || node.parentElement)
                    : node;
                const titleLink = isLink ? node : root.querySelector(
                    '[data-qa="serp-item__title"], '
                    '[data-qa="vacancy-serp__vacancy-title"], a[href*="/vacancy/"]'
                );
                const labels = [...root.querySelectorAll('[data-qa]')]
                    .map(el => ({qa: el.getAttribute('data-qa') || '', text: el.textContent?.trim() || ''}));
                const byQaPart = (part) => labels.find(x => x.qa.includes(part) && x.text)?.text || '';
                return {
                    title: titleLink?.textContent?.trim() || '',
                    url: titleLink?.href || '',
                    company: qa(root, ['vacancy-serp__vacancy-employer', 'vacancy-serp__vacancy-employer-text']),
                    salary: qa(root, ['vacancy-serp__vacancy-compensation', 'vacancy-compensation']),
                    location: qa(root, ['vacancy-serp__vacancy-address', 'vacancy-serp__vacancy-address-text']),
                    experience: byQaPart('experience'),
                    employment: byQaPart('employment'),
                    work_format: byQaPart('work-format'),
                    description: qa(root, ['vacancy-serp__vacancy_snippet_responsibility']),
                    requirements: qa(root, ['vacancy-serp__vacancy_snippet_requirement']),
                    published: byQaPart('publication') || byQaPart('date')
                };
            }"""
        )
        url = clean(data.get("url"))
        item_id = vacancy_id(url)
        dedupe_key = item_id or url
        if not clean(data.get("title")) or not dedupe_key or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        rows.append(
            Vacancy(
                id=item_id,
                title=clean(data.get("title")),
                company=clean(data.get("company")),
                salary=clean(data.get("salary")),
                location=clean(data.get("location")),
                experience=clean(data.get("experience")),
                employment=clean(data.get("employment")),
                work_format=clean(data.get("work_format")),
                description=clean(data.get("description")),
                requirements=clean(data.get("requirements")),
                published=clean(data.get("published")),
                url=url.split("?")[0],
                source_page=source_page,
            )
        )
    return rows


async def open_context(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    try:
        from playwright.async_api import async_playwright
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Playwright не установлен. Выполните: "
            "pip install -r hh_parser/requirements.txt"
        ) from error

    playwright = await async_playwright().start()
    if args.cdp_url:
        browser = await playwright.chromium.connect_over_cdp(args.cdp_url)
        if not browser.contexts:
            raise RuntimeError("В подключённом Chrome не найден контекст браузера.")
        context = browser.contexts[0]
        page = next((p for p in context.pages if "hh.ru" in p.url), None)
        page = page or await context.new_page()
        return playwright, browser, context, page

    profile = Path(args.profile).resolve()
    profile.mkdir(parents=True, exist_ok=True)
    context = await playwright.chromium.launch_persistent_context(
        user_data_dir=str(profile),
        channel="chrome",
        headless=args.headless,
        locale="ru-RU",
        viewport={"width": 1440, "height": 1000},
        args=["--disable-blink-features=AutomationControlled"],
    )
    page = context.pages[0] if context.pages else await context.new_page()
    return playwright, None, context, page


def save(rows: list[Vacancy], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"vacancies_{stamp}.json"
    csv_path = output_dir / f"vacancies_{stamp}.csv"
    payload = [asdict(row) for row in rows]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(payload)
    return csv_path, json_path


async def run(args: argparse.Namespace) -> int:
    playwright = browser = context = page = None
    try:
        playwright, browser, context, page = await open_context(args)
        await wait_for_login(page, args.url, args.login_timeout)

        vacancies: dict[str, Vacancy] = {}
        for number in range(1, args.pages + 1):
            target = page_url(args.url, number)
            if page.url != target:
                await page.goto(target, wait_until="domcontentloaded", timeout=90_000)
            page_rows = await extract_page(page, number)
            if not page_rows:
                print(f"Страница {number}: вакансии не найдены, останавливаюсь.")
                break
            for row in page_rows:
                vacancies[row.id or row.url] = row
            print(f"Страница {number}: {len(page_rows)} вакансий; всего {len(vacancies)}.")
            if number < args.pages:
                await page.wait_for_timeout(args.delay_ms)

        if not vacancies:
            raise RuntimeError(
                "Не удалось найти карточки вакансий. Проверьте авторизацию, URL и отсутствие CAPTCHA."
            )
        csv_path, json_path = save(list(vacancies.values()), Path(args.output))
        print(f"Готово: {len(vacancies)} вакансий")
        print(f"CSV:  {csv_path.resolve()}")
        print(f"JSON: {json_path.resolve()}")
        return 0
    finally:
        if context is not None and browser is None:
            await context.close()
        # In CDP mode do not close the user's Chrome/session.
        if playwright is not None:
            await playwright.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="URL поисковой выдачи hh.ru")
    parser.add_argument("--pages", type=int, default=3, help="Сколько страниц собрать")
    parser.add_argument("--delay-ms", type=int, default=2500, help="Пауза между страницами")
    parser.add_argument("--output", default="hh_parser/output", help="Каталог результатов")
    parser.add_argument("--profile", default="hh_parser/.chrome-profile", help="Профиль Chrome")
    parser.add_argument("--cdp-url", help="Chrome DevTools URL, например http://127.0.0.1:9222")
    parser.add_argument("--login-timeout", type=int, default=10, help="Ожидание ручного входа, минут")
    parser.add_argument("--headless", action="store_true", help="Без окна; только после сохранения сессии")
    args = parser.parse_args()
    if args.pages < 1 or args.pages > 50:
        parser.error("--pages должен быть от 1 до 50")
    if args.delay_ms < 1000:
        parser.error("--delay-ms должен быть не меньше 1000")
    if args.headless and not Path(args.profile).exists() and not args.cdp_url:
        parser.error("Первый запуск нужен без --headless, чтобы войти в hh.ru")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(run(parse_args())))
    except KeyboardInterrupt:
        print("Остановлено пользователем.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        raise SystemExit(1)
