#!/usr/bin/env python3
"""Local web interface for uploading PDFs and calculating facade area."""

from __future__ import annotations

from copy import deepcopy
import asyncio
from decimal import Decimal, InvalidOperation
import json
import logging
import mimetypes
import os
from pathlib import Path
import secrets
import threading
import time
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request as URLRequest, urlopen

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
import psutil
from pydantic import BaseModel

from pipeline_logic import analyze_pdf_file, analyze_pdf_legends, setup_logging


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
UPLOAD_ROOT = BASE_DIR / "output" / "web_uploads"
FEEDBACK_ROOT = BASE_DIR / "run"
SERVER_CONFIG_PATH = Path(
    os.getenv("PIPESTONE_SERVER_CONFIG", str(BASE_DIR / "server_config.local.json"))
)
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf", "application/octet-stream"}
SESSION_COOKIE = "pipestone_session"
SESSION_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
FEEDBACK_MAX_LENGTH = 512
TELEGRAM_API_BASE = "https://api.telegram.org"

HTTP_REQUESTS = Counter(
    "pipestone_http_requests_total",
    "HTTP requests handled by the application.",
    ("method", "endpoint", "status"),
)
HTTP_REQUEST_DURATION = Histogram(
    "pipestone_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("method", "endpoint"),
)
DOCUMENTS_UPLOADED = Counter(
    "pipestone_documents_uploaded_total",
    "PDF documents successfully uploaded.",
)
DOCUMENT_UPLOAD_BYTES = Counter(
    "pipestone_document_upload_bytes_total",
    "Total bytes of successfully uploaded PDF documents.",
)
DOCUMENT_UPLOAD_DURATION = Histogram(
    "pipestone_document_upload_duration_seconds",
    "Successful PDF upload duration in seconds.",
)
DOCUMENT_PROCESSING = Counter(
    "pipestone_document_processing_total",
    "Completed document processing operations.",
    ("operation", "status"),
)
PROCESSING_DURATION = Histogram(
    "pipestone_processing_duration_seconds",
    "Document processing operation duration in seconds.",
    ("operation", "status"),
)
STAGE_DURATION = Histogram(
    "pipestone_stage_duration_seconds",
    "Calculation stage duration in seconds.",
    ("stage",),
)
PROCESSING_ACTIVE = Gauge(
    "pipestone_processing_active",
    "Number of currently running document processing operations.",
    ("operation",),
)
SESSIONS_TOTAL = Counter(
    "pipestone_sessions_created_total",
    "Application sessions created.",
)
SESSIONS_ACTIVE = Gauge(
    "pipestone_sessions_active",
    "Sessions currently held in application memory.",
)
FEEDBACK_TOTAL = Counter(
    "pipestone_feedback_total",
    "Feedback submission outcomes.",
    ("status",),
)
TELEGRAM_REQUESTS = Counter(
    "pipestone_telegram_requests_total",
    "Telegram Bot API request outcomes.",
    ("method", "status"),
)
TELEGRAM_REQUEST_DURATION = Histogram(
    "pipestone_telegram_request_duration_seconds",
    "Telegram Bot API request duration in seconds.",
    ("method",),
)
APP_CPU_SECONDS = Gauge(
    "pipestone_process_cpu_seconds_total",
    "Total user and system CPU time consumed by the application process.",
)
APP_RESIDENT_MEMORY = Gauge(
    "pipestone_process_resident_memory_bytes",
    "Resident memory used by the application process.",
)
APP_THREADS = Gauge(
    "pipestone_process_threads",
    "Threads used by the application process.",
)
SYSTEM_CPU_PERCENT = Gauge(
    "pipestone_system_cpu_percent",
    "CPU utilization visible to the application container.",
)
SYSTEM_MEMORY_USED = Gauge(
    "pipestone_system_memory_used_bytes",
    "Used system memory visible to the application container.",
)
SYSTEM_MEMORY_PERCENT = Gauge(
    "pipestone_system_memory_percent",
    "Used system memory percentage visible to the application container.",
)

app = FastAPI(title="PipeStone — расчёт площади", version="1.0")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


@app.middleware("http")
async def prometheus_http_metrics(request: Request, call_next):
    started_at = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path)
        HTTP_REQUESTS.labels(request.method, endpoint, str(status_code)).inc()
        HTTP_REQUEST_DURATION.labels(request.method, endpoint).observe(
            time.perf_counter() - started_at
        )

logger = logging.getLogger("pipestone.web")
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()
sessions: dict[str, dict[str, str | None]] = {}
sessions_lock = threading.Lock()
calculation_state_lock = threading.Lock()
calculation_busy = threading.Event()
active_calculation_job_id: str | None = None
active_operation: str | None = None


class JobCancelled(RuntimeError):
    pass


class FeedbackPayload(BaseModel):
    description: str


def load_telegram_config() -> dict[str, str]:
    config: dict[str, Any] = {}
    if SERVER_CONFIG_PATH.exists():
        try:
            config = json.loads(SERVER_CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("Некорректная конфигурация сервера") from exc
    telegram_config = config.get("telegram", {})
    return {
        "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", telegram_config.get("bot_token", "")),
        "chat_id": str(os.getenv("TELEGRAM_CHAT_ID", telegram_config.get("chat_id", ""))),
    }


def telegram_api_request(
    method: str,
    fields: dict[str, str],
    document_path: Path | None = None,
) -> dict[str, Any]:
    config = load_telegram_config()
    if not config["bot_token"]:
        raise RuntimeError("токен Telegram-бота не настроен")
    if not config["chat_id"]:
        raise RuntimeError("chat_id Telegram не настроен")

    request_fields = {"chat_id": config["chat_id"], **fields}
    headers = {"Accept": "application/json"}
    if document_path is None:
        body = urlencode(request_fields).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded; charset=utf-8"
    else:
        if not document_path.is_file():
            raise RuntimeError("загруженный PDF не найден на сервере")
        boundary = f"----PipeStone{uuid.uuid4().hex}"
        body_parts: list[bytes] = []
        for name, value in request_fields.items():
            body_parts.extend(
                [
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    str(value).encode("utf-8"),
                    b"\r\n",
                ]
            )
        media_type = mimetypes.guess_type(document_path.name)[0] or "application/pdf"
        safe_filename = document_path.name.replace('"', "")
        body_parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="document"; filename="{safe_filename}"\r\n'.encode(),
                f"Content-Type: {media_type}\r\n\r\n".encode(),
                document_path.read_bytes(),
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        body = b"".join(body_parts)
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

    url = f"{TELEGRAM_API_BASE}/bot{config['bot_token']}/{method}"
    api_request = URLRequest(url, data=body, headers=headers, method="POST")
    started_at = time.perf_counter()
    try:
        with urlopen(api_request, timeout=15) as api_response:
            result = json.loads(api_response.read().decode("utf-8"))
        if not result.get("ok"):
            raise RuntimeError(result.get("description", "Telegram вернул ошибку"))
        TELEGRAM_REQUESTS.labels(method, "success").inc()
    except HTTPError as exc:
        try:
            api_message = exc.read().decode("utf-8", errors="replace")
        except OSError:
            api_message = ""
        logger.warning("Telegram rejected request: method=%s status=%s body=%s", method, exc.code, api_message[:500])
        TELEGRAM_REQUESTS.labels(method, f"http_{exc.code}").inc()
        raise RuntimeError(f"Telegram отклонил запрос (HTTP {exc.code})") from exc
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        logger.warning("Telegram request failed: method=%s error=%s", method, type(exc).__name__)
        TELEGRAM_REQUESTS.labels(method, "network_error").inc()
        raise RuntimeError("не удалось связаться с Telegram") from exc
    finally:
        TELEGRAM_REQUEST_DURATION.labels(method).observe(time.perf_counter() - started_at)
    return result


def send_telegram_feedback(
    description: str,
    session_id: str,
    pdf_path: Path | None = None,
) -> dict[str, Any]:
    caption = f"🛠 Обратная связь PipeStone\nСессия: {session_id}\n\n{description}"
    if pdf_path is not None:
        return telegram_api_request("sendDocument", {"caption": caption}, pdf_path)
    return telegram_api_request("sendMessage", {"text": caption})


def send_telegram_processing_success(
    filename: str,
    elapsed_seconds: float,
    session_id: str,
) -> dict[str, Any]:
    text = (
        "✅ Обработка PDF успешно завершена\n"
        f"Файл: {filename}\n"
        f"Время выполнения: {elapsed_seconds:.1f} с\n"
        f"Сессия: {session_id}"
    )
    return telegram_api_request("sendMessage", {"text": text})


def ensure_session(request: Request, response: Response) -> str:
    session_id = request.cookies.get(SESSION_COOKIE)
    with sessions_lock:
        if not session_id or session_id not in sessions:
            session_id = secrets.token_urlsafe(32)
            sessions[session_id] = {"job_id": None}
            response.set_cookie(
                SESSION_COOKIE,
                session_id,
                max_age=SESSION_MAX_AGE_SECONDS,
                httponly=True,
                samesite="lax",
                secure=False,
                path="/",
            )
            SESSIONS_TOTAL.inc()
            SESSIONS_ACTIVE.set(len(sessions))
    return session_id


def require_session_job(request: Request, job_id: str) -> None:
    session_id = request.cookies.get(SESSION_COOKIE)
    with sessions_lock:
        owned_job_id = sessions.get(session_id or "", {}).get("job_id")
    if owned_job_id != job_id:
        raise HTTPException(status_code=404, detail="Загрузка не найдена в текущей сессии")


def public_job_state(job: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(
        {
            key: value
            for key, value in job.items()
            if key not in {
                "pdf_path",
                "output_dir",
                "legend_started_at",
                "legend_finished_at",
                "calculation_started_at",
                "calculation_finished_at",
                "legend_analysis",
            }
        }
    )
    now = time.time()
    for operation in ("legend", "calculation"):
        started_at = job.get(f"{operation}_started_at")
        finished_at = job.get(f"{operation}_finished_at")
        elapsed = max(0.0, float((finished_at or now) - started_at)) if started_at else None
        state[f"{operation}_elapsed_seconds"] = round(elapsed, 1) if elapsed is not None else None
    return state


def claim_calculation_slot(job_id: str, operation: str = "calculation") -> bool:
    """Atomically reserve the single calculation slot without waiting."""
    global active_calculation_job_id, active_operation
    with calculation_state_lock:
        if calculation_busy.is_set():
            return False
        calculation_busy.set()
        active_calculation_job_id = job_id
        active_operation = operation
        return True


def release_calculation_slot(job_id: str) -> None:
    """Release the slot only from the job that currently owns it."""
    global active_calculation_job_id, active_operation
    with calculation_state_lock:
        if active_calculation_job_id != job_id:
            return
        active_calculation_job_id = None
        active_operation = None
        calculation_busy.clear()


def calculation_server_status() -> dict[str, Any]:
    with calculation_state_lock:
        busy = calculation_busy.is_set()
        return {
            "status": "busy" if busy else "free",
            "busy": busy,
            "active_job_id": active_calculation_job_id,
            "operation": active_operation,
            "poll_interval_seconds": 30,
        }


def public_server_status(request: Request) -> dict[str, Any]:
    internal = calculation_server_status()
    session_id = request.cookies.get(SESSION_COOKIE)
    with sessions_lock:
        session_job_id = sessions.get(session_id or "", {}).get("job_id")
    return {
        "status": internal["status"],
        "busy": internal["busy"],
        "operation": internal["operation"],
        "is_current_session": bool(
            internal["active_job_id"] and internal["active_job_id"] == session_job_id
        ),
        "poll_interval_seconds": internal["poll_interval_seconds"],
    }


def raise_if_cancelled(job_id: str) -> None:
    with jobs_lock:
        if jobs.get(job_id, {}).get("cancel_requested"):
            raise JobCancelled("Операция остановлена пользователем")


def initial_stages() -> list[dict[str, str]]:
    return [
        {"id": "legend", "label": "Поиск легенды", "status": "pending"},
        {"id": "symbol", "label": "Поиск условного обозначения", "status": "pending"},
        {"id": "pages", "label": "Поиск страниц в файле", "status": "pending"},
        {"id": "area", "label": "Подсчёт площади", "status": "pending"},
        {"id": "merge", "label": "Объединение результатов", "status": "pending"},
    ]


def update_progress(job_id: str, stage: str, percent: int, message: str) -> None:
    stage_order = ["legend", "symbol", "pages", "area", "merge"]
    with jobs_lock:
        job = jobs[job_id]
        if job.get("cancel_requested"):
            raise JobCancelled("Расчёт остановлен пользователем")
        job["progress"] = percent
        job["message"] = message
        job["status"] = "completed" if stage == "complete" else "running"
        current_index = stage_order.index(stage) if stage in stage_order else len(stage_order)
        for index, item in enumerate(job["stages"]):
            if index < current_index or stage == "complete":
                item["status"] = "completed"
            elif index == current_index:
                item["status"] = "running"
            else:
                item["status"] = "pending"


def normalize_bucket_name(name: Any) -> str:
    return " ".join(str(name).replace("ё", "е").replace("Ё", "Е").casefold().split())


def normalize_bucket_dimensions(values: Any) -> tuple[str, ...]:
    normalized: set[str] = set()
    for value in values or []:
        text = str(value).strip().replace(",", ".")
        if not text:
            continue
        try:
            number = Decimal(text)
            text = format(number.normalize(), "f")
        except InvalidOperation:
            text = " ".join(text.casefold().split())
        normalized.add(text)
    return tuple(sorted(normalized))


def merge_area_buckets(area_calculation: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[str, tuple[str, ...], tuple[str, ...]], dict[str, Any]] = {}
    for page_result in area_calculation.get("pages", []):
        page = int(page_result.get("page", 0))
        for name, values in page_result.get("unique_elements", {}).items():
            horizontal_dimensions = normalize_bucket_dimensions(values.get("horizontal_dimensions", []))
            vertical_dimensions = normalize_bucket_dimensions(values.get("vertical_dimensions", []))
            key = (
                normalize_bucket_name(name),
                horizontal_dimensions,
                vertical_dimensions,
            )
            item = grouped.setdefault(
                key,
                {
                    "name": str(name),
                    "count": 0,
                    "horizontal_dimensions": list(horizontal_dimensions),
                    "vertical_dimensions": list(vertical_dimensions),
                    "area_m2": 0.0,
                    "pages": set(),
                },
            )
            item["count"] += int(values.get("count", 0))
            area = values.get("total_area_m2")
            if isinstance(area, (int, float)):
                item["area_m2"] += float(area)
            item["pages"].add(page)

    result: list[dict[str, Any]] = []
    for item in grouped.values():
        result.append(
            {
                "name": item["name"],
                "count": item["count"],
                "horizontal_dimensions": item["horizontal_dimensions"],
                "vertical_dimensions": item["vertical_dimensions"],
                "area_m2": round(item["area_m2"], 3),
                "pages": sorted(item["pages"]),
            }
        )
    groups = sorted(
        result,
        key=lambda item: (
            -item["area_m2"],
            normalize_bucket_name(item["name"]),
            item["horizontal_dimensions"],
            item["vertical_dimensions"],
        ),
    )
    return {
        "groups": groups,
        "total_area_m2": round(sum(item["area_m2"] for item in groups), 3),
    }


def material_groups(area_calculation: dict[str, Any]) -> list[dict[str, Any]]:
    """Backward-compatible access to the merged web result groups."""
    return merge_area_buckets(area_calculation)["groups"]


def search_legends_job(job_id: str) -> None:
    """Wait for the processing slot, then run only the legend-search pass."""
    while not claim_calculation_slot(job_id, "legend_search"):
        with jobs_lock:
            if job_id not in jobs:
                return
            if jobs[job_id].get("cancel_requested"):
                jobs[job_id]["status"] = "legend_cancelled"
                jobs[job_id]["legend_message"] = "Поиск легенды остановлен"
                return
            jobs[job_id]["status"] = "legend_waiting"
            jobs[job_id]["legend_message"] = "Ожидание свободного сервера"
        time.sleep(1)

    metric_started_at = time.perf_counter()
    metric_status = "failed"
    PROCESSING_ACTIVE.labels("legend_search").inc()
    try:
        raise_if_cancelled(job_id)
        with jobs_lock:
            job = jobs[job_id]
            pdf_path = Path(job["pdf_path"])
            output_dir = Path(job["output_dir"]) / "legend_search"
            job["status"] = "legend_search"
            job["legend_started_at"] = time.time()
            job["legend_finished_at"] = None
            job["legend_progress"] = 1
            job["legend_message"] = "Поиск легенды в документе"

        def progress(stage: str, percent: int, message: str) -> None:
            with jobs_lock:
                if job_id in jobs:
                    if jobs[job_id].get("cancel_requested"):
                        raise JobCancelled("Поиск легенды остановлен пользователем")
                    jobs[job_id]["legend_progress"] = percent
                    jobs[job_id]["legend_message"] = message

        result = analyze_pdf_legends(
            pdf_path,
            output_dir=output_dir,
            ocr_backend="auto",
            progress_callback=progress,
        )
        legends = result.get("legends", [])
        legend_analysis = {
            "analysis_dpi": result.get("analysis_dpi"),
            "legend_pages": result.get("legend_pages", []),
            "material_lines": result.get("material_lines", []),
            "pattern_matches": result.get("pattern_matches", []),
        }
        with jobs_lock:
            job = jobs[job_id]
            job["legends"] = legends
            job["legend_analysis"] = legend_analysis
            job["legend_progress"] = 100
            job["legend_log_file"] = result.get("log_file", "")
            if legends:
                job["status"] = "ready"
                job["legend_message"] = f"Найдено легенд: {len(legends)}"
                metric_status = "ready"
            else:
                job["status"] = "no_legends"
                job["legend_message"] = "Выбранный файл не содержит сведений о камне"
                metric_status = "no_legends"
    except JobCancelled:
        metric_status = "cancelled"
        with jobs_lock:
            jobs[job_id]["status"] = "legend_cancelled"
            jobs[job_id]["legend_message"] = "Поиск легенды остановлен"
            jobs[job_id]["error"] = ""
    except Exception as exc:
        logger.exception("Legend search failed: job=%s", job_id)
        with jobs_lock:
            jobs[job_id]["status"] = "legend_failed"
            jobs[job_id]["legend_message"] = "Не удалось выполнить поиск легенды"
            jobs[job_id]["error"] = str(exc)
    finally:
        with jobs_lock:
            job = jobs.get(job_id)
            if job and job.get("legend_started_at") and not job.get("legend_finished_at"):
                job["legend_finished_at"] = time.time()
        release_calculation_slot(job_id)
        PROCESSING_ACTIVE.labels("legend_search").dec()
        DOCUMENT_PROCESSING.labels("legend_search", metric_status).inc()
        PROCESSING_DURATION.labels("legend_search", metric_status).observe(
            time.perf_counter() - metric_started_at
        )


def calculate_job(job_id: str) -> None:
    metric_started_at = time.perf_counter()
    metric_status = "failed"
    active_metric_stage: str | None = None
    metric_stage_started_at = metric_started_at
    PROCESSING_ACTIVE.labels("calculation").inc()

    def progress_with_metrics(stage: str, percent: int, message: str) -> None:
        nonlocal active_metric_stage, metric_stage_started_at
        now = time.perf_counter()
        if active_metric_stage and active_metric_stage != stage:
            STAGE_DURATION.labels(active_metric_stage).observe(now - metric_stage_started_at)
            metric_stage_started_at = now
        if stage == "complete":
            active_metric_stage = None
            return
        active_metric_stage = stage
        update_progress(job_id, stage, percent, message)

    try:
        raise_if_cancelled(job_id)
        with jobs_lock:
            pdf_path = Path(jobs[job_id]["pdf_path"])
            output_dir = Path(jobs[job_id]["output_dir"])
            legend_analysis = deepcopy(jobs[job_id].get("legend_analysis"))
        result = analyze_pdf_file(
            pdf_path,
            output_dir=output_dir,
            ocr_backend="auto",
            calculate_area=True,
            precomputed_legend_analysis=legend_analysis,
            progress_callback=progress_with_metrics,
        )
        area = result.get("area_calculation", {})
        progress_with_metrics("merge", 98, "Объединение одинаковых результатов")
        merge_started_at = time.perf_counter()
        merged_buckets = merge_area_buckets(area)
        STAGE_DURATION.labels("merge").observe(time.perf_counter() - merge_started_at)
        progress_with_metrics("complete", 100, "")
        web_result = {
            "legend_pages": result.get("legend_pages", []),
            "hatch_pages": result.get("hatch_pages", []),
            "hatch_pattern_box_pages": result.get("hatch_pattern_box_pages", []),
            "groups": merged_buckets["groups"],
            "page_areas": [
                {
                    "page": item.get("page"),
                    "area_m2": item.get("total_area_m2", 0.0),
                }
                for item in area.get("pages", [])
            ],
            "total_area_m2": merged_buckets["total_area_m2"],
            "pattern_image": area.get("pattern_image", ""),
            "run_dir": result.get("run_dir", ""),
            "log_file": result.get("log_file", ""),
            "warning": area.get("warning", ""),
        }
        with jobs_lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["message"] = "Расчёт успешно завершён"
            for stage in jobs[job_id]["stages"]:
                stage["status"] = "completed"
            jobs[job_id]["result"] = web_result
            notification_filename = str(jobs[job_id]["filename"])
            notification_session_id = str(jobs[job_id]["session_id"])
            metric_status = "completed"
        try:
            send_telegram_processing_success(
                notification_filename,
                time.perf_counter() - metric_started_at,
                notification_session_id,
            )
        except RuntimeError as exc:
            logger.warning("Telegram completion notification was not sent: %s", exc)
    except JobCancelled:
        metric_status = "cancelled"
        with jobs_lock:
            jobs[job_id]["status"] = "calculation_cancelled"
            jobs[job_id]["message"] = "Расчёт остановлен"
            jobs[job_id]["error"] = ""
    except Exception as exc:  # The UI receives a safe error; details remain in the run log.
        logger.exception("Web calculation failed: job=%s", job_id)
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = "Не удалось завершить расчёт"
            jobs[job_id]["error"] = str(exc)
    finally:
        if active_metric_stage:
            STAGE_DURATION.labels(active_metric_stage).observe(
                time.perf_counter() - metric_stage_started_at
            )
        with jobs_lock:
            job = jobs.get(job_id)
            if job and job.get("calculation_started_at") and not job.get("calculation_finished_at"):
                job["calculation_finished_at"] = time.time()
        release_calculation_slot(job_id)
        PROCESSING_ACTIVE.labels("calculation").dec()
        DOCUMENT_PROCESSING.labels("calculation", metric_status).inc()
        PROCESSING_DURATION.labels("calculation", metric_status).observe(
            time.perf_counter() - metric_started_at
        )


@app.on_event("startup")
def startup() -> None:
    setup_logging()
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    FEEDBACK_ROOT.mkdir(parents=True, exist_ok=True)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/logo.png", include_in_schema=False)
def logo() -> FileResponse:
    return FileResponse(WEB_DIR / "logo.png", media_type="image/png")


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    process = psutil.Process()
    cpu_times = process.cpu_times()
    memory = psutil.virtual_memory()
    APP_CPU_SECONDS.set(cpu_times.user + cpu_times.system)
    APP_RESIDENT_MEMORY.set(process.memory_info().rss)
    APP_THREADS.set(process.num_threads())
    SYSTEM_CPU_PERCENT.set(psutil.cpu_percent(interval=None))
    SYSTEM_MEMORY_USED.set(memory.used)
    SYSTEM_MEMORY_PERCENT.set(memory.percent)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/feedback", status_code=201)
async def submit_feedback(
    payload: FeedbackPayload,
    request: Request,
    response: Response,
) -> dict[str, Any]:
    description = payload.description.strip()
    if not description:
        raise HTTPException(status_code=422, detail="Опишите проблему")
    if len(description) > FEEDBACK_MAX_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Описание не должно превышать {FEEDBACK_MAX_LENGTH} символов",
        )

    session_id = ensure_session(request, response)
    session_feedback_root = FEEDBACK_ROOT / session_id
    session_feedback_root.mkdir(parents=True, exist_ok=True)
    feedback_id = uuid.uuid4().hex
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    filename_timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    feedback_path = session_feedback_root / f"feedback_{filename_timestamp}_{feedback_id}.txt"
    feedback_path.write_text(
        f"created_at: {created_at}\nsession_id: {session_id}\n\n{description}\n",
        encoding="utf-8",
    )
    FEEDBACK_TOTAL.labels("saved_locally").inc()

    pdf_path: Path | None = None
    with sessions_lock:
        current_job_id = sessions.get(session_id, {}).get("job_id")
    if current_job_id:
        with jobs_lock:
            current_job = jobs.get(current_job_id)
            if current_job:
                candidate_path = Path(str(current_job["pdf_path"]))
                if candidate_path.is_file():
                    pdf_path = candidate_path

    try:
        telegram_result = await asyncio.to_thread(
            send_telegram_feedback,
            description,
            session_id,
            pdf_path,
        )
    except RuntimeError as exc:
        FEEDBACK_TOTAL.labels("telegram_failed").inc()
        raise HTTPException(
            status_code=502,
            detail=f"Сообщение сохранено на сервере, но не отправлено в Telegram: {exc}",
        ) from exc

    FEEDBACK_TOTAL.labels("submitted").inc()
    return {
        "status": "submitted",
        "feedback_id": feedback_id,
        "telegram_message_id": telegram_result.get("result", {}).get("message_id"),
    }


@app.post("/api/upload", status_code=201)
async def upload_pdf(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    upload_started_at = time.perf_counter()
    session_id = ensure_session(request, response)
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Разрешены только файлы с расширением .pdf")
    if file.content_type and file.content_type.lower() not in PDF_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Файл должен иметь PDF MIME-тип")

    job_id = uuid.uuid4().hex
    job_dir = UPLOAD_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    session_run_dir = FEEDBACK_ROOT / session_id
    session_run_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = job_dir / filename
    size = 0
    first_chunk = True
    try:
        with pdf_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                if first_chunk:
                    first_chunk = False
                    if not chunk.startswith(b"%PDF-"):
                        raise HTTPException(status_code=400, detail="Содержимое файла не является PDF-документом")
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Размер PDF не должен превышать 100 МБ")
                destination.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Загружен пустой файл")
    except Exception:
        pdf_path.unlink(missing_ok=True)
        job_dir.rmdir()
        raise
    finally:
        await file.close()

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "session_id": session_id,
            "filename": filename,
            "size": size,
            "pdf_path": str(pdf_path),
            "output_dir": str(session_run_dir),
            "status": "legend_queued",
            "progress": 0,
            "message": "PDF успешно загружен",
            "legend_progress": 0,
            "legend_message": "Поиск легенды поставлен в очередь",
            "legends": [],
            "legend_analysis": None,
            "legend_log_file": "",
            "legend_started_at": None,
            "legend_finished_at": None,
            "calculation_started_at": None,
            "calculation_finished_at": None,
            "cancel_requested": False,
            "stages": initial_stages(),
            "result": None,
            "error": "",
        }
    with sessions_lock:
        sessions[session_id]["job_id"] = job_id
    background_tasks.add_task(search_legends_job, job_id)
    DOCUMENTS_UPLOADED.inc()
    DOCUMENT_UPLOAD_BYTES.inc(size)
    DOCUMENT_UPLOAD_DURATION.observe(time.perf_counter() - upload_started_at)
    return {"job_id": job_id, "filename": filename, "size": size, "status": "uploaded"}


@app.post("/api/jobs/{job_id}/calculate", status_code=202)
def start_calculation(job_id: str, request: Request, background_tasks: BackgroundTasks) -> dict[str, str]:
    require_session_job(request, job_id)
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Загрузка не найдена")
        if job["status"] in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="Расчёт уже выполняется")
        if job["status"] in {"legend_queued", "legend_waiting", "legend_search"}:
            raise HTTPException(status_code=409, detail="Сначала дождитесь завершения поиска легенды")
        if not job.get("legends"):
            raise HTTPException(status_code=422, detail="Выбранный файл не содержит сведений о камне")
    if not claim_calculation_slot(job_id):
        raise HTTPException(status_code=409, detail="Сервер занят другим расчётом. Повторите попытку после его завершения")
    try:
        with jobs_lock:
            job = jobs[job_id]
            job["status"] = "queued"
            job["calculation_started_at"] = time.time()
            job["calculation_finished_at"] = None
            job["progress"] = 1
            job["message"] = "Расчёт поставлен в очередь"
            job["error"] = ""
            job["result"] = None
            job["stages"] = initial_stages()
            job["cancel_requested"] = False
        background_tasks.add_task(calculate_job, job_id)
    except Exception:
        release_calculation_slot(job_id)
        raise
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/jobs/{job_id}/cancel", status_code=202)
def cancel_job(job_id: str, request: Request) -> dict[str, str]:
    require_session_job(request, job_id)
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Загрузка не найдена")
        if job["status"] not in {
            "legend_queued", "legend_waiting", "legend_search", "queued", "running"
        }:
            raise HTTPException(status_code=409, detail="Нет активной операции для остановки")
        job["cancel_requested"] = True
        if job["status"].startswith("legend"):
            job["legend_message"] = "Остановка поиска легенды…"
        else:
            job["message"] = "Остановка расчёта…"
    return {"job_id": job_id, "status": "cancellation_requested"}


@app.get("/api/server-status")
def server_status(request: Request) -> dict[str, Any]:
    return public_server_status(request)


@app.get("/api/session")
def session_status(request: Request, response: Response) -> dict[str, Any]:
    session_id = ensure_session(request, response)
    with sessions_lock:
        job_id = sessions[session_id].get("job_id")
    job = None
    if job_id:
        with jobs_lock:
            stored_job = jobs.get(job_id)
            if stored_job is not None:
                job = public_job_state(stored_job)
    return {"job": job, "server": public_server_status(request)}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, request: Request) -> dict[str, Any]:
    require_session_job(request, job_id)
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Загрузка не найдена")
        return public_job_state(job)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=False)
