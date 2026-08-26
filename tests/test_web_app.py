import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

import web_app


class WebAppTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(web_app.app)

    def setUp(self) -> None:
        with web_app.jobs_lock:
            web_app.jobs.clear()
        status = web_app.calculation_server_status()
        if status["active_job_id"]:
            web_app.release_calculation_slot(status["active_job_id"])

    def tearDown(self) -> None:
        status = web_app.calculation_server_status()
        if status["active_job_id"]:
            web_app.release_calculation_slot(status["active_job_id"])

    def upload_valid_pdf(self):
        with patch.object(web_app, "search_legends_job") as search:
            response = self.client.post(
                "/api/upload",
                files={"file": ("drawing.pdf", io.BytesIO(b"%PDF-1.7\n%%EOF"), "application/pdf")},
            )
        search.assert_called_once()
        return response

    @staticmethod
    def mark_legend_ready(job_id: str) -> None:
        with web_app.jobs_lock:
            web_app.jobs[job_id]["status"] = "ready"
            web_app.jobs[job_id]["legends"] = [{"page": 1, "name": "Гранит", "score": 1.0}]

    def test_calculation_slot_is_atomic(self) -> None:
        job_ids = [f"job-{index}" for index in range(16)]
        with ThreadPoolExecutor(max_workers=16) as pool:
            claims = list(pool.map(web_app.claim_calculation_slot, job_ids))

        self.assertEqual(claims.count(True), 1)
        winner = job_ids[claims.index(True)]
        self.assertEqual(web_app.calculation_server_status()["active_job_id"], winner)

    def test_server_status_reports_free_and_busy(self) -> None:
        self.assertEqual(self.client.get("/api/server-status").json()["status"], "free")
        upload = self.upload_valid_pdf().json()
        self.assertTrue(web_app.claim_calculation_slot(upload["job_id"], "calculation"))
        owner_status = self.client.get("/api/server-status").json()
        other_status = TestClient(web_app.app).get("/api/server-status").json()
        self.assertEqual(owner_status["status"], "busy")
        self.assertEqual(owner_status["operation"], "calculation")
        self.assertTrue(owner_status["is_current_session"])
        self.assertFalse(other_status["is_current_session"])
        self.assertNotIn("active_job_id", owner_status)

    def test_rejects_wrong_extension(self) -> None:
        response = self.client.post(
            "/api/upload",
            files={"file": ("drawing.txt", io.BytesIO(b"%PDF-1.7\n"), "application/pdf")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(".pdf", response.json()["detail"])

    def test_rejects_fake_pdf_content(self) -> None:
        response = self.client.post(
            "/api/upload",
            files={"file": ("drawing.pdf", io.BytesIO(b"not a pdf"), "application/pdf")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("не является PDF", response.json()["detail"])

    def test_uploads_pdf_and_exposes_job_status(self) -> None:
        with TemporaryDirectory() as temporary_directory, patch.object(
            web_app, "FEEDBACK_ROOT", Path(temporary_directory)
        ):
            response = self.upload_valid_pdf()

            self.assertEqual(response.status_code, 201)
            payload = response.json()
            self.assertEqual(payload["status"], "uploaded")
            status = self.client.get(f"/api/jobs/{payload['job_id']}").json()
            self.assertEqual(status["filename"], "drawing.pdf")
            self.assertEqual(status["status"], "legend_queued")
            self.assertNotIn("pdf_path", status)
            with web_app.jobs_lock:
                stored_job = web_app.jobs[payload["job_id"]]
                stored_pdf = Path(stored_job["pdf_path"])
                output_dir = Path(stored_job["output_dir"])
            session_id = self.client.cookies.get(web_app.SESSION_COOKIE)
            self.assertEqual(output_dir, Path(temporary_directory) / session_id)
            self.assertTrue(output_dir.is_dir())
            self.assertEqual(stored_pdf.name, "drawing.pdf")
            self.assertTrue(stored_pdf.exists())

    def test_brand_assets_are_served(self) -> None:
        logo = self.client.get("/logo.png")
        marble = self.client.get("/static/marble-background.png")

        self.assertEqual(logo.status_code, 200)
        self.assertEqual(logo.headers["content-type"], "image/png")
        self.assertEqual(marble.status_code, 200)
        self.assertEqual(marble.headers["content-type"], "image/png")

    def test_prometheus_metrics_are_exposed(self) -> None:
        response = self.client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/plain", response.headers["content-type"])
        self.assertIn("pipestone_documents_uploaded_total", response.text)
        self.assertIn("pipestone_processing_duration_seconds", response.text)
        self.assertIn("pipestone_process_cpu_seconds_total", response.text)
        self.assertIn("pipestone_process_resident_memory_bytes", response.text)
        self.assertIn("pipestone_system_memory_percent", response.text)

    def test_homepage_displays_alpha_notice(self) -> None:
        homepage = self.client.get("/")

        self.assertEqual(homepage.status_code, 200)
        self.assertIn("Альфа-версия", homepage.text)
        self.assertIn("тестовой эксплуатации", homepage.text)

    def test_feedback_is_saved_and_submitted_to_telegram_with_pdf(self) -> None:
        with TemporaryDirectory() as temporary_directory, patch.object(
            web_app, "FEEDBACK_ROOT", Path(temporary_directory)
        ), patch.object(
            web_app,
            "send_telegram_feedback",
            return_value={"ok": True, "result": {"message_id": 123456}},
        ) as send_telegram:
            upload = self.upload_valid_pdf().json()
            with web_app.jobs_lock:
                pdf_path = Path(web_app.jobs[upload["job_id"]]["pdf_path"])
            response = self.client.post(
                "/api/feedback",
                json={"description": "Не загружается чертёж"},
            )

            self.assertEqual(response.status_code, 201)
            payload = response.json()
            session_id = self.client.cookies.get(web_app.SESSION_COOKIE)
            session_directory = Path(temporary_directory) / session_id
            self.assertTrue(session_directory.is_dir())
            feedback_files = list(session_directory.glob("feedback_*.txt"))
            self.assertEqual(len(feedback_files), 1)
            contents = feedback_files[0].read_text(encoding="utf-8")
            self.assertIn("Не загружается чертёж", contents)
            self.assertIn(f"session_id: {session_id}", contents)
            self.assertEqual(payload["status"], "submitted")
            self.assertEqual(payload["telegram_message_id"], 123456)
            send_telegram.assert_called_once_with("Не загружается чертёж", session_id, pdf_path)

    def test_feedback_reports_telegram_failure_but_keeps_local_file(self) -> None:
        with TemporaryDirectory() as temporary_directory, patch.object(
            web_app, "FEEDBACK_ROOT", Path(temporary_directory)
        ), patch.object(
            web_app, "send_telegram_feedback", side_effect=RuntimeError("API недоступен")
        ):
            response = self.client.post(
                "/api/feedback",
                json={"description": "Ошибка"},
            )

            self.assertEqual(response.status_code, 502)
            session_id = self.client.cookies.get(web_app.SESSION_COOKIE)
            feedback_files = list((Path(temporary_directory) / session_id).glob("feedback_*.txt"))
            self.assertEqual(len(feedback_files), 1)

    def test_feedback_rejects_blank_description(self) -> None:
        response = self.client.post("/api/feedback", json={"description": "   "})

        self.assertEqual(response.status_code, 422)

    def test_job_status_reports_running_and_finished_operation_durations(self) -> None:
        upload = self.upload_valid_pdf().json()
        with web_app.jobs_lock:
            job = web_app.jobs[upload["job_id"]]
            job["legend_started_at"] = 100.0
            job["legend_finished_at"] = 112.3
            job["calculation_started_at"] = 200.0
            job["calculation_finished_at"] = None

        with patch.object(web_app.time, "time", return_value=207.8):
            status = self.client.get(f"/api/jobs/{upload['job_id']}").json()

        self.assertEqual(status["legend_elapsed_seconds"], 12.3)
        self.assertEqual(status["calculation_elapsed_seconds"], 7.8)
        self.assertNotIn("legend_started_at", status)
        self.assertNotIn("calculation_started_at", status)

    def test_cookie_session_restores_current_job(self) -> None:
        upload = self.upload_valid_pdf().json()
        with web_app.jobs_lock:
            job = web_app.jobs[upload["job_id"]]
            job["status"] = "running"
            job["progress"] = 55
            job["message"] = "Поиск страниц в файле"
            job["legends"] = [{"page": 1, "name": "Гранит", "score": 1.0}]

        response = self.client.get("/api/session")

        self.assertEqual(response.status_code, 200)
        restored = response.json()["job"]
        self.assertEqual(restored["job_id"], upload["job_id"])
        self.assertEqual(restored["progress"], 55)
        self.assertEqual(restored["legends"][0]["name"], "Гранит")
        cookie = self.client.cookies.get(web_app.SESSION_COOKIE)
        self.assertTrue(cookie)

    def test_another_cookie_session_cannot_read_job(self) -> None:
        upload = self.upload_valid_pdf().json()
        other_client = TestClient(web_app.app)

        response = other_client.get(f"/api/jobs/{upload['job_id']}")

        self.assertEqual(response.status_code, 404)

    def test_calculation_endpoint_queues_background_job(self) -> None:
        upload = self.upload_valid_pdf().json()
        self.mark_legend_ready(upload["job_id"])

        with patch.object(web_app, "calculate_job") as calculate:
            response = self.client.post(f"/api/jobs/{upload['job_id']}/calculate")

        self.assertEqual(response.status_code, 202)
        calculate.assert_called_once_with(upload["job_id"])

    def test_rejects_calculation_while_server_is_busy(self) -> None:
        upload = self.upload_valid_pdf().json()
        self.mark_legend_ready(upload["job_id"])
        self.assertTrue(web_app.claim_calculation_slot("another-job"))

        response = self.client.post(f"/api/jobs/{upload['job_id']}/calculate")

        self.assertEqual(response.status_code, 409)
        self.assertIn("Сервер занят", response.json()["detail"])

    def test_owner_can_request_legend_validation_cancellation(self) -> None:
        upload = self.upload_valid_pdf().json()

        response = self.client.post(f"/api/jobs/{upload['job_id']}/cancel")

        self.assertEqual(response.status_code, 202)
        with web_app.jobs_lock:
            self.assertTrue(web_app.jobs[upload["job_id"]]["cancel_requested"])

    def test_cancelled_calculation_stops_before_pipeline_call(self) -> None:
        upload = self.upload_valid_pdf().json()
        self.mark_legend_ready(upload["job_id"])
        with web_app.jobs_lock:
            web_app.jobs[upload["job_id"]]["status"] = "running"
            web_app.jobs[upload["job_id"]]["cancel_requested"] = True
        self.assertTrue(web_app.claim_calculation_slot(upload["job_id"], "calculation"))

        with patch.object(web_app, "analyze_pdf_file") as analyze:
            web_app.calculate_job(upload["job_id"])

        analyze.assert_not_called()
        with web_app.jobs_lock:
            self.assertEqual(web_app.jobs[upload["job_id"]]["status"], "calculation_cancelled")
        self.assertFalse(web_app.calculation_server_status()["busy"])

    def test_successful_calculation_sends_telegram_notification(self) -> None:
        upload = self.upload_valid_pdf().json()
        self.mark_legend_ready(upload["job_id"])
        self.assertTrue(web_app.claim_calculation_slot(upload["job_id"], "calculation"))
        with web_app.jobs_lock:
            job = web_app.jobs[upload["job_id"]]
            job["calculation_started_at"] = web_app.time.time()
            session_id = str(job["session_id"])
            legend_analysis = {
                "analysis_dpi": 220,
                "legend_pages": [1],
                "material_lines": [{"page": 1, "text": "Гранит", "bbox": [1, 2, 3, 4]}],
                "pattern_matches": [{"page": 1, "line_text": "Гранит"}],
            }
            job["legend_analysis"] = legend_analysis

        analysis_result = {
            "legend_pages": [1],
            "hatch_pages": [2],
            "hatch_pattern_box_pages": [],
            "area_calculation": {"pages": [], "total_area_m2": 0.0},
        }
        with patch.object(web_app, "analyze_pdf_file", return_value=analysis_result) as analyze, patch.object(
            web_app, "send_telegram_processing_success", return_value={"ok": True}
        ) as notify:
            web_app.calculate_job(upload["job_id"])

        self.assertEqual(
            analyze.call_args.kwargs["precomputed_legend_analysis"],
            legend_analysis,
        )
        notify.assert_called_once()
        filename, elapsed_seconds, notified_session_id = notify.call_args.args
        self.assertEqual(filename, "drawing.pdf")
        self.assertGreaterEqual(elapsed_seconds, 0)
        self.assertEqual(notified_session_id, session_id)
        with web_app.jobs_lock:
            self.assertEqual(web_app.jobs[upload["job_id"]]["status"], "completed")
            self.assertTrue(all(stage["status"] == "completed" for stage in web_app.jobs[upload["job_id"]]["stages"]))

    def test_legend_search_makes_job_ready_and_returns_legends(self) -> None:
        upload = self.upload_valid_pdf().json()
        legend_result = {
            "legends": [{"page": 2, "name": "Натуральный гранит", "score": 0.95}],
            "analysis_dpi": 220,
            "legend_pages": [2],
            "material_lines": [{"page": 2, "text": "Натуральный гранит", "bbox": [1, 2, 3, 4]}],
            "pattern_matches": [{"page": 2, "line_text": "Натуральный гранит"}],
            "log_file": "legend.log",
        }
        with patch.object(web_app, "analyze_pdf_legends", return_value=legend_result) as analyze:
            web_app.search_legends_job(upload["job_id"])

        status = self.client.get(f"/api/jobs/{upload['job_id']}").json()
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["legends"][0]["name"], "Натуральный гранит")
        self.assertNotIn("legend_analysis", status)
        with web_app.jobs_lock:
            self.assertEqual(web_app.jobs[upload["job_id"]]["legend_analysis"]["analysis_dpi"], 220)
        analyze.assert_called_once()

    def test_legend_search_reports_missing_stone_information(self) -> None:
        upload = self.upload_valid_pdf().json()
        with patch.object(web_app, "analyze_pdf_legends", return_value={"legends": [], "log_file": ""}):
            web_app.search_legends_job(upload["job_id"])

        status = self.client.get(f"/api/jobs/{upload['job_id']}").json()
        self.assertEqual(status["status"], "no_legends")
        self.assertEqual(status["legend_message"], "Выбранный файл не содержит сведений о камне")

    def test_material_groups_aggregate_page_areas(self) -> None:
        groups = web_app.material_groups(
            {
                "pages": [
                    {"page": 1, "unique_elements": {"Гранит": {"count": 2, "horizontal_dimensions": ["1000"], "vertical_dimensions": ["500"], "total_area_m2": 1.0}}},
                    {"page": 3, "unique_elements": {"Гранит": {"count": 1, "horizontal_dimensions": ["1000"], "vertical_dimensions": ["500"], "total_area_m2": 0.5}}},
                ]
            }
        )

        self.assertEqual(groups[0]["count"], 3)
        self.assertEqual(groups[0]["area_m2"], 1.5)
        self.assertEqual(groups[0]["pages"], [1, 3])

    def test_merge_area_buckets_matches_normalized_name_and_exact_dimensions(self) -> None:
        merged = web_app.merge_area_buckets(
            {
                "pages": [
                    {
                        "page": 1,
                        "unique_elements": {
                            "Гранит": {
                                "count": 2,
                                "horizontal_dimensions": ["1000"],
                                "vertical_dimensions": ["500"],
                                "total_area_m2": 1.0,
                            }
                        },
                    },
                    {
                        "page": 2,
                        "unique_elements": {
                            "  гранит ": {
                                "count": 3,
                                "horizontal_dimensions": ["1000.0"],
                                "vertical_dimensions": ["500,0"],
                                "total_area_m2": 1.5,
                            }
                        },
                    },
                    {
                        "page": 3,
                        "unique_elements": {
                            "Гранит": {
                                "count": 1,
                                "horizontal_dimensions": ["1200"],
                                "vertical_dimensions": ["500"],
                                "total_area_m2": 0.6,
                            }
                        },
                    },
                ]
            }
        )

        self.assertEqual(len(merged["groups"]), 2)
        same_size = next(group for group in merged["groups"] if group["horizontal_dimensions"] == ["1000"])
        self.assertEqual(same_size["count"], 5)
        self.assertEqual(same_size["area_m2"], 2.5)
        self.assertEqual(same_size["pages"], [1, 2])
        self.assertEqual(merged["total_area_m2"], 3.1)

    def test_web_stages_include_bucket_merge_after_area(self) -> None:
        stages = web_app.initial_stages()

        self.assertEqual([stage["id"] for stage in stages], ["legend", "symbol", "pages", "area", "merge"])
