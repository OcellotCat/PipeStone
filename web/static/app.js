const fileInput = document.querySelector("#fileInput");
const selectButton = document.querySelector("#selectButton");
const dropZone = document.querySelector("#dropZone");
const uploadBadge = document.querySelector("#uploadBadge");
const uploadError = document.querySelector("#uploadError");
const uploadProgressBlock = document.querySelector("#uploadProgressBlock");
const uploadBar = document.querySelector("#uploadBar");
const uploadPercent = document.querySelector("#uploadPercent");
const uploadLabel = document.querySelector("#uploadLabel");
const fileSummary = document.querySelector("#fileSummary");
const fileName = document.querySelector("#fileName");
const fileSize = document.querySelector("#fileSize");
const calculateButton = document.querySelector("#calculateButton");
const calculationIdle = document.querySelector("#calculationIdle");
const calculationProgress = document.querySelector("#calculationProgress");
const calculationBar = document.querySelector("#calculationBar");
const calculationPercent = document.querySelector("#calculationPercent");
const calculationMessage = document.querySelector("#calculationMessage");
const calculationError = document.querySelector("#calculationError");
const results = document.querySelector("#results");
const serverStatus = document.querySelector("#serverStatus");
const serverStatusText = document.querySelector("#serverStatusText");
const legendPanel = document.querySelector("#legendPanel");
const legendSearchProgress = document.querySelector("#legendSearchProgress");
const legendMessage = document.querySelector("#legendMessage");
const legendPercent = document.querySelector("#legendPercent");
const legendBar = document.querySelector("#legendBar");
const legendList = document.querySelector("#legendList");
const legendCount = document.querySelector("#legendCount");
const noLegends = document.querySelector("#noLegends");
const cancelLegendButton = document.querySelector("#cancelLegendButton");
const cancelCalculationButton = document.querySelector("#cancelCalculationButton");
const legendDuration = document.querySelector("#legendDuration");
const calculationDuration = document.querySelector("#calculationDuration");
const feedbackToggle = document.querySelector("#feedbackToggle");
const feedbackDrawer = document.querySelector("#feedbackDrawer");
const feedbackForm = document.querySelector("#feedbackForm");
const feedbackClose = document.querySelector("#feedbackClose");
const feedbackDescription = document.querySelector("#feedbackDescription");
const feedbackSubmit = document.querySelector("#feedbackSubmit");
const feedbackMessage = document.querySelector("#feedbackMessage");

let activeJobId = null;
let pollTimer = null;
let legendPollTimer = null;
let serverBusy = true;
let jobRunning = false;
let legendReady = false;

function updateCalculateAvailability() {
  calculateButton.disabled = !activeJobId || !legendReady || serverBusy || jobRunning;
  if (jobRunning) calculateButton.textContent = "Расчёт выполняется…";
  else if (serverBusy) calculateButton.textContent = "Сервер занят расчётом";
  else if (!legendReady) calculateButton.textContent = "Сначала найдите легенду";
  else calculateButton.textContent = results.hidden ? "Запустить расчёт площади" : "Запустить расчёт повторно";
}

async function checkServerStatus() {
  try {
    const response = await fetch("/api/server-status", { cache: "no-store" });
    const status = await response.json();
    if (!response.ok) throw new Error();
    serverBusy = Boolean(status.busy);
    serverStatus.classList.remove("checking", "free", "busy");
    serverStatus.classList.add(serverBusy ? "busy" : "free");
    serverStatusText.textContent = serverBusy
      ? (status.operation === "legend_search"
          ? "Идёт поиск легенды"
          : (status.is_current_session ? "Идёт ваш расчёт" : "Сервер занят — идёт расчёт"))
      : "Сервер свободен";
  } catch {
    serverBusy = true;
    serverStatus.classList.remove("free", "busy");
    serverStatus.classList.add("checking");
    serverStatusText.textContent = "Сервер недоступен";
  }
  updateCalculateAvailability();
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} КБ`;
  return `${(bytes / 1024 / 1024).toFixed(1)} МБ`;
}

function showError(element, message) {
  element.textContent = message;
  element.hidden = false;
}

function setFeedbackOpen(open) {
  feedbackDrawer.classList.toggle("open", open);
  feedbackDrawer.setAttribute("aria-hidden", String(!open));
  feedbackToggle.setAttribute("aria-expanded", String(open));
  if (open) setTimeout(() => feedbackDescription.focus(), 220);
}

async function submitFeedback(event) {
  event.preventDefault();
  const description = feedbackDescription.value.trim();
  if (!description) {
    feedbackMessage.textContent = "Опишите проблему";
    feedbackMessage.className = "feedback-message error";
    feedbackDescription.focus();
    return;
  }

  feedbackSubmit.disabled = true;
  feedbackSubmit.textContent = "Отправка…";
  feedbackMessage.textContent = "";
  feedbackMessage.className = "feedback-message";
  try {
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ description }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Не удалось отправить сообщение");
    feedbackDescription.value = "";
    feedbackMessage.textContent = "Спасибо, сообщение отправлено";
    feedbackMessage.className = "feedback-message success";
  } catch (error) {
    feedbackMessage.textContent = error.message;
    feedbackMessage.className = "feedback-message error";
  } finally {
    feedbackSubmit.disabled = false;
    feedbackSubmit.textContent = "Отправить";
  }
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remainder = total % 60;
  const clock = [minutes, remainder].map((value) => String(value).padStart(2, "0")).join(":");
  return hours ? `${String(hours).padStart(2, "0")}:${clock}` : clock;
}

function updateOperationDurations(job) {
  const legendSeconds = job.legend_elapsed_seconds;
  legendDuration.hidden = legendSeconds == null;
  if (legendSeconds != null) {
    legendDuration.textContent = `Время поиска легенды: ${formatDuration(legendSeconds)}`;
  }

  const calculationSeconds = job.calculation_elapsed_seconds;
  calculationDuration.hidden = calculationSeconds == null;
  if (calculationSeconds != null) {
    calculationDuration.textContent = `Время расчёта: ${formatDuration(calculationSeconds)}`;
  }
}

function clearUploadState() {
  activeJobId = null;
  legendReady = false;
  if (legendPollTimer) clearInterval(legendPollTimer);
  legendPollTimer = null;
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
  uploadError.hidden = true;
  fileSummary.hidden = true;
  legendPanel.hidden = true;
  results.hidden = true;
  calculationError.hidden = true;
  jobRunning = false;
  calculationIdle.hidden = false;
  calculationIdle.textContent = "Дождитесь загрузки файла и завершения поиска легенды.";
  calculationProgress.hidden = true;
  calculationBar.style.width = "0%";
  calculationPercent.textContent = "0%";
  calculationMessage.textContent = "Подготовка";
  legendDuration.hidden = true;
  calculationDuration.hidden = true;
  cancelCalculationButton.hidden = true;
  cancelCalculationButton.disabled = false;
  renderStages([
    { id: "legend", status: "pending" }, { id: "symbol", status: "pending" },
    { id: "pages", status: "pending" }, { id: "area", status: "pending" },
  ]);
  document.querySelector("#resultsBody").replaceChildren();
  document.querySelector("#totalArea").textContent = "0.000";
  document.querySelector("#resultMeta").textContent = "";
  document.querySelector("#resultWarning").hidden = true;
  updateCalculateAvailability();
  uploadBadge.textContent = "Файл не выбран";
  uploadBadge.classList.remove("success");
}

function validateFile(file) {
  if (!file) return "Файл не выбран.";
  if (!file.name.toLowerCase().endsWith(".pdf")) return "Неверное расширение. Выберите файл .pdf.";
  if (file.type && file.type !== "application/pdf" && file.type !== "application/x-pdf") return "Выбранный файл не имеет PDF-формат.";
  if (file.size === 0) return "Нельзя загрузить пустой файл.";
  if (file.size > 100 * 1024 * 1024) return "Размер PDF не должен превышать 100 МБ.";
  return "";
}

function uploadFile(file) {
  const error = validateFile(file);
  if (error) {
    showError(uploadError, error);
    fileInput.value = "";
    return;
  }
  clearUploadState();

  uploadProgressBlock.hidden = false;
  uploadLabel.textContent = "Загрузка файла";
  uploadBar.style.width = "0%";
  uploadPercent.textContent = "0%";
  uploadBadge.textContent = "Загрузка";

  const payload = new FormData();
  payload.append("file", file);
  const request = new XMLHttpRequest();
  request.open("POST", "/api/upload");
  request.upload.addEventListener("progress", (event) => {
    if (!event.lengthComputable) return;
    const percent = Math.round((event.loaded / event.total) * 100);
    uploadBar.style.width = `${percent}%`;
    uploadPercent.textContent = `${percent}%`;
  });
  request.addEventListener("load", () => {
    let response = {};
    try { response = JSON.parse(request.responseText); } catch { response = {}; }
    if (request.status < 200 || request.status >= 300) {
      uploadProgressBlock.hidden = true;
      uploadBadge.textContent = "Ошибка";
      showError(uploadError, response.detail || "Не удалось загрузить файл.");
      return;
    }
    activeJobId = response.job_id;
    uploadBar.style.width = "100%";
    uploadPercent.textContent = "100%";
    uploadLabel.textContent = "Загрузка завершена";
    uploadBadge.textContent = "Готово";
    uploadBadge.classList.add("success");
    fileName.textContent = response.filename;
    fileSize.textContent = formatBytes(response.size);
    fileSummary.hidden = false;
    legendPanel.hidden = false;
    legendSearchProgress.hidden = false;
    legendList.hidden = true;
    noLegends.hidden = true;
    legendCount.textContent = "0";
    legendBar.style.width = "0%";
    legendPercent.textContent = "0%";
    legendMessage.textContent = "Поиск легенды поставлен в очередь";
    cancelLegendButton.hidden = false;
    cancelLegendButton.disabled = false;
    calculationIdle.textContent = "Дождитесь завершения поиска легенды.";
    legendPollTimer = setInterval(pollLegendStatus, 1200);
    pollLegendStatus();
    checkServerStatus();
  });
  request.addEventListener("error", () => {
    uploadProgressBlock.hidden = true;
    uploadBadge.textContent = "Ошибка";
    showError(uploadError, "Сетевая ошибка при загрузке файла.");
  });
  request.send(payload);
}

function renderLegends(legends) {
  legendList.replaceChildren();
  legends.forEach((legend) => {
    const item = document.createElement("div");
    item.className = "legend-item";
    const page = document.createElement("span");
    page.className = "legend-page";
    page.textContent = `стр. ${legend.page}`;
    const name = document.createElement("span");
    name.className = "legend-name";
    name.textContent = legend.name || "Легенда без названия";
    const score = document.createElement("span");
    score.className = "legend-score";
    score.textContent = `совпадение ${Math.round(Number(legend.score || 0) * 100)}%`;
    item.append(page, name, score);
    legendList.append(item);
  });
  legendCount.textContent = String(legends.length);
  legendList.hidden = legends.length === 0;
  noLegends.hidden = legends.length !== 0;
}

function showRestoredFile(job) {
  activeJobId = job.job_id;
  uploadProgressBlock.hidden = false;
  uploadBar.style.width = "100%";
  uploadPercent.textContent = "100%";
  uploadLabel.textContent = "Загрузка завершена";
  uploadBadge.textContent = "Готово";
  uploadBadge.classList.add("success");
  fileName.textContent = job.filename;
  fileSize.textContent = formatBytes(job.size);
  fileSummary.hidden = false;
  legendPanel.hidden = false;
  legendCount.textContent = String((job.legends || []).length);
}

function restoreJob(job) {
  showRestoredFile(job);
  updateOperationDurations(job);
  const legendInProgress = ["legend_queued", "legend_waiting", "legend_search"].includes(job.status);
  if (legendInProgress) {
    legendReady = false;
    legendSearchProgress.hidden = false;
    legendBar.style.width = `${job.legend_progress || 0}%`;
    legendPercent.textContent = `${job.legend_progress || 0}%`;
    legendMessage.textContent = job.legend_message || "Поиск легенды";
    legendList.hidden = true;
    noLegends.hidden = true;
    cancelLegendButton.hidden = false;
    cancelLegendButton.disabled = false;
    calculationIdle.textContent = "Дождитесь завершения поиска легенды.";
    legendPollTimer = setInterval(pollLegendStatus, 1200);
    pollLegendStatus();
    return;
  }

  legendSearchProgress.hidden = true;
  cancelLegendButton.hidden = true;
  renderLegends(job.legends || []);
  legendReady = (job.legends || []).length > 0 && job.status !== "legend_failed";
  if (!legendReady && job.status !== "failed") {
    noLegends.hidden = false;
    noLegends.textContent = job.status === "legend_failed"
      ? (job.legend_message || "Не удалось выполнить поиск легенды")
      : "Выбранный файл не содержит сведений о камне";
  }

  if (["queued", "running"].includes(job.status)) {
    jobRunning = true;
    calculationIdle.hidden = true;
    calculationProgress.hidden = false;
    calculationBar.style.width = `${job.progress || 0}%`;
    calculationPercent.textContent = `${job.progress || 0}%`;
    calculationMessage.textContent = job.message || "Расчёт выполняется";
    renderStages(job.stages || []);
    pollTimer = setInterval(pollStatus, 1200);
    pollStatus();
  } else if (job.status === "completed" && job.result) {
    jobRunning = false;
    calculationIdle.hidden = true;
    calculationProgress.hidden = false;
    calculationBar.style.width = "100%";
    calculationPercent.textContent = "100%";
    calculationMessage.textContent = job.message;
    renderStages(job.stages || []);
    showResults(job.result);
  } else if (job.status === "failed") {
    jobRunning = false;
    calculationIdle.hidden = true;
    showError(calculationError, job.error || job.message);
  } else {
    calculationIdle.textContent = legendReady
      ? "Легенды найдены. Документ готов к расчёту площади."
      : "Расчёт площади недоступен: сведения о камне не найдены.";
  }
  updateCalculateAvailability();
}

async function restoreSession() {
  try {
    const response = await fetch("/api/session", { cache: "no-store" });
    const session = await response.json();
    if (!response.ok) throw new Error();
    if (session.job) restoreJob(session.job);
  } catch {
    showError(uploadError, "Не удалось восстановить предыдущую сессию.");
  }
  checkServerStatus();
}

async function pollLegendStatus() {
  const jobId = activeJobId;
  if (!jobId) return;
  try {
    const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
    const job = await response.json();
    if (jobId !== activeJobId) return;
    if (!response.ok) throw new Error(job.detail || "Не удалось получить статус поиска легенды.");
    updateOperationDurations(job);
    legendBar.style.width = `${job.legend_progress || 0}%`;
    legendPercent.textContent = `${job.legend_progress || 0}%`;
    legendMessage.textContent = job.legend_message || "Поиск легенды";
    if (["ready", "no_legends", "legend_failed", "legend_cancelled"].includes(job.status)) {
      clearInterval(legendPollTimer);
      legendPollTimer = null;
      legendSearchProgress.hidden = true;
      cancelLegendButton.hidden = true;
      legendReady = job.status === "ready" && job.legends.length > 0;
      renderLegends(job.legends || []);
      if (job.status === "legend_failed" || job.status === "legend_cancelled") {
        noLegends.hidden = false;
        noLegends.textContent = job.legend_message || "Поиск легенды остановлен";
      } else if (!legendReady) {
        noLegends.textContent = "Выбранный файл не содержит сведений о камне";
      }
      calculationIdle.textContent = legendReady
        ? "Легенды найдены. Документ готов к расчёту площади."
        : "Расчёт площади недоступен: сведения о камне не найдены.";
      setTimeout(checkServerStatus, 500);
    }
  } catch (error) {
    if (jobId !== activeJobId) return;
    clearInterval(legendPollTimer);
    legendPollTimer = null;
    legendSearchProgress.hidden = true;
    noLegends.hidden = false;
    noLegends.textContent = error.message;
    legendReady = false;
    updateCalculateAvailability();
  }
}

function renderStages(stages) {
  stages.forEach((stage) => {
    const item = document.querySelector(`[data-stage="${stage.id}"]`);
    item.classList.remove("running", "completed");
    if (stage.status !== "pending") item.classList.add(stage.status);
  });
}

function dimensions(group) {
  const horizontal = group.horizontal_dimensions.length ? group.horizontal_dimensions.join(", ") : "?";
  const vertical = group.vertical_dimensions.length ? group.vertical_dimensions.join(", ") : "?";
  return `${horizontal} × ${vertical}`;
}

function hasKnownDimensions(group) {
  const hasNumericValue = (values) => Array.isArray(values)
    && values.some((value) => /^\d+$/.test(String(value).trim()));
  return hasNumericValue(group.horizontal_dimensions)
    && hasNumericValue(group.vertical_dimensions);
}

function showResults(result) {
  const body = document.querySelector("#resultsBody");
  body.replaceChildren();
  const visibleGroups = result.groups.filter(hasKnownDimensions);
  if (!visibleGroups.length) {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="5" class="empty-cell">Группы материалов с указанными размерами не найдены</td>';
    body.append(row);
  } else {
    visibleGroups.forEach((group) => {
      const row = document.createElement("tr");
      [group.name, group.count, dimensions(group), group.pages.join(", "), group.area_m2.toFixed(3)].forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      body.append(row);
    });
  }
  document.querySelector("#totalArea").textContent = Number(result.total_area_m2).toFixed(3);
  const pages = result.hatch_pattern_box_pages.length ? result.hatch_pattern_box_pages.join(", ") : "—";
  document.querySelector("#resultMeta").textContent = `Обработанные страницы: ${pages}`;
  const warning = document.querySelector("#resultWarning");
  warning.hidden = !result.warning;
  warning.textContent = result.warning || "";
  results.hidden = false;
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function pollStatus() {
  const jobId = activeJobId;
  if (!jobId) return;
  try {
    const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
    const job = await response.json();
    if (jobId !== activeJobId) return;
    if (!response.ok) throw new Error(job.detail || "Не удалось получить статус расчёта.");
    updateOperationDurations(job);
    calculationBar.style.width = `${job.progress}%`;
    calculationPercent.textContent = `${job.progress}%`;
    calculationMessage.textContent = job.message;
    renderStages(job.stages);
    if (job.status === "completed") {
      clearInterval(pollTimer);
      pollTimer = null;
      jobRunning = false;
      cancelCalculationButton.hidden = true;
      showResults(job.result);
      setTimeout(checkServerStatus, 500);
    } else if (job.status === "failed") {
      clearInterval(pollTimer);
      pollTimer = null;
      jobRunning = false;
      cancelCalculationButton.hidden = true;
      checkServerStatus();
      showError(calculationError, job.error || job.message);
    } else if (job.status === "calculation_cancelled") {
      clearInterval(pollTimer);
      pollTimer = null;
      jobRunning = false;
      cancelCalculationButton.hidden = true;
      calculationMessage.textContent = job.message || "Расчёт остановлен";
      checkServerStatus();
    }
  } catch (error) {
    if (jobId !== activeJobId) return;
    clearInterval(pollTimer);
    pollTimer = null;
    jobRunning = false;
    checkServerStatus();
    showError(calculationError, error.message);
  }
}

async function startCalculation() {
  if (!activeJobId) return;
  jobRunning = true;
  serverBusy = true;
  updateCalculateAvailability();
  serverStatus.classList.remove("checking", "free");
  serverStatus.classList.add("busy");
  serverStatusText.textContent = "Запуск расчёта…";
  cancelCalculationButton.hidden = false;
  cancelCalculationButton.disabled = false;
  calculationIdle.hidden = true;
  calculationProgress.hidden = false;
  calculationError.hidden = true;
  results.hidden = true;
  calculationBar.style.width = "1%";
  calculationPercent.textContent = "1%";
  calculationMessage.textContent = "Расчёт поставлен в очередь";
  calculationDuration.hidden = false;
  calculationDuration.textContent = "Время расчёта: 00:00";
  renderStages([
    { id: "legend", status: "pending" }, { id: "symbol", status: "pending" },
    { id: "pages", status: "pending" }, { id: "area", status: "pending" },
  ]);
  try {
    const response = await fetch(`/api/jobs/${activeJobId}/calculate`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Не удалось запустить расчёт.");
    pollTimer = setInterval(pollStatus, 1200);
    pollStatus();
  } catch (error) {
    jobRunning = false;
    cancelCalculationButton.hidden = true;
    checkServerStatus();
    showError(calculationError, error.message);
  }
}

async function cancelActiveOperation(button) {
  if (!activeJobId) return;
  button.disabled = true;
  try {
    const response = await fetch(`/api/jobs/${activeJobId}/cancel`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Не удалось остановить операцию.");
    if (button === cancelLegendButton) legendMessage.textContent = "Остановка поиска легенды…";
    else calculationMessage.textContent = "Остановка расчёта…";
  } catch (error) {
    button.disabled = false;
    showError(calculationError, error.message);
  }
}

selectButton.addEventListener("click", (event) => { event.stopPropagation(); fileInput.click(); });
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", () => uploadFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault(); dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault(); dropZone.classList.remove("dragging");
}));
dropZone.addEventListener("drop", (event) => uploadFile(event.dataTransfer.files[0]));
calculateButton.addEventListener("click", startCalculation);
cancelLegendButton.addEventListener("click", () => cancelActiveOperation(cancelLegendButton));
cancelCalculationButton.addEventListener("click", () => cancelActiveOperation(cancelCalculationButton));
feedbackToggle.addEventListener("click", () => setFeedbackOpen(!feedbackDrawer.classList.contains("open")));
feedbackClose.addEventListener("click", () => setFeedbackOpen(false));
feedbackForm.addEventListener("submit", submitFeedback);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && feedbackDrawer.classList.contains("open")) setFeedbackOpen(false);
});
restoreSession();
setInterval(checkServerStatus, 30_000);
