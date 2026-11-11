import os
from pathlib import Path
import httpx
from nicegui import ui
from nicegui.events import UploadEventArguments
from src.config import settings

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
POLL_INTERVAL_SECONDS = 1.5
TERMINAL_STATUSES = {"completed", "failed", "success", "done"}

TIER_LABELS = {
    "Tier 0: Embedded Subtitle Match":      ("T0", "Subtitle"),
    "Tier 1: STT Acoustic Match":           ("T1", "Speech"),
    "Tier 2: Sparse Timeline OCR Match":    ("T2", "OCR Scan"),
    "Tier 3: Dense Onset OCR Confirmation": ("T3", "OCR Dense"),
    "Tier 4: VLM Arbiter Fallback":         ("T4", "Vision AI"),
}

def _tier_badge(tier_raw):
    if not tier_raw:
        return ("-", "-")
    return TIER_LABELS.get(tier_raw, ("T?", tier_raw))

def _confidence_color(score):
    if score >= 0.85: return "text-emerald-400"
    if score >= 0.65: return "text-yellow-400"
    return "text-red-400"

def _stat_card(label, value, color="text-white"):
    with ui.column().classes("items-center gap-0"):
        ui.label(value).classes(f"text-2xl font-bold {color}")
        ui.label(label).classes("text-xs text-gray-500 uppercase tracking-widest")

@ui.page("/")
def dashboard():
    ui.page_title("Quest1 - Dialogue Detection")

    state = {
        "job_id": None, "status": None, "target_dialogue": None,
        "formatted_timestamp": None, "timestamp_seconds": None,
        "frame_number": None, "confidence_score": None,
        "tier_executed": None, "extracted_text": None,
        "url": None, "error": None, "has_frame": False,
    }
    pending_upload = {}

    def _reset_state(url, target_text):
        state.update({
            "job_id": None, "status": "processing",
            "target_dialogue": target_text, "formatted_timestamp": None,
            "timestamp_seconds": None, "frame_number": None,
            "confidence_score": None, "tier_executed": None,
            "extracted_text": None, "url": url, "error": None, "has_frame": False,
        })

    async def submit_job(url, target_text):
        _reset_state(url, target_text)
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    f"{BACKEND_URL}/api/v1/jobs",
                    json={"url": url, "target_text": target_text},
                )
                r.raise_for_status()
                data = r.json()
                state["job_id"] = data.get("job_id")
                state["status"] = data.get("status")
        except Exception as exc:
            state["status"] = "failed"
            state["error"] = str(exc)

    async def submit_uploaded_bytes(filename, file_bytes, content_type, target_text):
        _reset_state(filename, target_text)
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                response = await client.post(
                    f"{BACKEND_URL}/api/v1/jobs/upload",
                    data={"target_text": target_text},
                    files={"file": (filename, file_bytes, content_type)},
                )
                response.raise_for_status()
            data = response.json()
            state["job_id"] = data.get("job_id")
            state["status"] = data.get("status")
        except Exception as exc:
            state["status"] = "failed"
            state["error"] = str(exc)

    async def poll_job_status():
        if not state["job_id"] or state["status"] in TERMINAL_STATUSES:
            return
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{BACKEND_URL}/api/v1/jobs/{state['job_id']}")
                data = r.json()
                state["status"]              = data.get("status")
                state["formatted_timestamp"] = data.get("formatted_timestamp")
                state["timestamp_seconds"]   = data.get("timestamp_seconds")
                state["frame_number"]        = data.get("frame_number")
                state["confidence_score"]    = data.get("confidence_score")
                state["tier_executed"]       = data.get("tier_executed")
                state["extracted_text"]      = data.get("extracted_text")
                state["error"]               = data.get("error_message")
                state["has_frame"]           = bool(data.get("frame_image_path"))
        except Exception as exc:
            state["error"] = str(exc)

    ui.add_head_html("""
    <style>
      body { background:#0f1117; }
      .q-field__label { color:#9ca3af !important; }
      .q-field__native, .q-field__input { color:#f9fafb !important; }
      .q-field--outlined .q-field__control { border-color:#374151 !important; }
      .q-field--outlined.q-field--focused .q-field__control {
        border-color:#6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.25);
      }
      .nicegui-content { padding:0 !important; }
    </style>
    """)

    with ui.column().classes("w-full min-h-screen items-center pb-16 px-4"):

        with ui.column().classes("items-center mt-14 mb-10 text-center"):
            ui.label("Quest1").classes(
                "text-6xl font-black tracking-tight"
                " bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400"
                " bg-clip-text text-transparent"
            )
            ui.label("Dialogue Detection Engine").classes(
                "text-xl font-semibold text-gray-300 mt-1"
            )
            ui.label(
                "Locate any spoken or visual dialogue inside a video"
                " — timestamp, frame, and confidence score."
            ).classes("text-sm text-gray-500 mt-2 max-w-lg")

        with ui.card().classes(
            "w-full max-w-2xl rounded-2xl p-8 gap-0 border border-gray-800"
        ).style("background:#161b27"):

            ui.label("New Detection Job").classes(
                "text-lg font-semibold text-gray-200 mb-6"
            )

            video_url = (
                ui.input(placeholder="https://youtu.be/...  or any video URL")
                .props('outlined dense label="Video URL" color="indigo"')
                .classes("w-full")
            )
            ui.space().classes("h-3")
            dialogue_text = (
                ui.input(placeholder='e.g. "in a world where"')
                .props('outlined dense label="Target Dialogue" color="indigo"')
                .classes("w-full mt-3")
            )

            ui.label("Or upload a video file (MP4, MOV, MKV, WebM)").classes(
                "text-xs text-gray-500 mt-4"
            )

            async def handle_upload(event: UploadEventArguments):
                text = dialogue_text.value.strip()
                uploaded_file = event.file
                file_bytes = await uploaded_file.read()
                content_type = uploaded_file.content_type or "video/mp4"
                filename = Path(uploaded_file.name).name

                if not text:
                    pending_upload.update({
                        "filename": filename,
                        "file_bytes": file_bytes,
                        "content_type": content_type,
                    })
                    ui.notify(
                        "Video ready. Enter Target Dialogue and click Run Detection.",
                        type="info",
                        position="top",
                    )
                    return

                await submit_uploaded_bytes(filename, file_bytes, content_type, text)
                refresh_ui()

            def handle_rejected():
                ui.notify(
                    "File rejected — ensure it is a valid video under 250 MB.",
                    type="negative",
                    position="top",
                )

            ui.upload(
                on_upload=handle_upload,
                on_rejected=handle_rejected,
                auto_upload=True,
            ).props('accept="video/*" max-file-size="262144000"').classes("w-full mt-2")

            run_btn = (
                ui.button("Run Detection")
                .classes("w-full mt-6 py-3 rounded-xl text-base font-semibold text-white")
                .style("background: linear-gradient(135deg,#6366f1,#8b5cf6); transition: opacity .2s;")
                .props("unelevated")
            )

        status_card = ui.card().classes(
            "w-full max-w-2xl rounded-2xl p-6 mt-4 border border-gray-800"
        ).style("background:#161b27; display:none")

        result_card = ui.card().classes(
            "w-full max-w-2xl rounded-2xl p-8 mt-4 border border-gray-800"
        ).style("background:#161b27; display:none")

        with status_card:
            with ui.row().classes("w-full items-center gap-3"):
                status_dot = ui.element("div").classes(
                    "w-2.5 h-2.5 rounded-full bg-yellow-400"
                ).style("flex-shrink:0")
                status_label = ui.label("Processing...").classes(
                    "text-sm font-semibold text-gray-200"
                )
                ui.space()
                job_id_label = ui.label("").classes("text-xs text-gray-600 font-mono")
            error_label = ui.label("").classes("text-sm text-red-400 mt-3 break-all")

        with result_card:
            result_inner = ui.column().classes("w-full gap-5")

        def refresh_ui():
            s = state["status"]
            status_card.style(
                "background:#161b27;" + ("display:block" if s else "display:none")
            )
            if s in ("completed", "success", "done"):
                status_dot.classes("bg-emerald-400", remove="bg-yellow-400 bg-red-400")
                status_label.text = "Completed"
                status_label.classes("text-emerald-300", remove="text-gray-200 text-red-300")
            elif s == "failed":
                status_dot.classes("bg-red-400", remove="bg-yellow-400 bg-emerald-400")
                status_label.text = "Failed"
                status_label.classes("text-red-300", remove="text-gray-200 text-emerald-300")
            else:
                status_dot.classes("bg-yellow-400 animate-pulse",
                                   remove="bg-emerald-400 bg-red-400")
                status_label.text = "Processing..."
                status_label.classes("text-gray-200", remove="text-emerald-300 text-red-300")

            job_id_label.text = state["job_id"] or ""
            error_label.text  = state["error"]  or ""
            result_inner.clear()

            if s == "completed":
                result_card.style("background:#161b27; display:block")
                score = float(state["confidence_score"] or 0.0)
                tier_code, tier_name = _tier_badge(state["tier_executed"])
                with result_inner:
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("Detection Result").classes(
                            "text-lg font-semibold text-gray-200"
                        )
                        ui.badge(f"OK  {tier_code} - {tier_name}").classes(
                            "bg-indigo-600 text-white text-xs px-3 py-1 rounded-full"
                        )
                    with ui.row().classes(
                        "w-full justify-around mt-2 py-4 rounded-xl"
                    ).style("background:#0f1117"):
                        _stat_card("Timestamp", state["formatted_timestamp"] or "-", "text-yellow-300")
                        _stat_card(
                            "Frame",
                            f"#{state['frame_number']}" if state["frame_number"] is not None else "-",
                            "text-blue-300",
                        )
                        _stat_card("Confidence", f"{score * 100:.1f}%", _confidence_color(score))
                        _stat_card("Tier", tier_code, "text-purple-300")

                    if state["extracted_text"]:
                        with ui.column().classes("w-full rounded-xl p-4").style("background:#0f1117"):
                            ui.label("Detected Text").classes(
                                "text-xs text-gray-500 uppercase tracking-widest mb-1"
                            )
                            ui.label(f'"{state["extracted_text"]}"').classes(
                                "text-gray-200 text-sm italic"
                            )

                    if state["has_frame"]:
                        with ui.column().classes("w-full rounded-xl overflow-hidden mt-1"):
                            ui.label("Extracted Frame").classes(
                                "text-xs text-gray-500 uppercase tracking-widest mb-2"
                            )
                            ui.image(
                                f"{BACKEND_URL}/api/v1/jobs/{state['job_id']}/frame"
                            ).classes("w-full rounded-xl").style("border:1px solid #374151")

            elif s == "failed":
                result_card.style("background:#161b27; display:block")
                with result_inner:
                    with ui.column().classes("w-full items-center py-8 gap-3"):
                        ui.label("Detection Failed").classes("text-lg font-semibold text-red-400")
                        if state["error"]:
                            ui.label(state["error"]).classes(
                                "text-xs text-gray-500 text-center max-w-sm break-all"
                            )
            else:
                result_card.style("background:#161b27; display:none")

        async def run_detection():
            url  = video_url.value.strip()
            text = dialogue_text.value.strip()

            if not text:
                ui.notify("Please enter a Target Dialogue.", type="warning", position="top")
                return

            if not url and pending_upload:
                run_btn.props("loading")
                upload = pending_upload.copy()
                pending_upload.clear()
                await submit_uploaded_bytes(
                    upload["filename"], upload["file_bytes"], upload["content_type"], text,
                )
                run_btn.props(remove="loading")
                refresh_ui()
                return

            if not url:
                ui.notify(
                    "Please enter a Video URL or upload a video file.",
                    type="warning", position="top",
                )
                return

            run_btn.props("loading")
            await submit_job(url, text)
            run_btn.props(remove="loading")
            refresh_ui()

        run_btn.on_click(run_detection)

        async def update_loop():
            await poll_job_status()
            refresh_ui()

        ui.timer(POLL_INTERVAL_SECONDS, update_loop)


ui.run(
    title="Quest1 - Dialogue Detection",
    port=8080,
    reload=False,
    dark=True,
)