import httpx
from nicegui import ui

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BACKEND_URL = "http://127.0.0.1:8000"
POLL_INTERVAL_SECONDS = 1.5

TERMINAL_STATUSES = {
    "completed",
    "failed",
    "success",
    "done",
}


# ---------------------------------------------------------
# Application State
# ---------------------------------------------------------

state = {
    "job_id": None,
    "status": None,
    "target_dialogue": None,
    "formatted_timestamp": None,
    "timestamp_seconds": None,
    "confidence_score": None,
    "tier_executed": None,
    "extracted_text": None,
    "url": None,
    "error": None,
    "has_frame": False,
}


# ---------------------------------------------------------
# Backend Communication
# ---------------------------------------------------------

async def submit_job(url: str, target_text: str):
    state.update(
        {
            "job_id": None,
            "status": "processing",
            "target_dialogue": target_text,
            "formatted_timestamp": None,
            "timestamp_seconds": None,
            "confidence_score": None,
            "tier_executed": None,
            "extracted_text": None,
            "url": url,
            "error": None,
            "has_frame": False,
        }
    )

    # Fixed: Uses "url" instead of "video_url" to match backend API schema
    payload = {
        "url": url,
        "target_text": target_text,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/jobs",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            state["job_id"] = data.get("job_id")
            state["status"] = data.get("status", "processing")
            state["target_dialogue"] = data.get("target_dialogue")

    except httpx.HTTPStatusError as error:
        state["status"] = "failed"
        state["error"] = f"{error.response.status_code}: {error.response.text}"

    except Exception as error:
        state["status"] = "failed"
        state["error"] = str(error)


async def poll_job_status():
    if not state["job_id"]:
        return

    current_status = (state["status"] or "").lower()
    if current_status in TERMINAL_STATUSES:
        return

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{BACKEND_URL}/api/v1/jobs/{state['job_id']}"
            )
            response.raise_for_status()
            data = response.json()

            state["status"] = data.get("status")
            state["target_dialogue"] = data.get("target_dialogue")
            state["formatted_timestamp"] = data.get("formatted_timestamp")
            state["timestamp_seconds"] = data.get("timestamp_seconds")
            state["confidence_score"] = data.get("confidence_score")
            state["tier_executed"] = data.get("tier_executed")
            state["extracted_text"] = data.get("extracted_text")
            state["error"] = data.get("error_message")
            state["has_frame"] = bool(data.get("frame_image_path"))

    except Exception as error:
        state["error"] = str(error)


# ---------------------------------------------------------
# NiceGUI Interface
# ---------------------------------------------------------

@ui.page("/")
def dashboard():
    ui.page_title("Quest1 Dashboard")

    # Header
    with ui.column().classes("w-full items-center"):
        ui.label("Quest1 Dialogue Detection Engine").classes("text-3xl font-bold text-white")
        ui.label("AI-powered dialogue localization and frame detection").classes("text-gray-400")

    with ui.column().classes("w-full items-center mt-6"):
        # Input Card
        with ui.card().classes("w-full max-w-3xl p-6 bg-gray-900 border border-gray-800 rounded-xl shadow-2xl"):
            ui.label("Create Detection Job").classes("text-xl font-semibold text-white")

            video_url = ui.input(
                label="Video URL",
                placeholder="Enter YouTube or HLS direct stream URL",
            ).classes("w-full mt-3 text-white")

            dialogue_text = ui.input(
                label="Target Dialogue",
                placeholder="Enter exact spoken phrase or on-screen text",
            ).classes("w-full text-white")

            async def run_detection():
                if not video_url.value or not dialogue_text.value:
                    ui.notify("Enter both video URL and target dialogue", type="warning")
                    return

                await submit_job(video_url.value.strip(), dialogue_text.value.strip())
                refresh_ui()

            ui.button(
                "Run Detection",
                on_click=run_detection,
            ).classes("mt-4 w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 rounded-lg")

        # Status Card
        with ui.card().classes("w-full max-w-3xl mt-4 bg-gray-900 border border-gray-800 rounded-xl p-4"):
            ui.label("Job Status").classes("text-xl font-semibold text-white")
            job_id_label = ui.label("Job ID: -").classes("text-xs text-gray-500 font-mono mt-1")
            status_label = ui.label("Status: Waiting").classes("text-sm font-semibold text-gray-400 mt-1")
            error_label = ui.label().classes("text-red-500 font-mono text-sm mt-2")

        # Result Card
        with ui.card().classes("w-full max-w-3xl mt-4 bg-gray-900 border border-gray-800 rounded-xl p-4"):
            ui.label("Detection Result").classes("text-xl font-semibold text-white mb-2")
            result_container = ui.column().classes("w-full gap-3")

    # -----------------------------------------------------
    # UI Refresh
    # -----------------------------------------------------

    def refresh_ui():
        job_id_label.text = f"Job ID: {state['job_id']}" if state["job_id"] else "Job ID: -"
        status_label.text = f"Status: {state['status'].upper()}" if state["status"] else "Status: Waiting"
        error_label.text = state["error"] if state["error"] else ""

        result_container.clear()
        st = (state["status"] or "").lower()

        with result_container:
            if st == "completed":
                ts_formatted = state["formatted_timestamp"] or "00:00:00.000"
                ts_seconds = state["timestamp_seconds"] if state["timestamp_seconds"] is not None else 0.0

                # PROMINENT TIMESTAMP CARD
                with ui.row().classes("w-full items-center justify-between bg-gray-800 p-3 rounded-lg border border-yellow-500/50"):
                    ui.label("DETECTED TIMESTAMP:").classes("text-yellow-400 font-bold text-sm")
                    ui.label(f"{ts_formatted} ({ts_seconds:.2f}s)").classes("text-xl font-extrabold text-yellow-300 font-mono")

                # EXECUTION TIER
                with ui.row().classes("w-full items-center justify-between bg-gray-800 p-3 rounded-lg border border-gray-700"):
                    ui.label("Execution Tier:").classes("text-gray-400 text-sm")
                    ui.label(str(state["tier_executed"] or "N/A")).classes("font-semibold text-white text-sm")

                # CONFIDENCE SCORE
                with ui.row().classes("w-full items-center justify-between bg-gray-800 p-3 rounded-lg border border-gray-700"):
                    ui.label("Confidence Score:").classes("text-gray-400 text-sm")
                    score = float(state["confidence_score"] or 0) * 100
                    ui.label(f"{score:.1f}%").classes("font-semibold text-green-400 text-sm")

                # MATCHED TEXT
                if state["extracted_text"]:
                    with ui.column().classes("w-full bg-gray-800 p-3 rounded-lg border border-gray-700"):
                        ui.label("Matched Text:").classes("text-xs text-gray-400")
                        ui.label(f'"{state["extracted_text"]}"').classes("italic text-gray-200 text-sm mt-1")

                # DIRECT YOUTUBE TIMESTAMP LINK
                if state["url"] and ("youtube.com" in state["url"] or "youtu.be" in state["url"]):
                    yt_link = f"{state['url']}&t={int(ts_seconds)}s"
                    ui.button("Open Video at Timestamp", on_click=lambda: ui.open(yt_link)).classes(
                        "w-full bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-1.5 rounded-lg text-sm"
                    )

                # EXTRACTED FRAME
                if state["has_frame"]:
                    frame_url = f"{BACKEND_URL}/api/v1/jobs/{state['job_id']}/frame"
                    ui.image(frame_url).classes("w-full rounded-lg shadow-lg border border-gray-700 mt-2")

            elif st == "failed":
                ui.label("Detection Failed").classes("text-lg font-bold text-red-500 mb-1")
                if state["error"]:
                    ui.label(state["error"]).classes("text-sm text-red-400 bg-red-950 p-3 rounded-lg w-full font-mono")

    async def update_loop():
        await poll_job_status()
        refresh_ui()

    ui.timer(POLL_INTERVAL_SECONDS, update_loop)


# ---------------------------------------------------------
# Application Start
# ---------------------------------------------------------

ui.run(
    title="Quest1 Dashboard",
    port=8080,
    reload=False,
)