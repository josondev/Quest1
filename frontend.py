"""
Quest1 NiceGUI Dashboard

Matches the real endpoints in src/app.py:
  POST /api/v1/jobs           -> {"job_id": ..., "status": ..., "target_dialogue": ...}
  GET  /api/v1/jobs/{job_id}  -> DetectionResult
  GET  /api/v1/jobs/{job_id}/frame -> raw JPEG of the detected frame (if any)

ASSUMPTION TO VERIFY: the JobRequest field name for the video source.
I couldn't fetch src/models/schemas.py, so this uses "video_url" and
"target_text" (the latter is confirmed -- app.py references
request.target_text directly). Check schemas.py and adjust JOB_REQUEST
below if the source field is named differently (e.g. "source", "url").

NOTE: file upload isn't wired up on the backend yet (create_detection_job
only accepts a JSON body, no UploadFile) -- so this dashboard only submits
by URL for now. Add multipart handling to src/app.py first if you want
local-file submission from here.
"""

import httpx
from nicegui import ui

BACKEND_URL = "http://127.0.0.1:8000"
POLL_INTERVAL_SECONDS = 2
TERMINAL_STATUSES = {"completed", "failed", "success", "done"}

state = {
    "job_id": None,
    "status": None,
    "target_dialogue": None,
    "error": None,
    "has_frame": False,
}


async def submit_job(video_url: str, target_text: str):
    state.update(job_id=None, status=None, target_dialogue=None, error=None, has_frame=False)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                f"{BACKEND_URL}/api/v1/jobs",
                json={"video_url": video_url, "target_text": target_text},
            )
            resp.raise_for_status()
            data = resp.json()
            state["job_id"] = data.get("job_id")
            state["status"] = data.get("status")
            state["target_dialogue"] = data.get("target_dialogue")
        except httpx.HTTPStatusError as e:
            state["error"] = f"{e.response.status_code}: {e.response.text}"
        except Exception as e:
            state["error"] = str(e)


async def poll_job_status():
    if not state["job_id"] or (state["status"] or "").lower() in TERMINAL_STATUSES:
        return
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(f"{BACKEND_URL}/api/v1/jobs/{state['job_id']}")
            resp.raise_for_status()
            data = resp.json()
            state["status"] = data.get("status")
            state["target_dialogue"] = data.get("target_dialogue")
            state["error"] = data.get("error_message")
            state["has_frame"] = bool(data.get("frame_image_path"))
        except Exception as e:
            state["error"] = str(e)


@ui.page("/")
def main_page():
    ui.label("Quest1 -- Dialogue Detection").classes("text-2xl font-bold")

    with ui.card().classes("w-full max-w-2xl"):
        video_url = ui.input("Video URL").classes("w-full")
        dialogue_text = ui.input("Target dialogue").classes("w-full")

        status_label = ui.label().classes("text-sm text-gray-600")
        job_id_label = ui.label().classes("text-xs text-gray-400")
        error_label = ui.label().classes("text-sm text-red-500")
        frame_image = ui.image().classes("w-64 h-64 object-cover rounded").props("visible=false")

        async def on_submit():
            if not video_url.value or not dialogue_text.value:
                ui.notify("Enter both a video URL and the target dialogue", type="warning")
                return
            frame_image.props("visible=false")
            await submit_job(video_url.value, dialogue_text.value)
            if state["error"]:
                ui.notify(f"Failed to submit: {state['error']}", type="negative")

        ui.button("Run Detection", on_click=on_submit).classes("mt-2")

    def refresh_ui():
        job_id_label.text = f"Job: {state['job_id']}" if state["job_id"] else ""
        status_label.text = f"Status: {state['status'] or '-'}"
        error_label.text = state["error"] or ""
        if state["has_frame"] and state["job_id"]:
            frame_image.set_source(f"{BACKEND_URL}/api/v1/jobs/{state['job_id']}/frame")
            frame_image.props("visible=true")

    async def tick():
        await poll_job_status()
        refresh_ui()

    ui.timer(POLL_INTERVAL_SECONDS, tick)


ui.run(title="Quest1 Dashboard", port=8080, reload=False)
