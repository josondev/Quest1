import os
import httpx
from nicegui import ui

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
POLL_INTERVAL_SECONDS = 1.5
TERMINAL_STATUSES = {"completed", "failed", "success", "done"}

# Tier labels → short readable names
TIER_LABELS = {
    "Tier 0: Embedded Subtitle Match":   ("T0", "Subtitle"),
    "Tier 1: STT Acoustic Match":        ("T1", "Speech"),
    "Tier 2: Sparse Timeline OCR Match": ("T2", "OCR Scan"),
    "Tier 3: Dense Onset OCR Confirmation": ("T3", "OCR Dense"),
    "Tier 4: VLM Arbiter Fallback":      ("T4", "Vision AI"),
}

# ──────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────
state: dict = {
    "job_id": None,
    "status": None,
    "target_dialogue": None,
    "formatted_timestamp": None,
    "timestamp_seconds": None,
    "frame_number": None,
    "confidence_score": None,
    "tier_executed": None,
    "extracted_text": None,
    "url": None,
    "error": None,
    "has_frame": False,
}

def _reset_state(url: str, target_text: str) -> None:
    state.update({
        "job_id": None, "status": "processing",
        "target_dialogue": target_text, "formatted_timestamp": None,
        "timestamp_seconds": None, "frame_number": None,
        "confidence_score": None, "tier_executed": None,
        "extracted_text": None, "url": url, "error": None, "has_frame": False,
    })

# ──────────────────────────────────────────────
# API
# ──────────────────────────────────────────────
async def submit_job(url: str, target_text: str) -> None:
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
            state["status"]  = data.get("status")
    except Exception as exc:
        state["status"] = "failed"
        state["error"]  = str(exc)

async def poll_job_status() -> None:
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

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def _tier_badge(tier_raw: str | None) -> tuple[str, str]:
    """Return (short_code, label) for a tier string."""
    if not tier_raw:
        return ("–", "–")
    return TIER_LABELS.get(tier_raw, ("T?", tier_raw))

def _confidence_color(score: float) -> str:
    if score >= 0.85: return "text-emerald-400"
    if score >= 0.65: return "text-yellow-400"
    return "text-red-400"

def _stat_card(label: str, value: str, color: str = "text-white") -> None:
    with ui.column().classes("items-center gap-0"):
        ui.label(value).classes(f"text-2xl font-bold {color}")
        ui.label(label).classes("text-xs text-gray-500 uppercase tracking-widest")

# ──────────────────────────────────────────────
# PAGE
# ──────────────────────────────────────────────
@ui.page("/")
def dashboard() -> None:
    ui.page_title("Quest1 · Dialogue Detection")

    # ── global dark background ──
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

    # ── outer wrapper ──
    with ui.column().classes("w-full min-h-screen items-center pb-16 px-4"):

        # ── HERO ──────────────────────────────────────────────────
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
                "Locate any spoken or visual dialogue inside a video — "
                "timestamp, frame, and confidence score."
            ).classes("text-sm text-gray-500 mt-2 max-w-lg")

        # ── INPUT CARD ────────────────────────────────────────────
        with ui.card().classes(
            "w-full max-w-2xl rounded-2xl p-8 gap-0"
            " border border-gray-800"
        ).style("background:#161b27"):

            ui.label("New Detection Job").classes(
                "text-lg font-semibold text-gray-200 mb-6"
            )

            video_url = (
                ui.input(placeholder="https://youtu.be/…  or any video URL")
                .props('outlined dense label="Video URL" color="indigo"')
                .classes("w-full")
            )

            ui.space().classes("h-3")

            dialogue_text = (
                ui.input(placeholder='e.g. "in a world where"')
                .props('outlined dense label="Target Dialogue" color="indigo"')
                .classes("w-full mt-3")
            )

            run_btn = (
                ui.button("🔍  Run Detection")
                .classes(
                    "w-full mt-6 py-3 rounded-xl text-base font-semibold"
                    " text-white"
                )
                .style(
                    "background: linear-gradient(135deg,#6366f1,#8b5cf6);"
                    "transition: opacity .2s;"
                )
                .props("unelevated")
            )

        # ── STATUS + RESULT AREA ──────────────────────────────────
        status_card   = ui.card().classes(
            "w-full max-w-2xl rounded-2xl p-6 mt-4"
            " border border-gray-800"
        ).style("background:#161b27; display:none")

        result_card   = ui.card().classes(
            "w-full max-w-2xl rounded-2xl p-8 mt-4"
            " border border-gray-800"
        ).style("background:#161b27; display:none")

        # ── STATUS CARD CONTENTS ──────────────────────────────────
        with status_card:
            with ui.row().classes("w-full items-center gap-3"):
                status_dot   = ui.element("div").classes(
                    "w-2.5 h-2.5 rounded-full bg-yellow-400"
                ).style("flex-shrink:0")
                status_label = ui.label("Processing…").classes(
                    "text-sm font-semibold text-gray-200"
                )
                ui.space()
                job_id_label = ui.label("").classes(
                    "text-xs text-gray-600 font-mono"
                )
            error_label = ui.label("").classes(
                "text-sm text-red-400 mt-3 break-all"
            )

        # ── RESULT CARD CONTENTS ──────────────────────────────────
        with result_card:
            result_inner = ui.column().classes("w-full gap-5")

        # ─────────────────────────────────────────────────────────
        # REFRESH UI
        # ─────────────────────────────────────────────────────────
        def refresh_ui() -> None:
            s = state["status"]

            # ── show/hide cards ──
            status_card.style(
                "background:#161b27;" +
                ("display:block" if s else "display:none")
            )

            # ── status dot colour ──
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
                status_label.text = "Processing…"
                status_label.classes("text-gray-200", remove="text-emerald-300 text-red-300")

            job_id_label.text  = state["job_id"] or ""
            error_label.text   = state["error"]  or ""

            # ── result ──
            result_inner.clear()

            if s == "completed":
                result_card.style("background:#161b27; display:block")
                score_raw = state["confidence_score"] or 0.0
                score     = float(score_raw)
                tier_code, tier_name = _tier_badge(state["tier_executed"])

                with result_inner:
                    # header row
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("Detection Result").classes(
                            "text-lg font-semibold text-gray-200"
                        )
                        ui.badge(f"✓  {tier_code} · {tier_name}").classes(
                            "bg-indigo-600 text-white text-xs px-3 py-1 rounded-full"
                        )

                    # stats row
                    with ui.row().classes(
                        "w-full justify-around mt-2 py-4 rounded-xl"
                    ).style("background:#0f1117"):
                        _stat_card(
                            "Timestamp",
                            state["formatted_timestamp"] or "–",
                            "text-yellow-300",
                        )
                        _stat_card(
                            "Frame",
                            f"#{state['frame_number']}" if state["frame_number"] is not None else "–",
                            "text-blue-300",
                        )
                        _stat_card(
                            "Confidence",
                            f"{score * 100:.1f}%",
                            _confidence_color(score),
                        )
                        _stat_card(
                            "Tier",
                            tier_code,
                            "text-purple-300",
                        )

                    # detected text
                    if state["extracted_text"]:
                        with ui.column().classes(
                            "w-full rounded-xl p-4"
                        ).style("background:#0f1117"):
                            ui.label("Detected Text").classes(
                                "text-xs text-gray-500 uppercase tracking-widest mb-1"
                            )
                            ui.label(f'"{state["extracted_text"]}"').classes(
                                "text-gray-200 text-sm italic"
                            )

                    # frame image
                    if state["has_frame"]:
                        with ui.column().classes("w-full rounded-xl overflow-hidden mt-1"):
                            ui.label("Extracted Frame").classes(
                                "text-xs text-gray-500 uppercase tracking-widest mb-2"
                            )
                            ui.image(
                                f"{BACKEND_URL}/api/v1/jobs/{state['job_id']}/frame"
                            ).classes("w-full rounded-xl").style(
                                "border:1px solid #374151"
                            )

            elif s == "failed":
                result_card.style("background:#161b27; display:block")
                with result_inner:
                    with ui.column().classes(
                        "w-full items-center py-8 gap-3"
                    ):
                        ui.label("✕").classes("text-5xl text-red-500")
                        ui.label("Detection Failed").classes(
                            "text-lg font-semibold text-red-400"
                        )
                        if state["error"]:
                            ui.label(state["error"]).classes(
                                "text-xs text-gray-500 text-center max-w-sm break-all"
                            )
            else:
                result_card.style("background:#161b27; display:none")

        # ─────────────────────────────────────────────────────────
        # BUTTON CLICK
        # ─────────────────────────────────────────────────────────
        async def run_detection() -> None:
            url  = video_url.value.strip()
            text = dialogue_text.value.strip()
            if not url or not text:
                ui.notify(
                    "Please enter both a Video URL and Target Dialogue.",
                    type="warning",
                    position="top",
                )
                return
            run_btn.props("loading")
            await submit_job(url, text)
            run_btn.props(remove="loading")
            refresh_ui()

        run_btn.on_click(run_detection)

        # ─────────────────────────────────────────────────────────
        # POLL TIMER
        # ─────────────────────────────────────────────────────────
        async def update_loop() -> None:
            await poll_job_status()
            refresh_ui()

        ui.timer(POLL_INTERVAL_SECONDS, update_loop)


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
ui.run(
    title="Quest1 · Dialogue Detection",
    port=8080,
    reload=False,
    dark=True,
)
