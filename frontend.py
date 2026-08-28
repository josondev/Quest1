import httpx
from nicegui import ui


BACKEND_URL = "http://127.0.0.1:8000"
POLL_INTERVAL_SECONDS = 1.5

TERMINAL_STATUSES = {
    "completed",
    "failed",
    "success",
    "done",
}


state = {
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



async def submit_job(url, target_text):

    state.update({
        "job_id": None,
        "status": "processing",
        "target_dialogue": target_text,
        "formatted_timestamp": None,
        "timestamp_seconds": None,
        "frame_number": None,
        "confidence_score": None,
        "tier_executed": None,
        "extracted_text": None,
        "url": url,
        "error": None,
        "has_frame": False,
    })


    try:

        async with httpx.AsyncClient(timeout=30) as client:

            response = await client.post(
                f"{BACKEND_URL}/api/v1/jobs",
                json={
                    "url": url,
                    "target_text": target_text,
                },
            )

            response.raise_for_status()

            data = response.json()

            state["job_id"] = data.get("job_id")
            state["status"] = data.get("status")


    except Exception as e:

        state["status"] = "failed"
        state["error"] = str(e)




async def poll_job_status():

    if not state["job_id"]:
        return


    if state["status"] in TERMINAL_STATUSES:
        return


    try:

        async with httpx.AsyncClient(timeout=10) as client:

            response = await client.get(
                f"{BACKEND_URL}/api/v1/jobs/{state['job_id']}"
            )

            data = response.json()


            state["status"] = data.get("status")
            state["formatted_timestamp"] = data.get("formatted_timestamp")
            state["timestamp_seconds"] = data.get("timestamp_seconds")
            state["frame_number"] = data.get("frame_number")
            state["confidence_score"] = data.get("confidence_score")
            state["tier_executed"] = data.get("tier_executed")
            state["extracted_text"] = data.get("extracted_text")
            state["error"] = data.get("error_message")
            state["has_frame"] = bool(data.get("frame_image_path"))



    except Exception as e:

        state["error"] = str(e)




@ui.page("/")
def dashboard():


    ui.page_title(
        "Quest1 Dashboard"
    )


    # GLOBAL CENTER WRAPPER

    with ui.column().classes(
        """
        w-full
        items-center
        justify-center
        """
    ):



        # HEADER


        with ui.column().classes(
            """
            items-center
            mt-10
            text-center
            """
        ):


            ui.label(
                "Quest1 Dialogue Detection Engine"
            ).classes(
                """
                text-5xl
                font-extrabold
                text-white
                """
            )


            ui.label(
                "AI powered dialogue localization and frame extraction"
            ).classes(
                """
                text-gray-400
                text-lg
                mt-2
                """
            )



        # INPUT CARD


        with ui.card().classes(
            """
            w-[700px]
            mt-10
            bg-gray-900
            rounded-2xl
            p-8
            """
        ):


            with ui.column().classes(
                "items-center w-full"
            ):


                ui.label(
                    "Create Detection Job"
                ).classes(
                    """
                    text-2xl
                    font-bold
                    text-white
                    """
                )


                video_url = ui.input(
                    label="Video URL"
                ).classes(
                    "w-full mt-5"
                )


                dialogue_text = ui.input(
                    label="Target Dialogue"
                ).classes(
                    "w-full mt-3"
                )



                async def run_detection():

                    await submit_job(
                        video_url.value,
                        dialogue_text.value
                    )

                    refresh_ui()



                ui.button(
                    "🔍 Run Detection",
                    on_click=run_detection
                ).classes(
                    """
                    w-full
                    mt-6
                    bg-blue-600
                    text-white
                    rounded-xl
                    py-3
                    """
                )



        # STATUS CARD


        with ui.card().classes(
            """
            w-[700px]
            mt-6
            bg-gray-900
            rounded-2xl
            p-6
            """
        ):


            with ui.column().classes(
                "items-center w-full"
            ):


                ui.label(
                    "Job Status"
                ).classes(
                    "text-2xl font-bold text-white"
                )


                job_id_label = ui.label(
                    "Job ID: -"
                )


                status_label = ui.label(
                    "Status: Waiting"
                )



                error_label = ui.label()



        # RESULT CARD


        with ui.card().classes(
            """
            w-[700px]
            mt-6
            mb-10
            bg-gray-900
            rounded-2xl
            p-8
            """
        ):


            with ui.column().classes(
                "items-center w-full"
            ):


                ui.label(
                    "🎯 Detection Result"
                ).classes(
                    "text-3xl font-bold text-white"
                )


                result_container = ui.column().classes(
                    "w-full mt-6"
                )




    def refresh_ui():


        job_id_label.text = (
            f"Job ID: {state['job_id']}"
            if state["job_id"]
            else
            "Job ID: -"
        )


        status_label.text = (
            f"Status: {state['status']}"
            if state["status"]
            else
            "Status: Waiting"
        )


        result_container.clear()



        if state["status"] == "completed":


            with result_container.classes(
                "items-center"
            ):


                ui.badge(
                    "✓ COMPLETED"
                ).classes(
                    """
                    bg-green-600
                    text-white
                    px-5 py-2
                    rounded-full
                    """
                )


                ui.label(
                    f"Timestamp: {state['formatted_timestamp']}"
                ).classes(
                    "text-xl text-yellow-400"
                )


                ui.label(
                    f"Frame Number: {state['frame_number']}"
                ).classes(
                    "text-xl text-blue-400"
                )


                ui.label(
                    f"Tier: {state['tier_executed']}"
                ).classes(
                    "text-white"
                )


                ui.label(
                    f"Confidence: {float(state['confidence_score'])*100:.1f}%"
                ).classes(
                    "text-green-400 text-xl"
                )


                ui.label(
                    state["extracted_text"]
                ).classes(
                    """
                    text-white
                    text-center
                    mt-4
                    """
                )


                if state["has_frame"]:

                    ui.image(
                        f"{BACKEND_URL}/api/v1/jobs/{state['job_id']}/frame"
                    ).classes(
                        """
                        w-full
                        rounded-xl
                        mt-5
                        """
                    )



    async def update_loop():

        await poll_job_status()

        refresh_ui()



    ui.timer(
        POLL_INTERVAL_SECONDS,
        update_loop
    )



ui.run(
    title="Quest1 Dashboard",
    port=8080,
    reload=False,
)