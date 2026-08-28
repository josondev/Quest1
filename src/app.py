import uuid
import logging
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.config import settings
from src.models.schemas import (
    JobRequest,
    DetectionResult,
    JobStatus,
)
from src.pipeline import PipelineOrchestrator


logger = logging.getLogger(__name__)


app = FastAPI(
    title="Quest1: Dynamic Dialogue Detector API",
    version="1.0.0",
    description="5-Tier Cascade Pipeline for Video Stream Dialogue Localization",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STATIC ARTIFACTS
# ============================================================

artifacts_path = getattr(
    settings,
    "artifacts_dir",
    getattr(
        settings,
        "artifact_storage_dir",
        Path("artifacts"),
    ),
)


artifacts_dir = Path(artifacts_path)

artifacts_dir.mkdir(
    parents=True,
    exist_ok=True,
)


app.mount(
    "/artifacts",
    StaticFiles(
        directory=str(artifacts_dir)
    ),
    name="artifacts",
)


# ============================================================
# STORAGE
# ============================================================

JOBS_DB: Dict[str, DetectionResult] = {}

orchestrator = PipelineOrchestrator()


# ============================================================
# BACKGROUND PIPELINE EXECUTION
# ============================================================

def _execute_pipeline_background(
    job_id: str,
    request: JobRequest,
) -> None:

    logger.info(
        "Background task started for %s",
        job_id,
    )

    try:

        result = orchestrator.run(
            job_id,
            request,
        )


        # ----------------------------------------
        # Convert local image path to API path
        # ----------------------------------------

        if result.frame_image_path:

            image_path = Path(
                result.frame_image_path
            )

            if image_path.exists():

                img_name = image_path.name

                result = result.model_copy(
                    update={
                        "frame_image_path":
                            f"/artifacts/{job_id}/{img_name}"
                    }
                )


        logger.info(
            "FINAL RESULT STORED: %s",
            result.model_dump()
        )


        JOBS_DB[job_id] = result


        logger.info(
            "Background task completed %s",
            job_id,
        )


    except Exception as exc:

        logger.exception(
            "Pipeline failed for %s",
            job_id,
        )


        JOBS_DB[job_id] = DetectionResult(
            job_id=job_id,
            status=JobStatus.FAILED,
            target_dialogue=request.target_text,
            error_message=str(exc),
        )



# ============================================================
# CREATE JOB
# ============================================================

@app.post(
    "/api/v1/jobs",
    status_code=202,
)
async def create_detection_job(
    request: JobRequest,
    background_tasks: BackgroundTasks,
):

    job_id = (
        f"job_{uuid.uuid4().hex[:10]}"
    )


    JOBS_DB[job_id] = DetectionResult(
        job_id=job_id,
        status=JobStatus.PROCESSING,
        target_dialogue=request.target_text,
    )


    background_tasks.add_task(
        _execute_pipeline_background,
        job_id,
        request,
    )


    return {
        "job_id": job_id,
        "status": "processing",
        "target_dialogue": request.target_text,
    }



# ============================================================
# GET JOB RESULT
# ============================================================

@app.get(
    "/api/v1/jobs/{job_id}",
    response_model=DetectionResult,
)
async def get_job_status(
    job_id: str,
):

    if job_id not in JOBS_DB:

        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found.",
        )


    result = JOBS_DB[job_id]


    logger.info(
        "API RESPONSE: %s",
        result.model_dump()
    )


    return result



# ============================================================
# DIRECT FRAME IMAGE ENDPOINT
# ============================================================

@app.get(
    "/api/v1/jobs/{job_id}/frame"
)
async def get_job_frame(
    job_id: str,
):

    if job_id not in JOBS_DB:

        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found.",
        )


    job = JOBS_DB[job_id]


    if not job.frame_image_path:

        raise HTTPException(
            status_code=400,
            detail="No frame artifact available.",
        )


    image_ref = job.frame_image_path


    # Handle API path:
    # /artifacts/job_x/image.jpg

    if image_ref.startswith(
        "/artifacts/"
    ):

        relative_path = image_ref.replace(
            "/artifacts/",
            "",
            1,
        )

        frame_path = (
            artifacts_dir /
            relative_path
        )

    else:

        frame_path = Path(
            image_ref
        )


    if not frame_path.exists():

        raise HTTPException(
            status_code=400,
            detail="Frame file missing on disk.",
        )


    return FileResponse(
        frame_path,
        media_type="image/jpeg",
    )