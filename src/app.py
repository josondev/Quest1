import uuid
import logging
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, File, Form, HTTPException, UploadFile
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
from src.storage import JobStore, ArtifactStore


logger = logging.getLogger(__name__)

VIDEO_UPLOAD_EXTENSIONS = {
    ".3gp", ".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg",
    ".mpg", ".ts", ".webm",
}


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
    allow_credentials=False,
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

job_store = JobStore(
    connection_string=settings.azure_storage_connection_string or None,
    table_name=settings.azure_table_name,
)
artifact_store = ArtifactStore(
    connection_string=settings.azure_storage_connection_string or None,
    container_name=settings.azure_blob_container,
    local_artifacts_dir=artifacts_dir,
)

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


        # Upload frame to Blob Storage if available
        if result.frame_image_path:
            local_frame = Path(result.frame_image_path) if not result.frame_image_path.startswith("http") else None
            if local_frame and local_frame.exists():
                blob_url = artifact_store.upload_frame(local_frame, job_id)
                result = result.model_copy(update={"frame_image_path": blob_url})


        logger.info(
            "FINAL RESULT STORED: %s",
            result.model_dump()
        )


        job_store.put(job_id, result)


        logger.info(
            "Background task completed %s",
            job_id,
        )


    except Exception as exc:

        logger.exception(
            "Pipeline failed for %s",
            job_id,
        )


        job_store.put(job_id, DetectionResult(
            job_id=job_id,
            status=JobStatus.FAILED,
            target_dialogue=request.target_text,
            error_message=str(exc),
        ))
    finally:
        if request.local_file_path:
            upload_root = (Path(settings.temp_storage_dir) / "uploads").resolve()
            uploaded_path = Path(request.local_file_path).resolve()
            if uploaded_path.is_relative_to(upload_root):
                try:
                    uploaded_path.unlink(missing_ok=True)
                except OSError:
                    logger.exception("Could not remove uploaded video for job %s", job_id)



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


    job_store.put(job_id, DetectionResult(
        job_id=job_id,
        status=JobStatus.PROCESSING,
        target_dialogue=request.target_text,
    ))


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


@app.post(
    "/api/v1/jobs/upload",
    status_code=202,
)
async def create_uploaded_detection_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_text: str = Form(...),
):
    target_text = target_text.strip()
    if not target_text:
        raise HTTPException(
            status_code=422,
            detail="Target dialogue cannot be empty.",
        )

    extension = Path(file.filename or "").suffix.lower()
    if extension not in VIDEO_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported video type. Upload an MP4, MOV, MKV, WebM, AVI, or supported video file.",
        )

    job_id = f"job_{uuid.uuid4().hex[:10]}"
    upload_root = Path(settings.temp_storage_dir) / "uploads"
    upload_root.mkdir(parents=True, exist_ok=True)
    upload_path = upload_root / f"{job_id}{extension}"
    total_bytes = 0

    try:
        with upload_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > settings.max_video_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Video exceeds the upload limit of "
                            f"{settings.max_video_upload_bytes // (1024 * 1024)} MB."
                        ),
                    )
                destination.write(chunk)
        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded video file is empty.",
            )
    except Exception:
        upload_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    request = JobRequest(
        local_file_path=str(upload_path),
        target_text=target_text,
    )
    job_store.put(job_id, DetectionResult(
        job_id=job_id,
        status=JobStatus.PROCESSING,
        target_dialogue=request.target_text,
    ))
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

    if job_id not in job_store:

        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found.",
        )


    result = job_store[job_id]


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

    if job_id not in job_store:

        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found.",
        )


    job = job_store[job_id]


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


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health", include_in_schema=False)
async def health_check():
    return {"status": "ok", "service": "quest1-api"}
