import logging
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse

from src.models.schemas import DetectionResult, JobRequest, JobStatus
from src.pipeline import PipelineOrchestrator

logger = logging.getLogger("quest1.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Quest1 Dynamic Dialogue Detection API",
    description="Multi-tier hybrid AI service for pinpointing video timestamps and frame artifacts.",
    version="1.0.0",
)

# In-memory storage for asynchronous job statuses and results
JOBS_DB: Dict[str, DetectionResult] = {}
orchestrator = PipelineOrchestrator()


def _process_job_task(job_id: str, request: JobRequest) -> None:
    """Background worker function executing the pipeline orchestrator."""
    logger.info("Background task started for job %s", job_id)
    try:
        result = orchestrator.run(job_id=job_id, request=request)
        JOBS_DB[job_id] = result
        logger.info("Background task completed for job %s with status %s", job_id, result.status)
    except Exception as exc:
        logger.error("Background task unhandled failure for job %s: %s", job_id, exc)
        JOBS_DB[job_id] = DetectionResult(
            job_id=job_id,
            status=JobStatus.FAILED,
            target_dialogue=request.target_text,
            error_message=f"Pipeline error: {str(exc)}",
        )


@app.post(
    "/api/v1/jobs",
    response_model=DetectionResult,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a dialogue localization job",
)
async def create_detection_job(request: JobRequest, background_tasks: BackgroundTasks):
    """Enqueues a dialogue search job and immediately returns job tracking metadata."""
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    
    initial_result = DetectionResult(
        job_id=job_id,
        status=JobStatus.PROCESSING,
        target_dialogue=request.target_text,
    )
    JOBS_DB[job_id] = initial_result

    background_tasks.add_task(_process_job_task, job_id, request)
    return initial_result


@app.get(
    "/api/v1/jobs/{job_id}",
    response_model=DetectionResult,
    summary="Poll job status and detection output",
)
async def get_job_status(job_id: str):
    """Retrieves current processing state, tier execution result, and confidence scores."""
    if job_id not in JOBS_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job ID '{job_id}' not found.",
        )
    return JOBS_DB[job_id]


@app.get(
    "/api/v1/jobs/{job_id}/frame",
    summary="Download verified candidate frame image",
)
async def get_job_frame(job_id: str):
    """Serves the persisted candidate frame JPEG artifact for completed jobs."""
    if job_id not in JOBS_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job ID '{job_id}' not found.",
        )

    job_result = JOBS_DB[job_id]
    if job_result.status != JobStatus.COMPLETED or not job_result.frame_image_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job '{job_id}' does not have a verified frame artifact available.",
        )

    frame_path = Path(job_result.frame_image_path)
    if not frame_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Frame artifact file missing from storage.",
        )

    return FileResponse(path=frame_path, media_type="image/jpeg", filename=frame_path.name)