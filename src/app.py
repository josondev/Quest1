import uuid
import logging
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.config import settings
from src.models.schemas import JobRequest, DetectionResult, JobStatus
from src.pipeline import PipelineOrchestrator

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quest1: Dynamic Dialogue Detector API",
    version="1.0.0",
    description="5-Tier Cascade Pipeline for Video Stream Dialogue Localization",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static artifact serving path setup
artifacts_path = getattr(
    settings,
    "artifacts_dir",
    getattr(settings, "artifact_storage_dir", Path("artifacts")),
)
artifacts_dir = Path(artifacts_path)
artifacts_dir.mkdir(parents=True, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=str(artifacts_dir)), name="artifacts")

# In-memory storage dictionary expected by pytest suite
JOBS_DB: Dict[str, DetectionResult] = {}
orchestrator = PipelineOrchestrator()


def _execute_pipeline_background(job_id: str, request: JobRequest) -> None:
    """Background task runner for executing 5-tier pipeline processing."""
    logger.info("Background task started for job %s", job_id)
    try:
        result = orchestrator.run(job_id, request)
        
        # Format image path for static access if frame artifact exists
        if result.frame_image_path and Path(result.frame_image_path).exists():
            img_name = Path(result.frame_image_path).name
            result.frame_image_path = f"/artifacts/{job_id}/{img_name}"
            
        JOBS_DB[job_id] = result
        logger.info("Background task completed for job %s with status %s", job_id, result.status)
    except Exception as exc:
        logger.exception("Background execution failed for job %s: %s", job_id, exc)
        JOBS_DB[job_id] = DetectionResult(
            job_id=job_id,
            status=JobStatus.FAILED,
            target_dialogue=request.target_text,
            error_message=str(exc),
        )


@app.post("/api/v1/jobs", status_code=202)
async def create_detection_job(
    request: JobRequest, background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Queues a dialogue detection job asynchronously."""
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    
    # Store initial pending result state
    JOBS_DB[job_id] = DetectionResult(
        job_id=job_id,
        status=JobStatus.PROCESSING,
        target_dialogue=request.target_text,
    )
    
    background_tasks.add_task(_execute_pipeline_background, job_id, request)
    return {
        "job_id": job_id,
        "status": "processing",
        "target_dialogue": request.target_text,
    }


@app.get("/api/v1/jobs/{job_id}", response_model=DetectionResult)
async def get_job_status(job_id: str) -> DetectionResult:
    """Retrieves pipeline execution status or final DetectionResult schema."""
    if job_id not in JOBS_DB:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JOBS_DB[job_id]


@app.get("/api/v1/jobs/{job_id}/frame")
async def get_job_frame(job_id: str):
    """Serves extracted candidate frame image binary directly."""
    if job_id not in JOBS_DB:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    
    job = JOBS_DB[job_id]
    if not job.frame_image_path:
        raise HTTPException(status_code=400, detail="No frame artifact available for this job.")
    
    frame_path = Path(job.frame_image_path)
    if not frame_path.is_absolute() and str(job.frame_image_path).startswith("/artifacts/"):
        rel_path = str(job.frame_image_path).replace("/artifacts/", "", 1)
        frame_path = artifacts_dir / rel_path

    if not frame_path.exists():
        raise HTTPException(status_code=400, detail="Frame artifact file does not exist on disk.")

    return FileResponse(frame_path, media_type="image/jpeg")