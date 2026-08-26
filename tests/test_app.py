from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.app import JOBS_DB, app
from src.models.schemas import DetectionResult, JobStatus, TierType


@pytest.fixture
def client():
    JOBS_DB.clear()
    return TestClient(app)


class TestFastAPIEndpoints:

    def test_create_job_accepted(self, client):
        payload = {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "target_text": "Never gonna give you up",
        }
        response = client.post("/api/v1/jobs", json=payload)
        
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "processing"
        assert data["target_dialogue"] == "Never gonna give you up"

    def test_create_job_invalid_url_rejected(self, client):
        payload = {
            "url": "ftp://invalid-scheme.com/video.mp4",
            "target_text": "Hello",
        }
        response = client.post("/api/v1/jobs", json=payload)
        assert response.status_code == 422

    def test_get_job_status_success(self, client):
        job_id = "job_test_123"
        JOBS_DB[job_id] = DetectionResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            target_dialogue="sample text",
            timestamp_seconds=12.5,
            tier_executed=TierType.TIER_0_SUBTITLE,
        )

        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "completed"

    def test_get_job_status_not_found(self, client):
        response = client.get("/api/v1/jobs/non_existent_job")
        assert response.status_code == 404

    def test_get_job_frame_success(self, client, tmp_path):
        frame_file = tmp_path / "frame.jpg"
        frame_file.write_bytes(b"FAKE_JPEG_BINARY")

        job_id = "job_frame_123"
        JOBS_DB[job_id] = DetectionResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            target_dialogue="sample text",
            frame_image_path=str(frame_file),
        )

        response = client.get(f"/api/v1/jobs/{job_id}/frame")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.content == b"FAKE_JPEG_BINARY"

    def test_get_job_frame_missing_artifact(self, client):
        job_id = "job_no_artifact"
        JOBS_DB[job_id] = DetectionResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            target_dialogue="sample text",
            frame_image_path=None,
        )

        response = client.get(f"/api/v1/jobs/{job_id}/frame")
        assert response.status_code == 400