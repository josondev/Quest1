"""
src/storage.py

Production storage layer for Quest1.

Two backends are supported:
- Azure (when AZURE_STORAGE_CONNECTION_STRING is set in env)
- Local in-memory / disk fallback (for local development and tests)

JobStore     â†’ Azure Table Storage  (replaces in-memory JOBS_DB dict)
ArtifactStore â†’ Azure Blob Storage  (replaces local artifacts/ disk writes)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from src.models.schemas import DetectionResult, JobStatus

logger = logging.getLogger(__name__)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HELPERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _result_to_entity(result: DetectionResult) -> dict:
    """Flatten a DetectionResult into a flat dict safe for Table Storage.

    Uses model_dump(mode="json") so enum values are serialised as their
    string values (e.g. "completed") not their Python reprs ("JobStatus.COMPLETED").
    """
    # mode="json" coerces enums → .value, Paths → str, etc.
    data = result.model_dump(mode="json")
    flat: dict = {"PartitionKey": "jobs", "RowKey": result.job_id}
    for k, v in data.items():
        if v is None:
            flat[k] = ""
        elif isinstance(v, (dict, list)):
            flat[k] = json.dumps(v)
        else:
            flat[k] = str(v)
    return flat


def _entity_to_result(entity: dict) -> DetectionResult:
    """Reconstruct a DetectionResult from a Table Storage entity dict."""
    data: dict = {}
    skip = {"PartitionKey", "RowKey", "etag", "Timestamp", "odata.etag"}
    for k, v in entity.items():
        if k in skip:
            continue
        if isinstance(v, str) and v == "":
            data[k] = None
        elif isinstance(v, str) and v.startswith("{"):
            try:
                data[k] = json.loads(v)
            except Exception:
                data[k] = v
        else:
            data[k] = v
    return DetectionResult(**{k: v for k, v in data.items() if v is not None})


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# JOB STORE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class JobStore:
    """
    Stores and retrieves DetectionResult objects.

    Azure mode  : uses Azure Table Storage (persists across restarts)
    Local mode  : uses an in-memory dict (identical to old JOBS_DB)
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        table_name: str = "quest1jobs",
    ) -> None:
        self._table_name = table_name
        self._local: Dict[str, DetectionResult] = {}
        self._client = None

        if connection_string:
            try:
                from azure.data.tables import TableServiceClient
                service = TableServiceClient.from_connection_string(connection_string)
                self._client = service.get_table_client(table_name)
                # Create table if it doesn't exist
                try:
                    self._client.create_table()
                    logger.info("Table Storage: created table '%s'", table_name)
                except Exception:
                    # Table already exists â€” normal
                    pass
                logger.info("JobStore: using Azure Table Storage (%s)", table_name)
            except Exception as exc:
                logger.warning(
                    "JobStore: Azure Table init failed (%s) â€” falling back to in-memory", exc
                )
                self._client = None
        else:
            logger.info("JobStore: no connection string â€” using in-memory store")

    # â”€â”€ public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def put(self, job_id: str, result: DetectionResult) -> None:
        """Upsert a job result."""
        if self._client is not None:
            try:
                entity = _result_to_entity(result)
                self._client.upsert_entity(entity)
                return
            except Exception as exc:
                logger.warning("JobStore.put Azure failed (%s) â€” using local", exc)
        self._local[job_id] = result

    def get(self, job_id: str) -> Optional[DetectionResult]:
        """Retrieve a job result by id. Returns None if not found."""
        if self._client is not None:
            try:
                entity = self._client.get_entity(
                    partition_key="jobs", row_key=job_id
                )
                return _entity_to_result(dict(entity))
            except Exception as exc:
                logger.warning("JobStore.get Azure failed (%s) â€” using local", exc)
        return self._local.get(job_id)

    def exists(self, job_id: str) -> bool:
        return self.get(job_id) is not None

    def clear(self) -> None:
        """Clear in-memory store â€” used by tests only."""
        self._local.clear()

    # â”€â”€ dict-like interface so existing code using JOBS_DB still works â”€â”€

    def __contains__(self, job_id: str) -> bool:
        return self.exists(job_id)

    def __setitem__(self, job_id: str, result: DetectionResult) -> None:
        self.put(job_id, result)

    def __getitem__(self, job_id: str) -> DetectionResult:
        result = self.get(job_id)
        if result is None:
            raise KeyError(job_id)
        return result


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ARTIFACT STORE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ArtifactStore:
    """
    Stores frame JPEG artifacts.

    Azure mode  : uploads to Azure Blob Storage, returns public HTTPS URL
    Local mode  : keeps files on disk, returns local path (existing behaviour)
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        container_name: str = "quest1-artifacts",
        local_artifacts_dir: Optional[Path] = None,
    ) -> None:
        self._container_name = container_name
        self._local_dir = local_artifacts_dir or Path("artifacts")
        self._client = None

        if connection_string:
            try:
                from azure.storage.blob import BlobServiceClient, PublicAccess
                service = BlobServiceClient.from_connection_string(connection_string)
                self._client = service.get_container_client(container_name)
                # Create container if it doesn't exist
                try:
                    self._client.create_container(public_access=PublicAccess.BLOB)
                    logger.info(
                        "ArtifactStore: created blob container '%s'", container_name
                    )
                except Exception:
                    # Already exists
                    pass
                logger.info(
                    "ArtifactStore: using Azure Blob Storage (%s)", container_name
                )
            except Exception as exc:
                logger.warning(
                    "ArtifactStore: Azure Blob init failed (%s) â€” using local disk", exc
                )
                self._client = None
        else:
            logger.info("ArtifactStore: no connection string â€” using local disk")

    # â”€â”€ public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def upload_frame(self, local_path: Path, job_id: str) -> str:
        """
        Upload a frame JPEG and return its accessible URL/path.

        Azure : uploads to Blob, returns permanent HTTPS URL
        Local : file already on disk, returns the local path string
        """
        if not local_path.exists():
            logger.warning("ArtifactStore.upload_frame: file not found %s", local_path)
            return str(local_path)

        if self._client is not None:
            try:
                blob_name = f"{job_id}/{local_path.name}"
                with open(local_path, "rb") as f:
                    self._client.upload_blob(
                        blob_name, f, overwrite=True,
                        content_settings=self._jpeg_content_settings(),
                    )
                url = f"{self._client.url}/{blob_name}"
                logger.info("ArtifactStore: uploaded %s â†’ %s", local_path.name, url)
                return url
            except Exception as exc:
                logger.warning(
                    "ArtifactStore.upload_frame Azure failed (%s) â€” using local path", exc
                )

        return str(local_path)

    def upload_metadata(self, local_path: Path, job_id: str) -> None:
        """Upload metadata.json to Blob (best-effort, never raises)."""
        if self._client is None or not local_path.exists():
            return
        try:
            blob_name = f"{job_id}/{local_path.name}"
            with open(local_path, "rb") as f:
                self._client.upload_blob(blob_name, f, overwrite=True)
        except Exception as exc:
            logger.warning("ArtifactStore.upload_metadata failed: %s", exc)

    @staticmethod
    def _jpeg_content_settings():
        try:
            from azure.storage.blob import ContentSettings
            return ContentSettings(content_type="image/jpeg")
        except Exception:
            return None
