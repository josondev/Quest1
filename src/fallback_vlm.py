import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from pydantic import BaseModel
from src.config import settings
from src.models.schemas import CandidateFrame

# Module-level attribute required for unittest.mock @patch("src.fallback_vlm.OpenAI")
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)


class VLMError(Exception):
    """Domain exception raised when VLM evaluation fails."""
    pass


class BoundingBox(BaseModel):
    ymin: float
    xmin: float
    ymax: float
    xmax: float


class VLMDecision(BaseModel):
    selected_candidate_id: str
    exact_detected_text: str
    confidence_score: float
    reasoning: Optional[str] = None
    bounding_box: Optional[BoundingBox] = None


def encode_image_to_base64(image_path: Union[str, Path]) -> Tuple[str, str]:
    """
    Encode an image file on disk to a base64 string for VLM payload transmission.
    Returns a tuple of (base64_string, mime_type).
    """
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise VLMError(f"Candidate frame image not found: {image_path}")

    ext = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    if ext not in mime_map:
        raise VLMError(f"Unsupported or undetected image type: {ext}")

    try:
        with open(path, "rb") as image_file:
            encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_str, mime_map[ext]
    except Exception as exc:
        if isinstance(exc, VLMError):
            raise
        raise VLMError(f"Failed to encode image {image_path}: {exc}") from exc


class VLMArbiterService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = getattr(settings, "vlm_api_key", "") or getattr(settings, "groq_api_key", "")

        self.model = model or getattr(settings, "vlm_model_name", "llama-3.2-11b-vision-preview")
        self.client = client

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise VLMError("API Key is not configured")
        if OpenAI is not None:
            self.client = OpenAI(api_key=self.api_key)
            return self.client
        raise VLMError("API Key is not configured")

    def evaluate_candidates(
        self, target_text: str, candidates: List[CandidateFrame]
    ) -> VLMDecision:
        if not candidates:
            return VLMDecision(
                selected_candidate_id="NONE",
                exact_detected_text="",
                confidence_score=0.0,
                reasoning="No candidates provided.",
            )

        if not self.api_key and self.client is None:
            raise VLMError("API Key is not configured")

        client = self._get_client()

        cand_ids = [c.candidate_id for c in candidates]
        if len(cand_ids) != len(set(cand_ids)):
            raise VLMError("Candidate IDs provided for VLM evaluation must be unique")

        user_prompt = (
            f'Target Dialogue to locate: "{target_text}"\n'
            "Return strictly JSON with keys: selected_candidate_id, exact_detected_text, confidence_score, reasoning, bounding_box."
        )

        content_items = [{"type": "text", "text": user_prompt}]
        for cand in candidates:
            b64_str, mime_type = encode_image_to_base64(cand.image_path)
            content_items.append({
                "type": "text",
                "text": f"\nCandidate ID: {cand.candidate_id}:"
            })
            content_items.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_str}"}
            })

        messages = [
            {
                "role": "system",
                "content": "You are a Vision-Language Model analyzer evaluating candidate video frames.",
            },
            {
                "role": "user",
                "content": content_items,
            },
        ]

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
        except Exception as exc:
            raise VLMError(f"VLM decision processing failed: {exc}") from exc

        try:
            raw_content = response.choices[0].message.content or ""
            json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if not json_match:
                raise VLMError("Response is not valid JSON")
            data = json.loads(json_match.group(0))
        except Exception as exc:
            if isinstance(exc, VLMError):
                raise
            raise VLMError("Response is not valid JSON") from exc

        required_fields = ["selected_candidate_id", "exact_detected_text", "confidence_score"]
        for field in required_fields:
            if field not in data:
                raise VLMError(f"missing required field '{field}'")

        selected_id = str(data["selected_candidate_id"])
        detected_text = str(data["exact_detected_text"])

        try:
            conf_score = float(data["confidence_score"])
        except (TypeError, ValueError):
            raise VLMError("Response is not valid JSON")

        if conf_score < 0.0 or conf_score > 1.0:
            raise VLMError(f"invalid confidence_score: {conf_score}")

        if selected_id != "NONE" and selected_id not in cand_ids:
            raise VLMError(
                f"Candidate ID {selected_id} which was not among the candidates provided"
            )

        bbox_data = data.get("bounding_box")
        bbox = None
        if isinstance(bbox_data, dict):
            try:
                bbox = BoundingBox(
                    ymin=float(bbox_data["ymin"]),
                    xmin=float(bbox_data["xmin"]),
                    ymax=float(bbox_data["ymax"]),
                    xmax=float(bbox_data["xmax"]),
                )
            except (KeyError, ValueError, TypeError):
                bbox = None

        if selected_id == "NONE":
            return VLMDecision(
                selected_candidate_id="NONE",
                exact_detected_text="",
                confidence_score=0.0,
                reasoning=data.get("reasoning", "No matching candidate found."),
                bounding_box=bbox,
            )

        return VLMDecision(
            selected_candidate_id=selected_id,
            exact_detected_text=detected_text,
            confidence_score=conf_score,
            reasoning=data.get("reasoning"),
            bounding_box=bbox,
        )