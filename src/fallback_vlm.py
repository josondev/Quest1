import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from pydantic import BaseModel
from src.config import settings
from src.models.schemas import CandidateFrame

# Required module-level symbol for unittest.mock @patch("src.fallback_vlm.OpenAI")
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


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
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = (
                getattr(settings, "nvidia_api_key", "")
                or getattr(settings, "vlm_api_key", "")
                or getattr(settings, "groq_api_key", "")
            )

        self.model = model or getattr(settings, "nvidia_vlm_model", "meta/llama-3.2-11b-vision-instruct")
        self.base_url = base_url or getattr(settings, "nvidia_base_url", "https://integrate.api.nvidia.com/v1")
        self.client = client

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise VLMError("API Key is not configured")
        if OpenAI is not None:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
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

        # NVIDIA NIM and most vision models support only 1 image per prompt.
        # Evaluate each candidate individually and pick the highest confidence match.
        best_decision: Optional[VLMDecision] = None

        for cand in candidates:
            user_prompt = (
                f'Target Dialogue to locate: "{target_text}"\n'
                f"Candidate ID: {cand.candidate_id}\n"
                "Does this frame contain the target dialogue as visible text? "
                "Return strictly JSON with keys: selected_candidate_id (use the candidate ID if found, else 'NONE'), "
                "exact_detected_text, confidence_score (0.0-1.0), reasoning, bounding_box (null if unknown)."
            )

            b64_str, mime_type = encode_image_to_base64(cand.image_path)

            content_items = [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_str}"},
                },
            ]

            messages = [
                {
                    "role": "system",
                    "content": "You are a Vision-Language Model analyzer evaluating video frames for visible dialogue text.",
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

            # Parse this candidate's response
            try:
                raw_content = response.choices[0].message.content or ""
                json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
                if not json_match:
                    continue
                data = json.loads(json_match.group(0))
            except Exception:
                continue

            required_fields = ["selected_candidate_id", "exact_detected_text", "confidence_score"]
            if not all(f in data for f in required_fields):
                continue

            selected_id = str(data["selected_candidate_id"])
            if selected_id == "NONE":
                continue

            try:
                conf_score = float(data["confidence_score"])
            except (TypeError, ValueError):
                continue

            if conf_score < 0.0 or conf_score > 1.0:
                continue

            # Override candidate id to match what was sent (model may echo it)
            data["selected_candidate_id"] = cand.candidate_id

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

            decision = VLMDecision(
                selected_candidate_id=cand.candidate_id,
                exact_detected_text=str(data.get("exact_detected_text", "")),
                confidence_score=conf_score,
                reasoning=data.get("reasoning"),
                bounding_box=bbox,
            )

            if best_decision is None or conf_score > best_decision.confidence_score:
                best_decision = decision

        # Return best match found, or NONE if nothing passed
        if best_decision is not None:
            return best_decision

        return VLMDecision(
            selected_candidate_id="NONE",
            exact_detected_text="",
            confidence_score=0.0,
            reasoning="No candidate contained the target dialogue.",
        )