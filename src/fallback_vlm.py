# src/fallback_vlm.py
import base64
import json
import logging
import math
import mimetypes
from pathlib import Path
from typing import List, Optional
from openai import OpenAI

from src.config import settings
from src.models.schemas import BoundingBox, CandidateFrame, VLMDecision

logger = logging.getLogger(__name__)


class VLMError(Exception):
    """Domain exception raised when Tier 4 VLM arbitration fails."""
    pass


def encode_image_to_base64(image_path: str) -> tuple[str, str]:
    """
    Read a local frame image file and encode it as base64.
    Returns (base64_string, mime_type) derived from file extension.
    """
    path = Path(image_path)
    if not path.exists():
        raise VLMError(f"Candidate frame image not found at path: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type not in ("image/jpeg", "image/png", "image/webp"):
        raise VLMError(
            f"Unsupported or undetected image type for '{image_path}' "
            f"(guessed: {mime_type}). Expected .jpg/.jpeg/.png/.webp."
        )

    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8"), mime_type
    except Exception as e:
        raise VLMError(f"Failed to encode frame image '{image_path}': {e}") from e


class VLMArbiterService:
    """
    Tier 4: VLM Arbiter Service via OpenAI-compatible API.
    Enforces bounded candidate selection: the model only selects from physically 
    supplied candidate frame IDs. Timestamps and frame numbers are mapped by the 
    application logic downstream upon candidate selection.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.nvidia_api_key
        self.base_url = base_url or settings.nvidia_base_url
        self.model_name = model_name or settings.nvidia_vlm_model

        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if not self._client:
            if not self.api_key:
                raise VLMError("NVIDIA API Key is not configured. Set NVIDIA_API_KEY in .env")
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._client

    def evaluate_candidates(
        self,
        target_dialogue: str,
        candidates: List[CandidateFrame],
    ) -> VLMDecision:
        """
        Send candidate frames to the VLM to verify subtitle text presence.
        Returns a VLMDecision selecting the optimal candidate_id or 'NONE'.
        """
        if not candidates:
            return VLMDecision(
                selected_candidate_id="NONE",
                confidence_score=0.0,
                reasoning="No candidate frames provided for VLM evaluation.",
            )

        # Enforce unique candidate IDs
        raw_ids = [cand.candidate_id for cand in candidates]
        if len(raw_ids) != len(set(raw_ids)):
            raise VLMError("Candidate IDs provided for VLM evaluation must be unique.")

        valid_ids = set(raw_ids) | {"NONE"}
        client = self._get_client()

        system_prompt = (
            "You are an expert video frame dialogue verifier.\n"
            "Your task is to inspect the provided video frame images and determine which frame "
            "visually displays the target dialogue phrase in its subtitles/on-screen text.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. Select ONLY one of the provided candidate IDs (e.g., C1, C2) or 'NONE'.\n"
            "2. Do NOT create candidate IDs that were not explicitly provided.\n"
            "3. Do NOT infer or output timestamps or frame numbers.\n"
            "4. Return your response STRICTLY as a valid JSON object matching this schema:\n"
            "{\n"
            '  "selected_candidate_id": "C1",\n'
            '  "exact_detected_text": "verbatim text visible in frame",\n'
            '  "bounding_box": {"ymin": 0.75, "xmin": 0.1, "ymax": 0.95, "xmax": 0.9},\n'
            '  "confidence_score": 0.95,\n'
            '  "reasoning": "Clear explanation of why this frame was chosen."\n'
            "}"
        )

        content_payload = [
            {
                "type": "text",
                "text": f"Target Dialogue to locate: \"{target_dialogue}\"\n\nInspect these candidate frames:",
            }
        ]

        # Attach candidate frames without leaking timestamps
        for cand in candidates:
            b64_img, mime_type = encode_image_to_base64(cand.image_path)
            content_payload.append({
                "type": "text",
                "text": f"\nCandidate ID: {cand.candidate_id}:",
            })
            content_payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{b64_img}"
                },
            })

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_payload},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            raw_response = response.choices[0].message.content.strip()

            if "```json" in raw_response:
                raw_response = raw_response.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_response:
                raw_response = raw_response.split("```")[1].split("```")[0].strip()

            try:
                parsed = json.loads(raw_response)
            except json.JSONDecodeError as e:
                raise VLMError(
                    f"VLM response was not valid JSON after fence-stripping: {e}. "
                    f"Raw response: {raw_response[:500]}"
                ) from e

            # Enforce mandatory response fields
            required_fields = ["selected_candidate_id", "confidence_score", "exact_detected_text", "reasoning"]
            for field in required_fields:
                if field not in parsed:
                    raise VLMError(f"VLM response missing required field '{field}'.")

            selected_id = str(parsed["selected_candidate_id"])
            if selected_id not in valid_ids:
                raise VLMError(
                    f"VLM selected candidate_id '{selected_id}', which was not "
                    f"among the candidates provided ({sorted(valid_ids)}). "
                    "Refusing to propagate an out-of-set selection."
                )

            # Strict non-clamping confidence validation
            try:
                raw_conf = float(parsed["confidence_score"])
            except (ValueError, TypeError) as e:
                raise VLMError(f"VLM returned non-numeric confidence_score: {parsed['confidence_score']}") from e

            if not math.isfinite(raw_conf) or not (0.0 <= raw_conf <= 1.0):
                raise VLMError(f"VLM returned invalid confidence_score: {raw_conf}. Expected float in range [0.0, 1.0].")

            # Validate bounding box coordinates if present
            bbox_data = parsed.get("bounding_box")
            bbox = None
            if isinstance(bbox_data, dict):
                try:
                    bbox = BoundingBox(**bbox_data)
                    if bbox.xmin > bbox.xmax or bbox.ymin > bbox.ymax:
                        raise VLMError(f"VLM returned inverted bounding box coordinates: {bbox}")
                except Exception as e:
                    raise VLMError(f"VLM returned malformed bounding_box data: {e}") from e

            return VLMDecision(
                selected_candidate_id=selected_id,
                exact_detected_text=str(parsed["exact_detected_text"]),
                bounding_box=bbox,
                confidence_score=round(raw_conf, 4),
                reasoning=str(parsed["reasoning"]),
            )

        except VLMError:
            raise
        except Exception as e:
            logger.error("VLM evaluation failed: %s", e)
            raise VLMError(f"VLM decision processing failed: {e}") from e