# src/fallback_vlm.py
import base64
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import requests

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.config import settings
from src.models.schemas import CandidateFrame, VLMDecision

logger = logging.getLogger(__name__)


class VLMError(Exception):
    """Domain exception raised when Tier 4 VLM evaluation fails."""
    pass


def encode_image_to_base64(image_path: str) -> Tuple[str, str]:
    """Encode local image file to base64 string and return (b64_string, mime_type)."""
    path = Path(image_path)
    if not path.exists():
        raise VLMError(f"Candidate frame image not found: {image_path}")

    ext = path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif ext == ".png":
        mime_type = "image/png"
    elif ext == ".webp":
        mime_type = "image/webp"
    else:
        raise VLMError(f"Unsupported or undetected image type for {image_path}")

    try:
        with open(path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
            return encoded, mime_type
    except Exception as exc:
        raise VLMError(f"Failed base64 encoding for image {image_path}: {exc}") from exc


class VLMArbiterService:
    """
    Tier 4 Bounded VLM Candidate Arbiter.
    Evaluates extracted candidate frames against target dialogue using NVIDIA NIM.
    Strictly enforces zero-hallucination candidate selection bounded to physical frames.
    """

    SYSTEM_PROMPT = """You are a strict visual frame arbiter for video dialogue localization.
Examine the provided candidate frames and determine which frame visually displays the target dialogue phrase in its subtitles or on-screen text.

CRITICAL CONSTRAINT:
You MUST select ONLY from the supplied candidate IDs (e.g., "C1", "C2", "C3") or return "NONE".
Do NOT invent new timestamps, frame numbers, or candidate IDs.

OUTPUT FORMAT:
Return ONLY a valid JSON object matching this schema. Do NOT include any preamble, conversational text, or narrative explanation before or after the JSON:
{
  "selected_candidate_id": "C1",
  "exact_detected_text": "text visible in subtitle ROI",
  "confidence_score": 0.95,
  "reasoning": "brief confirmation of visual match"
}"""

    encode_image_to_base64 = staticmethod(encode_image_to_base64)

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key if api_key is not None else settings.nvidia_api_key
        self.base_url = (base_url or settings.nvidia_base_url).rstrip("/")
        self.model = model or settings.nvidia_vlm_model

        if OpenAI is not None:
            self.client = OpenAI(api_key=self.api_key or "dummy", base_url=self.base_url)
        else:
            self.client = None

    def evaluate_candidates(
        self, target_dialogue: str, candidate_frames: List[CandidateFrame]
    ) -> VLMDecision:
        """
        Evaluate candidate frames against target dialogue using multimodal LLM.
        Forces candidate selection to remain strictly bounded to input IDs.
        """
        if not candidate_frames:
            return VLMDecision(
                selected_candidate_id="NONE",
                exact_detected_text="",
                confidence_score=0.0,
                reasoning="No candidate frames provided for evaluation.",
            )

        if not self.api_key:
            raise VLMError("API Key is not configured for VLM evaluation.")

        candidate_ids = [c.candidate_id for c in candidate_frames]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise VLMError("Candidate IDs provided for VLM evaluation must be unique")

        valid_ids = set(candidate_ids)

        content_payload: List[dict] = [
            {"type": "text", "text": f'Target Dialogue to locate: "{target_dialogue}"\nCandidates:'}
        ]

        for cand in candidate_frames:
            b64_str, mime_type = self.encode_image_to_base64(cand.image_path)
            b64_url = f"data:{mime_type};base64,{b64_str}"
            content_payload.append({
                "type": "text",
                "text": f"\nCandidate ID: {cand.candidate_id}: (Timestamp: {cand.timestamp_seconds}s)"
            })
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": b64_url}
            })

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": content_payload},
        ]

        try:
            if self.client is not None:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=512,
                )
                raw_content = response.choices[0].message.content
            else:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                body = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 512,
                }
                endpoint = f"{self.base_url}/chat/completions"
                resp = requests.post(endpoint, headers=headers, json=body, timeout=30)
                if resp.status_code != 200:
                    raise VLMError(f"VLM API endpoint returned status {resp.status_code}")
                res_data = resp.json()
                raw_content = res_data["choices"][0]["message"]["content"]

            return self._parse_vlm_response(raw_content, valid_ids)

        except VLMError:
            raise
        except Exception as exc:
            logger.error("VLM candidate arbitration failed: %s", exc)
            raise VLMError(f"VLM decision processing failed: {exc}") from exc

    @staticmethod
    def _parse_vlm_response(raw_text: str, valid_ids: set) -> VLMDecision:
        """
        Parse raw VLM output string into VLMDecision schema.
        Extracts embedded JSON objects via regex while enforcing schema guardrails.
        """
        if not raw_text or not raw_text.strip():
            raise VLMError("VLM response was empty")

        clean_text = raw_text.strip()
        clean_text = re.sub(r"^```(?:json)?", "", clean_text, flags=re.IGNORECASE).strip()
        clean_text = re.sub(r"```$", "", clean_text).strip()

        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if not json_match:
            raise VLMError(f"VLM response was not valid JSON: {raw_text[:100]}")

        json_str = json_match.group(0)

        try:
            data = json.loads(json_str)
        except Exception as exc:
            raise VLMError(f"VLM response was not valid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise VLMError("VLM response was not a valid JSON object")

        required_fields = ["selected_candidate_id", "exact_detected_text", "confidence_score", "reasoning"]
        for field in required_fields:
            if field not in data:
                raise VLMError(f"missing required field '{field}'")

        try:
            conf_score = float(data["confidence_score"])
        except (ValueError, TypeError):
            raise VLMError(f"invalid confidence_score: {data.get('confidence_score')}")

        if not (0.0 <= conf_score <= 1.0):
            raise VLMError(f"invalid confidence_score: {conf_score}")

        selected_id = str(data["selected_candidate_id"])
        if selected_id != "NONE" and selected_id not in valid_ids:
            raise VLMError(f"selected candidate ID '{selected_id}', which was not among the candidates provided")

        return VLMDecision(
            selected_candidate_id=selected_id,
            exact_detected_text=data.get("exact_detected_text", ""),
            bounding_box=data.get("bounding_box"),
            confidence_score=conf_score,
            reasoning=data.get("reasoning", ""),
        )