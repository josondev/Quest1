import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel
from src.models.schemas import CandidateFrame

logger = logging.getLogger(__name__)


class VLMError(Exception):
    """Domain exception raised when VLM evaluation fails."""
    pass


class VLMDecision(BaseModel):
    selected_candidate_id: str
    exact_detected_text: str
    confidence_score: float
    reasoning: Optional[str] = None


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file on disk to a base64 string for VLM payload transmission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class VLMArbiterService:
    def __init__(
        self,
        client: Optional[Any] = None,
        model: str = "llama-3.2-11b-vision-preview",
    ):
        self.client = client
        self.model = model

    @staticmethod
    def _encode_image(image_path: str) -> str:
        return encode_image_to_base64(image_path)

    def evaluate_candidates(
        self, target_text: str, candidates: List[CandidateFrame]
    ) -> VLMDecision:
        if not candidates or not self.client:
            return VLMDecision(
                selected_candidate_id="NONE",
                exact_detected_text="",
                confidence_score=0.0,
                reasoning="No candidates or VLM client uninitialized.",
            )

        best_candidate_id = "NONE"
        best_confidence = 0.0
        best_text = ""
        best_reasoning = ""

        # Sequential evaluation: sends 1 image per request to respect vision API limits
        for candidate in candidates:
            img_path = Path(candidate.image_path)
            if not img_path.exists():
                continue

            try:
                base64_img = encode_image_to_base64(str(img_path))
                prompt = (
                    f"Does this video frame contain on-screen text or subtitles matching: '{target_text}'?\n"
                    "Respond strictly in JSON format with no markdown wrappers:\n"
                    '{"found": true, "detected_text": "...", "confidence": 0.95, "reasoning": "..."}'
                )

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_img}"
                                    },
                                },
                            ],
                        }
                    ],
                    temperature=0.0,
                )

                content = response.choices[0].message.content or ""
                json_match = re.search(r"\{.*\}", content, re.DOTALL)

                if json_match:
                    data = json.loads(json_match.group(0))
                    found = bool(data.get("found", False))
                    confidence = float(data.get("confidence", 0.0))
                    detected_text = str(data.get("detected_text", ""))
                    reasoning = str(data.get("reasoning", ""))

                    if found and confidence > best_confidence:
                        best_candidate_id = candidate.candidate_id
                        best_confidence = confidence
                        best_text = detected_text
                        best_reasoning = reasoning

            except Exception as exc:
                logger.warning(
                    "VLM candidate evaluation failed for candidate %s: %s",
                    candidate.candidate_id,
                    exc,
                )

        if best_candidate_id != "NONE" and best_confidence >= 0.60:
            return VLMDecision(
                selected_candidate_id=best_candidate_id,
                exact_detected_text=best_text or target_text,
                confidence_score=best_confidence,
                reasoning=best_reasoning,
            )

        return VLMDecision(
            selected_candidate_id="NONE",
            exact_detected_text="",
            confidence_score=0.0,
            reasoning="No candidate met the evaluation threshold.",
        )