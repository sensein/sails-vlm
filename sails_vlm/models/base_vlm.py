"""Base classes and interfaces for VLM implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch


@dataclass
class VLMRawOutput:
    """Lowest-common-denominator output from any VLM backend."""

    raw_text: str
    raw_topk: List[str] = field(default_factory=list)  # <-- NEW (e.g., [top1, top2])
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseVLM(ABC):
    """Minimal interface your runners can rely on.

    - load(): must initialize model weights/tokenizer/processor
    - predict(): should return raw string (easy for CSV + downstream parsing)
    - generate(): can return VLMRawOutput (raw + meta)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the VLM with configuration.

        Args:
            config: Configuration dictionary for the model.
        """
        self.config = config or {}
        self.model = None
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._loaded: bool = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights, tokenizer, and processor."""

    def predict(self, video_path: str, prompt: str, labels: List[str]) -> str:
        """Predict based on video and prompt, calling generate method."""
        if not self._loaded:
            self.load()
        out = self.generate(video_path=video_path, prompt=prompt, labels=labels)

        # If model provided top-k, return "top1|top2" for downstream top2 accuracy
        if getattr(out, "raw_topk", None):
            # ensure at most 2; filter empties
            topk = [x for x in out.raw_topk if x]
            if len(topk) >= 2:
                return f"{topk[0]}|{topk[1]}"
            if len(topk) == 1:
                return topk[0]

        return out.raw_text

    @abstractmethod
    def generate(
        self,
        video_path: str,
        prompt: str,
        labels: List[str],
    ) -> VLMRawOutput:
        """Generate response for video and prompt."""
