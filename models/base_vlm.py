"""Base classes and interfaces for VLM implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class VLMRawOutput:
    """Lowest-common-denominator output from any VLM backend.

    Task-specific parsing (classification vs free text) happens elsewhere.
    """

    raw_text: str
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
        self.device: Optional[str] = None
        self._loaded: bool = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights, tokenizer, and processor."""

    def predict(self, video_path: str, prompt: str) -> str:
        """Default implementation: call generate() and return raw_text.

        You can override, but usually you don't need to.
        """
        if not self._loaded:
            self.load()
        return self.generate(video_path=video_path, prompt=prompt).raw_text

    @abstractmethod
    def generate(
        self,
        video_path: str,
        prompt: str,
        video_cfg: Optional[Dict[str, Any]] = None,
        gen_cfg: Optional[Dict[str, Any]] = None,
    ) -> VLMRawOutput:
        """Generate response for video and prompt."""
