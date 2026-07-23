"""Cosmos-Reason2 model wrapper using native video input.

Follows the official HF example for Cosmos-Reason2 video inference:
- Passes video path directly in message content with fps parameter
- Uses apply_chat_template with fps kwarg (no process_vision_info)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from .base_vlm import BaseVLM, VLMRawOutput


class CosmosVideo(BaseVLM):
    """Cosmos-Reason2 wrapper using native video input."""

    def __init__(self, model_config: Dict[str, Any]) -> None:
        super().__init__(model_config)

        self.model_id = model_config.get("hf_model_id", "nvidia/Cosmos-Reason2-8B")
        self.device = model_config.get("device", "cuda")
        self.fps = float(model_config.get("fps", 4.0))
        self.max_new_tokens = model_config.get("max_new_tokens", None)
        self.do_sample = model_config.get("do_sample", True)
        self.temperature = model_config.get("temperature", 0.7)
        self.top_p = model_config.get("top_p", 0.95)
        self.top_k = model_config.get("top_k", 20)
        self.local_files_only = model_config.get("local_files_only", True)
        self.label_aliases = model_config.get("label_aliases", None)

        self.model = None
        self.processor = None

    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"Loading Cosmos model: {self.model_id}...")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            local_files_only=self.local_files_only,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=self.local_files_only,
        )
        self.model.eval()
        self._loaded = True
        print("Cosmos model loaded.")

    def _extract_label(
        self,
        text: str,
        labels: Sequence[str],
        aliases: Optional[Dict[str, List[str]]] = None,
    ) -> Optional[str]:
        low = (text or "").lower()
        for label in sorted(labels, key=len, reverse=True):
            if label.lower() in low:
                return label
        if aliases:
            for canonical, syns in aliases.items():
                for syn in syns:
                    if syn.lower() in low:
                        return canonical
        return None

    def _strip_thinking(self, text: str) -> str:
        if "<think>" in text and "</think>" in text:
            return text.split("</think>")[-1].strip()
        return text.strip()

    def generate(
        self,
        video_path: str,
        prompt: str,
        labels: Optional[Sequence[str]] = None,
    ) -> VLMRawOutput:
        if not self._loaded:
            self.load()

        p = Path(str(video_path))
        if not p.exists():
            return VLMRawOutput(
                raw_text="",
                raw_topk=[],
                meta={"model": "cosmos-video", "error": f"FileNotFound: {p}"},
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(p),
                        "fps": self.fps,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Pass fps kwarg directly to apply_chat_template (HF Cosmos pattern)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            fps=self.fps,
        )
        inputs = inputs.to(self.model.device)

        gen_kwargs: Dict[str, Any] = {}
        if self.max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = self.max_new_tokens
        if self.do_sample:
            gen_kwargs.update(
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        raw_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        raw_text = self._strip_thinking(raw_text)

        labels_list = list(labels) if labels else []
        pred = self._extract_label(raw_text, labels_list, self.label_aliases) if labels_list else None

        return VLMRawOutput(
            raw_text=pred or raw_text,
            raw_topk=[pred] if pred else [],
            meta={
                "model": "cosmos-video",
                "hf_model_id": self.model_id,
                "video_path": str(p),
                "fps": self.fps,
                "completion": raw_text,
            },
        )
