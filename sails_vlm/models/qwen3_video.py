"""Qwen3-VL model wrapper using native video input (no manual frame extraction).

Instead of extracting PIL frames and voting across windows, this wrapper
passes the video file path directly to the processor via the `video` content
type. The model handles temporal sampling internally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from .base_vlm import BaseVLM, VLMRawOutput


class Qwen3Video(BaseVLM):
    """Qwen3-VL wrapper using native video input (whole-clip inference)."""

    def __init__(self, model_config: Dict[str, Any]) -> None:
        super().__init__(model_config)

        self.model_id = model_config.get("hf_model_id", "Qwen/Qwen3-VL-8B-Instruct")
        self.device = model_config.get("device", "cuda")
        self.max_new_tokens = model_config.get("max_new_tokens", 64)
        self.load_in_4bit = model_config.get("load_in_4bit", False)
        self.load_in_8bit = model_config.get("load_in_8bit", False)
        self.do_sample = model_config.get("do_sample", False)
        self.temperature = model_config.get("temperature", 0.7)
        self.top_p = model_config.get("top_p", 0.95)
        self.top_k = model_config.get("top_k", 20)
        self.local_files_only = model_config.get("local_files_only", True)
        self.label_aliases = model_config.get("label_aliases", None)
        # max_pixels controls how many video tokens the processor allocates
        self.max_pixels = model_config.get("max_pixels", 360 * 420)

        self.model = None
        self.processor = None

    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"Loading Qwen3 model: {self.model_id}...")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
            max_pixels=self.max_pixels,
        )

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        quantization_config = None
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )

        self._loaded = True
        print("Qwen3 model loaded successfully.")

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
                meta={"model": "qwen3-vl-video", "error": f"FileNotFound: {p}"},
            )

        from qwen_vl_utils import process_vision_info

        # Get total frame count so the processor uses every frame in the clip
        total_frames: Optional[int] = None
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(str(p), ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
        except Exception:
            try:
                import cv2
                cap = cv2.VideoCapture(str(p))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
                cap.release()
            except Exception:
                pass

        video_content: Dict[str, Any] = {"type": "video", "video": str(p)}
        if total_frames:
            # Cap at qwen_vl_utils FPS_MAX_FRAMES (768) and floor to multiple of FRAME_FACTOR (2)
            nframes = min(total_frames, 768)
            nframes = (nframes // 2) * 2  # floor to even number
            nframes = max(nframes, 2)
            video_content["nframes"] = nframes

        messages = [
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        # qwen_vl_utils may return fps as a list; processor expects a scalar
        if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
            video_kwargs["fps"] = video_kwargs["fps"][0]

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(self.model.device)

        gen_kwargs: Dict[str, Any] = dict(max_new_tokens=self.max_new_tokens)
        if self.do_sample:
            gen_kwargs.update(
                dict(do_sample=True, temperature=self.temperature,
                     top_p=self.top_p, top_k=self.top_k)
            )
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        raw_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        raw_text = self._strip_thinking(raw_text)

        labels_list = list(labels) if labels else []
        pred = self._extract_label(raw_text, labels_list, self.label_aliases) if labels_list else None

        top2 = [pred] if pred else []

        return VLMRawOutput(
            raw_text=pred or raw_text,
            raw_topk=top2,
            meta={
                "model": "qwen3-vl-video",
                "hf_model_id": self.model_id,
                "video_path": str(p),
                "completion": raw_text,
            },
        )
