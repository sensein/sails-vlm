"""Qwen3-VL model wrapper for video understanding tasks."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .base_vlm import BaseVLM, VLMRawOutput


def sample_frame_indices(
    total_frames: int, frames_per_sample: int, samples_per_clip: int
) -> List[np.ndarray]:
    """Same window-index logic as Cosmos wrapper."""
    if total_frames <= 0 or frames_per_sample <= 0 or samples_per_clip <= 0:
        return []
    max_start = max(total_frames - frames_per_sample, 0)
    starts = [int(round(s)) for s in np.linspace(0, max_start, num=samples_per_clip)]
    all_indices: List[np.ndarray] = []
    for start in starts:
        end = min(start + frames_per_sample, total_frames)
        idx = np.linspace(start, max(end - 1, start), frames_per_sample)
        idx = np.clip(idx, 0, total_frames - 1).astype("int64")
        all_indices.append(idx)
    return all_indices


class Qwen3(BaseVLM):
    """Qwen3-VL wrapper using HuggingFace Transformers."""

    def __init__(self, model_config: Dict[str, Any]) -> None:
        super().__init__(model_config)

        self.model_id = model_config.get("hf_model_id", "Qwen/Qwen3-VL-8B-Instruct")
        self.device = model_config.get("device", "cuda")
        self.frames_per_sample = model_config.get("frames_per_sample", 8)
        self.samples_per_clip = model_config.get("samples_per_clip", 4)
        self.max_new_tokens = model_config.get("max_new_tokens", 64)
        self.load_in_4bit = model_config.get("load_in_4bit", False)
        self.load_in_8bit = model_config.get("load_in_8bit", False)
        self.do_sample = model_config.get("do_sample", False)
        self.temperature = model_config.get("temperature", 0.7)
        self.top_p = model_config.get("top_p", 0.95)
        self.top_k = model_config.get("top_k", 20)
        self.local_files_only = model_config.get("local_files_only", True)
        self.label_aliases = model_config.get("label_aliases", None)

        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load the Qwen3-VL model and processor."""
        from transformers import AutoProcessor

        print(f"Loading Qwen3 model: {self.model_id}...")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        quantization_config = None
        device_map = "auto"
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

        from transformers import AutoModelForImageTextToText

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )

        self._loaded = True
        print("Qwen3 model loaded successfully.")

    def _build_messages(
        self, frames: List[Image.Image], prompt: str
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _strip_thinking_process(self, text: str) -> str:
        if "<think>" in text and "</think>" in text:
            parts = text.split("</think>")
            return parts[-1].strip()
        return text.strip()

    def _run_one_window(self, frames: List[Image.Image], prompt: str) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        from qwen_vl_utils import process_vision_info

        messages = self._build_messages(frames, prompt)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        gen_kwargs: Dict[str, Any] = dict(max_new_tokens=self.max_new_tokens)
        if self.do_sample:
            gen_kwargs.update(
                dict(do_sample=True, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
            )
        else:
            gen_kwargs.update(dict(do_sample=False))

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return self._strip_thinking_process(output_text)

    # -------------------------
    # Label extraction + voting (same as Cosmos)
    # -------------------------
    def _extract_label(
        self,
        text: str,
        labels: Sequence[str],
        aliases: Optional[Dict[str, List[str]]] = None,
    ) -> Optional[str]:
        low = (text or "").lower()
        labels_sorted = sorted([str(lab) for lab in labels], key=len, reverse=True)
        for label in labels_sorted:
            if label.lower() in low:
                return label
        if aliases:
            for canonical, syns in aliases.items():
                for syn in syns:
                    if syn.lower() in low:
                        return canonical
        return None

    def _compute_class_scores(
        self, sample_preds: List[Optional[str]], labels: Sequence[str]
    ) -> Dict[str, float]:
        valid = [p for p in sample_preds if p is not None]
        labels_list = [str(lab) for lab in labels]
        if not labels_list:
            return {}
        if not valid:
            return {lab: 1.0 / len(labels_list) for lab in labels_list}
        counts = Counter(valid)
        total = sum(counts.values())
        return {lab: counts.get(lab, 0) / total for lab in labels_list}

    # -------------------------
    # Frame extraction (decord)
    # -------------------------
    def _extract_frame_windows(
        self,
        video_path: str,
        frames_per_sample: int,
        samples_per_clip: int,
    ) -> Tuple[List[List[Image.Image]], Dict[str, Any]]:
        p = Path(str(video_path))
        if not p.exists():
            return [], {"frame_backend": None, "error": f"FileNotFound: {p}"}

        windows: List[List[Image.Image]] = []

        # 1) decord
        try:
            from decord import VideoReader, cpu

            vr = VideoReader(str(p), ctx=cpu(0), num_threads=1)
            total = len(vr)
            if total <= 0:
                return [], {"frame_backend": "decord", "error": "EmptyVideo(len=0)"}

            index_sets = sample_frame_indices(total, frames_per_sample, samples_per_clip)
            for idxs in index_sets:
                frames_np = vr.get_batch(idxs).asnumpy()
                windows.append(
                    [Image.fromarray(frames_np[i]) for i in range(frames_np.shape[0])]
                )

            return windows, {
                "frame_backend": "decord",
                "n_frames_video": total,
                "n_windows": len(windows),
            }
        except Exception as e:
            return [], {
                "frame_backend": "decord",
                "error": f"decord_failed={repr(e)}",
            }

    def generate(
        self,
        video_path: str,
        prompt: str,
        labels: Optional[Sequence[str]] = None,
    ) -> VLMRawOutput:
        """Generate prediction with multi-window voting (same as Cosmos)."""
        if not self._loaded:
            self.load()
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        labels_list: List[str] = list(labels) if labels else []

        windows, fmeta = self._extract_frame_windows(
            video_path=video_path,
            frames_per_sample=self.frames_per_sample,
            samples_per_clip=self.samples_per_clip,
        )

        if not windows:
            return VLMRawOutput(
                raw_text="",
                raw_topk=[],
                meta={
                    "model": "qwen3-vl",
                    "hf_model_id": self.model_id,
                    "empty_frames": True,
                    **fmeta,
                },
            )

        completions: List[str] = []
        sample_preds: List[Optional[str]] = []

        for win_frames in windows:
            txt = self._run_one_window(win_frames, prompt)
            completions.append(txt)

            if labels_list:
                pred = self._extract_label(txt, labels_list, aliases=self.label_aliases)
                sample_preds.append(pred)

        if labels_list:
            scores = self._compute_class_scores(sample_preds, labels_list)
            sorted_labels = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
            top2 = sorted_labels[:2]
            top1 = top2[0] if top2 else None

            meta = {
                "model": "qwen3-vl",
                "hf_model_id": self.model_id,
                "frames_per_sample": self.frames_per_sample,
                "samples_per_clip": self.samples_per_clip,
                "sample_predictions": sample_preds,
                "scores": scores,
                "top2": top2,
                "completions": completions,
                **fmeta,
            }
            return VLMRawOutput(raw_text=top1 or "", raw_topk=top2, meta=meta)

        return VLMRawOutput(
            raw_text=completions[0] if completions else "",
            raw_topk=[],
            meta={
                "model": "qwen3-vl",
                "hf_model_id": self.model_id,
                "frames_per_sample": self.frames_per_sample,
                "samples_per_clip": self.samples_per_clip,
                "completions": completions,
                **fmeta,
            },
        )
