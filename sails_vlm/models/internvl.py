"""InternVL3.5 model wrapper for video understanding tasks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from .base_vlm import BaseVLM, VLMRawOutput


class InternVL(BaseVLM):
    """InternVL3.5 wrapper using HuggingFace Transformers."""

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initialize InternVL model wrapper.

        Args:
            model_config: Configuration dictionary with keys:
                - hf_model_id: HuggingFace model ID (default: OpenGVLab/InternVL3_5-8B-HF)
                - device: Device to use (default: cuda)
                - frames_per_sample: Frames per video sample (default: 8)
                - samples_per_clip: Number of samples per clip (default: 1)
                - max_new_tokens: Maximum new tokens to generate (default: 512)
                - load_in_4bit: Whether to use 4-bit quantization (default: False)
                - load_in_8bit: Whether to use 8-bit quantization (default: False)
                - temperature: Sampling temperature (default: 0.0)
                - top_p: Top-p sampling (default: 1.0)
                - do_sample: Whether to sample (default: False)
        """
        super().__init__(model_config)

        self.model_id = model_config.get("hf_model_id", "OpenGVLab/InternVL3_5-8B-HF")
        self.device = model_config.get("device", "cuda")
        self.frames_per_sample = model_config.get("frames_per_sample", 8)
        self.samples_per_clip = model_config.get("samples_per_clip", 1)
        self.max_new_tokens = model_config.get("max_new_tokens", 512)
        self.load_in_4bit = model_config.get("load_in_4bit", False)
        self.load_in_8bit = model_config.get("load_in_8bit", False)
        self.temperature = model_config.get("temperature", 0.0)
        self.top_p = model_config.get("top_p", 1.0)
        self.do_sample = model_config.get("do_sample", False)
        self.local_files_only = model_config.get("local_files_only", True)  # Use cache by default

        self.model = None
        self.processor = None
        self.tokenizer = None

    def load(self) -> None:
        """Load the InternVL model and processor."""
        from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText

        print(f"Loading InternVL model: {self.model_id}...")

        # Load processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, local_files_only=self.local_files_only
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False, local_files_only=self.local_files_only
        )

        # Determine dtype
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Quantization config
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

        # Load model
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto" if quantization_config else None,
            local_files_only=self.local_files_only,
        ).eval()

        if not quantization_config and self.device == "cuda":
            self.model = self.model.to(self.device)

        self._loaded = True
        print("InternVL model loaded successfully.")

    def _extract_frame_windows(
        self, video_path: str, num_frames: int
    ) -> List[List[Image.Image]]:
        """Extract frame windows from video.

        Args:
            video_path: Path to video file.
            num_frames: Number of frames per window.

        Returns:
            List of frame windows, each containing PIL Images.
        """
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)

            if total_frames == 0:
                return []

            # Sample frames uniformly
            indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            indices = [min(i, total_frames - 1) for i in indices]

            frames = vr.get_batch(indices).asnumpy()
            pil_frames = [Image.fromarray(f) for f in frames]

            return [pil_frames]

        except Exception as e:
            print(f"Error extracting frames with decord: {e}")
            # Fallback to opencv
            import cv2
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                cap.release()
                return []

            indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            indices = [min(i, total_frames - 1) for i in indices]

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
            cap.release()

            return [frames] if frames else []

    def _run_one_window(self, frames: List[Image.Image], prompt: str) -> str:
        """Run inference on one frame window.

        Args:
            frames: List of PIL Image frames.
            prompt: Text prompt.

        Returns:
            Model output text.
        """
        if self.model is None or self.processor is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build prompt with image placeholders
        image_token = getattr(self.processor, "image_token", "<image>")
        image_placeholders = "".join([f"{image_token}\n" for _ in frames])
        full_prompt = f"{image_placeholders}{prompt}"

        # Process inputs
        inputs = self.processor(
            text=full_prompt,
            images=frames,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                top_p=self.top_p if self.do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else None,
            )

        # Decode output
        input_len = inputs["input_ids"].shape[1]
        output_text = self.tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )

        return output_text.strip()

    def generate(
        self,
        video_path: str,
        prompt: str,
        labels: List[str],
    ) -> VLMRawOutput:
        """Generate prediction for a video.

        Args:
            video_path: Path to video file.
            prompt: Text prompt for the model.
            labels: List of valid labels (not used for constrained decoding here).

        Returns:
            VLMRawOutput with model prediction.
        """
        if self.model is None:
            self.load()
            self._loaded = True

        # Extract frames
        windows = self._extract_frame_windows(video_path, self.frames_per_sample)
        if not windows:
            return VLMRawOutput(raw_text="ERROR: Could not extract frames from video")

        # Run inference on first window
        result = self._run_one_window(windows[0], prompt)
        return VLMRawOutput(raw_text=result)
