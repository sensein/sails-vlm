"""Qwen3-VL model wrapper for video understanding tasks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from .base_vlm import BaseVLM, VLMRawOutput


class Qwen3(BaseVLM):
    """Qwen3-VL wrapper using HuggingFace Transformers."""

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initialize Qwen3-VL model wrapper.

        Args:
            model_config: Configuration dictionary with keys:
                - hf_model_id: HuggingFace model ID (default: Qwen/Qwen3-VL-8B-Thinking)
                - device: Device to use (default: cuda)
                - frames_per_sample: Frames per video sample (default: 8)
                - samples_per_clip: Number of samples per clip (default: 1)
                - max_new_tokens: Maximum new tokens to generate (default: 1024)
                - load_in_4bit: Whether to use 4-bit quantization (default: False)
                - load_in_8bit: Whether to use 8-bit quantization (default: False)
                - temperature: Sampling temperature (default: 0.6)
                - top_p: Top-p sampling (default: 0.95)
                - top_k: Top-k sampling (default: 20)
        """
        super().__init__(model_config)

        self.model_id = model_config.get("hf_model_id", "Qwen/Qwen3-VL-8B-Thinking")
        self.device = model_config.get("device", "cuda")
        self.frames_per_sample = model_config.get("frames_per_sample", 8)
        self.samples_per_clip = model_config.get("samples_per_clip", 1)
        self.max_new_tokens = model_config.get("max_new_tokens", 1024)
        self.load_in_4bit = model_config.get("load_in_4bit", False)
        self.load_in_8bit = model_config.get("load_in_8bit", False)
        self.temperature = model_config.get("temperature", 0.6)
        self.top_p = model_config.get("top_p", 0.95)
        self.top_k = model_config.get("top_k", 20)
        self.local_files_only = model_config.get("local_files_only", True)  # Use cache by default

        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load the Qwen3-VL model and processor."""
        from transformers import AutoProcessor, AutoModelForCausalLM
        from qwen_vl_utils import process_vision_info

        print(f"Loading Qwen3 model: {self.model_id}...")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )

        # Determine dtype
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Quantization config
        quantization_config = None
        device_map = None
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            device_map = "auto"
        elif self.load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            device_map = "auto"
        else:
            device_map = "auto"

        model_cls = None
        if "moe" in self.model_id.lower() or "a22b" in self.model_id.lower():
            try:
                from transformers import Qwen3VLMoeForConditionalGeneration
                model_cls = Qwen3VLMoeForConditionalGeneration
                print("Using Qwen3VLMoeForConditionalGeneration for MoE model")
            except ImportError:
                model_cls = AutoModelForCausalLM
        else:
            try:
                from transformers import Qwen3VLForConditionalGeneration
                model_cls = Qwen3VLForConditionalGeneration
                print("Using Qwen3VLForConditionalGeneration for non-MoE model")
            except ImportError:
                model_cls = AutoModelForCausalLM

        # Load model
        self.model = model_cls.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )

        self._loaded = True
        print("Qwen3 model loaded successfully.")

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

    def _build_messages(
        self, frames: List[Image.Image], prompt: str
    ) -> List[Dict[str, Any]]:
        """Build chat messages with images.

        Args:
            frames: List of PIL Image frames.
            prompt: Text prompt.

        Returns:
            List of message dictionaries for chat template.
        """
        content: List[Dict[str, Any]] = []
        for frame in frames:
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    def _strip_thinking_process(self, text: str) -> str:
        """Strip <think>...</think> blocks from output to get the final answer.

        Args:
            text: Raw model output that may contain thinking blocks.

        Returns:
            The final answer after the thinking process.
        """
        if "<think>" in text and "</think>" in text:
            parts = text.split("</think>")
            return parts[-1].strip()
        return text.strip()

    def _run_one_window(self, frames: List[Image.Image], prompt: str) -> str:
        """Run inference on one frame window.

        Args:
            frames: List of PIL Image frames.
            prompt: Text prompt.

        Returns:
            Model output text (with thinking stripped).
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        from qwen_vl_utils import process_vision_info

        # Build messages
        messages = self._build_messages(frames, prompt)

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )

        # Decode - trim input tokens
        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Strip thinking process
        return self._strip_thinking_process(output_text)

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
