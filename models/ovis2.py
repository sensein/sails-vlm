"""Ovis2 VLM model wrapper for the baseline framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
from moviepy.editor import VideoFileClip
from PIL import Image
from transformers import AutoModelForCausalLM

from .base_vlm import BaseVLM, VLMRawOutput


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").lower().strip()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _get_cfg(
    cfg: Dict[str, Any],
    *keys: str,
    default: Union[str, int, bool, Dict[str, Any], None] = None,
) -> Union[str, int, bool, Dict[str, Any], None]:
    """Return first found key in cfg (supports multiple aliases)."""
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    return default


class Ovis2VLM(BaseVLM):
    """Runner-compatible wrapper for Ovis2 VLM.

    Provides a standardized interface for the Ovis2 model with methods:
    - __init__(model_config)
    - load()
    - predict(video_path, prompt) -> str
    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initialize the Ovis2 VLM wrapper.

        Args:
            model_config: Configuration dictionary for the model.
        """
        self.config = model_config or {}
        self.model: Optional[Any] = None
        self.device: str = str(
            cast(str, _get_cfg(self.config, "device", default="cpu"))
        )
        self._loaded = False

        # Runner config uses these keys:
        #   precision: "bf16"
        #   max_frames: 16
        # Allow aliases for convenience.
        self.precision = str(
            cast(str, _get_cfg(self.config, "precision", "torch_dtype", default="bf16"))
        )
        self.max_frames = int(
            cast(int, _get_cfg(self.config, "max_frames", default=16))
        )

        # HF id / path aliases
        self.model_path = str(
            cast(
                str,
                _get_cfg(
                    self.config,
                    "hf_model_id",
                    "model_path",
                    default="AIDC-AI/Ovis2.5-9B",
                ),
            )
        )

        self.trust_remote_code = bool(
            cast(bool, _get_cfg(self.config, "trust_remote_code", default=True))
        )

        # Optional sub-config blocks (kept for compatibility with your existing pattern)
        video_cfg_raw = cast(Dict[str, Any], _get_cfg(self.config, "video", default={}))
        self.video_cfg = dict(video_cfg_raw) if video_cfg_raw is not None else {}
        gen_cfg_raw = cast(
            Dict[str, Any], _get_cfg(self.config, "generation", default={})
        )
        self.gen_cfg = dict(gen_cfg_raw) if gen_cfg_raw is not None else {}

    def load(self) -> None:
        """Load the Ovis2 model and move it to the specified device."""
        if self._loaded:
            return

        torch_dtype = _dtype_from_str(self.precision)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )

        # Device handling
        if self.device.startswith("cuda") and torch.cuda.is_available():
            self.model = self.model.to(self.device)
        else:
            self.device = "cpu"
            self.model = self.model.to("cpu")

        self.model.eval()
        self._loaded = True

    # -------------------------
    # Public API expected by runner
    # -------------------------
    # Note: Using base class predict() which calls generate().raw_text

    # -------------------------
    # Core generation
    # -------------------------
    def _build_messages(
        self, frames: List[Image.Image], prompt: str
    ) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def generate(
        self,
        video_path: str,
        prompt: str,
        video_cfg: Optional[Dict[str, Any]] = None,
        gen_cfg: Optional[Dict[str, Any]] = None,
    ) -> VLMRawOutput:
        """Generate a response for the given video and prompt.

        Args:
            video_path: Path to the video file.
            prompt: Text prompt for the model.
            video_cfg: Optional video processing configuration.
            gen_cfg: Optional generation configuration.

        Returns:
            VLMRawOutput containing the generated text and metadata.
        """
        if self.model is None or self.device is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        video_cfg = video_cfg or self.config.get("video", {}) or {}
        gen_cfg = gen_cfg or self.config.get("generation", {}) or {}

        frames, fmeta = self._extract_frames(video_path, video_cfg)
        if not frames:
            return VLMRawOutput(
                raw_text="", meta={"model": "ovis2", "empty_frames": True, **fmeta}
            )

        messages = self._build_messages(frames, prompt)

        # 1) PREPROCESS (same as your snippet)
        max_pixels = int(video_cfg.get("max_pixels", 896 * 896))
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
            max_pixels=max_pixels,
        )
        # 2) MOVE TO DEVICE (equivalent to .cuda(), but device-agnostic)
        input_ids = input_ids.to(self.device)
        pixel_values = (
            pixel_values.to(self.device).to(self.model.dtype)
            if pixel_values is not None
            else None
        )
        grid_thws = grid_thws.to(self.device) if grid_thws is not None else None

        # 3) INFERENCE (same as your snippet)
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 128))
        do_sample = bool(gen_cfg.get("do_sample", False))

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                eos_token_id=self.model.text_tokenizer.eos_token_id,
                pad_token_id=self.model.text_tokenizer.pad_token_id,
            )

            # 4) DECODE (same as your snippet)
            answer = self.model.text_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()
        return VLMRawOutput(raw_text=answer, meta={"model": "ovis2", **fmeta})

        # --- Frame extraction (keep here for now; move later if you want) ---

    def _extract_frames(
        self, video_path: str, video_cfg: Dict[str, Any]
    ) -> Tuple[List[Image.Image], Dict[str, Any]]:
        num_frames = int(video_cfg.get("num_frames", 16))
        sampling = str(video_cfg.get("sampling", "uniform")).lower()

        p = Path(str(video_path))
        if not p.exists():
            # IMPORTANT: don’t hide this — it’s usually the real bug on clusters
            return [], {"frame_backend": None, "error": f"FileNotFound: {p}"}

        # 1) decord
        try:
            from decord import VideoReader, cpu  # type: ignore

            vr = VideoReader(str(p), ctx=cpu(0))
            n = len(vr)
            if n <= 0:
                return [], {"frame_backend": "decord", "error": "EmptyVideo(len=0)"}

            if sampling != "uniform":
                raise ValueError(f"Unknown sampling strategy: {sampling}")

            if n <= num_frames:
                idxs = list(range(n))
            else:
                step = (n - 1) / max(num_frames - 1, 1)
                idxs = [int(round(i * step)) for i in range(num_frames)]

            frames_np = vr.get_batch(idxs).asnumpy()  # (T,H,W,C) RGB
            frames_decord = [
                Image.fromarray(frames_np[i]) for i in range(frames_np.shape[0])
            ]
            return frames_decord, {"frame_backend": "decord", "n_frames_video": n}
        except Exception as e:
            decord_err = repr(e)

        # 2) OpenCV fallback (often works even when moviepy/ffmpeg is weird)
        try:
            import cv2  # type: ignore

            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                return [], {
                    "frame_backend": "opencv",
                    "error": f"VideoCaptureNotOpened; decord_error={decord_err}",
                }

            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            if sampling != "uniform":
                cap.release()
                raise ValueError(f"Unknown sampling strategy: {sampling}")

            frames_opencv: List[Image.Image] = []

            if n > 0:
                # sample indices
                if n <= num_frames:
                    idxs = list(range(n))
                else:
                    step = (n - 1) / max(num_frames - 1, 1)
                    idxs = [int(round(i * step)) for i in range(num_frames)]
                wanted = set(idxs)

                i = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if i in wanted:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames_opencv.append(Image.fromarray(frame))
                        if len(frames_opencv) >= len(idxs):
                            break
                    i += 1
            else:
                # unknown frame count: just take first num_frames
                while len(frames_opencv) < num_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_opencv.append(Image.fromarray(frame))

            cap.release()
            if not frames_opencv:
                return [], {
                    "frame_backend": "opencv",
                    "error": f"NoFramesDecoded; decord_error={decord_err}",
                }
            return frames_opencv, {
                "frame_backend": "opencv",
                "n_frames_video": n,
                "decord_error": decord_err,
            }
        except Exception as e:
            opencv_err = repr(e)

        # 3) moviepy fallback (relies on ffmpeg)
        try:
            # type: ignore

            frames_moviepy: List[Image.Image] = []
            with VideoFileClip(str(p)) as clip:
                if clip.duration is None or clip.fps is None:
                    return [], {
                        "frame_backend": "moviepy",
                        "error": (
                            f"NoDurationOrFPS; decord_error={decord_err}; "
                            f"opencv_error={opencv_err}"
                        ),
                    }

                total_frames = int(clip.fps * clip.duration)
                if total_frames <= 0:
                    return [], {
                        "frame_backend": "moviepy",
                        "error": (
                            f"total_frames<=0; decord_error={decord_err}; "
                            f"opencv_error={opencv_err}"
                        ),
                    }

                if sampling != "uniform":
                    raise ValueError(f"Unknown sampling strategy: {sampling}")

                idxs = [
                    int(i * (total_frames - 1) / max(num_frames - 1, 1))
                    for i in range(num_frames)
                ]
                for idx in idxs:
                    t = min(max(idx / clip.fps, 0.0), clip.duration - 1e-3)
                    frame = clip.get_frame(t)
                    frames_moviepy.append(Image.fromarray(frame))

            if not frames_moviepy:
                return [], {
                    "frame_backend": "moviepy",
                    "error": (
                        f"NoFramesDecoded; decord_error={decord_err}; "
                        f"opencv_error={opencv_err}"
                    ),
                }
            return frames_moviepy, {
                "frame_backend": "moviepy",
                "total_frames_est": total_frames,
                "decord_error": decord_err,
                "opencv_error": opencv_err,
            }
        except Exception as e:
            return [], {
                "frame_backend": "moviepy",
                "error": (
                    f"moviepy_failed={repr(e)}; decord_error={decord_err}; "
                    f"opencv_error={opencv_err}"
                ),
            }
