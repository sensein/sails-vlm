"""Ovis2 VLM model for the baseline framework (window sampling + max-pool voting)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
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
    default: Union[str, int, bool, Dict[str, Any], List[Any], None] = None,
) -> Union[str, int, bool, Dict[str, Any], List[Any], None]:
    """Return first found key in cfg (supports multiple aliases)."""
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    return default


def sample_frame_indices(
    total_frames: int, frames_per_sample: int, samples_per_clip: int
) -> List[np.ndarray]:
    """Sample multiple windows across the clip.

    Returns a list of arrays; each array is length frames_per_sample.
    """
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


class Ovis2VLM(BaseVLM):
    """Runner-compatible wrapper for Ovis2 VLM."""

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initialize Ovis2 VLM with configuration."""
        self.config = model_config or {}
        self.model: Optional[Any] = None
        self.device: str = str(
            cast(str, _get_cfg(self.config, "device", default="cpu"))
        )
        self._loaded = False

        self.precision = str(
            cast(str, _get_cfg(self.config, "precision", "torch_dtype", default="bf16"))
        )
        self.frames_per_sample = int(
            cast(int, _get_cfg(self.config, "frames_per_sample", default=6))
        )
        self.samples_per_clip = int(
            cast(int, _get_cfg(self.config, "samples_per_clip", default=4))
        )

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

    def load(self) -> None:
        """Load Ovis2 model weights and prepare for inference."""
        if self._loaded:
            return

        torch_dtype = _dtype_from_str(self.precision)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )

        if self.device.startswith("cuda") and torch.cuda.is_available():
            self.model = self.model.to(self.device)
        else:
            self.device = "cpu"
            self.model = self.model.to("cpu")

        self.model.eval()
        self._loaded = True

    # -------------------------
    # Message building / inference
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

    def _run_one_window(
        self,
        frames: List[Image.Image],
        prompt: str,
    ) -> str:
        """Run Ovis2 on one window of frames and return decoded string."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = self._build_messages(frames, prompt)

        max_pixels = int(self.config.get("max_pixels", 896 * 896))
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
            max_pixels=max_pixels,
        )

        input_ids = input_ids.to(self.device)
        pixel_values = (
            pixel_values.to(self.device).to(self.model.dtype)
            if pixel_values is not None
            else None
        )
        grid_thws = grid_thws.to(self.device) if grid_thws is not None else None

        max_new_tokens = int(self.config.get("max_new_tokens", 600))
        do_sample = bool(self.config.get("do_sample", False))

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
        return self.model.text_tokenizer.decode(
            outputs[0], skip_special_tokens=True
        ).strip()

    # -------------------------
    # Label extraction + voting (Qwen-style)
    # -------------------------
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

    def _compute_class_scores(
        self, sample_preds: List[Optional[str]], labels: Sequence[str]
    ) -> Dict[str, float]:
        valid = [p for p in sample_preds if p is not None]
        if not valid:
            return {lab: 1.0 / max(len(labels), 1) for lab in labels}

        counts = Counter(valid)
        total = sum(counts.values())
        return {lab: counts.get(lab, 0) / total for lab in labels}

    # -------------------------
    # Public API
    # -------------------------
    def generate(
        self,
        video_path: str,
        prompt: str,
        labels: Optional[Sequence[str]] = None,
    ) -> VLMRawOutput:
        """Generate output for video + prompt."""
        labels_list: List[str] = list(labels) if labels else []

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Window sampling params (mirror Qwen script behavior)
        frames_per_sample = self.frames_per_sample
        samples_per_clip = self.samples_per_clip

        # Optional label set for voting
        label_aliases = cast(
            Optional[Dict[str, List[str]]],
            self.config.get("label_aliases", None),
        )

        windows, fmeta = self._extract_frame_windows(
            video_path=video_path,
            frames_per_sample=frames_per_sample,
            samples_per_clip=samples_per_clip,
        )

        if not windows:
            return VLMRawOutput(
                raw_text="",
                meta={"model": "ovis2", "empty_frames": True, **fmeta},
            )

        # Run model per window
        completions: List[str] = []
        sample_preds: List[Optional[str]] = []

        for win_frames in windows:
            txt = self._run_one_window(win_frames, prompt)
            completions.append(txt)

            if labels_list:
                pred = self._extract_label(
                    txt, labels=labels_list, aliases=label_aliases
                )
                sample_preds.append(pred)

        # If we have labels -> do Qwen-style scores + max-pool pick
        if labels_list:
            scores = self._compute_class_scores(sample_preds, labels_list)
            sorted_labels = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
            top2 = sorted_labels[:2]
            top1 = top2[0] if top2 else None

            meta = {
                "model": "ovis2",
                "frames_per_sample": frames_per_sample,
                "samples_per_clip": samples_per_clip,
                "sample_predictions": sample_preds,
                "scores": scores,
                "top2": top2,
                "completions": completions,
                **fmeta,
            }
            return VLMRawOutput(raw_text=top1 or "", raw_topk=top2, meta=meta)

        # Otherwise: no voting possible
        return VLMRawOutput(
            raw_text=completions[0] if completions else "",
            meta={
                "model": "ovis2",
                "frames_per_sample": frames_per_sample,
                "samples_per_clip": samples_per_clip,
                "completions": completions,
                "warning": "No labels provided; skipping voting.",
                **fmeta,
            },
        )

    # -------------------------
    # Frame sampling: extract multiple windows across clip
    # -------------------------
    def _extract_frame_windows(
        self,
        video_path: str,
        frames_per_sample: int,
        samples_per_clip: int,
    ) -> Tuple[List[List[Image.Image]], Dict[str, Any]]:
        """Returns: list of windows of length frames_per_sample."""
        p = Path(str(video_path))
        if not p.exists():
            return [], {"frame_backend": None, "error": f"FileNotFound: {p}"}

        windows: List[List[Image.Image]] = []  # define ONCE
        decord_err: str = ""
        opencv_err: str = ""

        # 1) decord
        try:
            from decord import VideoReader, cpu  # type: ignore

            vr = VideoReader(str(p), ctx=cpu(0), num_threads=1)
            total = len(vr)
            if total <= 0:
                return [], {"frame_backend": "decord", "error": "EmptyVideo(len=0)"}

            index_sets = sample_frame_indices(
                total, frames_per_sample, samples_per_clip
            )
            windows = []
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
            decord_err = repr(e)

        # 2) OpenCV fallback
        try:
            import cv2  # type: ignore

            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                return [], {
                    "frame_backend": "opencv",
                    "error": "" "" f"VideoCaptureNotOpened; decord_error={decord_err}",
                }

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            if total <= 0:
                frames_all: List[Image.Image] = []
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_all.append(Image.fromarray(frame))
                cap.release()

                total = len(frames_all)
                if total <= 0:
                    return [], {
                        "frame_backend": "opencv",
                        "error": "" "" f"NoFramesDecoded; decord_error={decord_err}",
                    }

                index_sets = sample_frame_indices(
                    total, frames_per_sample, samples_per_clip
                )
                windows = [[frames_all[int(i)] for i in idxs] for idxs in index_sets]
                return windows, {
                    "frame_backend": "opencv",
                    "n_frames_video": total,
                    "n_windows": len(windows),
                    "decord_error": decord_err,
                }

            index_sets = sample_frame_indices(
                total, frames_per_sample, samples_per_clip
            )
            windows = []
            for idxs in index_sets:
                win_frames_local: List[Image.Image] = []  # renamed to avoid reuse
                for idx in idxs.tolist():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    win_frames_local.append(Image.fromarray(frame))

                if win_frames_local:
                    while len(win_frames_local) < frames_per_sample:
                        win_frames_local.append(win_frames_local[-1])
                    windows.append(win_frames_local)

            cap.release()

            if not windows:
                return [], {
                    "frame_backend": "opencv",
                    "error": f"NoWindowsDecoded; decord_error={decord_err}",
                }

            return windows, {
                "frame_backend": "opencv",
                "n_frames_video": total,
                "n_windows": len(windows),
                "decord_error": decord_err,
            }
        except Exception as e:
            opencv_err = repr(e)

        # 3) moviepy fallback
        try:
            from moviepy.editor import VideoFileClip

            windows = []
            with VideoFileClip(str(p)) as clip:
                if clip.duration is None or clip.fps is None:
                    return [], {
                        "frame_backend": "moviepy",
                        "error": "NoDurationOrFPS;"
                        f"decord_error={decord_err}; opencv_error={opencv_err}",
                    }

                total = int(clip.fps * clip.duration)
                if total <= 0:
                    return [], {
                        "frame_backend": "moviepy",
                        "error": "total_frames<=0;"
                        f"decord_error={decord_err}; opencv_error={opencv_err}",
                    }

                index_sets = sample_frame_indices(
                    total, frames_per_sample, samples_per_clip
                )
                for idxs in index_sets:
                    win_frames_local_: List[
                        Image.Image
                    ] = []  # use same local name here too
                    for idx in idxs.tolist():
                        t = min(max(idx / clip.fps, 0.0), clip.duration - 1e-3)
                        frame = clip.get_frame(t)
                        win_frames_local_.append(Image.fromarray(frame))
                    if win_frames_local_:
                        windows.append(win_frames_local_)

            if not windows:
                return [], {
                    "frame_backend": "moviepy",
                    "error": "NoWindowsDecoded;"
                    f"decord_error={decord_err};"
                    f"opencv_error={opencv_err}",
                }

            return windows, {
                "frame_backend": "moviepy",
                "total_frames_est": total,
                "n_windows": len(windows),
                "decord_error": decord_err,
                "opencv_error": opencv_err,
            }
        except Exception as e:
            return [], {
                "frame_backend": "moviepy",
                "error": f"moviepy_failed={repr(e)};"
                f"decord_error={decord_err};"
                f"opencv_error={opencv_err}",
            }
