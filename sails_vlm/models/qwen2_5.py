"""Qwen2.5-VL model wrapper for the baseline framework."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info  # from your RMM script
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .base_vlm import BaseVLM, VLMRawOutput


def _get_cfg(
    cfg: Dict[str, Any],
    *keys: str,
    default: Union[str, int, float, bool, Dict[str, Any], List[Any], None] = None,
) -> Union[str, int, float, bool, Dict[str, Any], List[Any], None]:
    """Return first found key in cfg (supports multiple aliases)."""
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    return default


def sample_frame_indices(
    total_frames: int, frames_per_sample: int, samples_per_clip: int
) -> List[np.ndarray]:
    """Same window-index logic as your Qwen RMM script.

    Returns a list of arrays; each array length == frames_per_sample.
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


class Qwen25VLM(BaseVLM):
    """Runner-compatible wrapper for Qwen2.5-VL."""

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initialize with model configuration dictionary."""
        self.config = model_config or {}
        self._loaded = False

        # Model + device
        self.model_id = str(
            cast(
                str,
                _get_cfg(
                    self.config,
                    "hf_model_id",
                    "model_id",
                    "model_path",
                    default="Qwen/Qwen2.5-VL-7B-Instruct",
                ),
            )
        )
        self.device_str = str(
            cast(str, _get_cfg(self.config, "device", default="cuda"))
        )
        self.device = torch.device(
            self.device_str
            if ("cuda" in self.device_str and torch.cuda.is_available())
            else "cpu"
        )

        # Sampling params (top-level, like your current ovis2.py)
        self.frames_per_sample = int(
            cast(int, _get_cfg(self.config, "frames_per_sample", default=6))
        )
        self.samples_per_clip = int(
            cast(int, _get_cfg(self.config, "samples_per_clip", default=4))
        )

        # Generation params (top-level)
        self.max_new_tokens = int(
            cast(int, _get_cfg(self.config, "max_new_tokens", default=600))
        )
        self.do_sample = bool(
            cast(bool, _get_cfg(self.config, "do_sample", default=False))
        )
        self.temperature = float(
            cast(float, _get_cfg(self.config, "temperature", default=1.0))
        )
        self.top_p = float(cast(float, _get_cfg(self.config, "top_p", default=1.0)))

        # Optional: avoid HF hub checks (like your RMM script)
        self.local_files_only = bool(
            cast(bool, _get_cfg(self.config, "local_files_only", default=False))
        )

        # Optional aliases for label extraction (top-level)
        self.label_aliases = cast(
            Optional[Dict[str, List[str]]],
            self.config.get("label_aliases", None),
        )

        self.model: Optional[Qwen2_5_VLForConditionalGeneration] = None
        self.processor: Optional[AutoProcessor] = None

    def _infer_model_device(self) -> torch.device:
        assert self.model is not None
        # safest: embeddings device
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            # fallback: first parameter device
            return next(self.model.parameters()).device

    def load(self) -> None:
        """Method to load model from HuggingFace."""
        if self._loaded:
            return

        # Qwen VL is often large; "device_map=auto" can be helpful.
        # If you want strict single-device placement, set device_map=None in config.
        device_map = self.config.get("device_map", None)
        if device_map is None and self.device.type == "cuda":
            device_map = "auto"

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map=device_map,
            local_files_only=self.local_files_only,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            local_files_only=self.local_files_only,
        )

        # If NOT using device_map, move explicitly
        if device_map is None:
            self.model = self.model.to(self.device)

        self.model.eval()
        self._loaded = True

    # -------------------------
    # Qwen message format + inference
    # -------------------------
    def _build_messages(
        self, frames: List[Image.Image], prompt: str
    ) -> List[Dict[str, Any]]:
        # Qwen chat format: list of {"role":..., "content":[...]}
        content: List[Dict[str, Any]] = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _run_one_window(self, frames: List[Image.Image], prompt: str) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = self._build_messages(frames, prompt)

        # Build text with chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision inputs
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        # fps edge case (copied from your RMM script)
        if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
            video_kwargs["fps"] = (
                video_kwargs["fps"][0] if video_kwargs["fps"] else None
            )
        if video_kwargs.get("fps") is None:
            video_kwargs.pop("fps", None)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

        target_device = self._infer_model_device()
        inputs = {
            k: (v.to(target_device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                top_p=self.top_p if self.do_sample else None,
            )

        # Trim prompt tokens
        trimmed = [
            out[len(in_ids) :]
            for in_ids, out in zip(inputs["input_ids"], generated_ids)
        ]

        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return decoded.strip()

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

        # canonical labels first (longest first)
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

    def generate(
        self,
        video_path: str,
        prompt: str,
        labels: Optional[Sequence[str]] = None,
    ) -> VLMRawOutput:
        """Model method generate to predict based on video and prompt."""
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
                meta={"model": "qwen2.5-vl", "empty_frames": True, **fmeta},
            )

        completions: List[str] = []
        sample_preds: List[Optional[str]] = []

        for win_frames in windows:
            txt = self._run_one_window(win_frames, prompt)
            completions.append(txt)

            if labels_list:
                pred = self._extract_label(txt, labels_list, aliases=self.label_aliases)
                sample_preds.append(pred)

        # Classification path: voting + top2
        if labels_list:
            scores = self._compute_class_scores(sample_preds, labels_list)
            sorted_labels = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
            top2 = sorted_labels[:2]
            top1 = top2[0] if top2 else None

            meta = {
                "model": "qwen2.5-vl",
                "frames_per_sample": self.frames_per_sample,
                "samples_per_clip": self.samples_per_clip,
                "sample_predictions": sample_preds,
                "scores": scores,
                "top2": top2,
                "completions": completions,
                **fmeta,
            }
            return VLMRawOutput(raw_text=top1 or "", raw_topk=top2, meta=meta)

        # Description path: no voting
        return VLMRawOutput(
            raw_text=completions[0] if completions else "",
            raw_topk=[],
            meta={
                "model": "qwen2.5-vl",
                "frames_per_sample": self.frames_per_sample,
                "samples_per_clip": self.samples_per_clip,
                "completions": completions,
                "warning": "No labels provided; skipping voting.",
                **fmeta,
            },
        )

    # -------------------------
    # Frame windows
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
