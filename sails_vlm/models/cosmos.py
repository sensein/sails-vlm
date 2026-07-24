"""Cosmos-Reason2 model wrapper (Qwen3-VL-based) for the baseline framework."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from PIL import Image

# Cosmos-Reason2 is Qwen3-VL-based and supported in transformers>=4.57.0
try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Cosmos-Reason2 requires a recent Transformers with Qwen3-VL support.\n"
        'Please install: pip install "transformers>=4.57.0"\n'
        f"Original import error: {e!r}"
    )

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
    """Same window-index logic as your original script.

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


class CosmosReason2VLM(BaseVLM):
    """Runner-compatible wrapper for NVIDIA Cosmos-Reason2 (Qwen3-VL based)."""

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initialization of Cosmos w configuration file."""
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
                    default="nvidia/Cosmos-Reason2-2B",
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

        # Frame sampling (same as your wrapper)
        self.frames_per_sample = int(
            cast(int, _get_cfg(self.config, "frames_per_sample", default=6))
        )
        self.samples_per_clip = int(
            cast(int, _get_cfg(self.config, "samples_per_clip", default=4))
        )

        # Optional system prompt (Cosmos model card recommends a reasoning format)
        self.system_prompt = cast(
            str,
            _get_cfg(
                self.config,
                "system_prompt",
                default="",
            ),
        )

        # Generation params
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

        # Optional perf knobs
        self.attn_implementation = cast(
            Optional[str],
            _get_cfg(self.config, "attn_implementation", default=None),
        )

        # Optional: avoid HF hub checks
        self.local_files_only = bool(
            cast(bool, _get_cfg(self.config, "local_files_only", default=False))
        )

        # Optional aliases for label extraction
        self.label_aliases = cast(
            Optional[Dict[str, List[str]]], self.config.get("label_aliases", None)
        )

        self.model: Optional[torch.nn.Module] = None
        self.processor: Optional[AutoProcessor] = None

    def _infer_model_device(self) -> torch.device:
        assert self.model is not None
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return next(self.model.parameters()).device

    def load(self) -> None:
        """Method to load the HF model."""
        if self._loaded:
            return

        # If you want strict single-device placement, set device_map=None in config.
        device_map = self.config.get("device_map", None)
        if device_map is None and self.device.type == "cuda":
            device_map = "auto"

        kwargs: Dict[str, Any] = dict(
            torch_dtype="auto",
            device_map=device_map,
            local_files_only=self.local_files_only,
        )
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        # Cosmos-Reason2 is an Image-Text-to-Text model (Qwen3-VL family)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, local_files_only=self.local_files_only
        )

        if device_map is None:
            self.model = self.model.to(self.device)

        self.model.eval()
        self._loaded = True

    # -------------------------
    # Message format + inference (Qwen3-VL style)
    # -------------------------
    def _build_messages(
        self, frames: List[Image.Image], prompt: str
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []

        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )

        content: List[Dict[str, Any]] = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def _run_one_window(self, frames: List[Image.Image], prompt: str) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = self._build_messages(frames, prompt)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        # Some Qwen3-VL processors may return token_type_ids; safe to drop.
        try:
            inputs.pop("token_type_ids", None)
        except Exception:
            pass

        target_device = self._infer_model_device()
        if hasattr(inputs, "to"):
            inputs = inputs.to(target_device)
        else:
            inputs = {
                k: (v.to(target_device) if torch.is_tensor(v) else v)
                for k, v in inputs.items()
            }

        gen_kwargs: Dict[str, Any] = dict(max_new_tokens=self.max_new_tokens)
        if self.do_sample:
            gen_kwargs.update(
                dict(do_sample=True, temperature=self.temperature, top_p=self.top_p)
            )
        else:
            gen_kwargs.update(dict(do_sample=False))

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)  # type: ignore[arg-type]

        # Trim prompt tokens
        input_ids = (
            inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        )
        trimmed = [out[len(in_ids) :] for in_ids, out in zip(input_ids, generated_ids)]

        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return decoded.strip()

    # -------------------------
    # Label extraction + voting
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

    def generate(
        self,
        video_path: str,
        prompt: str,
        labels: Optional[Sequence[str]] = None,
    ) -> VLMRawOutput:
        """Method to generate prediction based on video +prompt."""
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
                    "model": "cosmos-reason2",
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
                "model": "cosmos-reason2",
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
                "model": "cosmos-reason2",
                "hf_model_id": self.model_id,
                "frames_per_sample": self.frames_per_sample,
                "samples_per_clip": self.samples_per_clip,
                "completions": completions,
                "warning": "No labels provided; skipping voting.",
                **fmeta,
            },
        )

    # -------------------------
    # Frame windows (unchanged)
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
                    "error": f"VideoCaptureNotOpened; decord_error={decord_err}",
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
                        "error": f"NoFramesDecoded; decord_error={decord_err}",
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
                win_frames_local: List[Image.Image] = []
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
            from moviepy.editor import VideoFileClip  # type: ignore

            windows = []
            with VideoFileClip(str(p)) as clip:
                if clip.duration is None or clip.fps is None:
                    return [], {
                        "frame_backend": "moviepy",
                        "error": f"NoDurationOrFPS; decord_error={decord_err};"
                        f" opencv_error={opencv_err}",
                    }

                total = int(clip.fps * clip.duration)
                if total <= 0:
                    return [], {
                        "frame_backend": "moviepy",
                        "error": f"total_frames<=0; decord_error={decord_err}; "
                        f"opencv_error={opencv_err}",
                    }

                index_sets = sample_frame_indices(
                    total, frames_per_sample, samples_per_clip
                )
                for idxs in index_sets:
                    win_frames_local_: List[Image.Image] = []
                    for idx in idxs.tolist():
                        t = min(max(idx / clip.fps, 0.0), clip.duration - 1e-3)
                        frame = clip.get_frame(t)
                        win_frames_local_.append(Image.fromarray(frame))
                    if win_frames_local_:
                        windows.append(win_frames_local_)

            if not windows:
                return [], {
                    "frame_backend": "moviepy",
                    "error": f"NoWindowsDecoded; decord_error={decord_err}; "
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
                "error": f"moviepy_failed={repr(e)}; decord_error={decord_err}; "
                f"opencv_error={opencv_err}",
            }
