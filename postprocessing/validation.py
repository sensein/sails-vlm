"""Postprocessing and validation functions for VLM outputs."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

INVALID_LABEL = "INVALID"


def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def _normalize_label(s: str) -> str:
    return _normalize_space(s).lower()


def validate_classification_output(
    raw_output: str,
    allowed_labels: List[str],
    *,
    invalid_label: str = INVALID_LABEL,
) -> Tuple[str, Dict]:
    """Returns: (final_label, debug).

    - final_label is always a string:
        - one of allowed_labels, OR invalid_label
    """
    debug: Dict = {"raw_output": raw_output}

    if raw_output is None:
        debug["reason"] = "raw_output_none"
        return invalid_label, debug

    out = _normalize_space(raw_output)
    if not out:
        debug["reason"] = "empty_output"
        return invalid_label, debug

    allowed_norm_to_orig = {_normalize_label(label): label for label in allowed_labels}
    out_norm = _normalize_label(out)

    # 1) Exact match
    if out_norm in allowed_norm_to_orig:
        label = allowed_norm_to_orig[out_norm]
        debug["mode"] = "exact"
        debug["label"] = label
        return label, debug

    # 2) Extract if exactly one allowed label is present
    hits = []
    for label_norm, label_orig in allowed_norm_to_orig.items():
        pattern = r"(?:^|[^a-z0-9])" + re.escape(label_norm) + r"(?:$|[^a-z0-9])"
        if re.search(pattern, out_norm):
            hits.append(label_orig)

    hits = list(dict.fromkeys(hits))
    debug["hits"] = hits

    if len(hits) == 1:
        debug["mode"] = "single_hit"
        debug["label"] = hits[0]
        return hits[0], debug

    debug["reason"] = (
        "no_unique_label_found" if len(hits) == 0 else "multiple_labels_found"
    )
    return invalid_label, debug
