"""Postprocessing and validation functions for VLM outputs."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Union

INVALID_LABEL = "INVALID"

# Word-to-number mapping for extracting counts from text
WORD_TO_NUM = {
    "zero": 0, "none": 0, "no": 0,
    "one": 1, "a": 1, "an": 1, "single": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


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


def validate_counting_output(
    raw_output: str,
) -> Tuple[Optional[int], Dict]:
    """Extract integer count from raw model output.

    Handles:
    - Direct digits: "3", "0", "10"
    - Word numbers: "one", "two", "three", etc.
    - Numbers in sentences: "There are 2 adults", "I see three children"

    Args:
        raw_output: Raw text output from the model.

    Returns:
        Tuple of (extracted_count, debug_dict).
        If extraction fails, extracted_count is None.
    """
    debug: Dict = {"raw_output": raw_output}

    if raw_output is None:
        debug["reason"] = "raw_output_none"
        return None, debug

    out = _normalize_space(raw_output)
    if not out:
        debug["reason"] = "empty_output"
        return None, debug

    out_lower = out.lower()

    # Strategy 1: Try to find a standalone digit/number at the start or as the whole output
    # This handles cases like "3" or "3 adults"
    match = re.match(r"^(\d+)", out_lower)
    if match:
        value = int(match.group(1))
        debug["mode"] = "leading_digit"
        debug["value"] = value
        return value, debug

    # Strategy 2: Look for word numbers at the start
    for word, num in WORD_TO_NUM.items():
        if out_lower.startswith(word) and (
            len(out_lower) == len(word) or not out_lower[len(word)].isalnum()
        ):
            debug["mode"] = "leading_word"
            debug["value"] = num
            debug["matched_word"] = word
            return num, debug

    # Strategy 3: Extract any digit from the text (take the first one found)
    digit_match = re.search(r"\b(\d+)\b", out_lower)
    if digit_match:
        value = int(digit_match.group(1))
        debug["mode"] = "extracted_digit"
        debug["value"] = value
        return value, debug

    # Strategy 4: Look for word numbers anywhere in the text
    found_words = []
    for word, num in WORD_TO_NUM.items():
        # Look for word boundaries
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, out_lower):
            found_words.append((word, num))

    # Filter out words like "a", "an", "no" if there are better matches
    if found_words:
        # Prefer longer/more specific matches
        found_words.sort(key=lambda x: len(x[0]), reverse=True)
        # Skip "a"/"an" if they appear in context like "a video"
        for word, num in found_words:
            if word in ("a", "an"):
                # Only use if it's like "a single" or appears to be counting
                continue
            if word == "no" and "no " in out_lower:
                # "no adults" likely means 0
                debug["mode"] = "word_in_text"
                debug["value"] = 0
                debug["matched_word"] = "no"
                return 0, debug
            debug["mode"] = "word_in_text"
            debug["value"] = num
            debug["matched_word"] = word
            return num, debug

    debug["reason"] = "no_number_found"
    return None, debug
