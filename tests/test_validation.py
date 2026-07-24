"""Characterization tests for postprocessing.validation (existing behavior)."""
from sails_vlm.postprocessing.validation import (
    INVALID_LABEL,
    validate_classification_output,
    validate_counting_output,
)

LABELS = ["hands flapping", "rocking", "spinning", "jumping"]


class TestValidateClassification:
    def test_exact_match(self):
        label, debug = validate_classification_output("rocking", LABELS)
        assert label == "rocking"
        assert debug["mode"] == "exact"

    def test_exact_match_is_case_and_space_insensitive(self):
        label, _ = validate_classification_output("  Rocking ", LABELS)
        assert label == "rocking"

    def test_single_label_embedded_in_sentence(self):
        label, debug = validate_classification_output(
            "RMM: the child is spinning around", LABELS
        )
        assert label == "spinning"
        assert debug["mode"] == "single_hit"

    def test_multiple_labels_is_invalid(self):
        label, debug = validate_classification_output(
            "could be rocking or jumping", LABELS
        )
        assert label == INVALID_LABEL
        assert debug["reason"] == "multiple_labels_found"

    def test_no_label_is_invalid(self):
        label, debug = validate_classification_output("unclear video", LABELS)
        assert label == INVALID_LABEL
        assert debug["reason"] == "no_unique_label_found"

    def test_none_and_empty_are_invalid(self):
        assert validate_classification_output(None, LABELS)[0] == INVALID_LABEL
        assert validate_classification_output("   ", LABELS)[0] == INVALID_LABEL


class TestValidateCounting:
    def test_leading_digit(self):
        value, debug = validate_counting_output("3 adults")
        assert value == 3
        assert debug["mode"] == "leading_digit"

    def test_bare_digit(self):
        assert validate_counting_output("0")[0] == 0

    def test_word_number(self):
        value, _ = validate_counting_output("three children visible")
        assert value == 3

    def test_number_inside_sentence(self):
        value, _ = validate_counting_output("There are 2 adults")
        assert value == 2

    def test_unparseable_returns_none(self):
        value, _ = validate_counting_output("no idea")
        assert value is None

    def test_none_returns_none(self):
        assert validate_counting_output(None)[0] is None
