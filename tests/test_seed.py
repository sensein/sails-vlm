"""experiment.seed is honored when present."""
from sails_vlm.runners.run_prediction import apply_seed


def test_apply_seed_sets_all_rngs():
    import random

    import numpy as np
    import torch

    apply_seed(1234)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    apply_seed(1234)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_apply_seed_none_is_noop():
    apply_seed(None)  # must not raise
