"""Independent small examples for the published-cohort AUC calculation."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_paper import delong_interval


def test_delong_ties_and_variance_match_hand_calculation():
    # Positive placements: (1/4, 3/4); negative placements: (3/4, 1/4).
    # Both sample variances are 1/8, giving total AUC variance 1/8.
    auc, interval = delong_interval([1, 1, 0, 0], [0.2, 0.8, 0.2, 0.8])
    assert auc == 0.5
    width = norm.ppf(0.975) * np.sqrt(0.125)
    np.testing.assert_allclose(interval, [max(0, 0.5 - width), min(1, 0.5 + width)])


def test_delong_perfect_ranking_and_monotone_transform():
    labels, scores = [1, 1, 0, 0], np.array([0.9, 0.8, 0.1, 0.2])
    assert delong_interval(labels, scores) == (1.0, [1.0, 1.0])
    assert delong_interval(labels, scores) == delong_interval(labels, np.exp(scores))


def test_delong_rejects_insufficient_class_support():
    with pytest.raises(ValueError):
        delong_interval([1, 0, 0], [0.8, 0.2, 0.1])
