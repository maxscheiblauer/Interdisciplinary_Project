"""Validation utilities: chi-square test of distribution shift and CUSUM
change-point detection. Used in Phase 1 (event validation) and Phase 2
(degradation trend / change-point detection)."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2_contingency


def chi_square_shift(observed_counts, baseline_counts):
    """Chi-square test comparing an observed categorical distribution against a
    baseline. Inputs are count vectors over the same categories.

    Returns (chi2, p_value, dof). Builds a 2xK contingency table.
    """
    obs = np.asarray(observed_counts, dtype="float64")
    base = np.asarray(baseline_counts, dtype="float64")
    table = np.vstack([obs, base])
    # drop all-zero columns to keep the test well-defined
    table = table[:, table.sum(axis=0) > 0]
    chi2, p, dof, _ = chi2_contingency(table)
    return float(chi2), float(p), int(dof)


def cusum(values, threshold, drift=0.0):
    """Two-sided CUSUM change-point detector on a 1-D series.

    Standardizes the series, accumulates positive/negative deviations beyond a
    small ``drift`` slack, and flags an index when |cumulative| exceeds
    ``threshold`` (in std units), then resets. Returns a list of change-point
    indices.
    """
    x = np.asarray(values, dtype="float64")
    x = x[np.isfinite(x)]
    if x.size < 2:
        return []
    mu, sd = x.mean(), x.std()
    if sd == 0:
        return []
    z = (x - mu) / sd
    s_pos = s_neg = 0.0
    cps = []
    for i, v in enumerate(z):
        s_pos = max(0.0, s_pos + v - drift)
        s_neg = min(0.0, s_neg + v + drift)
        if s_pos > threshold or s_neg < -threshold:
            cps.append(i)
            s_pos = s_neg = 0.0
    return cps
