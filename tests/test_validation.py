import numpy as np

from metroat.validation import chi_square_shift, cusum


def test_chi_square_on_known_table():
    # strongly different distributions -> small p
    chi2, p, dof = chi_square_shift([90, 10], [10, 90])
    assert chi2 > 50
    assert p < 1e-3
    assert dof == 1


def test_chi_square_identical_distributions():
    chi2, p, dof = chi_square_shift([50, 50], [500, 500])
    assert p > 0.9  # same proportions -> no evidence of shift


def test_cusum_detects_step():
    rng = np.random.default_rng(0)
    series = np.concatenate([rng.normal(0, 1, 200), rng.normal(5, 1, 200)])
    cps = cusum(series, threshold=5.0, drift=0.5)
    assert len(cps) >= 1
    # first detected change point should be near the step at index 200
    assert any(180 <= c <= 260 for c in cps)


def test_cusum_no_change_on_flat():
    rng = np.random.default_rng(1)
    series = rng.normal(0, 1, 400)
    # a positive drift slack keeps CUSUM stable on zero-mean noise
    cps = cusum(series, threshold=8.0, drift=0.5)
    assert len(cps) == 0
