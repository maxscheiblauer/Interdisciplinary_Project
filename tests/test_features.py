import numpy as np
import pandas as pd

from metroat import features as F
from metroat import schema as S


def _mini_schema():
    cols = {
        "TRAIN_SPEED_ACTUAL": "continuous_sensor",
        "CW1_MAIN_RESERVOIR_PRESSURE": "continuous_sensor",
        "CW1_BRAKE_CYLINDER_PRESSURE_BOGIE1": "continuous_sensor",
        "CW2_BRAKE_CYLINDER_PRESSURE_BOGIE1": "continuous_sensor",
        "CW1_LOAD_PRESSURE_BOGIE1": "continuous_sensor",
        "B_SENSOR": "binary_sensor",
    }
    return {"columns": {c: {"category": cat} for c, cat in cols.items()}}


def _frame(n=10, dt=1):
    t0 = pd.Timestamp("2024-06-01 00:00:00")
    ts = [t0 + pd.Timedelta(seconds=i * dt) for i in range(n)]
    vel = np.arange(n, dtype="float64") * 0.1  # constant accel of 0.1/s
    return pd.DataFrame({
        "TIMESTAMP": ts,
        "TRAIN_SPEED_ACTUAL": vel,
        "CW1_MAIN_RESERVOIR_PRESSURE": np.full(n, 0.8),
        "CW1_BRAKE_CYLINDER_PRESSURE_BOGIE1": np.linspace(0, 0.5, n),
        "CW2_BRAKE_CYLINDER_PRESSURE_BOGIE1": np.linspace(0, 0.3, n),
        "CW1_LOAD_PRESSURE_BOGIE1": np.full(n, 0.4),
        "B_SENSOR": np.zeros(n),
    })


def test_acceleration_and_jerk_on_synthetic():
    sch = _mini_schema()
    df = F.add_derived_features(_frame(), sch)
    # constant velocity ramp 0.1/s -> acceleration ~0.1 (first sample 0)
    assert np.allclose(df["acceleration"].iloc[1:], 0.1, atol=1e-9)
    assert df["acceleration"].iloc[0] == 0.0
    # constant acceleration -> jerk ~0 in steady state (a start transient is
    # expected because acceleration[0] is pinned to 0)
    assert np.allclose(df["jerk"].iloc[4:], 0.0, atol=1e-9)
    # brake cylinder max/mean across the two BC cols
    assert np.allclose(df["brake_cylinder_pressure_max"],
                       df[["CW1_BRAKE_CYLINDER_PRESSURE_BOGIE1",
                           "CW2_BRAKE_CYLINDER_PRESSURE_BOGIE1"]].max(axis=1))


def test_no_inf_or_nan_in_derived():
    sch = _mini_schema()
    df = F.add_derived_features(_frame(), sch)
    for c in F.DERIVED_COLS:
        assert np.isfinite(df[c]).all(), c


def test_gap_zeroes_rate_features():
    sch = _mini_schema()
    df = _frame()
    # introduce a 100s gap before the last sample
    df.loc[df.index[-1], "TIMESTAMP"] = df["TIMESTAMP"].iloc[-2] + pd.Timedelta(seconds=100)
    out = F.add_derived_features(df, sch)
    assert out["acceleration"].iloc[-1] == 0.0  # rate zeroed across the gap


def test_binary_cols_excluded_from_scaled_features():
    binary = "B_SENSOR"
    assert binary not in F.DERIVED_COLS
    # scaled feature set is continuous + derived only
    from metroat.windowing import scaled_feature_cols
    sch = _mini_schema()
    assert binary not in scaled_feature_cols(sch)
