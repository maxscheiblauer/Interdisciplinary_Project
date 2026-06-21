import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from metroat import features as F
from metroat import schema as S
from metroat import windowing as W


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
    sch = _mini_schema()
    assert binary not in W.scaled_feature_cols(sch)


# --- windowing ---------------------------------------------------------------

def _win_schema():
    cols = {
        "TRAIN_SPEED_ACTUAL": "continuous_sensor",
        "CW1_MAIN_RESERVOIR_PRESSURE": "continuous_sensor",
        "CW1_BRAKE_CYLINDER_PRESSURE_BOGIE1": "continuous_sensor",
        "CW2_BRAKE_CYLINDER_PRESSURE_BOGIE1": "continuous_sensor",
        "CW1_LOAD_PRESSURE_BOGIE1": "continuous_sensor",
        "B_SENSOR": "binary_sensor",
        "TRAIN_AUTOMATIC_MODE": "operational_state",
    }
    return {"columns": {c: {"category": cat} for c, cat in cols.items()}}


def _win_scaler(sch):
    feats = W.scaled_feature_cols(sch)
    sc = StandardScaler()
    sc.mean_ = np.zeros(len(feats))
    sc.scale_ = np.ones(len(feats))
    sc.var_ = np.ones(len(feats))
    sc.n_features_in_ = len(feats)
    sc.feature_names_in_ = np.array(feats)
    return sc


def _win_frame(n):
    t0 = pd.Timestamp("2024-06-01 00:00:00")
    ts = [t0 + pd.Timedelta(seconds=i) for i in range(n)]
    df = pd.DataFrame({
        "TIMESTAMP": pd.Series(ts).astype("datetime64[ms]"),
        "TRAIN_SPEED_ACTUAL": np.linspace(0, 1, n),
        "CW1_MAIN_RESERVOIR_PRESSURE": np.full(n, 0.8),
        "CW1_BRAKE_CYLINDER_PRESSURE_BOGIE1": np.linspace(0, 0.5, n),
        "CW2_BRAKE_CYLINDER_PRESSURE_BOGIE1": np.linspace(0, 0.3, n),
        "CW1_LOAD_PRESSURE_BOGIE1": np.full(n, 0.4),
        "B_SENSOR": np.tile([0, 1], n // 2 + 1)[:n],
        "TRAIN_AUTOMATIC_MODE": np.ones(n),
        "TRAIN_IS_IN_FAILURE": np.zeros(n, dtype=bool),
        "TRAIN_IS_IN_MAINTENANCE": np.zeros(n, dtype=bool),
        "TRAIN_FAILURE_TYPE": ["No Failure"] * n,
    })
    return F.add_derived_features(df, _win_schema())


def test_window_count_10s_bins():
    sch = _win_schema()
    df = _win_frame(100)  # 100 seconds -> 10 windows of 10s
    w = W.window_day(df, _win_scaler(sch), sch)
    assert len(w) == 10
    assert (w["n_rows"] == 10).all()


def test_window_has_mean_std_and_flags():
    sch = _win_schema()
    w = W.window_day(_win_frame(50), _win_scaler(sch), sch)
    assert "TRAIN_SPEED_ACTUAL__mean" in w.columns
    assert "TRAIN_SPEED_ACTUAL__std" in w.columns
    assert "failure_flag" in w.columns
    # binary majority and operational mode columns present (not clustering inputs)
    assert "B_SENSOR__maj" in w.columns
    assert "TRAIN_AUTOMATIC_MODE__mode" in w.columns
    cin = W.cluster_input_cols(w, sch)
    assert not any("__maj" in c or "__mode" in c for c in cin)
