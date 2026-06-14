import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from metroat import features as F
from metroat import windowing as W


def _schema():
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


def _scaler(sch):
    feats = W.scaled_feature_cols(sch)
    sc = StandardScaler()
    sc.mean_ = np.zeros(len(feats))
    sc.scale_ = np.ones(len(feats))
    sc.var_ = np.ones(len(feats))
    sc.n_features_in_ = len(feats)
    sc.feature_names_in_ = np.array(feats)
    return sc


def _frame(n):
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
    return F.add_derived_features(df, _schema())


def test_window_count_10s_bins():
    sch = _schema()
    df = _frame(100)  # 100 seconds -> 10 windows of 10s
    w = W.window_day(df, _scaler(sch), sch)
    assert len(w) == 10
    assert (w["n_rows"] == 10).all()


def test_window_has_mean_std_and_flags():
    sch = _schema()
    w = W.window_day(_frame(50), _scaler(sch), sch)
    assert "TRAIN_SPEED_ACTUAL__mean" in w.columns
    assert "TRAIN_SPEED_ACTUAL__std" in w.columns
    assert "failure_flag" in w.columns
    # binary majority and operational mode columns present (not clustering inputs)
    assert "B_SENSOR__maj" in w.columns
    assert "TRAIN_AUTOMATIC_MODE__mode" in w.columns
    cin = W.cluster_input_cols(w, sch)
    assert not any("__maj" in c or "__mode" in c for c in cin)
