"""Phase 1.3 — windowing (10 s non-overlapping, the modelling unit).

Windows are fixed 10 s wall-clock bins (epoch // window_s). Bins with no rows
simply don't exist, so the natural service gaps never produce empty windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as F
from . import schema as S

WINDOW_S = 10

# Signals that get explicit min/max per window (in scaled space).
MINMAX_SIGNALS = [
    S.VELOCITY_COL,
    "acceleration",
    "brake_cylinder_pressure_mean",
    "main_reservoir_pressure_mean",
]


def scaled_feature_cols(schema: dict | None = None) -> list[str]:
    return S.continuous_cols(schema) + F.DERIVED_COLS


def window_day(df: pd.DataFrame, scaler, schema: dict | None = None) -> pd.DataFrame:
    """Aggregate one *cleaned+derived* daily frame into 10 s windows.

    Continuous+derived features are scaled (via ``scaler``) before aggregation.
    Returns one row per non-empty 10 s bin.
    """
    feats = list(scaler.feature_names_in_)
    binr = S.binary_cols(schema)
    oper = S.operational_cols(schema)

    df = df.copy()
    df[feats] = scaler.transform(df[feats])
    # Floor to seconds first so binning is independent of timestamp resolution
    # (the column is datetime64[ms]).
    epoch_s = df["TIMESTAMP"].values.astype("datetime64[s]").astype("int64")
    df["_win"] = epoch_s // WINDOW_S

    g = df.groupby("_win", sort=True)
    out = pd.DataFrame(index=g.size().index)
    out["window_start"] = g["TIMESTAMP"].first().values
    out["window_end"] = g["TIMESTAMP"].last().values
    out["n_rows"] = g.size().values

    # mean + std of scaled continuous/derived
    means = g[feats].mean()
    stds = g[feats].std().fillna(0.0)
    means.columns = [f"{c}__mean" for c in feats]
    stds.columns = [f"{c}__std" for c in feats]

    # min/max of key signals (scaled)
    mins = g[MINMAX_SIGNALS].min()
    maxs = g[MINMAX_SIGNALS].max()
    mins.columns = [f"{c}__min" for c in MINMAX_SIGNALS]
    maxs.columns = [f"{c}__max" for c in MINMAX_SIGNALS]

    # binary majority vote
    binmaj = (g[binr].mean() >= 0.5).astype(int)
    binmaj.columns = [f"{c}__maj" for c in binr]

    # operational mode (validation only)
    opmode = g[oper].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
    opmode.columns = [f"{c}__mode" for c in oper]

    # failure / maintenance flag: any active row in window
    fflag = g["TRAIN_IS_IN_FAILURE"].max().fillna(False).astype(int)
    mflag = g["TRAIN_IS_IN_MAINTENANCE"].max().fillna(False).astype(int)
    ftype = g["TRAIN_FAILURE_TYPE"].agg(
        lambda s: s[s != "No Failure"].mode().iloc[0]
        if (s != "No Failure").any() else "No Failure"
    )

    flags = pd.DataFrame({
        "failure_flag": fflag.values,
        "maintenance_flag": mflag.values,
        "failure_type": ftype.values,
    })
    res = pd.concat(
        [out.reset_index(drop=True),
         means.reset_index(drop=True), stds.reset_index(drop=True),
         mins.reset_index(drop=True), maxs.reset_index(drop=True),
         binmaj.reset_index(drop=True), opmode.reset_index(drop=True),
         flags],
        axis=1,
    )
    return res


def cluster_input_cols(window_df: pd.DataFrame, schema: dict | None = None) -> list[str]:
    """Columns fed to PCA/K-means: scaled continuous/derived mean & std + min/max.
    Excludes binary majority, operational mode, and failure flags."""
    feats = scaled_feature_cols(schema)
    cols = [f"{c}__mean" for c in feats] + [f"{c}__std" for c in feats]
    cols += [f"{c}__min" for c in MINMAX_SIGNALS] + [f"{c}__max" for c in MINMAX_SIGNALS]
    return [c for c in cols if c in window_df.columns]
