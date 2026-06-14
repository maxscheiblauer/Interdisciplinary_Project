"""Feature engineering for Phase 1.

All rate-based features are **dt-aware**: derivatives use the actual time delta
between consecutive (time-sorted) samples, and are zeroed across gaps > GAP_S so
that overnight / out-of-service gaps do not create spurious acceleration spikes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema as S

GAP_S = 2.0  # samples more than this far apart are treated as a discontinuity

# Derived feature names (continuous, fed to scaler/PCA alongside raw continuous).
DERIVED_COLS = [
    "acceleration",
    "jerk",
    "velocity_change_rate",
    "main_reservoir_pressure_rate",
    "brake_cylinder_pressure_max",
    "brake_cylinder_pressure_mean",
    "air_suspension_mean",
    "main_reservoir_pressure_mean",
]


def add_derived_features(df: pd.DataFrame, schema: dict | None = None) -> pd.DataFrame:
    """Add derived features to a *time-sorted* daily frame. Returns same frame."""
    bc = S.brake_cylinder_cols(schema)
    mr = S.main_reservoir_cols(schema)
    air = S.air_suspension_cols(schema)
    vel = df[S.VELOCITY_COL].to_numpy("float64")

    # dt in seconds between consecutive samples; first sample -> NaN.
    t = df["TIMESTAMP"].to_numpy("datetime64[ns]")
    dt = np.empty(len(df), dtype="float64")
    dt[0] = np.nan
    dt[1:] = (t[1:] - t[:-1]) / np.timedelta64(1, "s")
    gap = dt > GAP_S  # discontinuity mask (also True where dt is huge)

    def deriv(x: np.ndarray) -> np.ndarray:
        d = np.empty(len(x), dtype="float64")
        d[0] = 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            d[1:] = (x[1:] - x[:-1]) / dt[1:]
        d[~np.isfinite(d)] = 0.0
        d[gap] = 0.0  # no valid rate across a gap
        return d

    accel = deriv(vel)
    jerk_raw = deriv(accel)
    mr_mean = df[mr].mean(axis=1).to_numpy("float64")

    # Build all derived columns at once to avoid DataFrame fragmentation.
    new_cols = {
        "acceleration": accel,
        "velocity_change_rate": np.abs(deriv(vel)),
        "jerk": pd.Series(jerk_raw).rolling(3, min_periods=1).mean().to_numpy(),
        "main_reservoir_pressure_mean": mr_mean,
        "main_reservoir_pressure_rate": deriv(mr_mean),
        "brake_cylinder_pressure_max": df[bc].max(axis=1).to_numpy(),
        "brake_cylinder_pressure_mean": df[bc].mean(axis=1).to_numpy(),
        "air_suspension_mean": df[air].mean(axis=1).to_numpy(),
    }
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def count_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows describing timestamp gaps > GAP_S (for logging)."""
    t = df["TIMESTAMP"]
    dt = t.diff().dt.total_seconds()
    g = dt[dt > GAP_S]
    if g.empty:
        return pd.DataFrame(columns=["gap_start", "gap_end", "gap_seconds"])
    return pd.DataFrame({
        "gap_start": t.shift().loc[g.index].values,
        "gap_end": t.loc[g.index].values,
        "gap_seconds": g.values,
    })
