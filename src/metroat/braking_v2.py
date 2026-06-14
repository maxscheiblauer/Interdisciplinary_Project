"""Phase 2 v2 — honest braking analysis (pure logic).

This module replaces the circular-classification core of v1. Its central job is
the **feature-role ledger**: every window / event feature is assigned exactly one
*role* so that no task ever lets a predictor leak its own target.

Roles (see ``execution_plan_phase2_v2.md`` "central design principle"):

* ``velocity``  — kinematics (speed, acceleration, jerk, deceleration, delta_v,
  velocity-at-start/end). These *define* braking and intensity; they are only ever
  **targets**, never predictors in the classification/regression tasks.
* ``actuation`` — the brake *command* pneumatics (brake-cylinder, proportional
  valve, spring brake, pneumatic braking force). Physically near-tautological with
  braking, so always reported separately (tier B) so a high score isn't over-claimed.
* ``auxiliary`` — non-command health pneumatics (main reservoir, load pressure,
  load signal, energy-braking-resistance, ambient temperature, pressure-rate). The
  *interesting* independent predictors (tier A).
* ``other``     — bookkeeping / labels / operational mode (window timing, flags,
  cluster, state, event ids, calendar). Never a predictor or a target here.

Only pure, unit-testable logic lives here; I/O stays in the scripts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema as S

# --- group detection -------------------------------------------------------

# Window aggregation suffixes appended by ``windowing.window_day``.
_SUFFIXES = ("__mean", "__std", "__min", "__max", "__maj", "__mode")

# Substring -> sensor group, checked in order. Velocity first; the two binary
# "active/available" command signals are mapped to their actuation group before
# the generic BRAKE_CYLINDER check (none of them contain "BRAKE_CYLINDER").
_GROUP_PATTERNS: list[tuple[str, str]] = [
    ("TRAIN_SPEED_ACTUAL", "VELOCITY"),
    ("PNEUMATIC_BRAKING_FORCE", "PNEUMATIC_BRAKING_FORCE"),
    ("PNEUMATIC_BRAKE_ACTIVE", "PNEUMATIC_BRAKING_FORCE"),
    ("PROPORTIONAL_VALVE", "PROPORTIONAL_VALVE"),
    ("SPRING_BRAKE", "SPRING_BRAKE"),
    ("BRAKE_CYLINDER", "BRAKE_CYLINDER"),
    ("MAIN_RESERVOIR", "MAIN_RESERVOIR"),
    ("COMPRESSOR_RUNNING", "MAIN_RESERVOIR"),  # compressor charges the reservoir
    ("LOAD_PRESSURE", "LOAD_PRESSURE"),
    ("LOAD_SIGNAL", "LOAD_SIGNAL"),
    ("ENERGY_BRAKING_RESISTANCE", "ENERGY_BRAKING_RESISTANCE"),
    ("AMBIENT_TEMPERATURE", "AMBIENT_TEMPERATURE"),
]

# Derived 1 Hz columns (from features.DERIVED_COLS) -> sensor group.
_DERIVED_GROUP: dict[str, str] = {
    "acceleration": "VELOCITY",
    "jerk": "VELOCITY",
    "velocity_change_rate": "VELOCITY",
    "main_reservoir_pressure_rate": "MAIN_RESERVOIR",
    "main_reservoir_pressure_mean": "MAIN_RESERVOIR",
    "brake_cylinder_pressure_max": "BRAKE_CYLINDER",
    "brake_cylinder_pressure_mean": "BRAKE_CYLINDER",
    "air_suspension_mean": "LOAD_PRESSURE",  # air-spring load pressure
}

# Event-level features produced by Task 1 -> sensor group.
_EVENT_GROUP: dict[str, str] = {
    # velocity / target side
    "peak_deceleration": "VELOCITY",
    "mean_deceleration": "VELOCITY",
    "delta_v": "VELOCITY",
    "velocity_at_start": "VELOCITY",
    "velocity_at_end": "VELOCITY",
    "jerk_peak": "VELOCITY",
    "jerk_rms": "VELOCITY",
    "duration_seconds": "DURATION",  # ambiguous, see note below
    # actuation
    "brake_cylinder_pressure_peak": "BRAKE_CYLINDER",
    "brake_cylinder_pressure_mean": "BRAKE_CYLINDER",
    "brake_cylinder_pressure_integral": "BRAKE_CYLINDER",
    "proportional_valve_pressure_peak": "PROPORTIONAL_VALVE",
    "proportional_valve_pressure_mean": "PROPORTIONAL_VALVE",
    "proportional_valve_pressure_integral": "PROPORTIONAL_VALVE",
    "spring_brake_pressure_peak": "SPRING_BRAKE",
    "spring_brake_pressure_mean": "SPRING_BRAKE",
    "spring_brake_pressure_integral": "SPRING_BRAKE",
    "pneumatic_braking_force_peak": "PNEUMATIC_BRAKING_FORCE",
    "pneumatic_braking_force_mean": "PNEUMATIC_BRAKING_FORCE",
    "pneumatic_braking_force_integral": "PNEUMATIC_BRAKING_FORCE",
    # auxiliary
    "main_reservoir_pressure_mean": "MAIN_RESERVOIR",
    "main_reservoir_pressure_min": "MAIN_RESERVOIR",
    "main_reservoir_pressure_rate_mean": "MAIN_RESERVOIR",
    "main_reservoir_pressure_drop": "MAIN_RESERVOIR",
    "load_pressure_mean": "LOAD_PRESSURE",
    "load_pressure_std": "LOAD_PRESSURE",
    "load_signal_mean": "LOAD_SIGNAL",
    "energy_braking_resistance_mean": "ENERGY_BRAKING_RESISTANCE",
    "energy_braking_resistance_peak": "ENERGY_BRAKING_RESISTANCE",
    "ambient_temperature_mean": "AMBIENT_TEMPERATURE",
}

# Bookkeeping / label / operational columns -> role "other".
_OTHER_EXACT = {
    "window_start", "window_end", "n_rows",
    "failure_flag", "maintenance_flag", "failure_type",
    "cluster", "state", "standing_substate",
    "event_id", "start_timestamp", "end_timestamp", "date",
    "month", "is_long", "n_windows",
    "failure_within_7_days", "failure_within_30_days",
}

_ROLE_BY_GROUP: dict[str, str] = {
    "VELOCITY": "velocity",
    "BRAKE_CYLINDER": "actuation",
    "PROPORTIONAL_VALVE": "actuation",
    "SPRING_BRAKE": "actuation",
    "PNEUMATIC_BRAKING_FORCE": "actuation",
    "MAIN_RESERVOIR": "auxiliary",
    "LOAD_PRESSURE": "auxiliary",
    "LOAD_SIGNAL": "auxiliary",
    "ENERGY_BRAKING_RESISTANCE": "auxiliary",
    "AMBIENT_TEMPERATURE": "auxiliary",
    # DURATION is kinematic-adjacent (a structural property of the stop). It is
    # treated as a velocity-side feature by default so it can never silently leak
    # into a deceleration predictor; Task 5 reports with/without it explicitly.
    "DURATION": "velocity",
}


def base_name(feature: str) -> str:
    """Strip a window aggregation suffix to recover the underlying column name."""
    for suf in _SUFFIXES:
        if feature.endswith(suf):
            return feature[: -len(suf)]
    return feature


def feature_role(feature: str) -> tuple[str, str]:
    """Return ``(sensor_group, role)`` for a window or event feature name.

    Resolution order: explicit bookkeeping set -> event-feature table ->
    derived-column table -> raw-sensor substring patterns -> ``("OTHER", "other")``.
    """
    base = base_name(feature)

    if base in _OTHER_EXACT or feature in _OTHER_EXACT:
        return "OTHER", "other"

    if feature in _EVENT_GROUP:
        grp = _EVENT_GROUP[feature]
        return grp, _ROLE_BY_GROUP[grp]
    if base in _EVENT_GROUP:
        grp = _EVENT_GROUP[base]
        return grp, _ROLE_BY_GROUP[grp]

    if base in _DERIVED_GROUP:
        grp = _DERIVED_GROUP[base]
        return grp, _ROLE_BY_GROUP[grp]

    for pat, grp in _GROUP_PATTERNS:
        if pat in base:
            return grp, _ROLE_BY_GROUP[grp]

    return "OTHER", "other"


def build_role_ledger(features: list[str], level: str = "window") -> pd.DataFrame:
    """Build a role-ledger frame: one row per feature with group + role.

    Columns: ``feature, level, sensor_group, role, predictor_eligible, note``.
    ``predictor_eligible`` is True only for actuation/auxiliary features (the pools
    allowed as predictors in the cross-modal tasks).
    """
    rows = []
    for f in features:
        grp, role = feature_role(f)
        note = ""
        if base_name(f) == "duration_seconds":
            note = ("kinematic-adjacent: structural stop property; default role "
                    "velocity. Task 5 reports peak/mean-decel R2 with AND without it.")
        rows.append({
            "feature": f,
            "level": level,
            "sensor_group": grp,
            "role": role,
            "predictor_eligible": role in ("actuation", "auxiliary"),
            "note": note,
        })
    return pd.DataFrame(rows)


# Canonical event-feature names Task 1 will emit, so the ledger and every
# downstream task agree on roles before the events parquet exists.
EVENT_FEATURE_NAMES: list[str] = list(_EVENT_GROUP.keys())


# --- event extraction (Task 1) --------------------------------------------

# Events are one contiguous occupancy of the Phase-1 braking state. We merge
# only *immediately adjacent* 10 s windows; the max gap that still counts as
# adjacent is one window length (so a single missing window splits the event).
EVENT_GAP_S = 10.0          # = window length; immediate adjacency only
EVENT_MIN_DURATION_S = 10.0  # drop events shorter than one full window
EVENT_LONG_FLAG_S = 300.0    # flag (keep) very long events

# ~45% of braking-STATE events are stationary brake-holds (train already stopped,
# brake applied) with zero deceleration. A *real deceleration* event is one where
# the train was moving at the start and actually lost speed. Tasks 5/6 use this to
# scope to "how hard a MOVING train brakes" (confirmed scope with Maximilian).
REAL_DECEL_MIN_DELTA_V = 0.01
REAL_DECEL_MIN_V0 = 0.02


def is_real_deceleration(events: pd.DataFrame) -> pd.Series:
    """Boolean mask: events where the train was moving and actually slowed."""
    return ((events["delta_v"] > REAL_DECEL_MIN_DELTA_V)
            & (events["velocity_at_start"] > REAL_DECEL_MIN_V0))


def braking_events_from_windows(
    windows: pd.DataFrame, gap_s: float = EVENT_GAP_S, min_dur: float = EVENT_MIN_DURATION_S
) -> pd.DataFrame:
    """Contiguous braking-state occupancies from labeled 10 s windows.

    Returns one row per event: ``start, end, duration_s, n_windows, is_long``.
    Two braking windows belong to the same event iff the gap between them is
    ``<= gap_s`` (default one window length). Events shorter than ``min_dur`` are
    dropped (a single full window passes); long events are flagged, not dropped.
    """
    b = windows[windows["state"] == "braking"]
    cols = ["start", "end", "duration_s", "n_windows", "is_long"]
    if b.empty:
        return pd.DataFrame(columns=cols)
    s = pd.to_datetime(b["window_start"]).to_numpy()
    e = pd.to_datetime(b["window_end"]).to_numpy()
    order = np.argsort(s)
    s, e = s[order], e[order]
    groups: list[list] = [[s[0], e[0], 1]]
    for i in range(1, len(s)):
        gap = (s[i] - groups[-1][1]) / np.timedelta64(1, "s")
        if gap <= gap_s:
            groups[-1][1] = max(groups[-1][1], e[i])
            groups[-1][2] += 1
        else:
            groups.append([s[i], e[i], 1])
    rows = []
    for start, end, nwin in groups:
        dur = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() + 1.0
        if dur < min_dur:
            continue
        rows.append(dict(start=pd.Timestamp(start), end=pd.Timestamp(end),
                         duration_s=dur, n_windows=int(nwin),
                         is_long=dur > EVENT_LONG_FLAG_S))
    return pd.DataFrame(rows, columns=cols)


def _group_series(df: pd.DataFrame, substring: str, schema: dict | None = None) -> np.ndarray:
    """Row-wise mean across all continuous sensors of a group (1 Hz series)."""
    cont = S.continuous_cols(schema)
    cols = [c for c in cont if substring in c and c in df.columns]
    if not cols:
        return np.full(len(df), np.nan)
    return df[cols].mean(axis=1).to_numpy("float64")


def event_features_v2(
    day_df: pd.DataFrame, intervals: pd.DataFrame, schema: dict | None = None
) -> pd.DataFrame:
    """Role-tagged per-event features from a *single day's* 1 Hz frame.

    ``intervals`` are the events (from :func:`braking_events_from_windows`) whose
    span lies within ``day_df``. Velocity features are the target side; actuation
    and auxiliary features are the predictor pools (see the role ledger). Integral
    features use dt = 1 s and therefore scale with duration (noted for Task 5).
    """
    df = day_df.sort_values("TIMESTAMP").reset_index(drop=True)
    ts = pd.to_datetime(df["TIMESTAMP"]).to_numpy()
    vel = df[S.VELOCITY_COL].to_numpy("float64")
    acc = df["acceleration"].to_numpy("float64")
    jerk = df["jerk"].to_numpy("float64")
    mr_rate = df["main_reservoir_pressure_rate"].to_numpy("float64")

    grp = {
        "brake_cylinder": _group_series(df, "BRAKE_CYLINDER_PRESSURE", schema),
        "proportional_valve": _group_series(df, "PROPORTIONAL_VALVE_PRESSURE", schema),
        "spring_brake": _group_series(df, "SPRING_BRAKE_PRESSURE", schema),
        "pneumatic_braking_force": _group_series(df, "PNEUMATIC_BRAKING_FORCE", schema),
        "main_reservoir": _group_series(df, "MAIN_RESERVOIR_PRESSURE", schema),
        "load_pressure": _group_series(df, "LOAD_PRESSURE", schema),
        "load_signal": _group_series(df, "LOAD_SIGNAL", schema),
        "energy_braking_resistance": _group_series(df, "ENERGY_BRAKING_RESISTANCE", schema),
        "ambient": df["AMBIENT_TEMPERATURE"].to_numpy("float64")
        if "AMBIENT_TEMPERATURE" in df.columns else np.full(len(df), np.nan),
    }

    rows = []
    for _, ev in intervals.iterrows():
        lo, hi = np.datetime64(ev["start"]), np.datetime64(ev["end"])
        m = (ts >= lo) & (ts <= hi)
        if m.sum() < 2:
            continue
        decel = -acc[m]
        bc, pv, sb, pbf = (grp["brake_cylinder"][m], grp["proportional_valve"][m],
                           grp["spring_brake"][m], grp["pneumatic_braking_force"][m])
        mr, lp, ls = grp["main_reservoir"][m], grp["load_pressure"][m], grp["load_signal"][m]
        ebr, amb = grp["energy_braking_resistance"][m], grp["ambient"][m]

        def _peak(x):
            return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan

        def _mean(x):
            return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

        def _sum(x):
            return float(np.nansum(x)) if np.isfinite(x).any() else np.nan

        rows.append(dict(
            event_id=pd.Timestamp(ev["start"]).isoformat(),
            start_timestamp=pd.Timestamp(ev["start"]),
            end_timestamp=pd.Timestamp(ev["end"]),
            duration_seconds=float(ev["duration_s"]),
            n_windows=int(ev.get("n_windows", 0)),
            is_long=bool(ev["is_long"]),
            month=pd.Timestamp(ev["start"]).month,
            # --- velocity / target ---
            peak_deceleration=_peak(decel),
            mean_deceleration=float(np.nanmean(decel[decel > 0])) if (decel > 0).any() else 0.0,
            delta_v=float(vel[m][0] - vel[m][-1]),
            velocity_at_start=float(vel[m][0]),
            velocity_at_end=float(vel[m][-1]),
            jerk_peak=_peak(np.abs(jerk[m])),
            jerk_rms=float(np.sqrt(np.nanmean(jerk[m] ** 2))),
            # --- actuation ---
            brake_cylinder_pressure_peak=_peak(bc),
            brake_cylinder_pressure_mean=_mean(bc),
            brake_cylinder_pressure_integral=_sum(bc),
            proportional_valve_pressure_peak=_peak(pv),
            proportional_valve_pressure_mean=_mean(pv),
            proportional_valve_pressure_integral=_sum(pv),
            spring_brake_pressure_peak=_peak(sb),
            spring_brake_pressure_mean=_mean(sb),
            spring_brake_pressure_integral=_sum(sb),
            pneumatic_braking_force_peak=_peak(pbf),
            pneumatic_braking_force_mean=_mean(pbf),
            pneumatic_braking_force_integral=_sum(pbf),
            # --- auxiliary ---
            main_reservoir_pressure_mean=_mean(mr),
            main_reservoir_pressure_min=float(np.nanmin(mr)) if np.isfinite(mr).any() else np.nan,
            main_reservoir_pressure_rate_mean=_mean(mr_rate[m]),
            main_reservoir_pressure_drop=(float(mr[0] - np.nanmin(mr))
                                          if np.isfinite(mr).any() else np.nan),
            load_pressure_mean=_mean(lp),
            load_pressure_std=float(np.nanstd(lp)) if np.isfinite(lp).any() else np.nan,
            load_signal_mean=_mean(ls),
            energy_braking_resistance_mean=_mean(ebr),
            energy_braking_resistance_peak=_peak(ebr),
            ambient_temperature_mean=_mean(amb),
        ))
    return pd.DataFrame(rows)
