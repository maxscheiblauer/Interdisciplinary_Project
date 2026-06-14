"""Phase 2.1 — feature-role ledger, leakage audit, and braking-event extraction.

Operational states are defined by KINEMATICS ONLY (Phase 1, Option A), so every
pneumatic sensor is an independent predictor. This script:
  1. builds the feature-role ledger (velocity / actuation / auxiliary / other),
  2. writes a leakage audit proving the predictor pools are disjoint from both the
     velocity targets and the kinematic features that defined the states,
  3. extracts braking events as contiguous occupancies of the Phase-1 `braking`
     state (now ~all real decelerations) with role-tagged per-event features.

Outputs:
  results/tables/feature_roles.csv
  logs/phase2/leakage_audit.txt
  data/processed/braking_state_events.parquet
  results/plots/phase2/event_duration_distribution.png
  results/tables/event_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from loguru import logger  # noqa: E402

from metroat import braking_v2 as BV2  # noqa: E402
from metroat import schema as S  # noqa: E402

PROC = ROOT / "data" / "processed"
FEAT_ROOT = PROC / "train_features"
PLOT_DIR = ROOT / "results" / "plots" / "phase2"
TBL_DIR = ROOT / "results" / "tables"
LOG_DIR = ROOT / "logs" / "phase2"
MODEL_DIR = ROOT / "models" / "phase2"
for d in (PLOT_DIR, TBL_DIR, LOG_DIR, MODEL_DIR, ROOT / "results" / "plots" / "phase2"):
    d.mkdir(parents=True, exist_ok=True)

LEDGER = TBL_DIR / "feature_roles.csv"
AUDIT = LOG_DIR / "leakage_audit.txt"
OUT = PROC / "braking_state_events.parquet"
OLD_COUNT = 157823  # original (v1) braking_events count, for the >2x sanity gate

# Phase-1 state-defining features (Option A: kinematic only).
STATE_DEF_FEATS = [
    "TRAIN_SPEED_ACTUAL__mean", "TRAIN_SPEED_ACTUAL__std",
    "TRAIN_SPEED_ACTUAL__min", "TRAIN_SPEED_ACTUAL__max",
    "acceleration__mean", "acceleration__std",
    "acceleration__min", "acceleration__max",
    "jerk__mean", "jerk__std",
    "velocity_change_rate__mean", "velocity_change_rate__std",
]
DAY_COLS = ["TIMESTAMP", S.VELOCITY_COL, "acceleration", "jerk",
            "main_reservoir_pressure_rate", "AMBIENT_TEMPERATURE"]

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def build_ledger() -> pd.DataFrame:
    lw = pd.read_parquet(PROC / "train_windows_labeled.parquet")
    win_ledger = BV2.build_role_ledger(list(lw.columns), level="window")
    ev_ledger = BV2.build_role_ledger(BV2.EVENT_FEATURE_NAMES, level="event")
    ledger = pd.concat([win_ledger, ev_ledger], ignore_index=True)
    leftover = ledger[(ledger.sensor_group == "OTHER")].copy()
    leftover = leftover[~leftover["feature"].apply(
        lambda f: (BV2.base_name(f) in BV2._OTHER_EXACT) or ("__mode" in f))]
    if len(leftover):
        logger.warning(f"[1] {len(leftover)} feature(s) fell to OTHER unexpectedly: "
                       f"{leftover['feature'].tolist()}")
    ledger.to_csv(LEDGER, index=False)
    counts = ledger.groupby(["level", "role"]).size().unstack(fill_value=0)
    logger.info(f"[1] feature-role ledger ({len(ledger)} rows):\n{counts}")
    return ledger


def leakage_audit(ledger: pd.DataFrame) -> None:
    win = ledger[ledger.level == "window"]
    predictor_pool = set(win[win.role.isin(["actuation", "auxiliary"])]["feature"])
    target_pool = set(win[win.role == "velocity"]["feature"])
    overlap = predictor_pool & target_pool
    overlap_state = predictor_pool & set(STATE_DEF_FEATS)
    assert not overlap, f"LEAKAGE: predictor/target overlap: {overlap}"
    assert not overlap_state, f"LEAKAGE: predictor/state-definer overlap: {overlap_state}"

    lines = [
        "Phase 2 -- Leakage audit", "=" * 60, "",
        "Rule: the modality that DEFINES a target must never predict it.",
        "Velocity/kinematics -> targets. Actuation + auxiliary -> predictors.", "",
        f"Window features classified: {len(win)}",
        f"  role counts: {win['role'].value_counts().to_dict()}",
        f"  predictor pool (actuation+auxiliary): {len(predictor_pool)}",
        f"  velocity target pool: {len(target_pool)}",
        f"  predictor/target overlap: {len(overlap)} -> DISJOINT OK", "",
        "Phase-1 states are defined by KINEMATICS ONLY (Option A). The 12",
        "state-defining features (velocity + acceleration family):",
    ]
    lines += [f"    - {c}" for c in STATE_DEF_FEATS]
    lines += [
        "",
        f"  predictor pool overlap with state-definers: {len(overlap_state)} -> DISJOINT OK",
        "",
        "Interpretation: `state` (and `braking`) is built from kinematics ALONE -- no",
        "pneumatic sensor entered the clustering. So predicting braking from pneumatics",
        "(state classifier) and recovering deceleration from pneumatics (regression) are",
        "BOTH fully cross-modal: every predictor is independent of the kinematic features",
        "that defined the target. Tier A (auxiliary-only) and Tier B (+actuation) are clean.",
        "",
        "RESULT: predictor and target/state-definer pools are disjoint. PASS.",
    ]
    AUDIT.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"[1] leakage audit -> {AUDIT.relative_to(ROOT)} (PASS)")


def _feature_file(day: pd.Timestamp) -> Path:
    return (FEAT_ROOT / f"year={day.year}" / f"month={day.month:02d}"
            / f"day={day.day:02d}" / "day.parquet")


def extract_events(schema) -> None:
    if OUT.exists():
        logger.info(f"[1] {OUT.name} exists -- loading (skip extraction, refresh summary)")
        ev = pd.read_parquet(OUT)
        ev["start_timestamp"] = pd.to_datetime(ev["start_timestamp"])
    else:
        cont = S.continuous_cols(schema)
        read_cols = list(dict.fromkeys(DAY_COLS + cont))
        windows = pd.read_parquet(PROC / "train_windows_labeled.parquet",
                                  columns=["window_start", "window_end", "state"])
        windows["window_start"] = pd.to_datetime(windows["window_start"])
        windows["window_end"] = pd.to_datetime(windows["window_end"])
        events = BV2.braking_events_from_windows(windows)
        events["date"] = events["start"].dt.normalize()
        logger.info(f"[1] {len(events):,} contiguous braking events "
                    f"(gap_s={BV2.EVENT_GAP_S:g}, min_dur={BV2.EVENT_MIN_DURATION_S:g})")

        feats = []
        for date, grp in events.groupby("date"):
            f = _feature_file(date)
            if not f.exists():
                continue
            df = pd.read_parquet(f, columns=read_cols, engine="pyarrow")
            feats.append(BV2.event_features_v2(df, grp, schema))
            del df
        ev = pd.concat(feats, ignore_index=True)
        ev["start_timestamp"] = pd.to_datetime(ev["start_timestamp"])

        inv = pd.read_csv(ROOT / "logs" / "data_profiling" / "event_inventory.csv")
        inv = inv[(inv.event_class == "failure") & (inv.type == "Brake System Failure")]
        fail_starts = pd.to_datetime(inv["start"]).to_numpy()

        def within(days: int) -> np.ndarray:
            out = np.zeros(len(ev), dtype=int)
            st = ev["start_timestamp"].to_numpy()
            for i, t in enumerate(st):
                for ft in fail_starts:
                    d = (ft - t) / np.timedelta64(1, "D")
                    if 0 <= d <= days:
                        out[i] = 1
                        break
            return out

        ev["failure_within_7_days"] = within(7)
        ev["failure_within_30_days"] = within(30)
        ev["is_real_deceleration"] = BV2.is_real_deceleration(ev).to_numpy()
        ev.to_parquet(OUT, engine="pyarrow", compression="snappy", index=False)

    logger.info(f"[1] braking_state_events: {len(ev):,} events | "
                f"real-decel={ev['is_real_deceleration'].mean():.1%} | "
                f"7d-pre={int(ev['failure_within_7_days'].sum())} "
                f"30d-pre={int(ev['failure_within_30_days'].sum())}")

    dur = ev["duration_seconds"].to_numpy("float64")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(dur[dur <= np.quantile(dur, 0.99)], bins=60, color="steelblue")
    ax.axvline(float(np.median(dur)), color="red", ls="--", label=f"median={np.median(dur):.0f}s")
    ax.set_xlabel("event duration (seconds)"); ax.set_ylabel("count")
    ax.set_title(f"Braking-event duration (n={len(ev):,}, contiguous braking-state occupancy)")
    ax.legend(); fig.tight_layout()
    fig.savefig(PLOT_DIR / "event_duration_distribution.png", dpi=150); plt.close(fig)

    summary = pd.DataFrame({
        "metric": ["n_events", "n_real_deceleration", "frac_real_deceleration",
                   "duration_mean_s", "duration_median_s", "duration_p95_s",
                   "events_7d_pre_failure", "events_30d_pre_failure",
                   "v1_event_count", "ratio_vs_v1"],
        "value": [len(ev), int(ev["is_real_deceleration"].sum()),
                  round(float(ev["is_real_deceleration"].mean()), 4),
                  float(np.mean(dur)), float(np.median(dur)), float(np.quantile(dur, 0.95)),
                  int(ev["failure_within_7_days"].sum()),
                  int(ev["failure_within_30_days"].sum()),
                  OLD_COUNT, round(len(ev) / OLD_COUNT, 3)],
    })
    summary.to_csv(TBL_DIR / "event_summary.csv", index=False)
    logger.info(f"[1] duration mean={np.mean(dur):.1f}s median={np.median(dur):.0f}s | "
                f"count vs v1 = {len(ev)/OLD_COUNT:.2f}x")


def main() -> None:
    schema = S.load_schema()
    ledger = build_ledger()
    leakage_audit(ledger)
    extract_events(schema)
    logger.info("[1] Phase 2.1 complete.")


if __name__ == "__main__":
    main()
