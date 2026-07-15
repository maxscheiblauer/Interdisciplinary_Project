"""Window-size sensitivity check (Phase-1 scoped, read-only).

The 10 s window is the canonical modelling unit. A supervisor question: does that
choice drive the operational-state picture, and does a shorter window make the state
label "chatter" (flip every window)? This re-windows the SAME cleaned features at
5/10/15/20 s, re-runs the kinematic k=4 clustering per size (a fresh scaler+KMeans;
no canonical model or artifact is touched), and reports, per size: state fractions,
dwell mean/std, state persistence (transition-matrix diagonal) and a chatter rate
(fraction of adjacent windows that change state). The 10 s row must reproduce the
canonical state fractions -- asserted as a sanity gate.

Outputs: results/tables/window_sensitivity.csv, results/plots/phase1/window_sensitivity.png.

Re-windowing one size sweeps every daily feature file (~15 min/size), so the sizes
run independently and each writes its own part file:

    python scripts/window_sensitivity.py 5    # one size -> window_sensitivity_part_5s.csv
    python scripts/window_sensitivity.py 10
    python scripts/window_sensitivity.py 15
    python scripts/window_sensitivity.py 20
    python scripts/window_sensitivity.py combine   # merge parts -> csv + png + sanity

Running with no argument does all four sizes then combines (the slow single-shot path).
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from loguru import logger  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from metroat import schema as S  # noqa: E402
from metroat import windowing as W  # noqa: E402
from metroat.io import discover_files  # noqa: E402
from metroat.states import KINEMATIC_FEATS, STATES, name_states_by_physics  # noqa: E402

FEAT_ROOT = ROOT / "data" / "processed" / "train_features"
MODEL_DIR = ROOT / "models" / "phase1"
PLOT_DIR = ROOT / "results" / "plots" / "phase1"
TBL_DIR = ROOT / "results" / "tables"
for d in (PLOT_DIR, TBL_DIR):
    d.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

WINDOW_SIZES = (5, 10, 15, 20)
GAP_LIMIT_S = 30       # adjacent-window flip only counts across a small gap
RNG = 42


def windows_at(size: int, scaler, schema) -> pd.DataFrame:
    """Re-window every daily feature frame at ``size`` seconds (scaled features)."""
    parts = [W.window_day(pd.read_parquet(f, engine="pyarrow"), scaler, schema, window_s=size)
             for f in discover_files(FEAT_ROOT)]
    return (pd.concat(parts, ignore_index=True)
            .sort_values("window_start").reset_index(drop=True))


def label_states(windows: pd.DataFrame) -> pd.DataFrame:
    """Fresh kinematic k=4 clustering + physics naming (StandardScaler ordering is
    monotone per feature, so naming on scaled medians matches unscaled)."""
    X = np.nan_to_num(windows[KINEMATIC_FEATS].to_numpy("float64"))
    Xz = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=4, n_init=10, random_state=RNG).fit(Xz)
    windows = windows.copy()
    windows["cluster"] = km.labels_
    prof = (pd.DataFrame({"cluster": km.labels_,
                          "velocity_mean": windows["TRAIN_SPEED_ACTUAL__mean"].to_numpy(),
                          "accel_mean": windows["acceleration__mean"].to_numpy()})
            .groupby("cluster").median())
    labels = name_states_by_physics(prof)
    windows["state"] = windows["cluster"].map(labels)
    return windows


def size_metrics(windows: pd.DataFrame, size: int) -> list[dict]:
    """State fractions, dwell mean/std, state persistence, chatter rate for one size."""
    w = windows.sort_values("window_start").reset_index(drop=True)
    w["window_start"] = pd.to_datetime(w["window_start"])
    w["window_end"] = pd.to_datetime(w["window_end"])
    gap = (w["window_start"].shift(-1) - w["window_end"]).dt.total_seconds()
    contiguous = gap.fillna(np.inf) <= GAP_LIMIT_S
    cur = w["state"].to_numpy()
    nxt = w["state"].shift(-1).to_numpy()

    n = len(w)
    rows = [dict(window_s=size, metric="n_windows", value=float(n))]

    frac = w["state"].value_counts(normalize=True)
    for s in STATES:
        rows.append(dict(window_s=size, metric=f"frac_{s}", value=float(frac.get(s, 0.0))))

    # dwell: run length of a state (contiguous windows) x window seconds
    runs = []
    run_state, run_len = cur[0], 1
    for i in range(1, n):
        if contiguous.iloc[i - 1] and cur[i] == run_state:
            run_len += 1
        else:
            runs.append((run_state, run_len * size))
            run_state, run_len = cur[i], 1
    runs.append((run_state, run_len * size))
    rdf = pd.DataFrame(runs, columns=["state", "dwell_s"])
    dwell = rdf.groupby("state")["dwell_s"].agg(["mean", "std"])
    for s in STATES:
        m = float(dwell.loc[s, "mean"]) if s in dwell.index else np.nan
        sd = float(dwell.loc[s, "std"]) if s in dwell.index else np.nan
        rows.append(dict(window_s=size, metric=f"dwell_mean_{s}", value=m))
        rows.append(dict(window_s=size, metric=f"dwell_std_{s}", value=sd))

    # transition matrix -> persistence (diagonal), and chatter (any flip)
    flips = 0
    adj = 0
    persist_num = {s: 0 for s in STATES}
    persist_den = {s: 0 for s in STATES}
    for a, b, ok in zip(cur, nxt, contiguous):
        if ok and isinstance(b, str):
            adj += 1
            persist_den[a] += 1
            if a == b:
                persist_num[a] += 1
            else:
                flips += 1
    for s in STATES:
        p = persist_num[s] / persist_den[s] if persist_den[s] else np.nan
        rows.append(dict(window_s=size, metric=f"persist_{s}", value=float(p)))
    rows.append(dict(window_s=size, metric="chatter_rate",
                     value=float(flips / adj) if adj else np.nan))
    logger.info(f"size={size}s: n={n:,} chatter={flips/adj:.3f} "
                f"fracs={ {s: round(float(frac.get(s,0)),3) for s in STATES} }")
    return rows


def sanity_check(long_df: pd.DataFrame) -> None:
    """The 10 s fractions must match the canonical phase1_state_summary.csv."""
    canon = TBL_DIR / "phase1_state_summary.csv"
    if not canon.exists():
        logger.warning("phase1_state_summary.csv absent -- skipping 10 s reproduction check")
        return
    summ = pd.read_csv(canon, index_col=0)
    got = long_df[(long_df.window_s == 10) & long_df.metric.str.startswith("frac_")]
    for _, r in got.iterrows():
        state = r.metric.removeprefix("frac_")
        want = float(summ.loc[state, "pct"]) / 100.0
        assert abs(r.value - want) < 0.02, (
            f"10 s {state} fraction {r.value:.3f} != canonical {want:.3f}")
    logger.info("10 s row reproduces canonical state fractions (sanity OK)")


def plot(long_df: pd.DataFrame) -> None:
    piv = long_df.pivot(index="window_s", columns="metric", values="value")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for s in STATES:
        axes[0].plot(piv.index, piv[f"frac_{s}"], "o-", label=s)
    axes[0].set_xlabel("window size (s)"); axes[0].set_ylabel("fraction of windows")
    axes[0].set_title("Operational-state fractions vs window size")
    axes[0].set_xticks(list(WINDOW_SIZES)); axes[0].legend()
    axes[1].plot(piv.index, piv["chatter_rate"], "s-", color="firebrick")
    axes[1].set_xlabel("window size (s)"); axes[1].set_ylabel("chatter rate (state flips / adj)")
    axes[1].set_title("State chatter vs window size\n(smaller windows flip more)")
    axes[1].set_xticks(list(WINDOW_SIZES))
    axes[1].axvline(10, color="grey", ls=":", alpha=0.7, label="canonical 10 s")
    axes[1].legend()
    fig.suptitle("Window-size sensitivity (kinematic k=4, re-clustered per size)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "window_sensitivity.png", dpi=150)
    plt.close(fig)


def _part_path(size: int) -> Path:
    return TBL_DIR / f"window_sensitivity_part_{size}s.csv"


def run_size(size: int) -> None:
    """Compute one window size and write its part file."""
    schema = S.load_schema()
    scaler = joblib.load(MODEL_DIR / "standard_scaler.joblib")
    windows = windows_at(size, scaler, schema)
    windows = label_states(windows)
    pd.DataFrame(size_metrics(windows, size)).to_csv(_part_path(size), index=False)
    logger.info(f"size {size}s written -> {_part_path(size).name}")


def combine() -> None:
    """Merge per-size part files into the canonical CSV, sanity-check, and plot."""
    parts = [pd.read_csv(_part_path(s)) for s in WINDOW_SIZES if _part_path(s).exists()]
    missing = [s for s in WINDOW_SIZES if not _part_path(s).exists()]
    assert parts, "no window_sensitivity part files found -- run the sizes first"
    if missing:
        logger.warning(f"combining without sizes {missing} (part files absent)")
    long_df = pd.concat(parts, ignore_index=True)
    long_df.to_csv(TBL_DIR / "window_sensitivity.csv", index=False)
    sanity_check(long_df)
    plot(long_df)
    logger.info("window sensitivity complete -> window_sensitivity.csv + .png")


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "combine":
        combine()
    elif arg is not None:
        run_size(int(arg))
    else:
        for size in WINDOW_SIZES:
            run_size(size)
        combine()


if __name__ == "__main__":
    main()
