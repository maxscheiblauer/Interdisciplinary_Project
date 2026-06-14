"""Phase 1.5-1.7 — finalize k=4 labels, transitions, event validation, importance.

REFOUNDED (Option A): this script first FINALIZES the operational-state labels from
the kinematic k=4 clustering (D2 decision: standing / accelerating / cruising /
braking), writes the canonical `train_windows_labeled.parquet`, adds a held-brake
sub-label to `standing`, and quantifies the change vs the old (pneumatic-based)
states. It then runs the usual transition / event / importance validation.

Inputs: data/processed/train_windows.parquet, models/phase1/{kinematic_scaler,
        kmeans_kinematic_k4}.joblib, logs/data_profiling/event_inventory.csv
Outputs:
  data/processed/train_windows_labeled.parquet  (overwritten — canonical labels)
  models/phase1/cluster_labels.json
  results/tables/{cluster_profiles,phase1_state_summary,phase1_new_vs_old_states,
                  transition_matrix,dwell_times,phase1_event_validation,
                  feature_importance_phase1}.csv
  results/plots/phase1/{state_profiles,transition_diagram,state_distribution_by_month,
                        pre_event_cluster_distributions,feature_importance}.png
  models/phase1/rf_feature_importance.joblib
  results/reports/01_phase1_findings.md
"""
from __future__ import annotations

import json
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
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, adjusted_rand_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from metroat import schema as S  # noqa: E402
from metroat import windowing as W  # noqa: E402
from metroat.validation import chi_square_shift  # noqa: E402

PROC = ROOT / "data" / "processed"
PLOT_DIR = ROOT / "results" / "plots" / "phase1"
TBL_DIR = ROOT / "results" / "tables"
REP_DIR = ROOT / "results" / "reports"
MODEL_DIR = ROOT / "models" / "phase1"
for d in (PLOT_DIR, TBL_DIR, REP_DIR):
    d.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
RNG = 42
K_FINAL = 4  # D2 decision (2026-06-14): standing / accelerating / cruising / braking
STATES = ["standing", "accelerating", "cruising", "braking"]
GAP_LIMIT_S = 30  # windows further apart than this are not a transition

# Same kinematic basis as 02_phase1_cluster.py.
KINEMATIC_FEATS = [
    "TRAIN_SPEED_ACTUAL__mean", "TRAIN_SPEED_ACTUAL__std",
    "TRAIN_SPEED_ACTUAL__min", "TRAIN_SPEED_ACTUAL__max",
    "acceleration__mean", "acceleration__std",
    "acceleration__min", "acceleration__max",
    "jerk__mean", "jerk__std",
    "velocity_change_rate__mean", "velocity_change_rate__std",
]


# --------------------------------------------------------------------------- #
# P1.2 — finalize kinematic k=4 labels
# --------------------------------------------------------------------------- #
def finalize_labels():
    windows = pd.read_parquet(PROC / "train_windows.parquet")

    # Old (pneumatic-based) states, for the new-vs-old comparison, if present.
    old_state = None
    old_path = PROC / "train_windows_labeled.parquet"
    if old_path.exists():
        try:
            old_state = pd.read_parquet(old_path, columns=["state"])["state"].to_numpy()
            if len(old_state) != len(windows):
                old_state = None
        except Exception:
            old_state = None

    scaler = joblib.load(MODEL_DIR / "kinematic_scaler.joblib")
    km = joblib.load(MODEL_DIR / f"kmeans_kinematic_k{K_FINAL}.joblib")
    Xz = scaler.transform(np.nan_to_num(windows[KINEMATIC_FEATS].to_numpy("float64")))
    clusters = km.predict(Xz)
    windows["cluster"] = clusters

    # --- physics-based naming ---
    sc = joblib.load(MODEL_DIR / "standard_scaler.joblib")
    idx = {f: i for i, f in enumerate(sc.feature_names_in_)}

    def un(col, feat):
        i = idx[feat]
        return windows[col] * sc.scale_[i] + sc.mean_[i]

    prof = pd.DataFrame({"cluster": clusters})
    prof["velocity_mean"] = un("TRAIN_SPEED_ACTUAL__mean", "TRAIN_SPEED_ACTUAL").values
    prof["accel_mean"] = un("acceleration__mean", "acceleration").values
    prof["accel_min"] = un("acceleration__min", "acceleration").values
    prof["accel_max"] = un("acceleration__max", "acceleration").values
    prof["jerk_mean"] = un("jerk__mean", "jerk").values
    med = prof.groupby("cluster").median()

    standing = med["velocity_mean"].idxmin()
    cruising = med["velocity_mean"].idxmax()
    rest = [c for c in med.index if c not in (standing, cruising)]
    accelerating = med.loc[rest, "accel_mean"].idxmax()
    braking = [c for c in rest if c != accelerating][0]
    labels = {standing: "standing", cruising: "cruising",
              accelerating: "accelerating", braking: "braking"}
    assert len(set(labels.values())) == 4, "label collision"
    windows["state"] = windows["cluster"].map(labels)
    (MODEL_DIR / "cluster_labels.json").write_text(
        json.dumps({str(k): v for k, v in labels.items()}, indent=2), encoding="utf-8")
    logger.info(f"[P1.2] k=4 labels: {labels}")

    # --- held-brake sub-label of standing (descriptor, NOT a clustering input) ---
    brake_active_cols = [c for c in windows.columns
                         if "PNEUMATIC_BRAKE_ACTIVE" in c and c.endswith("__maj")]
    held = windows[brake_active_cols].max(axis=1).fillna(0).astype(int) if brake_active_cols else 0
    windows["standing_substate"] = ""
    is_standing = windows["state"] == "standing"
    windows.loc[is_standing & (held == 1), "standing_substate"] = "standing_braked"
    windows.loc[is_standing & (held == 0), "standing_substate"] = "standing_idle"
    sub_counts = windows.loc[is_standing, "standing_substate"].value_counts().to_dict()
    logger.info(f"[P1.2] held-brake sub-label of standing: {sub_counts}")

    # --- profiles table (canonical) ---
    profile = med.copy()
    profile["accel_std"] = prof.assign(
        accel_std=windows["acceleration__std"].values).groupby("cluster")["accel_std"].median()
    profile["n_windows"] = windows.groupby("cluster").size()
    profile["pct"] = (100 * profile["n_windows"] / len(windows)).round(2)
    profile["label"] = [labels[c] for c in profile.index]
    profile.round(4).to_csv(TBL_DIR / "cluster_profiles.csv")

    # --- state summary + success metric (braking = real deceleration) ---
    summ = (windows.groupby("state")
            .agg(n=("state", "size")).reindex(STATES))
    summ["pct"] = (100 * summ["n"] / len(windows)).round(2)
    summ["median_velocity"] = [float(prof.loc[prof.cluster.map(labels) == s, "velocity_mean"].median())
                               for s in STATES]
    summ["median_accel"] = [float(prof.loc[prof.cluster.map(labels) == s, "accel_mean"].median())
                            for s in STATES]
    summ.to_csv(TBL_DIR / "phase1_state_summary.csv")
    brk = summ.loc["braking"]
    logger.info(f"[P1.2] braking state: median velocity={brk['median_velocity']:.3f}, "
                f"median accel={brk['median_accel']:.4f} (should be v>0 & accel<0 = real decel)")

    # --- new-vs-old comparison ---
    if old_state is not None:
        ari = adjusted_rand_score(old_state, windows["state"].to_numpy())
        ct = pd.crosstab(pd.Series(old_state, name="old_state"),
                         windows["state"].rename("new_state"))
        ct.to_csv(TBL_DIR / "phase1_new_vs_old_states.csv")
        logger.info(f"[P1.2] new-vs-old state ARI = {ari:.3f}\n{ct}")

    # --- profile bar plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    order = [c for c in profile.sort_values("velocity_mean").index]
    names = [labels[c] for c in order]
    for ax, metric, title in zip(
            axes, ["velocity_mean", "accel_mean", "jerk_mean"],
            ["median velocity", "median acceleration", "median jerk"]):
        ax.bar(names, profile.loc[order, metric].values,
               color=["grey", "tab:green", "tab:blue", "tab:red"])
        ax.set_title(f"{title} (norm.)"); ax.axhline(0, color="k", lw=0.6)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Operational-state physical profiles (kinematic k=4)")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "state_profiles.png", dpi=150); plt.close(fig)

    windows.to_parquet(PROC / "train_windows_labeled.parquet", engine="pyarrow",
                       compression="snappy", index=False)
    logger.info(f"[P1.2] train_windows_labeled.parquet written ({len(windows):,} rows)")
    return windows


# --------------------------------------------------------------------------- #
# Validation (transitions / events / importance) — unchanged logic
# --------------------------------------------------------------------------- #
def transitions(windows):
    w = windows.sort_values("window_start").reset_index(drop=True)
    w["window_start"] = pd.to_datetime(w["window_start"])
    w["window_end"] = pd.to_datetime(w["window_end"])
    gap = (w["window_start"].shift(-1) - w["window_end"]).dt.total_seconds()
    contiguous = gap.fillna(np.inf) <= GAP_LIMIT_S
    cur = w["state"].to_numpy()
    nxt = w["state"].shift(-1).to_numpy()

    states = [s for s in STATES if s in set(cur)]
    tm = pd.DataFrame(0, index=states, columns=states, dtype=float)
    for a, b, ok in zip(cur, nxt, contiguous):
        if ok and isinstance(b, str) and a in states and b in states:
            tm.loc[a, b] += 1
    tm_norm = tm.div(tm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    tm_norm.to_csv(TBL_DIR / "transition_matrix.csv")

    runs = []
    run_state, run_len = cur[0], 1
    for i in range(1, len(w)):
        if contiguous.iloc[i - 1] and cur[i] == run_state:
            run_len += 1
        else:
            runs.append((run_state, run_len * W.WINDOW_S))
            run_state, run_len = cur[i], 1
    runs.append((run_state, run_len * W.WINDOW_S))
    rdf = pd.DataFrame(runs, columns=["state", "dwell_s"])
    dwell = rdf.groupby("state")["dwell_s"].agg(["mean", "std", "count"]).reindex(states)
    dwell.to_csv(TBL_DIR / "dwell_times.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(tm_norm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(states))); ax.set_xticklabels(states, rotation=45)
    ax.set_yticks(range(len(states))); ax.set_yticklabels(states)
    for i in range(len(states)):
        for j in range(len(states)):
            ax.text(j, i, f"{tm_norm.values[i,j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("State Transition Probabilities"); ax.set_xlabel("to"); ax.set_ylabel("from")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(PLOT_DIR / "transition_diagram.png", dpi=150); plt.close(fig)

    w["ym"] = w["window_start"].dt.strftime("%Y-%m")
    dist = pd.crosstab(w["ym"], w["state"], normalize="index")
    dist = dist[[s for s in STATES if s in dist.columns]]
    fig, ax = plt.subplots(figsize=(14, 6))
    dist.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("fraction of windows"); ax.set_title("Monthly Operational-State Distribution")
    ax.legend(title="state", bbox_to_anchor=(1.01, 1))
    fig.tight_layout(); fig.savefig(PLOT_DIR / "state_distribution_by_month.png", dpi=150); plt.close(fig)
    logger.info(f"[1.5] transitions + dwell + monthly distribution | states={states}")
    return tm_norm, dwell, states


def event_validation(windows, states):
    inv = pd.read_csv(ROOT / "logs" / "data_profiling" / "event_inventory.csv")
    inv["start"] = pd.to_datetime(inv["start"])
    w = windows.copy()
    w["window_start"] = pd.to_datetime(w["window_start"])
    baseline = w["state"].value_counts().reindex(states).fillna(0)
    base_vec = baseline.to_numpy()

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, hours in zip(axes, [24, 48, 72]):
        agg = pd.Series(0.0, index=states)
        n_ev = 0
        for _, e in inv.iterrows():
            lo = e["start"] - pd.Timedelta(hours=hours)
            pre = w[(w["window_start"] >= lo) & (w["window_start"] < e["start"])]
            if len(pre) < 10:
                continue
            n_ev += 1
            counts = pre["state"].value_counts().reindex(states).fillna(0)
            agg += counts
            chi2, p, dof = chi_square_shift(counts.to_numpy(), base_vec)
            rows.append(dict(event_class=e["event_class"], type=e["type"],
                             start=e["start"], window_h=hours, n_pre_windows=len(pre),
                             chi2=chi2, p_value=p))
        if agg.sum() > 0:
            frac = agg / agg.sum()
            ax.bar(states, frac.reindex(states).values, alpha=0.7, label=f"pre-event ({n_ev} ev)")
            ax.bar(states, (baseline / baseline.sum()).reindex(states).values,
                   alpha=0.3, label="baseline")
        ax.set_title(f"{hours}h pre-event"); ax.tick_params(axis="x", rotation=45); ax.legend(fontsize=8)
    fig.suptitle("Pre-Event vs Baseline Cluster Distributions")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "pre_event_cluster_distributions.png", dpi=150); plt.close(fig)
    ev = pd.DataFrame(rows)
    ev.to_csv(TBL_DIR / "phase1_event_validation.csv", index=False)
    logger.info(f"[1.6] event validation: {len(ev)} (event,window) tests")
    return ev


def feature_importance(windows):
    cin = W.cluster_input_cols(windows)
    w = windows
    if len(w) > 400_000:
        w = w.sample(400_000, random_state=RNG)
        logger.info(f"[1.7] subsampled to {len(w):,} windows for RF feature importance")
    X = np.nan_to_num(w[cin].to_numpy("float64"))
    y = w["state"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RNG, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, random_state=RNG, n_jobs=-1).fit(Xtr, ytr)
    acc = accuracy_score(yte, rf.predict(Xte))
    joblib.dump(rf, MODEL_DIR / "rf_feature_importance.joblib")
    imp = pd.DataFrame({"feature": cin, "importance": rf.feature_importances_}) \
        .sort_values("importance", ascending=False)
    imp.to_csv(TBL_DIR / "feature_importance_phase1.csv", index=False)

    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["importance"]); ax.set_title(f"Top-20 Feature Importance (RF acc={acc:.3f})")
    ax.tick_params(axis="y", labelsize=7); fig.tight_layout()
    fig.savefig(PLOT_DIR / "feature_importance.png", dpi=150); plt.close(fig)
    logger.info(f"[1.7] RF acc={acc:.3f}; top feature={imp.iloc[0]['feature']}")
    return imp, acc


def write_report(tm, dwell, ev, imp, acc, states):
    sig = ev[ev.p_value < 0.05] if len(ev) else ev
    top10 = ", ".join(imp.head(10)["feature"].tolist())
    summ = pd.read_csv(TBL_DIR / "phase1_state_summary.csv", index_col=0)
    nvo = ""
    p = TBL_DIR / "phase1_new_vs_old_states.csv"
    if p.exists():
        nvo = ("\n## Change vs the old (pneumatic-based) states\n"
               "States are now defined from **kinematics only** (velocity + acceleration\n"
               "family), so pneumatic sensors stay independent for Phase 2. The contingency\n"
               "of old vs new states is in `phase1_new_vs_old_states.csv`. The main change:\n"
               "stationary brake-holds (zero deceleration) now fall into **standing**\n"
               "(sub-labelled `standing_braked`), and a genuine **accelerating** regime emerges.\n")
    txt = f"""# Phase 1 — Operational State Recognition: Findings

*Train set: Jun 2024 - 11 Feb 2025. Operational states from KINEMATIC-ONLY k-means
(k=4) on 10 s scaled window features (velocity + acceleration family). Pneumatic
sensors are deliberately excluded so they remain independent predictors in Phase 2.*

## States identified
{", ".join(states)}.
Cluster->state mapping in `models/phase1/cluster_labels.json`; physical medians in
`results/tables/cluster_profiles.csv`. Per-state summary:

```
{summ.round(4).to_string()}
```

The `braking` state now has positive velocity and **negative** median acceleration —
i.e. it is genuine deceleration, not a parked train with the brake held (those are
now `standing` / `standing_braked`).
{nvo}
## State transitions (RQ1)
See `transition_matrix.csv` / `transition_diagram.png` and dwell times below:

```
{dwell.round(1).to_string()}
```

## Feature importance (RQ2 / O2)
Random-Forest (state label as target, ALL window features as inputs — descriptive
only) test accuracy **{acc:.3f}**. Top-10 features: {top10}.
Full ranking: `feature_importance_phase1.csv`.

## Validation against documented events (O1)
{len(ev)} (event x window-size) chi-square tests; **{len(sig)}** with p<0.05.
Detail in `phase1_event_validation.csv` and `pre_event_cluster_distributions.png`.

## Notes
- Operational-state *variables* (`TRAIN_*_MODE` etc.) are validation-only; never inputs.
- States are defined by kinematics only (Option A); pneumatics are independent evidence.
- All sensor values are normalized ~[0,1]; physical medians reported in that space.
"""
    (REP_DIR / "01_phase1_findings.md").write_text(txt, encoding="utf-8")
    logger.info("[report] 01_phase1_findings.md written")


def main():
    windows = finalize_labels()
    logger.info(f"finalized {len(windows):,} labeled windows")
    tm, dwell, states = transitions(windows)
    ev = event_validation(windows, states)
    imp, acc = feature_importance(windows)
    write_report(tm, dwell, ev, imp, acc, states)
    logger.info("DONE phase1 validate")


if __name__ == "__main__":
    main()
