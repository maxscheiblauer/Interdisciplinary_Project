"""Phase 1.2-1.4 — operational-state audit, windowing, KINEMATIC-ONLY clustering.

REFOUNDED (Option A): operational states are clustered from KINEMATICS ALONE
(velocity + acceleration family, 12 window features). Pneumatic sensors are NOT
used to define states, so they remain fully independent predictors in Phase 2.
This fixes two problems in the previous version:
  (1) a parked train with the brake held was mislabelled `braking` (it is
      kinematically `standing`); kinematic clustering puts it in `standing`.
  (2) sensors that defined the states were then used to predict them (circular).

The 10 s windows are unchanged (reuse `train_windows.parquet`); only the
clustering inputs and the resulting labels change.

This script produces BOTH k=3 and k=4 candidate models + selection evidence.
The number of states (D2: is there a separable `accelerating` regime?) is
DECIDED in `03_phase1_validate.py` after a human look at the profiles.

Outputs:
  models/phase1/{kinematic_scaler,kmeans_kinematic_k3,kmeans_kinematic_k4}.joblib
  results/tables/{phase1_kselection,phase1_cluster_profiles}.csv
  results/plots/phase1/{dendrogram,kinematic_kselection,kinematic_cluster_scatter}.png
  (operational_state_audit.csv + train_windows.parquet as before)
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
import psutil  # noqa: E402
from loguru import logger  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402

from metroat import schema as S  # noqa: E402
from metroat import windowing as W  # noqa: E402
from metroat.io import discover_files  # noqa: E402

FEAT_ROOT = ROOT / "data" / "processed" / "train_features"
PROC = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "phase1"
PLOT_DIR = ROOT / "results" / "plots" / "phase1"
TBL_DIR = ROOT / "results" / "tables"
for d in (MODEL_DIR, PLOT_DIR, TBL_DIR):
    d.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add(ROOT / "logs" / "phase1" / "cluster.log", level="INFO")
proc = psutil.Process()
RNG = 42

# Kinematic clustering basis (velocity-role window features). Signed accel
# min/max are kept on purpose -- they are what could separate accelerate vs brake.
KINEMATIC_FEATS = [
    "TRAIN_SPEED_ACTUAL__mean", "TRAIN_SPEED_ACTUAL__std",
    "TRAIN_SPEED_ACTUAL__min", "TRAIN_SPEED_ACTUAL__max",
    "acceleration__mean", "acceleration__std",
    "acceleration__min", "acceleration__max",
    "jerk__mean", "jerk__std",
    "velocity_change_rate__mean", "velocity_change_rate__std",
]
HIER_SUBSAMPLE = 8000   # Ward linkage is O(n^2) memory; subsample for the dendrogram
SIL_SAMPLE = 20000


def rss():
    return proc.memory_info().rss / 1e9


def operational_audit(schema):
    if (TBL_DIR / "operational_state_audit.csv").exists():
        logger.info("[1.2] checkpoint: operational_state_audit.csv exists — skipping")
        return
    oper = S.operational_cols(schema)
    vel = S.VELOCITY_COL
    files = discover_files(FEAT_ROOT)
    rng = np.random.default_rng(RNG)
    samples = []
    for f in files:
        df = pd.read_parquet(f, columns=oper + [vel, "brake_cylinder_pressure_mean"],
                             engine="pyarrow")
        k = max(1, int(len(df) * 0.05))
        samples.append(df.iloc[rng.choice(len(df), k, replace=False)])
    s = pd.concat(samples, ignore_index=True)
    s[oper].apply(lambda c: c.astype("float64")).corr().to_csv(
        TBL_DIR / "operational_state_crosscorr.csv")
    rows = []
    for var in oper:
        for val, g in s.groupby(var):
            rows.append(dict(variable=var, value=float(val), n=len(g),
                             median_velocity=float(g[vel].median()),
                             median_brake_pressure=float(g["brake_cylinder_pressure_mean"].median())))
    pd.DataFrame(rows).to_csv(TBL_DIR / "operational_state_audit.csv", index=False)
    logger.info("[1.2] operational_state_audit.csv (validation-only)")


def build_windows(schema):
    out = PROC / "train_windows.parquet"
    if out.exists():
        logger.info("[1.3] checkpoint: train_windows.parquet exists — loading")
        return pd.read_parquet(out)
    scaler = joblib.load(MODEL_DIR / "standard_scaler.joblib")
    files = discover_files(FEAT_ROOT)
    parts = []
    for i, f in enumerate(files):
        df = pd.read_parquet(f, engine="pyarrow")
        parts.append(W.window_day(df, scaler, schema))
        if i % 30 == 0 or i == len(files) - 1:
            logger.info(f"[1.3] windowed {i+1}/{len(files)} RSS={rss():.2f}GB")
    windows = pd.concat(parts, ignore_index=True).sort_values("window_start").reset_index(drop=True)
    windows.to_parquet(out, engine="pyarrow", compression="snappy", index=False)
    logger.info(f"[1.3] train_windows.parquet: {len(windows):,} windows")
    return windows


def _unscale_profiles(windows, labels_col):
    """Median physics per cluster in ORIGINAL normalized space."""
    scaler = joblib.load(MODEL_DIR / "standard_scaler.joblib")
    idx = {f: i for i, f in enumerate(scaler.feature_names_in_)}

    def un(col, feat):
        i = idx[feat]
        return windows[col] * scaler.scale_[i] + scaler.mean_[i]

    keys = {
        "velocity_mean": ("TRAIN_SPEED_ACTUAL__mean", "TRAIN_SPEED_ACTUAL"),
        "velocity_max": ("TRAIN_SPEED_ACTUAL__max", "TRAIN_SPEED_ACTUAL"),
        "accel_mean": ("acceleration__mean", "acceleration"),
        "accel_min": ("acceleration__min", "acceleration"),
        "accel_max": ("acceleration__max", "acceleration"),
        "jerk_mean": ("jerk__mean", "jerk"),
    }
    tmp = pd.DataFrame({"cluster": windows[labels_col].values})
    for name, (col, feat) in keys.items():
        tmp[name] = un(col, feat).values
    tmp["accel_std"] = windows["acceleration__std"].values
    prof = tmp.groupby("cluster").median()
    prof["n_windows"] = windows.groupby(labels_col).size().values
    prof["pct"] = (100 * prof["n_windows"] / len(windows)).round(2)
    return prof


def kinematic_cluster(windows, schema):
    missing = [c for c in KINEMATIC_FEATS if c not in windows.columns]
    assert not missing, f"missing kinematic feats: {missing}"
    X = np.nan_to_num(windows[KINEMATIC_FEATS].to_numpy("float64"))
    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)
    joblib.dump(scaler, MODEL_DIR / "kinematic_scaler.joblib")
    logger.info(f"[1.4] kinematic clustering on {len(KINEMATIC_FEATS)} feats, {len(Xz):,} windows")

    rng = np.random.default_rng(RNG)
    sub = rng.choice(len(Xz), size=min(HIER_SUBSAMPLE, len(Xz)), replace=False)
    sil_idx = rng.choice(len(Xz), size=min(SIL_SAMPLE, len(Xz)), replace=False)

    # --- hierarchical (Ward) on subsample: structure + dendrogram ---
    Z = linkage(Xz[sub], method="ward")
    hier_labels = {k: fcluster(Z, t=k, criterion="maxclust") for k in (3, 4)}

    # --- k-means sweep for selection + ARI vs hierarchical ---
    rows = []
    km_models = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=RNG).fit(Xz)
        sil = silhouette_score(Xz[sil_idx], km.labels_[sil_idx])
        db = davies_bouldin_score(Xz[sil_idx], km.labels_[sil_idx])
        ari = (adjusted_rand_score(hier_labels[k], km.labels_[sub])
               if k in hier_labels else np.nan)
        rows.append(dict(k=k, silhouette=sil, davies_bouldin=db,
                         kmeans_vs_hier_ari=ari))
        logger.info(f"[1.4] k={k}: sil={sil:.3f} db={db:.3f} "
                    f"ari(km vs hier)={ari if isinstance(ari, float) else float('nan'):.3f}")
        if k in (3, 4):
            km_models[k] = km
            joblib.dump(km, MODEL_DIR / f"kmeans_kinematic_k{k}.joblib")
    pd.DataFrame(rows).to_csv(TBL_DIR / "phase1_kselection.csv", index=False)

    # --- candidate profiles for k=3 and k=4 ---
    profs = []
    for k, km in km_models.items():
        windows[f"_k{k}"] = km.labels_
        p = _unscale_profiles(windows, f"_k{k}").reset_index()
        p.insert(0, "n_states", k)
        profs.append(p)
    prof_all = pd.concat(profs, ignore_index=True)
    prof_all.to_csv(TBL_DIR / "phase1_cluster_profiles.csv", index=False)
    logger.info("[1.4] candidate profiles (k=3 and k=4):\n" + prof_all.round(4).to_string(index=False))

    # --- plots ---
    fig, ax = plt.subplots(figsize=(11, 4.5))
    dendrogram(Z, truncate_mode="lastp", p=20, ax=ax, no_labels=True)
    ax.set_title("Hierarchical (Ward) dendrogram — kinematic features (8k subsample)")
    ax.set_ylabel("merge distance")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "dendrogram.png", dpi=150); plt.close(fig)

    sel = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(sel.k, sel.silhouette, "o-", color="green")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("silhouette"); axes[0].set_title("Silhouette (higher=better)")
    axes[1].plot(sel.k, sel.davies_bouldin, "o-", color="purple")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Davies-Bouldin"); axes[1].set_title("Davies-Bouldin (lower=better)")
    fig.suptitle("Kinematic-clustering model selection")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "kinematic_kselection.png", dpi=150); plt.close(fig)

    pca = PCA(n_components=2, random_state=RNG).fit(Xz)
    pcs = pca.transform(Xz[sil_idx])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, k in zip(axes, (3, 4)):
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=km_models[k].labels_[sil_idx],
                        cmap="viridis", s=5, alpha=0.4)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
        ax.set_title(f"k={k}")
        plt.colorbar(sc, ax=ax, label="cluster")
    fig.suptitle("Kinematic clusters in 2-D PCA space")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "kinematic_cluster_scatter.png", dpi=150); plt.close(fig)

    windows.drop(columns=[c for c in windows.columns if c.startswith("_k")], inplace=True)
    logger.info("[1.4] candidates saved; DECIDE k in 03_phase1_validate.py (D2)")


def main():
    schema = S.load_schema()
    logger.info(f"START phase1 kinematic cluster | RSS={rss():.2f}GB")
    operational_audit(schema)
    windows = build_windows(schema)
    kinematic_cluster(windows, schema)
    logger.info(f"DONE phase1 cluster (candidates) | RSS={rss():.2f}GB")


if __name__ == "__main__":
    main()
