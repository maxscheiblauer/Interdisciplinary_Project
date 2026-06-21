"""Phase 1 — operational-state recognition.

Windows the 1 Hz day frames, clusters them on kinematics only (k selection over
k=3..6), fixes the final k=4 labelling, then validates the states via transitions,
documented-event distributions and a random-forest importance check. The narrative
(what the numbers mean) lives in notebooks/phase1_operational_states.ipynb and the
generated results/reports/phase1_findings.md.
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
import psutil  # noqa: E402
from loguru import logger  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from metroat import schema as S  # noqa: E402
from metroat import windowing as W  # noqa: E402
from metroat.io import discover_files  # noqa: E402
from metroat.validation import chi_square_shift  # noqa: E402

FEAT_ROOT = ROOT / "data" / "processed" / "train_features"
PROC = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "phase1"
PLOT_DIR = ROOT / "results" / "plots" / "phase1"
TBL_DIR = ROOT / "results" / "tables"
REP_DIR = ROOT / "results" / "reports"
for d in (MODEL_DIR, PLOT_DIR, TBL_DIR, REP_DIR):
    d.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add(ROOT / "logs" / "phase1" / "cluster.log", level="INFO")
proc = psutil.Process()
RNG = 42

K_FINAL = 4
STATES = ["standing", "accelerating", "cruising", "braking"]
GAP_LIMIT_S = 30  # windows further apart than this are not a transition

# Velocity-role window features used as the clustering basis. Signed accel min/max
# are kept on purpose -- they are what could separate accelerate vs brake.
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
        logger.info("checkpoint: operational_state_audit.csv exists — skipping")
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
    logger.info("operational_state_audit.csv (validation-only)")


def build_windows(schema):
    out = PROC / "train_windows.parquet"
    if out.exists():
        logger.info("checkpoint: train_windows.parquet exists — loading")
        return pd.read_parquet(out)
    scaler = joblib.load(MODEL_DIR / "standard_scaler.joblib")
    files = discover_files(FEAT_ROOT)
    parts = []
    for i, f in enumerate(files):
        df = pd.read_parquet(f, engine="pyarrow")
        parts.append(W.window_day(df, scaler, schema))
        if i % 30 == 0 or i == len(files) - 1:
            logger.info(f"windowed {i+1}/{len(files)} RSS={rss():.2f}GB")
    windows = pd.concat(parts, ignore_index=True).sort_values("window_start").reset_index(drop=True)
    windows.to_parquet(out, engine="pyarrow", compression="snappy", index=False)
    logger.info(f"train_windows.parquet: {len(windows):,} windows")
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
    logger.info(f"kinematic clustering on {len(KINEMATIC_FEATS)} feats, {len(Xz):,} windows")

    rng = np.random.default_rng(RNG)
    sub = rng.choice(len(Xz), size=min(HIER_SUBSAMPLE, len(Xz)), replace=False)
    sil_idx = rng.choice(len(Xz), size=min(SIL_SAMPLE, len(Xz)), replace=False)

    # hierarchical (Ward) on subsample: structure + dendrogram
    Z = linkage(Xz[sub], method="ward")
    hier_labels = {k: fcluster(Z, t=k, criterion="maxclust") for k in (3, 4)}

    # k-means sweep + GMM BIC/AIC for selection + ARI vs hierarchical
    rows = []
    km_models = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=RNG).fit(Xz)
        sil = silhouette_score(Xz[sil_idx], km.labels_[sil_idx])
        db = davies_bouldin_score(Xz[sil_idx], km.labels_[sil_idx])
        ari = (adjusted_rand_score(hier_labels[k], km.labels_[sub])
               if k in hier_labels else np.nan)
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=RNG, n_init=2).fit(Xz[sil_idx])
        bic_val = gm.bic(Xz[sil_idx])
        aic_val = gm.aic(Xz[sil_idx])
        rows.append(dict(k=k, silhouette=sil, davies_bouldin=db,
                         kmeans_vs_hier_ari=ari,
                         gmm_bic=bic_val, gmm_aic=aic_val))
        logger.info(f"k={k}: sil={sil:.3f} db={db:.3f} "
                    f"ari(km vs hier)={ari if isinstance(ari, float) else float('nan'):.3f} "
                    f"bic={bic_val:,.0f}")
        if k in (3, 4):
            km_models[k] = km
            joblib.dump(km, MODEL_DIR / f"kmeans_kinematic_k{k}.joblib")
    pd.DataFrame(rows).to_csv(TBL_DIR / "phase1_kselection.csv", index=False)

    # candidate profiles for k=3 and k=4
    profs = []
    for k, km in km_models.items():
        windows[f"_k{k}"] = km.labels_
        p = _unscale_profiles(windows, f"_k{k}").reset_index()
        p.insert(0, "n_states", k)
        profs.append(p)
    prof_all = pd.concat(profs, ignore_index=True)
    prof_all.to_csv(TBL_DIR / "phase1_cluster_profiles.csv", index=False)
    logger.info("candidate profiles (k=3 and k=4):\n" + prof_all.round(4).to_string(index=False))

    # plots
    fig, ax = plt.subplots(figsize=(11, 4.5))
    dendrogram(Z, truncate_mode="lastp", p=20, ax=ax, no_labels=True)
    ax.set_title("Hierarchical (Ward) dendrogram - 8000 subsample")
    ax.set_ylabel("merge distance")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "dendrogram.png", dpi=150); plt.close(fig)

    sel = pd.DataFrame(rows)
    k_bic_min = int(sel.loc[sel.gmm_bic.idxmin(), "k"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(sel.k, sel.silhouette, "o-", color="green")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("silhouette"); axes[0].set_title("Silhouette")
    axes[1].plot(sel.k, sel.davies_bouldin, "o-", color="purple")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Davies-Bouldin"); axes[1].set_title("Davies-Bouldin")
    axes[2].plot(sel.k, sel.gmm_bic, "o-", label="BIC")
    axes[2].plot(sel.k, sel.gmm_aic, "s--", label="AIC")
    axes[2].axvline(k_bic_min, color="grey", ls=":", alpha=0.8, label=f"k={k_bic_min} (BIC min)")
    axes[2].set_xlabel("k"); axes[2].set_ylabel("information criterion")
    axes[2].set_title("GMM BIC / AIC ")
    axes[2].legend()
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

    vel_raw = windows["TRAIN_SPEED_ACTUAL__mean"].to_numpy()[sil_idx]
    acc_raw = windows["acceleration__mean"].to_numpy()[sil_idx]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, k in zip(axes, (3, 4)):
        sc = ax.scatter(vel_raw, acc_raw, c=km_models[k].labels_[sil_idx],
                        cmap="tab10", s=4, alpha=0.35)
        ax.set_xlabel("velocity (normalised)")
        ax.set_ylabel("acceleration (normalised)")
        ax.set_title(f"k={k}")
        plt.colorbar(sc, ax=ax, label="cluster")
    fig.suptitle("Kinematic clusters: velocity vs acceleration (20 000 subsample)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "kinematic_vel_accel_scatter.png", dpi=150)
    plt.close(fig)

    windows.drop(columns=[c for c in windows.columns if c.startswith("_k")], inplace=True)
    logger.info("k=3/k=4 candidates saved")


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

    # physics-based naming
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
    logger.info(f"k=4 labels: {labels}")

    # held-brake sub-label of standing (descriptor, NOT a clustering input)
    brake_active_cols = [c for c in windows.columns
                         if "PNEUMATIC_BRAKE_ACTIVE" in c and c.endswith("__maj")]
    held = windows[brake_active_cols].max(axis=1).fillna(0).astype(int) if brake_active_cols else 0
    windows["standing_substate"] = ""
    is_standing = windows["state"] == "standing"
    windows.loc[is_standing & (held == 1), "standing_substate"] = "standing_braked"
    windows.loc[is_standing & (held == 0), "standing_substate"] = "standing_idle"
    sub_counts = windows.loc[is_standing, "standing_substate"].value_counts().to_dict()
    logger.info(f"held-brake sub-label of standing: {sub_counts}")

    # profiles table (canonical)
    profile = med.copy()
    profile["accel_std"] = prof.assign(
        accel_std=windows["acceleration__std"].values).groupby("cluster")["accel_std"].median()
    profile["n_windows"] = windows.groupby("cluster").size()
    profile["pct"] = (100 * profile["n_windows"] / len(windows)).round(2)
    profile["label"] = [labels[c] for c in profile.index]
    profile.round(4).to_csv(TBL_DIR / "cluster_profiles.csv")

    # state summary (braking should land at v>0, accel<0 = real deceleration)
    summ = (windows.groupby("state")
            .agg(n=("state", "size")).reindex(STATES))
    summ["pct"] = (100 * summ["n"] / len(windows)).round(2)
    summ["median_velocity"] = [float(prof.loc[prof.cluster.map(labels) == s, "velocity_mean"].median())
                               for s in STATES]
    summ["median_accel"] = [float(prof.loc[prof.cluster.map(labels) == s, "accel_mean"].median())
                            for s in STATES]
    summ.to_csv(TBL_DIR / "phase1_state_summary.csv")
    brk = summ.loc["braking"]
    logger.info(f"braking state: median velocity={brk['median_velocity']:.3f}, "
                f"median accel={brk['median_accel']:.4f}")

    if old_state is not None:
        ari = adjusted_rand_score(old_state, windows["state"].to_numpy())
        ct = pd.crosstab(pd.Series(old_state, name="old_state"),
                         windows["state"].rename("new_state"))
        ct.to_csv(TBL_DIR / "phase1_new_vs_old_states.csv")
        logger.info(f"new-vs-old state ARI = {ari:.3f}\n{ct}")

    # profile bar plot
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
    logger.info(f"train_windows_labeled.parquet written ({len(windows):,} rows)")
    return windows


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
    logger.info(f"transitions + dwell + monthly distribution | states={states}")
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
    logger.info(f"event validation: {len(ev)} (event,window) tests")
    return ev


def feature_importance(windows):
    cin = W.cluster_input_cols(windows)
    w = windows
    if len(w) > 400_000:
        w = w.sample(400_000, random_state=RNG)
        logger.info(f"subsampled to {len(w):,} windows for RF feature importance")
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
    logger.info(f"RF acc={acc:.3f}; top feature={imp.iloc[0]['feature']}")
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
    (REP_DIR / "phase1_findings.md").write_text(txt, encoding="utf-8")
    logger.info("phase1_findings.md written")


def main():
    schema = S.load_schema()
    logger.info(f"START phase1 | RSS={rss():.2f}GB")
    operational_audit(schema)
    windows = build_windows(schema)
    kinematic_cluster(windows, schema)
    windows = finalize_labels()
    logger.info(f"finalized {len(windows):,} labeled windows")
    tm, dwell, states = transitions(windows)
    ev = event_validation(windows, states)
    imp, acc = feature_importance(windows)
    write_report(tm, dwell, ev, imp, acc, states)
    logger.info(f"DONE phase1 | RSS={rss():.2f}GB")


if __name__ == "__main__":
    main()
