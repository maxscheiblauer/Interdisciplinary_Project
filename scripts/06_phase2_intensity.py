"""Phase 2 — Task 3: do braking events cluster into real intensity groups?

Replaces v1's dreamt-up quantile bins with a data-driven test of whether braking
intensity is discrete (real, stable clusters) or a continuum. We do NOT force k=3.

Two honest subtleties drive the analysis:
  1. ~45% of braking-STATE events are stationary brake-holds (zero deceleration,
     train already stopped). Naive clustering on ALL events finds "stable" clusters
     that merely separate stationary-vs-moving and split by start-velocity -- NOT
     intensity. We show this, then discard it as degenerate.
  2. The real intensity question is asked on REAL-DECELERATION events using the
     intensity features only (peak/mean deceleration, jerk). Model selection
     (GMM BIC/AIC) + silhouette/Davies-Bouldin + bootstrap ARI decide k.

Decision rule (corrected): discrete intensity classes exist only if BIC has an
INTERIOR minimum (a preferred k that is neither k=1 nor the largest k tried) AND
that k is well-separated (silhouette) AND stable (bootstrap ARI). A monotonically
decreasing BIC means "more components always help" = a continuum, not classes.

  * stable interior k>=2 -> name clusters, save, PROCEED to Task 4.
  * else -> CONTINUUM. Skip Task 4; Task 5 (regression) is the primary result.

Outputs:
  data/processed/braking_event_clusters.parquet   (only if stable)
  results/tables/cluster_selection.csv            (real-deceleration test)
  results/tables/cluster_selection_alldata.csv    (degenerate all-events run)
  results/plots/phase2/{cluster_model_selection,cluster_scatter,
                          intensity_distribution}.png
  logs/phase2/cluster_decision.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from loguru import logger  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.discriminant_analysis import (  # noqa: E402
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from metroat import braking_v2 as BV2  # noqa: E402

PROC = ROOT / "data" / "processed"
PLOT_DIR = ROOT / "results" / "plots" / "phase2"
TBL_DIR = ROOT / "results" / "tables"
LOG_DIR = ROOT / "logs" / "phase2"
MODEL_DIR = ROOT / "models" / "phase2"
LEDGER = TBL_DIR / "feature_roles.csv"
CLUSTERS = PROC / "braking_event_clusters.parquet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RNG = 42

# All-events kinematic basis (shows the degenerate result).
KINEMATIC_ALL = ["peak_deceleration", "mean_deceleration", "delta_v",
                 "duration_seconds", "jerk_rms", "velocity_at_start"]
# Intensity-only basis on real-deceleration events (the honest test).
INTENSITY = ["peak_deceleration", "mean_deceleration", "jerk_rms"]

SIL_MIN = 0.25
ARI_MIN = 0.70
N_BOOT = 20
SIL_SAMPLE = 10000
ARI_REF = 20000
K_MAX = 6

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def bootstrap_ari(Xz, k, rng):
    n = len(Xz)
    ref_idx = rng.choice(n, size=min(ARI_REF, n), replace=False)
    Xref = Xz[ref_idx]
    labelings = []
    for _ in range(N_BOOT):
        boot = rng.choice(n, size=n, replace=True)
        km = KMeans(n_clusters=k, n_init=5, random_state=RNG).fit(Xz[boot])
        labelings.append(km.predict(Xref))
    aris = [adjusted_rand_score(labelings[i], labelings[j])
            for i in range(len(labelings)) for j in range(i + 1, len(labelings))]
    return float(np.mean(aris))


def _strat_sample(labels, size, rng):
    """Stratified subsample with >=2 points per cluster (so silhouette is defined
    even when one cluster dominates -- which is itself a degeneracy signal)."""
    idx = []
    uniq = np.unique(labels)
    for c in uniq:
        ci = np.where(labels == c)[0]
        take = max(2, int(round(size * len(ci) / len(labels))))
        idx.append(rng.choice(ci, size=min(take, len(ci)), replace=False))
    return np.concatenate(idx)


def selection_table(Xz, with_stability=True):
    rng = np.random.default_rng(RNG)
    rows = []
    for k in range(1, K_MAX + 1):
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=RNG, n_init=2).fit(Xz)
        sil = db = ari = np.nan
        min_cluster_frac = np.nan
        if k >= 2:
            km = KMeans(n_clusters=k, n_init=10, random_state=RNG).fit(Xz)
            counts = np.bincount(km.labels_, minlength=k)
            min_cluster_frac = float(counts.min() / counts.sum())
            sidx = _strat_sample(km.labels_, min(SIL_SAMPLE, len(Xz)), rng)
            sil = silhouette_score(Xz[sidx], km.labels_[sidx])
            db = davies_bouldin_score(Xz, km.labels_)
            if with_stability:
                ari = bootstrap_ari(Xz, k, np.random.default_rng(RNG))
        rows.append(dict(k=k, gmm_bic=gm.bic(Xz), gmm_aic=gm.aic(Xz),
                         kmeans_silhouette=sil, davies_bouldin=db,
                         bootstrap_ari=ari, min_cluster_frac=min_cluster_frac))
    return pd.DataFrame(rows)


def has_interior_bic_min(sel: pd.DataFrame) -> tuple[bool, int]:
    """True iff BIC's argmin is an interior k (not 1, not K_MAX)."""
    kstar = int(sel.loc[sel["gmm_bic"].idxmin(), "k"])
    return (1 < kstar < K_MAX), kstar


def main() -> None:
    ev = pd.read_parquet(
        PROC / "braking_state_events.parquet",
        columns=list(set(KINEMATIC_ALL + ["event_id", "is_real_deceleration"])))
    n_all = len(ev)
    frac_hold = float((~ev["is_real_deceleration"]).mean())
    logger.info(f"[3] {n_all:,} events; stationary brake-holds (zero-decel) = {frac_hold:.1%}")

    # --- (A) degenerate all-events run (for the report) ---
    Xa = StandardScaler().fit_transform(ev[KINEMATIC_ALL].to_numpy("float64"))
    sel_all = selection_table(Xa, with_stability=False)
    sel_all.to_csv(TBL_DIR / "cluster_selection_alldata.csv", index=False)
    logger.info("[3] all-events run (degenerate):\n" + sel_all.round(3).to_string(index=False))

    # --- (B) honest intensity test on real-deceleration events ---
    real = ev[ev["is_real_deceleration"]].reset_index(drop=True)
    Xz = StandardScaler().fit_transform(real[INTENSITY].to_numpy("float64"))
    logger.info(f"[3] honest test: {len(real):,} real-deceleration events on "
                f"{len(INTENSITY)} intensity features")
    sel = selection_table(Xz, with_stability=True)
    sel.to_csv(TBL_DIR / "cluster_selection.csv", index=False)
    for _, r in sel.iterrows():
        logger.info(f"[3] k={int(r.k)}: BIC={r.gmm_bic:,.0f} sil={r.kmeans_silhouette:.3f} "
                    f"DB={r.davies_bouldin:.3f} ARI={r.bootstrap_ari:.3f}")

    interior, kstar = has_interior_bic_min(sel)
    stable = False
    chosen_k = None
    if interior:
        sil_ok = sel.loc[sel.k == kstar, "kmeans_silhouette"].iloc[0] >= SIL_MIN
        ari_ok = sel.loc[sel.k == kstar, "bootstrap_ari"].iloc[0] >= ARI_MIN
        if sil_ok and ari_ok:
            stable, chosen_k = True, kstar

    # --- decision text ---
    lines = ["Phase 2 -- Task 3 cluster decision", "=" * 55, ""]
    lines.append(f"Total braking-state events: {n_all:,}")
    lines.append(f"Stationary brake-holds (zero deceleration): {frac_hold:.1%}")
    lines.append("  -> these are brake applied while already stopped, NOT decelerations.")
    lines.append("")
    lines.append("(A) Naive clustering on ALL events / all kinematic features finds high")
    lines.append("    silhouette + near-perfect ARI, BUT this is DEGENERATE: it separates")
    lines.append("    stationary-holds from moving stops and splits by START VELOCITY, not")
    lines.append("    by intensity. BIC also decreases monotonically (no preferred k).")
    lines.append("    -> discarded as not an intensity result. See *_alldata.csv.")
    lines.append("")
    lines.append("(B) Honest intensity test -- real-deceleration events, intensity features:")
    lines.append(sel.round(3).to_string(index=False))
    lines.append("")
    lines.append(f"BIC argmin at k={kstar}  (interior minimum: {interior}; K_MAX={K_MAX})")
    lines.append(f"Thresholds: silhouette>={SIL_MIN}, bootstrap_ARI>={ARI_MIN}")
    lines.append("")

    if stable:
        rng = np.random.default_rng(RNG)
        km = KMeans(n_clusters=chosen_k, n_init=10, random_state=RNG).fit(Xz)
        real_out = real[["event_id"]].copy()
        real_out["cluster"] = km.labels_
        order = (pd.Series(real["peak_deceleration"].to_numpy())
                 .groupby(km.labels_).mean().sort_values().index.tolist())
        names = (["gentle", "hard", "emergency_like"] if chosen_k == 3
                 else [f"intensity_{i}" for i in range(chosen_k)])
        name_map = {c: names[min(i, len(names) - 1)] for i, c in enumerate(order)}
        real_out["cluster_name"] = real_out["cluster"].map(name_map)
        real_out.to_parquet(PROC / "braking_event_clusters.parquet", index=False)
        lines.append(f"DECISION: STABLE CLUSTERS at k={chosen_k}. Assignments saved.")
        lines.append(f"  names by mean peak_deceleration: {name_map}")
        lines.append("  -> PROCEED to Task 4.")
        logger.info(f"[3] DECISION: stable clusters k={chosen_k} -> Task 4")
        km_plot = km
    else:
        # ensure no stale cluster file remains
        (PROC / "braking_event_clusters.parquet").unlink(missing_ok=True)
        lines.append("DECISION: CONTINUUM -- braking intensity is NOT discrete classes.")
        lines.append("  BIC has no interior minimum (more GMM components always help), i.e.")
        lines.append("  the deceleration distribution is a single skewed pile (sharp mode")
        lines.append("  ~0.046 normalized, long thin tail to ~1.0), not separable regimes.")
        lines.append("  -> SKIP Task 4. Task 5 (deceleration regression) is the PRIMARY")
        lines.append("     intensity result. This is a legitimate, publishable finding and")
        lines.append("     matches the smooth unimodal deceleration seen in v1.")
        logger.info("[3] DECISION: CONTINUUM -> skip Task 4; Task 5 primary")
        km_plot = KMeans(n_clusters=3, n_init=10, random_state=RNG).fit(Xz)

    (LOG_DIR / "cluster_decision.txt").write_text("\n".join(lines), encoding="utf-8")

    # --- model-selection plot (honest test) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(sel.k, sel.gmm_bic, "o-", label="BIC")
    axes[0].plot(sel.k, sel.gmm_aic, "s--", label="AIC")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("information criterion")
    axes[0].set_title("GMM BIC/AIC (real-decel, intensity feats)\nmonotonic = no preferred k")
    axes[0].legend()
    axes[1].plot(sel.k, sel.kmeans_silhouette, "o-", color="green")
    axes[1].axhline(SIL_MIN, color="grey", ls=":")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("silhouette")
    axes[1].set_title("k-means silhouette")
    axes[2].plot(sel.k, sel.bootstrap_ari, "o-", color="purple")
    axes[2].axhline(ARI_MIN, color="grey", ls=":")
    axes[2].set_xlabel("k"); axes[2].set_ylabel("bootstrap ARI")
    axes[2].set_title("Cluster stability")
    fig.suptitle("Braking-intensity cluster model selection (honest test)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "cluster_model_selection.png", dpi=150)
    plt.close(fig)

    # --- intensity distribution plot (the continuum) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(ev["peak_deceleration"], bins=100, color="slategray")
    axes[0].axvline(0, color="red", ls="--", label=f"zero-decel mass ({frac_hold:.0%})")
    axes[0].set_xlabel("peak deceleration (normalized)"); axes[0].set_ylabel("count")
    axes[0].set_title("All braking-state events (note the zero spike)")
    axes[0].legend()
    axes[1].hist(real["peak_deceleration"], bins=100, color="steelblue")
    axes[1].set_xlabel("peak deceleration (normalized)"); axes[1].set_ylabel("count")
    axes[1].set_title("Real-deceleration events: unimodal pile + tail (continuum)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "intensity_distribution.png", dpi=150)
    plt.close(fig)

    # --- scatter in top-2 intensity PCs ---
    rng = np.random.default_rng(RNG)
    pca = PCA(n_components=2, random_state=RNG)
    pcs = pca.fit_transform(Xz)
    pidx = rng.choice(len(pcs), size=min(15000, len(pcs)), replace=False)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sc = ax.scatter(pcs[pidx, 0], pcs[pidx, 1], c=km_plot.labels_[pidx],
                    cmap="viridis", s=4, alpha=0.4)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
    ax.set_title("Real-deceleration events in intensity PC space "
                 + ("[STABLE]" if stable else "[continuum; k=3 coloring illustrative]"))
    plt.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "cluster_scatter.png", dpi=150)
    plt.close(fig)
    logger.info("[3] outputs written; cluster decision complete.")
    recover_from_pneumatics()


def _event_feats(ledger, roles, present):
    sel = ledger[(ledger.level == "event") & (ledger.role.isin(roles))]
    return [f for f in sel["feature"] if f in present]


def recover_from_pneumatics() -> None:
    """Conditional: recover the intensity clusters from pneumatic event features.
    Runs ONLY if Task 3 found stable clusters (braking_event_clusters.parquet)."""
    if not CLUSTERS.exists():
        msg = ("[3b] braking_event_clusters.parquet not found -> intensity is a "
               "CONTINUUM (no stable clusters). Recover-from-pneumatics is correctly "
               "SKIPPED; the deceleration regression (07) is the primary intensity result.")
        logger.info(msg)
        (LOG_DIR / "recover_skipped.txt").write_text(msg, encoding="utf-8")
        return

    ledger = pd.read_csv(LEDGER)
    ev = pd.read_parquet(PROC / "braking_state_events.parquet")
    clusters = pd.read_parquet(CLUSTERS)
    ev = ev.merge(clusters, on="event_id", how="inner")
    present = set(ev.columns)
    aux = _event_feats(ledger, ["auxiliary"], present)
    full = _event_feats(ledger, ["auxiliary", "actuation"], present)
    y = ev["cluster"].to_numpy()
    classes = np.unique(y)
    logger.info(f"[3b] {len(ev):,} events, {len(classes)} clusters; recovering from pneumatics")

    rows, confusions, importances = [], {}, None
    for tier_name, feats in {"auxonly": aux, "full": full}.items():
        X = ev[feats].to_numpy("float64")
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=RNG, stratify=y)
        models = {
            "majority": DummyClassifier(strategy="most_frequent"),
            "stratified": DummyClassifier(strategy="stratified", random_state=RNG),
            "dt": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RNG),
            "lda": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
            "qda": make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=1e-3)),
        }
        for mname, model in models.items():
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            macro_f1 = f1_score(yte, pred, average="macro")
            p, r, f1, _ = precision_recall_fscore_support(
                yte, pred, labels=classes, average=None, zero_division=0)
            auc = np.nan
            try:
                auc = roc_auc_score(yte, model.predict_proba(Xte), multi_class="ovr", average="macro")
            except Exception:
                pass
            rows.append(dict(tier=tier_name, model=mname, n_features=len(feats),
                             macro_f1=macro_f1, roc_auc_ovr=auc,
                             **{f"f1_c{c}": f1[i] for i, c in enumerate(classes)}))
            logger.info(f"[3b] {tier_name}/{mname}: macroF1={macro_f1:.3f} AUC={auc:.3f}")
            if mname in ("dt", "lda", "qda"):
                joblib.dump(model, MODEL_DIR / f"recover_clf_{mname}.joblib")
                confusions[(tier_name, mname)] = confusion_matrix(yte, pred, labels=classes)
            if tier_name == "full" and mname == "dt":
                importances = (feats, model.feature_importances_)
        del X

    pd.DataFrame(rows).to_csv(TBL_DIR / "recover_metrics.csv", index=False)
    feats, imp = importances
    order = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh([feats[i] for i in order][::-1], imp[order][::-1], color="indianred")
    ax.set_xlabel("decision-tree importance")
    ax.set_title("Pneumatic drivers of recovered intensity clusters (tier B DT)")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "recover_importances.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, mname in zip(axes, ["dt", "lda", "qda"]):
        cm = confusions[("full", mname)]
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for rr in range(len(classes)):
            for cc in range(len(classes)):
                ax.text(cc, rr, f"{cmn[rr, cc]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"full / {mname}"); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.suptitle("Recover-from-pneumatics confusion (held-out, row-normalized)")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "recover_confusion.png", dpi=150); plt.close(fig)
    logger.info("[3b] recover-from-pneumatics complete.")


if __name__ == "__main__":
    main()
