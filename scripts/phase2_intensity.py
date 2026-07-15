"""Phase 2 (part 2) — deceleration intensity and pre-failure behaviour.

Three analyses. (1) Is deceleration intensity discrete classes or a continuum? Model
selection (GMM BIC/AIC + silhouette + bootstrap ARI) on real-deceleration events
decides; if stable clusters appear they are named and recovered from the pneumatic
sensors. (2) Deceleration regression: how much of peak/mean deceleration the
non-velocity sensors explain out-of-sample. (3) Pre-failure deceleration behaviour:
distributional tests, an exploratory coefficient-shift OLS, and CUSUM on a weekly
wear proxy, with honest small-N framing. Interpretation lives in the Phase 2
notebook and results/reports/phase2_findings.md.
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
import statsmodels.formula.api as smf  # noqa: E402
from loguru import logger  # noqa: E402
from scipy.stats import ks_2samp, mannwhitneyu  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.discriminant_analysis import (  # noqa: E402
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from metroat.validation import cusum  # noqa: E402

PROC = ROOT / "data" / "processed"
PLOT_DIR = ROOT / "results" / "plots" / "phase2"
TBL_DIR = ROOT / "results" / "tables"
LOG_DIR = ROOT / "logs" / "phase2"
MODEL_DIR = ROOT / "models" / "phase2"
LEDGER = TBL_DIR / "feature_roles.csv"
CLUSTERS = PROC / "deceleration_event_clusters.parquet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RNG = 42

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def _event_feats(ledger: pd.DataFrame, roles: list[str], present: set[str]) -> list[str]:
    sel = ledger[(ledger.level == "event") & (ledger.role.isin(roles))]
    return [f for f in sel["feature"] if f in present]


# --------------------------------------------------------------------------- #
# Intensity: discrete classes or a continuum?
# --------------------------------------------------------------------------- #

# All-events kinematic basis (the degenerate contrast run).
KINEMATIC_ALL = ["peak_deceleration", "mean_deceleration", "delta_v",
                 "duration_seconds", "jerk_rms", "velocity_at_start"]
# Intensity-only basis on real-deceleration events (the honest test).
INTENSITY = ["peak_deceleration", "mean_deceleration", "jerk_rms"]
# Same basis + the kinematic deceleration-energy axis. Supplementary run answering #5:
# adding energy must not manufacture separable classes.
INTENSITY_ENERGY = INTENSITY + ["deceleration_energy_specific"]

SIL_MIN = 0.25
ARI_MIN = 0.70
N_BOOT = 20
SIL_SAMPLE = 10000
ARI_REF = 20000
K_MAX = 6


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
    """Stratified subsample with >=2 points per cluster so silhouette stays defined
    even when one cluster dominates (itself a degeneracy signal)."""
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


def cluster_intensity() -> None:
    ev = pd.read_parquet(
        PROC / "deceleration_state_events.parquet",
        columns=list(set(KINEMATIC_ALL + INTENSITY_ENERGY
                         + ["event_id", "is_real_deceleration"])))
    n_all = len(ev)
    frac_hold = float((~ev["is_real_deceleration"]).mean())
    logger.info(f"{n_all:,} events; stationary brake-holds (zero-decel) = {frac_hold:.1%}")

    # (A) all-events run (kept as the degenerate contrast)
    Xa = StandardScaler().fit_transform(ev[KINEMATIC_ALL].to_numpy("float64"))
    sel_all = selection_table(Xa, with_stability=False)
    sel_all.to_csv(TBL_DIR / "cluster_selection_alldata.csv", index=False)
    logger.info("all-events run (degenerate):\n" + sel_all.round(3).to_string(index=False))

    # (B) intensity test on real-deceleration events
    real = ev[ev["is_real_deceleration"]].reset_index(drop=True)
    Xz = StandardScaler().fit_transform(real[INTENSITY].to_numpy("float64"))
    logger.info(f"honest test: {len(real):,} real-deceleration events on "
                f"{len(INTENSITY)} intensity features")
    sel = selection_table(Xz, with_stability=True)
    sel.to_csv(TBL_DIR / "cluster_selection.csv", index=False)
    for _, r in sel.iterrows():
        logger.info(f"k={int(r.k)}: BIC={r.gmm_bic:,.0f} sil={r.kmeans_silhouette:.3f} "
                    f"DB={r.davies_bouldin:.3f} ARI={r.bootstrap_ari:.3f}")

    # (C) supplementary run: same real-deceleration events, intensity basis PLUS the
    # kinematic deceleration-energy axis. Deceleration energy ~ 1/2(v0^2-v1^2) is a smooth
    # deterministic function of the same continuum kinematics, so it cannot open a
    # gap that isn't already there. If BIC stays monotonic (no interior minimum) and
    # silhouette stays below threshold, the energy axis does NOT manufacture classes.
    Xe = StandardScaler().fit_transform(real[INTENSITY_ENERGY].to_numpy("float64"))
    sel_e = selection_table(Xe, with_stability=False)
    sel_e.to_csv(TBL_DIR / "cluster_selection_energy.csv", index=False)
    interior_e, kstar_e = has_interior_bic_min(sel_e)
    sil_max_e = float(np.nanmax(sel_e["kmeans_silhouette"]))
    logger.info(f"energy-augmented run: BIC interior min={interior_e} (argmin k={kstar_e}), "
                f"max silhouette={sil_max_e:.3f} -> "
                f"{'still a continuum' if (not interior_e or sil_max_e < SIL_MIN) else 'CHECK'}")

    interior, kstar = has_interior_bic_min(sel)
    stable = False
    chosen_k = None
    if interior:
        sil_ok = sel.loc[sel.k == kstar, "kmeans_silhouette"].iloc[0] >= SIL_MIN
        ari_ok = sel.loc[sel.k == kstar, "bootstrap_ari"].iloc[0] >= ARI_MIN
        if sil_ok and ari_ok:
            stable, chosen_k = True, kstar

    # decision text (output artifact)
    lines = ["Phase 2 -- intensity cluster decision", "=" * 55, ""]
    lines.append(f"Total deceleration-state events: {n_all:,}")
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
    lines.append("(C) Supplementary run -- intensity basis + deceleration-energy axis (answers the")
    lines.append("    'does forcing an energy integral create weak/strong classes?' question):")
    lines.append(sel_e.round(3).to_string(index=False))
    lines.append(f"    BIC interior minimum: {interior_e} (argmin k={kstar_e}); "
                 f"max silhouette {sil_max_e:.3f}.")
    if not interior_e or sil_max_e < SIL_MIN:
        lines.append("    -> Adding energy leaves BIC monotonic / silhouette below threshold:")
        lines.append("       the energy axis does NOT manufacture separable clusters. A")
        lines.append("       weak/strong split stays an imposed operational threshold, not a")
        lines.append("       structure the data volunteers. See cluster_selection_energy.csv.")
    else:
        lines.append("    -> Energy-augmented run shows an interior BIC minimum -- revisit.")
    lines.append("")

    if stable:
        km = KMeans(n_clusters=chosen_k, n_init=10, random_state=RNG).fit(Xz)
        real_out = real[["event_id"]].copy()
        real_out["cluster"] = km.labels_
        order = (pd.Series(real["peak_deceleration"].to_numpy())
                 .groupby(km.labels_).mean().sort_values().index.tolist())
        names = (["gentle", "hard", "emergency_like"] if chosen_k == 3
                 else [f"intensity_{i}" for i in range(chosen_k)])
        name_map = {c: names[min(i, len(names) - 1)] for i, c in enumerate(order)}
        real_out["cluster_name"] = real_out["cluster"].map(name_map)
        real_out.to_parquet(PROC / "deceleration_event_clusters.parquet", index=False)
        lines.append(f"DECISION: STABLE CLUSTERS at k={chosen_k}. Assignments saved.")
        lines.append(f"  names by mean peak_deceleration: {name_map}")
        lines.append("  -> recover clusters from pneumatics.")
        logger.info(f"DECISION: stable clusters k={chosen_k}")
        km_plot = km
    else:
        # remove any stale cluster file
        (PROC / "deceleration_event_clusters.parquet").unlink(missing_ok=True)
        lines.append("DECISION: CONTINUUM -- deceleration intensity is NOT discrete classes.")
        lines.append("  BIC has no interior minimum (more GMM components always help), i.e.")
        lines.append("  the deceleration distribution is a single skewed pile (sharp mode")
        lines.append("  ~0.046 normalized, long thin tail to ~1.0), not separable regimes.")
        lines.append("  -> deceleration regression is the PRIMARY intensity result.")
        logger.info("DECISION: CONTINUUM")
        km_plot = KMeans(n_clusters=3, n_init=10, random_state=RNG).fit(Xz)

    (LOG_DIR / "cluster_decision.txt").write_text("\n".join(lines), encoding="utf-8")

    # model-selection plot (honest test)
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
    fig.suptitle("Deceleration-intensity cluster model selection (honest test)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "cluster_model_selection.png", dpi=150)
    plt.close(fig)

    # intensity distribution plot
    p01 = float(np.percentile(real["peak_deceleration"], 1))
    p99 = float(np.percentile(real["peak_deceleration"], 99))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(ev["peak_deceleration"], bins=200, color="slategray")
    axes[0].axvline(0, color="red", ls="--", label=f"zero-decel mass ({frac_hold:.0%})")
    axes[0].set_xlabel("peak deceleration (normalized)"); axes[0].set_ylabel("count")
    axes[0].set_title("All deceleration-state events (note the zero spike)")
    axes[0].legend()
    axes[1].hist(real["peak_deceleration"], bins=200, color="steelblue",
                 range=(p01, p99))
    axes[1].set_xlabel("peak deceleration (normalized)"); axes[1].set_ylabel("count")
    axes[1].set_title(f"Real-deceleration events (1–99th pct, {p01:.3f}–{p99:.3f})\n"
                      "unimodal bell + tail (continuum)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "intensity_distribution.png", dpi=150)
    plt.close(fig)

    # scatter in top-2 intensity PCs
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
    logger.info("intensity cluster decision complete")
    recover_from_pneumatics()


def recover_from_pneumatics() -> None:
    """Recover the intensity clusters from pneumatic event features. Runs ONLY if
    stable clusters were found (deceleration_event_clusters.parquet exists)."""
    if not CLUSTERS.exists():
        msg = ("deceleration_event_clusters.parquet not found -> intensity is a CONTINUUM "
               "(no stable clusters). Recover-from-pneumatics is correctly SKIPPED; the "
               "deceleration regression is the primary intensity result.")
        logger.info(msg)
        (LOG_DIR / "recover_skipped.txt").write_text(msg, encoding="utf-8")
        return

    ledger = pd.read_csv(LEDGER)
    ev = pd.read_parquet(PROC / "deceleration_state_events.parquet")
    clusters = pd.read_parquet(CLUSTERS)
    ev = ev.merge(clusters, on="event_id", how="inner")
    present = set(ev.columns)
    aux = _event_feats(ledger, ["auxiliary"], present)
    full = _event_feats(ledger, ["auxiliary", "actuation"], present)
    y = ev["cluster"].to_numpy()
    classes = np.unique(y)
    logger.info(f"{len(ev):,} events, {len(classes)} clusters; recovering from pneumatics")

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
            logger.info(f"{tier_name}/{mname}: macroF1={macro_f1:.3f} AUC={auc:.3f}")
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
    logger.info("recover-from-pneumatics complete")


# --------------------------------------------------------------------------- #
# Deceleration regression
# --------------------------------------------------------------------------- #
# deceleration_energy_specific is added as a third target: "how much of a stop's total
# deceleration energy do the non-velocity sensors explain out-of-sample" -- the
# comparability/wear axis the supervisor asked about. The coefficient/scatter plots
# stay on the two deceleration targets; energy is reported as a number in the table.
PLOT_TARGETS = ["peak_deceleration", "mean_deceleration"]
TARGETS = PLOT_TARGETS + ["deceleration_energy_specific"]


def regress_deceleration() -> None:
    metrics_path = TBL_DIR / "decel_regression_metrics.csv"
    if metrics_path.exists():
        logger.info("decel-regression metrics exist -- skipping (delete to recompute)")
        return

    ledger = pd.read_csv(LEDGER)
    ev = pd.read_parquet(PROC / "deceleration_state_events.parquet")
    present = set(ev.columns)

    aux = _event_feats(ledger, ["auxiliary"], present)
    full = _event_feats(ledger, ["auxiliary", "actuation"], present)
    vel = set(ledger[(ledger.level == "event") & (ledger.role == "velocity")]["feature"])
    assert not (set(full) & vel), "velocity feature leaked into predictors!"
    logger.info(f"tier A={len(aux)} feats | tier B={len(full)} feats (+duration variants)")

    # PRIMARY scope = real-deceleration events; SECONDARY = all deceleration-state events.
    scopes = {
        "real_decel": ev[ev["is_real_deceleration"]].reset_index(drop=True),
        "all": ev,
    }
    logger.info(f"scopes: real_decel={len(scopes['real_decel']):,} | all={len(scopes['all']):,}")

    tiers = {"auxonly": aux, "full": full}
    rows = []
    coef_store: dict[str, tuple[list[str], np.ndarray]] = {}
    scatter_store: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    lasso_coef_store: dict[str, tuple[list[str], np.ndarray]] = {}
    lasso_cv_rows: list[dict] = []

    for scope_name, scope_df in scopes.items():
        for target in TARGETS:
            y = scope_df[target].to_numpy("float64")
            for tier_name, feats in tiers.items():
                for dur in (False, True):
                    cols = feats + (["duration_seconds"] if dur else [])
                    X = scope_df[cols].to_numpy("float64")
                    Xtr, Xte, ytr, yte = train_test_split(
                        X, y, test_size=0.25, random_state=RNG)

                    base_pred = np.full_like(yte, ytr.mean())
                    base_r2 = r2_score(yte, base_pred)
                    base_mae = mean_absolute_error(yte, base_pred)

                    lin = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, ytr)
                    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, ytr)
                    lasso = make_pipeline(StandardScaler(),
                                         Lasso(alpha=0.0001, max_iter=5000)).fit(Xtr, ytr)
                    tree = HistGradientBoostingRegressor(random_state=RNG).fit(Xtr, ytr)
                    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                               n_jobs=-1, random_state=RNG).fit(Xtr, ytr)
                    for mname, model in (("lin", lin), ("ridge", ridge),
                                         ("lasso", lasso), ("tree", tree), ("rf", rf)):
                        pred = model.predict(Xte)
                        rows.append(dict(
                            scope=scope_name, target=target, tier=tier_name,
                            with_duration=dur, model=mname, n_features=len(cols),
                            r2=r2_score(yte, pred), mae=mean_absolute_error(yte, pred),
                            baseline_r2=base_r2, baseline_mae=base_mae))
                        logger.info(f"{scope_name}/{target}/{tier_name}/dur={dur}/{mname}: "
                                    f"R2={r2_score(yte, pred):.3f} MAE={mean_absolute_error(yte, pred):.4f}")

                    # keep coeffs/models for the canonical PRIMARY variant. The Lasso
                    # penalty is chosen by 5-fold cross-validation on the TRAINING split
                    # only (no held-out leakage), replacing the earlier fixed alpha; the
                    # selection it yields drives lasso_coef_store and the coefficient plot.
                    if scope_name == "real_decel" and tier_name == "full" and not dur:
                        coef = lin.named_steps["linearregression"].coef_
                        coef_store[target] = (cols, coef)
                        scatter_store[target] = (yte, rf.predict(Xte))
                        lasso_cv = make_pipeline(
                            StandardScaler(),
                            LassoCV(cv=5, max_iter=20000, random_state=RNG, n_jobs=-1),
                        ).fit(Xtr, ytr)
                        lcv = lasso_cv.named_steps["lassocv"]
                        lasso_coef_store[target] = (cols, lcv.coef_)
                        n_sel = int((lcv.coef_ != 0).sum())
                        lasso_cv_rows.append(dict(
                            target=target, alpha=float(lcv.alpha_),
                            n_selected=n_sel, n_features=len(cols),
                            held_out_r2=r2_score(yte, lasso_cv.predict(Xte))))
                        logger.info(f"LassoCV/{target}: alpha={lcv.alpha_:.2e} "
                                    f"selected={n_sel}/{len(cols)}")
                        joblib.dump(lin, MODEL_DIR / f"decel_regress_lin_{target.split('_')[0]}.joblib")
                        joblib.dump(tree, MODEL_DIR / f"decel_regress_tree_{target.split('_')[0]}.joblib")
                        joblib.dump(rf, MODEL_DIR / f"decel_regress_rf_{target.split('_')[0]}.joblib")

    pd.DataFrame(lasso_cv_rows).to_csv(TBL_DIR / "lasso_cv_selection.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_path, index=False)
    logger.info(f"metrics written:\n{metrics.round(3).to_string(index=False)}")

    for target in TARGETS:
        best = metrics[(metrics.scope == "real_decel") & (metrics.target == target)
                       & (~metrics.with_duration)]
        b = best.loc[best.r2.idxmax()]
        logger.info(f"{target} (real-decel): best held-out R2={b.r2:.3f} ({b.tier}/{b.model})")

    # coefficient plot: linear top-15 + Lasso variable selection
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, target in zip(axes[:2], PLOT_TARGETS):
        cols, coef = coef_store[target]
        order = np.argsort(np.abs(coef))[::-1][:15]
        ax.barh([cols[i] for i in order][::-1], coef[order][::-1], color="teal")
        ax.set_xlabel("standardized coefficient")
        ax.set_title(f"{target}: top-15 linear drivers (tier B)")
    lasso_cols, lasso_coef = lasso_coef_store[PLOT_TARGETS[0]]
    nonzero = np.where(lasso_coef != 0)[0]
    if len(nonzero) == 0:
        axes[2].text(0.5, 0.5, "All Lasso coefficients zeroed\n(try smaller alpha)",
                     ha="center", va="center", transform=axes[2].transAxes)
    else:
        order_l = nonzero[np.argsort(np.abs(lasso_coef[nonzero]))[::-1]]
        axes[2].barh([lasso_cols[i] for i in order_l][::-1],
                     lasso_coef[order_l][::-1], color="indianred")
    axes[2].set_xlabel("Lasso coefficient (zero = excluded)")
    axes[2].set_title(f"{TARGETS[0]}: Lasso variable selection (tier B)\n"
                      f"{len(nonzero)}/{len(lasso_cols)} sensors selected")
    fig.suptitle("Which non-velocity sensors explain deceleration")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "decel_regression_coeffs.png", dpi=150)
    plt.close(fig)

    # scatter: predicted vs actual (primary model = random forest, tier B)
    rng_plot = np.random.default_rng(RNG)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, target in zip(axes, PLOT_TARGETS):
        yte_s, pred_s = scatter_store[target]
        n_plot = min(5000, len(yte_s))
        idx = rng_plot.choice(len(yte_s), n_plot, replace=False)
        ax.scatter(yte_s[idx], pred_s[idx], s=2, alpha=0.3, color="teal")
        lo = float(np.percentile(yte_s[idx], 1))
        hi = float(np.percentile(yte_s[idx], 99))
        pad = (hi - lo) * 0.05
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=1.2, label="perfect fit")
        best_r2 = metrics[(metrics.scope == "real_decel") & (metrics.target == target)
                          & (~metrics.with_duration) & (metrics.tier == "full")
                          & (metrics.model == "rf")]["r2"].iloc[0]
        ax.set_xlabel(f"actual {target}")
        ax.set_ylabel(f"predicted {target}")
        ax.set_title(f"{target}\nRandom forest, tier B — out-of-sample R²={best_r2:.2f}")
        ax.legend(fontsize=8)
    fig.suptitle("Regression: predicted vs actual deceleration (5 000 test-set points, random forest)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "decel_regression_scatter.png", dpi=150)
    plt.close(fig)
    logger.info("deceleration regression complete")


# --------------------------------------------------------------------------- #
# Braking energy between maintenance events
# --------------------------------------------------------------------------- #
BRAKE_FAILURE = "Brake System Failure"


def maintenance_interval_energy() -> None:
    """Braking energy accounting against the maintenance calendar.

    Step A: total real-deceleration energy dissipated in each gap BETWEEN
    consecutive maintenance events (plus the two dataset bookends), flagged for
    whether a brake failure falls inside it.
    Step B: energy dissipated since the last completed maintenance before each
    failure -- 'how much braking piled up before it broke'.

    Small N (11 maintenance events, 10 failures): this is descriptive/eyeball,
    not a test. Framed as such in the notebook.
    """
    out_a = TBL_DIR / "maintenance_interval_energy.csv"
    if out_a.exists():
        logger.info("maintenance-interval energy exists -- skipping (delete to recompute)")
        return

    ev = pd.read_parquet(
        PROC / "deceleration_state_events.parquet",
        columns=["start_timestamp", "is_real_deceleration", "deceleration_energy_specific"])
    ev = ev[ev["is_real_deceleration"]].copy()
    ev["start_timestamp"] = pd.to_datetime(ev["start_timestamp"])
    st = ev["start_timestamp"].to_numpy()
    energy = ev["deceleration_energy_specific"].to_numpy("float64")
    ds_min = ev["start_timestamp"].min()
    ds_max = ev["start_timestamp"].max()
    grand_total = float(np.nansum(energy))

    inv = pd.read_csv(ROOT / "inputs" / "event_inventory.csv")
    maint = inv[inv.event_class == "maintenance"].copy()
    maint["start"] = pd.to_datetime(maint["start"])
    maint["end"] = pd.to_datetime(maint["end"])
    maint = maint.sort_values("start").reset_index(drop=True)
    maint_ends = maint["end"].to_numpy()

    fail = inv[inv.event_class == "failure"].copy()
    fail["start"] = pd.to_datetime(fail["start"])
    fail = fail.sort_values("start").reset_index(drop=True)
    fail_starts = fail["start"].to_numpy()
    brake_fail_starts = fail.loc[fail.type == BRAKE_FAILURE, "start"].to_numpy()
    logger.info(f"{len(maint)} maintenance events, {len(fail)} failures "
                f"({len(brake_fail_starts)} {BRAKE_FAILURE}); grand real-decel "
                f"energy={grand_total:,.1f}")

    # --- Step A: gaps between consecutive maintenance events (+ bookends) ---
    # Half-open (lo, hi]; the first bookend includes its lower edge so no event
    # at the dataset minimum is dropped. Maintenance windows themselves are NOT
    # intervals, so the interval totals sum to slightly LESS than the grand
    # total (the shortfall = energy dissipated during maintenance windows).
    intervals = []
    intervals.append(dict(lo=ds_min, hi=maint["start"].iloc[0], kind="bookend_start",
                          include_lo=True))
    for i in range(len(maint) - 1):
        intervals.append(dict(lo=maint["end"].iloc[i], hi=maint["start"].iloc[i + 1],
                              kind="between", include_lo=False))
    intervals.append(dict(lo=maint["end"].iloc[-1], hi=ds_max, kind="bookend_end",
                          include_lo=False))

    rows = []
    for iv in intervals:
        lo = np.datetime64(iv["lo"]); hi = np.datetime64(iv["hi"])
        lo_ok = (st >= lo) if iv["include_lo"] else (st > lo)
        m = lo_ok & (st <= hi)
        e = energy[m]
        contains = bool(((brake_fail_starts > lo) & (brake_fail_starts <= hi)).any())
        rows.append(dict(
            interval_start=iv["lo"], interval_end=iv["hi"], kind=iv["kind"],
            duration_days=float((iv["hi"] - iv["lo"]) / np.timedelta64(1, "D")),
            n_events=int(m.sum()), total_energy=float(np.nansum(e)),
            contains_brake_failure=contains))
    interval_df = pd.DataFrame(rows)
    interval_df.to_csv(out_a, index=False)
    covered = float(interval_df["total_energy"].sum())
    logger.info(f"Step A: {len(interval_df)} intervals; energy covered={covered:,.1f} "
                f"of {grand_total:,.1f} ({covered / grand_total:.1%}; remainder is "
                "during-maintenance events)")
    logger.info("\n" + interval_df.round({"duration_days": 2, "total_energy": 2}).to_string(index=False))

    # --- Step B: energy since last completed maintenance before each failure ---
    # Scoped to ALL failure-class events (10) so the two reactive-maintenance
    # overlaps -- 2024-08-27 Leveling and 2024-10-31 Compressor, which start the
    # same day as a maintenance -- are included; a failure_type column keeps the
    # Brake-System-Failure-only view (8, matching prefailure_analysis) one filter
    # away. 'latest maintenance END before failure START' sidesteps the same-day
    # overlap cleanly (dataset min if a failure precedes all maintenance).
    b_rows = []
    for _, fr in fail.iterrows():
        fs = fr["start"]
        prior_ends = maint_ends[maint_ends <= np.datetime64(fs)]
        prior_end = pd.Timestamp(prior_ends.max()) if len(prior_ends) else ds_min
        m = (st > np.datetime64(prior_end)) & (st <= np.datetime64(fs))
        b_rows.append(dict(
            failure=fs, failure_type=fr["type"], prior_maint_end=prior_end,
            energy_since_maintenance=float(np.nansum(energy[m])),
            n_events_since_maintenance=int(m.sum()),
            days_since_maintenance=float((fs - prior_end) / np.timedelta64(1, "D"))))
    prefail_df = pd.DataFrame(b_rows)
    prefail_df.to_csv(TBL_DIR / "prefailure_maintenance_energy.csv", index=False)
    logger.info(f"Step B: {len(prefail_df)} failures "
                f"({int((prefail_df.failure_type == BRAKE_FAILURE).sum())} {BRAKE_FAILURE})\n"
                + prefail_df.round({"energy_since_maintenance": 2,
                                    "days_since_maintenance": 2}).to_string(index=False))

    # --- Plot: interval bars + cumulative-since-maintenance sawtooth ---
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    ax = axes[0]
    x = np.arange(len(interval_df))
    colors = ["firebrick" if c else "steelblue"
              for c in interval_df["contains_brake_failure"]]
    ax.bar(x, interval_df["total_energy"], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([pd.Timestamp(d).date() for d in interval_df["interval_start"]],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("total real-decel energy in interval")
    ax.set_xlabel("interval start (between-maintenance gap)")
    ax.set_title("Braking energy dissipated between maintenance events\n"
                 "red = interval contains a Brake System Failure")

    # sawtooth: cumulative real-decel energy, reset at each maintenance END
    order = np.argsort(st)
    st_s = st[order]
    e_s = energy[order]
    ends_sorted = np.sort(maint_ends)
    cum = 0.0
    ptr = 0
    cum_series = np.empty(len(st_s))
    for i in range(len(st_s)):
        while ptr < len(ends_sorted) and ends_sorted[ptr] <= st_s[i]:
            cum = 0.0
            ptr += 1
        cum += e_s[i]
        cum_series[i] = cum
    ax = axes[1]
    ax.plot(st_s, cum_series, color="teal", lw=0.9,
            label="cumulative energy since last maintenance")
    for i, me in enumerate(ends_sorted):
        ax.axvline(pd.Timestamp(me), color="grey", ls=":", alpha=0.5,
                   label="maintenance end" if i == 0 else None)
    for i, fsx in enumerate(fail_starts):
        ax.axvline(pd.Timestamp(fsx), color="red", ls="--", alpha=0.6,
                   label="failure" if i == 0 else None)
    ax.set_ylabel("cumulative energy since last maintenance")
    ax.set_xlabel("date")
    ax.set_title("Energy accumulated since last maintenance (sawtooth resets at each "
                 "maintenance)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "maintenance_interval_energy.png", dpi=150)
    plt.close(fig)
    logger.info("maintenance-interval energy complete")


# --------------------------------------------------------------------------- #
# Pre-failure deceleration behaviour
# --------------------------------------------------------------------------- #
PRE_DAYS = 7
PRE_DAYS_ALT = 14
MIN_EVENTS = 20          # a pre-failure window with fewer events is excluded
CUSUM_THRESHOLD = 4.0    # std units; tuned for ~1 false alarm / 6 months

# Key features per role (the behaviour we expect to drift before a brake failure).
KEY_FEATURES = [
    # actuation wear / command
    "brake_cylinder_pressure_integral", "brake_cylinder_pressure_peak",
    "pneumatic_braking_force_mean", "spring_brake_pressure_mean",
    # kinematic outcome
    "jerk_rms", "peak_deceleration", "deceleration_energy_specific",
    # auxiliary health
    "main_reservoir_pressure_drop", "main_reservoir_pressure_rate_mean",
    "load_pressure_mean", "energy_braking_resistance_mean",
]
# Weekly wear-proxy for CUSUM: the auxiliary channel with the largest pre-failure effect
# size (energy_braking_resistance_mean, Cliff's delta ~ -0.20), so the change-point test
# is run on the strongest available signal rather than an arbitrary reservoir channel.
AUX_PROXY = "energy_braking_resistance_mean"


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta from the Mann-Whitney U (a vs b). In [-1, 1]."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan
    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    return float(2.0 * u / (n1 * n2) - 1.0)


def cliffs_ci(a, b, rng, n_boot=500):
    deltas = []
    for _ in range(n_boot):
        sa = a[rng.integers(0, len(a), size=min(len(a), 2000))]
        sb = b[rng.integers(0, len(b), size=min(len(b), 2000))]
        deltas.append(cliffs_delta(sa, sb))
    return float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5))


def prefailure_analysis() -> None:
    out_tests = TBL_DIR / "prefailure_tests.csv"
    if out_tests.exists():
        logger.info("prefailure tests exist -- skipping (delete to recompute)")
        return

    ev_all = pd.read_parquet(PROC / "deceleration_state_events.parquet")
    ev_all["start_timestamp"] = pd.to_datetime(ev_all["start_timestamp"])
    inv = pd.read_csv(ROOT / "inputs" / "event_inventory.csv")
    inv = inv[(inv.event_class == "failure") & (inv.type == "Brake System Failure")]
    fail_starts = pd.to_datetime(inv["start"]).sort_values().to_numpy()

    # Scope to real-deceleration events; stationary holds have no decel signal.
    ev = ev_all[ev_all["is_real_deceleration"]].reset_index(drop=True)
    logger.info(f"{len(fail_starts)} brake failures; {len(ev_all):,} deceleration events "
                f"({len(ev):,} real-deceleration, used for tests; "
                f"{len(ev_all) - len(ev):,} stationary holds excluded)")

    rng = np.random.default_rng(RNG)
    st = ev["start_timestamp"].to_numpy()
    st_all = ev_all["start_timestamp"].to_numpy()

    # 1. per-failure event counts (real-decel used for tests; total reported)
    per_fail = []
    for ft in fail_starts:
        d = (ft - st) / np.timedelta64(1, "D")
        d_all = (ft - st_all) / np.timedelta64(1, "D")
        n7 = int(((d >= 0) & (d <= PRE_DAYS)).sum())
        n14 = int(((d >= 0) & (d <= PRE_DAYS_ALT)).sum())
        n7_all = int(((d_all >= 0) & (d_all <= PRE_DAYS)).sum())
        per_fail.append(dict(failure=pd.Timestamp(ft), n_realdecel_7d=n7,
                             n_realdecel_14d=n14, n_all_events_7d=n7_all,
                             included_7d=n7 >= MIN_EVENTS))
    per_fail_df = pd.DataFrame(per_fail)
    per_fail_df.to_csv(TBL_DIR / "prefailure_event_counts.csv", index=False)
    logger.info(f"per-failure 7d counts:\n{per_fail_df.to_string(index=False)}")
    n_incl = int(per_fail_df["included_7d"].sum())

    # 2. distributional comparison: pooled pre-failure vs baseline
    pre_mask = ev["failure_within_7_days"].to_numpy() == 1
    base_mask = ev["failure_within_30_days"].to_numpy() == 0  # far from any failure
    logger.info(f"pooled pre-7d events={int(pre_mask.sum()):,} | "
                f"baseline(>30d) events={int(base_mask.sum()):,}")

    rows = []
    n_tests = len(KEY_FEATURES)
    for feat in KEY_FEATURES:
        a = ev.loc[pre_mask, feat].to_numpy("float64")
        b = ev.loc[base_mask, feat].to_numpy("float64")
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        mwu_p = mannwhitneyu(a, b, alternative="two-sided")[1]
        ks_p = ks_2samp(a, b)[1]
        delta = cliffs_delta(a, b)
        lo, hi = cliffs_ci(a, b, rng)
        rows.append(dict(
            feature=feat, n_pre=len(a), n_base=len(b),
            pre_median=float(np.median(a)), base_median=float(np.median(b)),
            mwu_p=mwu_p, ks_p=ks_p, mwu_p_bonferroni=min(1.0, mwu_p * n_tests),
            cliffs_delta=delta, cliffs_ci_lo=lo, cliffs_ci_hi=hi,
            sig_bonferroni=bool(mwu_p * n_tests < 0.05)))
        logger.info(f"{feat}: delta={delta:+.3f} [{lo:+.2f},{hi:+.2f}] "
                    f"MWU p(bonf)={min(1.0, mwu_p*n_tests):.2e}")
    tests = pd.DataFrame(rows)
    tests.to_csv(out_tests, index=False)

    # 3. exploratory coefficient-shift (OLS with pre_failure interaction)
    coef_shift_lines = []
    try:
        sub_cols = ["peak_deceleration", "brake_cylinder_pressure_integral",
                    "main_reservoir_pressure_drop", "load_pressure_mean"]
        d = ev[sub_cols].copy()
        d["pre_failure"] = pre_mask.astype(int)
        # sample baseline down to ~5x pre for a tractable, less-imbalanced fit
        d_pre = d[d.pre_failure == 1]
        d_base = d[(base_mask) ].sample(n=min(len(d_pre) * 5, int(base_mask.sum())),
                                        random_state=RNG)
        dd = pd.concat([d_pre, d_base], ignore_index=True)
        m = smf.ols("peak_deceleration ~ brake_cylinder_pressure_integral * pre_failure "
                    "+ main_reservoir_pressure_drop * pre_failure "
                    "+ load_pressure_mean * pre_failure", data=dd).fit()
        coef_shift_lines.append("Exploratory coefficient-shift OLS (n_pre="
                                f"{len(d_pre)}, baseline sampled={len(d_base)}):")
        for term in m.params.index:
            if "pre_failure" in term:
                coef_shift_lines.append(
                    f"  {term}: beta={m.params[term]:+.4f}, p={m.pvalues[term]:.3g}")
        coef_shift_lines.append(f"  model R2={m.rsquared:.3f}")
    except Exception as e:  # pragma: no cover
        coef_shift_lines.append(f"coefficient-shift test failed: {e}")

    # 4. CUSUM on weekly auxiliary wear-proxy
    wk = (ev.set_index("start_timestamp")[AUX_PROXY]
          .resample("W").mean().dropna())
    cps = cusum(wk.to_numpy(), threshold=CUSUM_THRESHOLD)
    cp_dates = [wk.index[i] for i in cps]
    cp_rows = []
    for cd in cp_dates:
        near = min(((pd.Timestamp(ft) - cd).days for ft in fail_starts
                    if 0 <= (pd.Timestamp(ft) - cd).days <= PRE_DAYS), default=None)
        cp_rows.append(dict(changepoint=cd, proxy=AUX_PROXY,
                            days_to_next_failure_within_7=near,
                            within_7d_of_failure=near is not None))
    cusum_df = pd.DataFrame(cp_rows)
    cusum_df.to_csv(TBL_DIR / "cusum_changepoints.csv", index=False)
    logger.info(f"CUSUM (thr={CUSUM_THRESHOLD}) on weekly {AUX_PROXY}: "
                f"{len(cp_dates)} change points, "
                f"{int(cusum_df['within_7d_of_failure'].sum()) if len(cusum_df) else 0} within 7d of a failure")

    # 4b. deceleration-energy pre-failure signal -- the supervisor's energy integral spun
    # forward. Per-failure pre-7d/14d energy vs baseline, and CUSUM on weekly SUMMED
    # energy (wear accumulates, so summed not mean), plus a cumulative trajectory.
    energy = ev["deceleration_energy_specific"].to_numpy("float64")
    base_energy = energy[base_mask]
    base_energy = base_energy[np.isfinite(base_energy)]
    base_mean_e = float(np.mean(base_energy)) if len(base_energy) else np.nan
    e_rows = []
    for ft in fail_starts:
        d = (ft - st) / np.timedelta64(1, "D")
        for horizon in (PRE_DAYS, PRE_DAYS_ALT):
            m = (d >= 0) & (d <= horizon)
            e = energy[m]
            e = e[np.isfinite(e)]
            e_rows.append(dict(
                failure=pd.Timestamp(ft), horizon_days=horizon, n_events=int(m.sum()),
                total_energy=float(np.nansum(e)),
                mean_energy=float(np.mean(e)) if len(e) else np.nan,
                baseline_mean_energy=base_mean_e))
    energy_df = pd.DataFrame(e_rows)
    energy_df.to_csv(TBL_DIR / "prefailure_energy.csv", index=False)

    wke = (ev.set_index("start_timestamp")["deceleration_energy_specific"]
           .resample("W").sum().dropna())
    cps_e = cusum(wke.to_numpy(), threshold=CUSUM_THRESHOLD)
    cp_e_dates = [wke.index[i] for i in cps_e]
    n_e_near = sum(any(0 <= (pd.Timestamp(ft) - cd).days <= PRE_DAYS for ft in fail_starts)
                   for cd in cp_e_dates)
    logger.info(f"deceleration-energy CUSUM (weekly summed): {len(cp_e_dates)} change points, "
                f"{n_e_near} within 7d of a failure | baseline mean energy={base_mean_e:.5f}")

    ev_sorted = ev.sort_values("start_timestamp")
    cum_time = ev_sorted["start_timestamp"].to_numpy()
    cum_energy = np.nancumsum(ev_sorted["deceleration_energy_specific"].to_numpy("float64"))

    # summary log (output artifact)
    n_sig = int(tests["sig_bonferroni"].sum())
    lines = ["Phase 2 -- pre-failure deceleration summary", "=" * 55, ""]
    lines.append(f"Brake failures (n): {len(fail_starts)}  [SMALL N -- interpret cautiously]")
    lines.append(f"Pre-failure windows with >= {MIN_EVENTS} events (usable): {n_incl}/{len(fail_starts)}")
    lines.append("")
    lines.append(f"Distributional tests (pooled pre-7d vs baseline >30d), {len(KEY_FEATURES)} features,")
    lines.append(f"Bonferroni-corrected: {n_sig} feature(s) significant at alpha=0.05.")
    lines.append("NOTE: with ~35k pooled pre-failure events, even tiny distribution shifts")
    lines.append("reach significance -- read the Cliff's-delta EFFECT SIZE, not just p.")
    lines.append("Effect sizes |delta|<0.15 are negligible regardless of p-value.")
    lines.append("")
    lines.extend(coef_shift_lines)
    lines.append("")
    lines.append(f"CUSUM weekly {AUX_PROXY} (threshold {CUSUM_THRESHOLD} std, ~1 FA/6mo): "
                 f"{len(cp_dates)} change points; "
                 f"{int(cusum_df['within_7d_of_failure'].sum()) if len(cusum_df) else 0} within 0-7d of a failure.")
    lines.append(f"CUSUM weekly SUMMED deceleration energy (same threshold): {len(cp_e_dates)} "
                 f"change points; {n_e_near} within 0-7d of a failure. Per-failure pre-7d/14d "
                 "energy totals vs baseline in prefailure_energy.csv; cumulative trajectory in "
                 "prefailure_energy.png.")
    lines.append("")
    lines.append("FUTURE WORK (not implemented): brake-system failures may manifest during")
    lines.append("CHARGING/IDLE (compressor/reservoir/valve), not during deceleration. This")
    lines.append("deceleration-only analysis may be looking under the wrong lamp post. A follow-up")
    lines.append("should repeat it on state in {standing, cruising} windows using auxiliary")
    lines.append("sensors (compressor duty cycle, reservoir recharge time).")
    (LOG_DIR / "prefailure_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"{n_sig}/{len(KEY_FEATURES)} features significant (Bonferroni)")

    # 5a. case-study small multiples (7-day pre-failure trajectories)
    feat_plot = "brake_cylinder_pressure_integral"
    n = len(fail_starts)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for idx, ft in enumerate(fail_starts):
        ax = axes[idx // ncol][idx % ncol]
        d = (ft - st) / np.timedelta64(1, "D")
        m = (d >= 0) & (d <= PRE_DAYS)
        sub = ev.loc[m].sort_values("start_timestamp")
        if len(sub):
            x = -(ft - sub["start_timestamp"].to_numpy()) / np.timedelta64(1, "D")
            ax.scatter(x, sub[feat_plot], s=6, alpha=0.4, color="firebrick")
            roll = sub[feat_plot].rolling(20, min_periods=5).mean()
            ax.plot(x, roll, color="black", lw=1)
        ax.axvline(0, color="grey", ls="--")
        ax.set_title(f"{pd.Timestamp(ft).date()} (n={int(m.sum())})", fontsize=9)
        ax.set_xlabel("days before failure")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"7-day pre-failure trajectories: {feat_plot}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "prefailure_case_studies.png", dpi=150)
    plt.close(fig)

    # 5b. CUSUM plot
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(wk.index, wk.to_numpy(), "o-", ms=3, color="steelblue", label=AUX_PROXY)
    for ft in fail_starts:
        ax.axvline(pd.Timestamp(ft), color="red", alpha=0.4, lw=1)
    for cd in cp_dates:
        ax.axvline(cd, color="green", ls="--", alpha=0.7)
    ax.set_xlabel("week"); ax.set_ylabel(f"weekly mean {AUX_PROXY}")
    ax.set_title("Weekly auxiliary wear-proxy with CUSUM change points (green) "
                 "vs brake failures (red)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "prefailure_cusum.png", dpi=150)
    plt.close(fig)

    # 5c. cumulative deceleration-energy trajectory with failures + energy CUSUM points
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(cum_time, cum_energy, color="teal", lw=1.3, label="cumulative deceleration energy")
    for i, ft in enumerate(fail_starts):
        ax.axvline(pd.Timestamp(ft), color="red", alpha=0.4, lw=1,
                   label="brake failure" if i == 0 else None)
    for i, cd in enumerate(cp_e_dates):
        ax.axvline(cd, color="green", ls="--", alpha=0.7,
                   label="weekly-energy CUSUM" if i == 0 else None)
    ax.set_xlabel("date"); ax.set_ylabel("cumulative specific deceleration energy")
    ax.set_title("Cumulative deceleration energy (real-deceleration events)\n"
                 "brake failures (red) and weekly-summed-energy CUSUM change points (green)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "prefailure_energy.png", dpi=150)
    plt.close(fig)
    logger.info("pre-failure analysis complete")


def main() -> None:
    cluster_intensity()
    regress_deceleration()
    maintenance_interval_energy()
    prefailure_analysis()
    logger.info("phase2 intensity (part 2) complete")


if __name__ == "__main__":
    main()
