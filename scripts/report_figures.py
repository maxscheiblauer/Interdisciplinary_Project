"""Render the report figures at publication resolution.

This is a presentation layer only: it reads the analysis artefacts already on
disk (results/tables, models, and the processed event/window parquet) and
re-draws the sixteen figures the written report includes, at 300 DPI with a
single consistent style and clean, professional labels. It does not re-run any
analysis, so every number matches the tables produced by the pipeline scripts.

Figures are written to results/plots/report and copied into latex/Bilder.

    python scripts/report_figures.py
"""
from __future__ import annotations

import shutil
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
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

TBL = ROOT / "results" / "tables"
PROC = ROOT / "data" / "processed"
INPUTS = ROOT / "inputs"
M1 = ROOT / "models" / "phase1"
M2 = ROOT / "models" / "phase2"
OUT = ROOT / "results" / "plots" / "report"
BILDER = ROOT / "latex" / "Bilder"
OUT.mkdir(parents=True, exist_ok=True)

RNG = 42

# Canonical operating-cycle order and a fixed colour per state, used everywhere.
STATE_ORDER = ["standing", "accelerating", "cruising", "deceleration"]
STATE_COLOR = {
    "standing": "#6c757d",
    "accelerating": "#2a9d8f",
    "cruising": "#264b96",
    "deceleration": "#c1121f",
}
ACCENT = "#264b96"
ACCENT2 = "#c1121f"


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    })


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {name}")


# --------------------------------------------------------------------------- #
# Chapter 2 — preprocessing
# --------------------------------------------------------------------------- #
def fig_pca() -> None:
    pca = joblib.load(INPUTS / "pca.joblib")
    evr = np.asarray(pca.explained_variance_ratio_)
    cum = np.cumsum(evr)
    x = np.arange(1, len(evr) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x, evr * 100, color=ACCENT, alpha=0.35, label="Individual component")
    ax.plot(x, cum * 100, "o-", color=ACCENT2, label="Cumulative")
    for thr in (90, 95):
        k = int(np.argmax(cum >= thr / 100) + 1)
        ax.axhline(thr, color="grey", ls=":", lw=0.8)
        ax.annotate(f"{thr}% at {k} comp.", xy=(k, thr), xytext=(k + 1.5, thr - 8),
                    fontsize=9, color="grey",
                    arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Explained variance of the pneumatic sensor set")
    ax.set_ylim(0, 105)
    ax.legend(loc="center right")
    save(fig, "pca_explained_variance.png")


# --------------------------------------------------------------------------- #
# Chapter 3 — operational states
# --------------------------------------------------------------------------- #
def fig_kselection() -> None:
    sel = pd.read_csv(TBL / "phase1_kselection.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    axes[0].plot(sel.k, sel.silhouette, "o-", color=STATE_COLOR["accelerating"])
    axes[0].set_title("(a) Silhouette")
    axes[0].set_ylabel("Silhouette coefficient")

    axes[1].plot(sel.k, sel.davies_bouldin, "o-", color="#7b2cbf")
    axes[1].set_title("(b) Davies–Bouldin index")
    axes[1].set_ylabel("Davies–Bouldin index")

    axes[2].plot(sel.k, sel.gmm_bic / 1000, "o-", color=ACCENT, label="BIC")
    axes[2].plot(sel.k, sel.gmm_aic / 1000, "s--", color="grey", label="AIC")
    axes[2].axvline(4, color=ACCENT2, ls=":", lw=1.2)
    axes[2].annotate("largest BIC gain\nat $k=3\\rightarrow4$", xy=(4, sel.gmm_bic.iloc[2] / 1000),
                     xytext=(4.2, sel.gmm_bic.min() / 1000 + 40), fontsize=9, color=ACCENT2)
    axes[2].set_title("(c) Gaussian-mixture information criterion")
    axes[2].set_ylabel("Criterion ($\\times 10^{3}$)")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.set_xlabel("Number of clusters $k$")
        ax.set_xticks(sel.k)
    fig.suptitle("Cluster-count selection on the motion features", fontweight="bold")
    fig.tight_layout()
    save(fig, "kinematic_kselection.png")


def fig_state_profiles() -> None:
    prof = pd.read_csv(TBL / "cluster_profiles.csv")
    prof = prof.set_index("label").reindex(STATE_ORDER)
    colors = [STATE_COLOR[s] for s in STATE_ORDER]
    metrics = [("velocity_mean", "(a) Median velocity"),
               ("accel_mean", "(b) Median acceleration"),
               ("jerk_mean", "(c) Median jerk")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(STATE_ORDER, prof[col].to_numpy(), color=colors)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(title)
        ax.set_ylabel("Normalised units")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Motion profiles of the four operational states", fontweight="bold")
    fig.tight_layout()
    save(fig, "state_profiles.png")


def fig_vel_accel_scatter() -> None:
    cols = ["state", "TRAIN_SPEED_ACTUAL__mean", "acceleration__mean"]
    df = pd.read_parquet(PROC / "train_windows_labeled.parquet", columns=cols)
    rng = np.random.default_rng(RNG)
    idx = rng.choice(len(df), size=min(25000, len(df)), replace=False)
    df = df.iloc[idx]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for s in STATE_ORDER:
        m = df["state"] == s
        ax.scatter(df.loc[m, "TRAIN_SPEED_ACTUAL__mean"],
                   df.loc[m, "acceleration__mean"],
                   s=6, alpha=0.35, color=STATE_COLOR[s], label=s, edgecolors="none")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Window-mean velocity (normalised units)")
    ax.set_ylabel("Window-mean acceleration (normalised units)")
    ax.set_title("The velocity–acceleration operating cycle")
    leg = ax.legend(title="State", markerscale=2.5, loc="upper right")
    leg.get_title().set_fontweight("bold")
    save(fig, "kinematic_vel_accel_scatter.png")


def fig_transition_diagram() -> None:
    tm = pd.read_csv(TBL / "transition_matrix.csv", index_col=0)
    tm = tm.reindex(index=STATE_ORDER, columns=STATE_ORDER)
    pos = {"standing": (0, 1), "accelerating": (1, 1),
           "cruising": (1, 0), "deceleration": (0, 0)}
    cx, cy = 0.5, 0.5
    fig, ax = plt.subplots(figsize=(8, 7))
    for a in STATE_ORDER:
        for b in STATE_ORDER:
            if a == b:
                continue
            p = float(tm.loc[a, b])
            if p < 0.05:
                continue
            (x0, y0), (x1, y1) = pos[a], pos[b]
            rad = 0.2 if (STATE_ORDER.index(b) - STATE_ORDER.index(a)) % 4 <= 2 else -0.2
            arrow = FancyArrowPatch(
                (x0, y0), (x1, y1), connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>", mutation_scale=16, shrinkA=24, shrinkB=24,
                lw=0.8 + 4.5 * p, color="#495057", alpha=0.5 + 0.5 * p, zorder=2)
            ax.add_patch(arrow)
            mx, my = (x0 + x1) / 2 + rad * (y1 - y0), (y0 + y1) / 2 - rad * (x1 - x0)
            ax.text(mx, my, f"{p:.2f}", fontsize=10, ha="center", va="center",
                    color="#212529", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    for s, (x, y) in pos.items():
        ax.scatter([x], [y], s=2600, color=STATE_COLOR[s], zorder=3, alpha=0.95)
        lx = x + 0.30 * np.sign(x - cx)
        ly = y + 0.24 * np.sign(y - cy)
        ax.annotate(s, xy=(x, y), xytext=(lx, ly), ha="center", va="center",
                    color=STATE_COLOR[s], fontweight="bold", fontsize=12, zorder=4)
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylim(-0.7, 1.7)
    ax.axis("off")
    ax.set_title("Operational-state transition probabilities", fontweight="bold")
    save(fig, "transition_diagram.png")


def fig_by_month() -> None:
    df = pd.read_parquet(PROC / "train_windows_labeled.parquet",
                         columns=["window_start", "state"])
    df["month"] = pd.to_datetime(df["window_start"]).dt.to_period("M").astype(str)
    frac = (pd.crosstab(df["month"], df["state"], normalize="index")
            .reindex(columns=STATE_ORDER))
    months = list(frac.index)
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(frac))
    for s in STATE_ORDER:
        ax.bar(months, frac[s].to_numpy() * 100, bottom=bottom,
               color=STATE_COLOR[s], label=s, width=0.8)
        bottom += frac[s].to_numpy() * 100
    ax.set_ylabel("Share of windows (%)")
    ax.set_xlabel("Month")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="x", visible=False)
    leg = ax.legend(title="State", ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    leg.get_title().set_fontweight("bold")
    ax.set_title("Operational-state mix by month", fontweight="bold", pad=28)
    save(fig, "state_distribution_by_month.png")


def fig_window_sensitivity() -> None:
    ws = pd.read_csv(TBL / "window_sensitivity.csv")
    piv = ws.pivot(index="window_s", columns="metric", values="value")
    sizes = piv.index.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    for s in STATE_ORDER:
        col = f"frac_{s}"
        if col in piv:
            axes[0].plot(sizes, piv[col] * 100, "o-", color=STATE_COLOR[s], label=s)
    axes[0].set_xlabel("Window length (s)")
    axes[0].set_ylabel("Share of windows (%)")
    axes[0].set_title("State fractions vs window length")
    axes[0].set_xticks(sizes)
    leg = axes[0].legend(title="State")
    leg.get_title().set_fontweight("bold")

    axes[1].plot(sizes, piv["chatter_rate"], "o-", color=ACCENT2)
    axes[1].set_xlabel("Window length (s)")
    axes[1].set_ylabel("Chatter rate")
    axes[1].set_title("State chatter vs window length")
    axes[1].set_xticks(sizes)

    fig.suptitle("Sensitivity of the states to window length", fontweight="bold")
    fig.tight_layout()
    save(fig, "window_sensitivity.png")


# --------------------------------------------------------------------------- #
# Chapter 4 — classification
# --------------------------------------------------------------------------- #
MODEL_LABEL = {"dt": "Decision tree", "lda": "LDA", "qda": "QDA",
               "rf": "Random forest", "lr": "Logistic reg."}
TIER_LABEL = {"auxonly": "Auxiliary only", "full": "Full (aux + actuation)"}
TIER_COLOR = {"auxonly": "#2a9d8f", "full": "#264b96"}


def fig_clf_comparison() -> None:
    m = pd.read_csv(TBL / "state_classification_metrics.csv")
    order = ["rf", "lda", "dt", "qda", "lr"]
    labels = [MODEL_LABEL[k] for k in order]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for ax, metric, mtitle in zip(
            axes, ["roc_auc", "f1_deceleration"], ["ROC-AUC", "$F_1$ (deceleration)"]):
        for off, tier in zip((-0.21, 0.21), ["auxonly", "full"]):
            vals = [m[(m.tier == tier) & (m.model == k)][metric].iloc[0] for k in order]
            ax.bar(x + off, vals, width=0.4, color=TIER_COLOR[tier], label=TIER_LABEL[tier])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.tick_params(axis="x", labelsize=9)
        ax.set_ylabel(mtitle)
        ax.set_ylim(0, 1)
        ax.margins(x=0.04)
        ax.set_title(mtitle + " by model")
    axes[0].axhline(0.5, color="grey", ls=":", lw=0.9)
    # legend outside, to the right of the second panel, clear of the bars
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Deceleration-state classifier comparison (held-out window)", fontweight="bold")
    fig.tight_layout()
    save(fig, "state_clf_comparison.png")


def _tier_windows(roles):
    ledger = pd.read_csv(TBL / "feature_roles.csv")
    df = pd.read_parquet(PROC / "train_windows_labeled.parquet")
    present = set(df.columns)
    feats = [f for f in ledger[(ledger.level == "window")
             & (ledger.role.isin(roles))]["feature"] if f in present]
    y = (df["state"] == "deceleration").astype(int).to_numpy()
    X = df[feats].to_numpy("float32")
    return X, y, feats


def fig_clf_confusion() -> None:
    """The four most telling confusion matrices in a 2x2 grid: the full-tier random
    forest (best overall), the two interpretable linear/tree models on the full tier
    (LDA, decision tree), and the auxiliary-only random forest (the leakage-free test
    with the brake command withheld). Full per-model metrics live in the appendix."""
    # reproduce the two held-out splits used in training (same seed / stratify)
    Xf, yf, _ = _tier_windows(["auxiliary", "actuation"])
    _, Xfte, _, yfte = train_test_split(Xf, yf, test_size=0.25, random_state=RNG, stratify=yf)
    del Xf
    Xa, ya, _ = _tier_windows(["auxiliary"])
    _, Xate, _, yate = train_test_split(Xa, ya, test_size=0.25, random_state=RNG, stratify=ya)
    del Xa

    panels = [
        ("Full tier: random forest", "state_clf_rf_full.joblib", Xfte, yfte),
        ("Full tier: LDA", "state_clf_lda_full.joblib", Xfte, yfte),
        ("Full tier: decision tree", "state_clf_dt_full.joblib", Xfte, yfte),
        ("Auxiliary only: random forest", "state_clf_rf_auxonly.joblib", Xate, yate),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.6), constrained_layout=True)
    im = None
    for i, (ax, (title, model_file, Xte, yte)) in enumerate(zip(axes.ravel(), panels)):
        row, col = i // 2, i % 2
        pred = joblib.load(M2 / model_file).predict(Xte)
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(yte, pred):
            cm[t, p] += 1
        cmn = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm[r, c]:,}\n({cmn[r, c]:.1%})", ha="center", va="center",
                        fontsize=11, color="white" if cmn[r, c] > 0.5 else "#212529")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Not decel.", "Deceleration"])
        ax.set_yticklabels(["Not decel.", "Deceleration"], rotation=90, va="center")
        # only the outer edges carry axis labels, so the inner ones do not collide
        ax.set_xlabel("Predicted state" if row == 1 else "")
        ax.set_ylabel("True state" if col == 0 else "")
        ax.grid(False)
        ax.set_title(title, fontweight="bold", fontsize=11)
    cb = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cb.set_label("Row-normalised share")
    fig.suptitle("Deceleration classifiers on the held-out window (row-normalised)",
                 fontweight="bold")
    save(fig, "state_clf_confusion.png")


# --------------------------------------------------------------------------- #
# Chapter 5 — braking events
# --------------------------------------------------------------------------- #
def _events():
    return pd.read_parquet(PROC / "deceleration_state_events.parquet")


def fig_cluster_model_selection() -> None:
    sel = pd.read_csv(TBL / "cluster_selection.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].plot(sel.k, sel.gmm_bic / 1000, "o-", color=ACCENT, label="BIC")
    axes[0].plot(sel.k, sel.gmm_aic / 1000, "s--", color="grey", label="AIC")
    axes[0].set_ylabel("Criterion ($\\times 10^{3}$)")
    axes[0].set_title("Information criterion")
    axes[0].legend()

    axes[1].plot(sel.k, sel.kmeans_silhouette, "o-", color=STATE_COLOR["accelerating"])
    axes[1].set_ylabel("Silhouette coefficient")
    axes[1].set_title("Silhouette")

    axes[2].plot(sel.k, sel.bootstrap_ari, "o-", color="#7b2cbf")
    axes[2].axhline(0.7, color="grey", ls=":", lw=0.9)
    axes[2].set_ylabel("Bootstrap ARI")
    axes[2].set_title("Resampling stability")

    for ax in axes:
        ax.set_xlabel("Number of clusters $k$")
        ax.set_xticks(sel.k)
    fig.suptitle("Model selection for braking intensity: no natural class count",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "cluster_model_selection.png")


def fig_intensity_distribution() -> None:
    ev = _events()
    real = ev[ev["is_real_deceleration"]]
    peak = real["peak_deceleration"].to_numpy("float64")
    p01, p99 = np.percentile(peak, [1, 99])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(peak, bins=200, range=(p01, p99), color=ACCENT, alpha=0.85)
    ax.axvline(float(np.median(peak)), color=ACCENT2, ls="--",
               label=f"Median = {np.median(peak):.3f}")
    ax.set_xlabel("Peak deceleration (normalised units)")
    ax.set_ylabel("Number of events")
    ax.set_title("Distribution of deceleration intensity (real events, 1–99th pct)",
                 fontweight="bold")
    ax.legend()
    save(fig, "intensity_distribution.png")


def fig_regression_scatter() -> None:
    ledger = pd.read_csv(TBL / "feature_roles.csv")
    ev = _events()
    real = ev[ev["is_real_deceleration"]].reset_index(drop=True)
    present = set(real.columns)
    feats = [f for f in ledger[(ledger.level == "event")
             & (ledger.role.isin(["auxiliary", "actuation"]))]["feature"] if f in present]
    metrics = pd.read_csv(TBL / "decel_regression_metrics.csv")
    targets = [("peak_deceleration", "Peak deceleration", "peak"),
               ("mean_deceleration", "Mean deceleration", "mean")]
    rng = np.random.default_rng(RNG)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, (target, title, stem) in zip(axes, targets):
        y = real[target].to_numpy("float64")
        X = real[feats].to_numpy("float64")
        _, Xte, _, yte = train_test_split(X, y, test_size=0.25, random_state=RNG)
        model = joblib.load(M2 / f"decel_regress_rf_{stem}.joblib")
        pred = model.predict(Xte)
        r2 = metrics[(metrics.scope == "real_decel") & (metrics.target == target)
                     & (~metrics.with_duration) & (metrics.tier == "full")
                     & (metrics.model == "rf")]["r2"].iloc[0]
        n = min(5000, len(yte))
        idx = rng.choice(len(yte), n, replace=False)
        ax.scatter(yte[idx], pred[idx], s=4, alpha=0.25, color=ACCENT, edgecolors="none")
        lo, hi = np.percentile(yte[idx], [1, 99])
        pad = (hi - lo) * 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", color=ACCENT2,
                label="Perfect prediction")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"Actual {title.lower()} (normalised)")
        ax.set_ylabel(f"Predicted {title.lower()} (normalised)")
        ax.set_title(f"{title}  ($R^2 = {r2:.2f}$, out-of-sample)")
        ax.legend(loc="upper left")
    fig.suptitle("Braking intensity predicted from the air-system sensors "
                 "(random forest)", fontweight="bold")
    fig.tight_layout()
    save(fig, "decel_regression_scatter.png")


# --------------------------------------------------------------------------- #
# Chapter 6 — failure / maintenance
# --------------------------------------------------------------------------- #
def _brake_failures() -> np.ndarray:
    inv = pd.read_csv(ROOT / "inputs" / "event_inventory.csv")
    inv = inv[(inv.event_class == "failure") & (inv.type == "Brake System Failure")]
    return pd.to_datetime(inv["start"]).to_numpy()


def _maintenance_dates() -> np.ndarray:
    inv = pd.read_csv(ROOT / "inputs" / "event_inventory.csv")
    inv = inv[inv.event_class == "maintenance"]
    return pd.to_datetime(inv["start"]).sort_values().to_numpy()


CUSUM_PROXY = "energy_braking_resistance_mean"
CUSUM_THRESHOLD = 4.0


def fig_cusum() -> None:
    # The raw weekly proxy climbs almost monotonically over the year, which would
    # only make sense if brake wear never reversed. We instead assume the brakes
    # are serviced at each maintenance stop, so the proxy is reset to its post-
    # service level at every maintenance date and only the within-interval drift
    # is read. This is an assumption -- we cannot confirm the brakes are actually
    # touched at every stop -- and it is applied here only, for this figure.
    from metroat.validation import cusum

    ev = _events()
    ev["start_timestamp"] = pd.to_datetime(ev["start_timestamp"])
    wk = ev.set_index("start_timestamp")[CUSUM_PROXY].resample("W").mean()
    maint = _maintenance_dates()
    fails = _brake_failures()

    vals = wk.to_numpy(dtype=float).copy()
    seg = np.searchsorted(maint, wk.index.to_numpy(), side="right")
    reset = vals.copy()
    for s in np.unique(seg):
        m = seg == s
        finite = np.isfinite(vals[m])
        if finite.any():
            reset[m] = vals[m] - vals[m][finite][0]
    reset = pd.Series(reset, index=wk.index)

    r = reset.dropna()
    cp_idx = cusum(r.to_numpy(), threshold=CUSUM_THRESHOLD)
    cps = [r.index[i] for i in cp_idx]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    # draw each between-maintenance segment as its own line so the trace never
    # jumps across a reset (the source of the earlier stray line skip)
    first_seg = True
    for s in np.unique(seg):
        m = seg == s
        sub = reset[m].dropna()
        if sub.empty:
            continue
        ax.plot(sub.index, sub.to_numpy(), "o-", ms=3, color=ACCENT,
                label="Weekly proxy (reset at maintenance)" if first_seg else None)
        first_seg = False
    for i, mt in enumerate(maint):
        ax.axvline(pd.Timestamp(mt), color="#6c757d", ls=":", alpha=0.7, lw=1.0,
                   label="Maintenance (assumed reset)" if i == 0 else None)
    for i, ft in enumerate(fails):
        ax.axvline(pd.Timestamp(ft), color=ACCENT2, alpha=0.5, lw=1.1,
                   label="Brake failure" if i == 0 else None)
    for i, cd in enumerate(cps):
        ax.axvline(cd, color="#2a9d8f", ls="--", alpha=0.9,
                   label="CUSUM change-point" if i == 0 else None)
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.4)
    ax.set_xlabel("Week")
    ax.set_ylabel("Proxy relative to post-service level")
    ax.set_title("Energy-braking-resistance since last service: change-points vs brake failures",
                 fontweight="bold")
    # legend outside the axes so it never overlaps the trace or the vertical markers
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.autofmt_xdate()
    save(fig, "prefailure_cusum.png")


def fig_maintenance_energy() -> None:
    mi = pd.read_csv(TBL / "maintenance_interval_energy.csv")
    mi = mi[mi["kind"] == "between"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [ACCENT2 if c else ACCENT for c in mi["contains_brake_failure"]]
    # every failure-bearing interval ran longer than the longest failure-free one;
    # the gap between the two groups sits at roughly three weeks.
    with_fail = mi.loc[mi["contains_brake_failure"], "duration_days"]
    without = mi.loc[~mi["contains_brake_failure"], "duration_days"]
    split = 0.5 * (float(without.max()) + float(with_fail.min()))
    ax.axvspan(0, split, color=ACCENT, alpha=0.06, zorder=0)
    ax.axvline(split, color="grey", ls=":", lw=1.1, zorder=1)
    ax.annotate(f"~{split:.0f}-day gap", xy=(split, ax.get_ylim()[1]),
                xytext=(split + 1.0, mi["total_energy"].max() * 0.55),
                fontsize=9, color="grey")
    sc = ax.scatter(mi["duration_days"], mi["total_energy"], s=90, c=colors,
                    edgecolors="white", linewidth=0.6, zorder=3)
    ax.set_xlabel("Interval length (days)")
    ax.set_ylabel("Total deceleration energy in interval")
    ax.set_title("Braking energy per between-maintenance interval", fontweight="bold")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", markerfacecolor=ACCENT2,
                      markeredgecolor="white", markersize=10, label="Contains brake failure"),
               Line2D([0], [0], marker="o", ls="", markerfacecolor=ACCENT,
                      markeredgecolor="white", markersize=10, label="No brake failure")]
    ax.legend(handles=handles, loc="upper left")
    save(fig, "maintenance_interval_energy.png")


FIGURES = [
    fig_pca,
    fig_kselection, fig_state_profiles, fig_vel_accel_scatter,
    fig_transition_diagram, fig_by_month, fig_window_sensitivity,
    fig_clf_comparison, fig_clf_confusion,
    fig_cluster_model_selection, fig_intensity_distribution,
    fig_regression_scatter,
    fig_cusum, fig_maintenance_energy,
]


def main() -> None:
    set_style()
    for fn in FIGURES:
        print(f"[{fn.__name__}]")
        fn()
    if BILDER.exists():
        for png in OUT.glob("*.png"):
            shutil.copy2(png, BILDER / png.name)
        print(f"copied {len(list(OUT.glob('*.png')))} figures to {BILDER}")


if __name__ == "__main__":
    main()
