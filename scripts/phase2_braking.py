"""Phase 2 (part 1) — braking events and the pneumatics-only state classifier.

Three steps: build the feature-role ledger and prove the predictor pools are
disjoint from the kinematic targets (leakage audit); extract braking events as
contiguous occupancies of the Phase-1 `braking` state with role-tagged features;
then test whether the pneumatic sensors alone can tell a braking window from a
non-braking one (auxiliary-only tier A vs aux+actuation tier B, on a held-out
split). Interpretation lives in notebooks/phase2_braking_analysis.ipynb and
results/reports/phase2_findings.md.
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
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.discriminant_analysis import (  # noqa: E402
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, plot_tree  # noqa: E402

from metroat import braking as bv  # noqa: E402
from metroat import schema as S  # noqa: E402

PROC = ROOT / "data" / "processed"
FEAT_ROOT = PROC / "train_features"
PLOT_DIR = ROOT / "results" / "plots" / "phase2"
TBL_DIR = ROOT / "results" / "tables"
LOG_DIR = ROOT / "logs" / "phase2"
MODEL_DIR = ROOT / "models" / "phase2"
for d in (PLOT_DIR, TBL_DIR, LOG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

LEDGER = TBL_DIR / "feature_roles.csv"
AUDIT = LOG_DIR / "leakage_audit.txt"
OUT = PROC / "braking_state_events.parquet"
OLD_COUNT = 157823  # earlier braking_events count, kept as a sanity gate on the new total
RNG = 42

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
    win_ledger = bv.build_role_ledger(list(lw.columns), level="window")
    ev_ledger = bv.build_role_ledger(bv.EVENT_FEATURE_NAMES, level="event")
    ledger = pd.concat([win_ledger, ev_ledger], ignore_index=True)
    leftover = ledger[(ledger.sensor_group == "OTHER")].copy()
    leftover = leftover[~leftover["feature"].apply(
        lambda f: (bv.base_name(f) in bv._OTHER_EXACT) or ("__mode" in f))]
    if len(leftover):
        logger.warning(f"{len(leftover)} feature(s) fell to OTHER unexpectedly: "
                       f"{leftover['feature'].tolist()}")
    ledger.to_csv(LEDGER, index=False)
    counts = ledger.groupby(["level", "role"]).size().unstack(fill_value=0)
    logger.info(f"feature-role ledger ({len(ledger)} rows):\n{counts}")
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
    logger.info(f"leakage audit -> {AUDIT.relative_to(ROOT)} (PASS)")


def _feature_file(day: pd.Timestamp) -> Path:
    return (FEAT_ROOT / f"year={day.year}" / f"month={day.month:02d}"
            / f"day={day.day:02d}" / "day.parquet")


def extract_events(schema) -> None:
    if OUT.exists():
        logger.info(f"{OUT.name} exists -- loading (skip extraction, refresh summary)")
        ev = pd.read_parquet(OUT)
        ev["start_timestamp"] = pd.to_datetime(ev["start_timestamp"])
    else:
        cont = S.continuous_cols(schema)
        read_cols = list(dict.fromkeys(DAY_COLS + cont))
        windows = pd.read_parquet(PROC / "train_windows_labeled.parquet",
                                  columns=["window_start", "window_end", "state"])
        windows["window_start"] = pd.to_datetime(windows["window_start"])
        windows["window_end"] = pd.to_datetime(windows["window_end"])
        events = bv.braking_events_from_windows(windows)
        events["date"] = events["start"].dt.normalize()
        logger.info(f"{len(events):,} contiguous braking events "
                    f"(gap_s={bv.EVENT_GAP_S:g}, min_dur={bv.EVENT_MIN_DURATION_S:g})")

        feats = []
        for date, grp in events.groupby("date"):
            f = _feature_file(date)
            if not f.exists():
                continue
            df = pd.read_parquet(f, columns=read_cols, engine="pyarrow")
            feats.append(bv.event_features(df, grp, schema))
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
        ev["is_real_deceleration"] = bv.is_real_deceleration(ev).to_numpy()
        ev.to_parquet(OUT, engine="pyarrow", compression="snappy", index=False)

    logger.info(f"braking_state_events: {len(ev):,} events | "
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
    logger.info(f"duration mean={np.mean(dur):.1f}s median={np.median(dur):.0f}s | "
                f"count ratio vs old = {len(ev)/OLD_COUNT:.2f}x")


def _tier_features(ledger: pd.DataFrame, roles: list[str], present: set[str]) -> list[str]:
    sel = ledger[(ledger.level == "window") & (ledger.role.isin(roles))]
    return [f for f in sel["feature"] if f in present]


def _eval(name, model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    p, r, f1, _ = precision_recall_fscore_support(
        yte, pred, labels=[1], average=None, zero_division=0)
    row = dict(
        accuracy=accuracy_score(yte, pred),
        balanced_accuracy=balanced_accuracy_score(yte, pred),
        precision_braking=float(p[0]), recall_braking=float(r[0]), f1_braking=float(f1[0]),
        roc_auc=roc_auc_score(yte, proba),
    )
    logger.info(f"{name}: bal_acc={row['balanced_accuracy']:.3f} "
                f"F1(brake)={row['f1_braking']:.3f} AUC={row['roc_auc']:.3f}")
    return model, row, confusion_matrix(yte, pred)


def classify_states() -> None:
    metrics_path = TBL_DIR / "state_classification_metrics.csv"
    if metrics_path.exists():
        logger.info("state-classification metrics exist -- skipping (delete to recompute)")
        return

    ledger = pd.read_csv(LEDGER)
    df = pd.read_parquet(PROC / "train_windows_labeled.parquet")
    present = set(df.columns)

    aux = _tier_features(ledger, ["auxiliary"], present)
    full = _tier_features(ledger, ["auxiliary", "actuation"], present)
    # Leakage guard: no velocity feature may appear in either tier.
    vel = set(ledger[(ledger.level == "window") & (ledger.role == "velocity")]["feature"])
    assert not (set(full) & vel), "velocity feature leaked into predictors!"
    logger.info(f"tier A (auxiliary)={len(aux)} feats | tier B (aux+actuation)={len(full)} feats")

    y = (df["state"] == "braking").astype(int).to_numpy()
    base_rate = y.mean()
    logger.info(f"is_braking base rate = {base_rate:.3f} ({y.sum():,}/{len(y):,})")

    tiers = {"auxonly": aux, "full": full}
    rows = []
    confusions: dict[tuple[str, str], np.ndarray] = {}
    saved_lda_full = None
    saved_dt_full = None
    ALL_MODELS = ["dt", "lda", "qda", "rf", "lr", "pca_lda"]

    for tier_name, feats in tiers.items():
        X = df[feats].to_numpy("float32")
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=RNG, stratify=y)
        n_pca = min(15, len(feats) - 1)
        models = {
            "dt": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RNG),
            "lda": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
            "qda": make_pipeline(StandardScaler(),
                                 QuadraticDiscriminantAnalysis(reg_param=1e-3)),
            "rf": RandomForestClassifier(n_estimators=50, max_depth=10,
                                         class_weight="balanced", n_jobs=-1, random_state=RNG),
            "lr": make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=1000, class_weight="balanced", solver="saga", random_state=RNG)),
            "pca_lda": make_pipeline(StandardScaler(), PCA(n_components=n_pca),
                                     LinearDiscriminantAnalysis()),
        }
        for mname, model in models.items():
            fitted, row, cm = _eval(f"{tier_name}/{mname}", model, Xtr, Xte, ytr, yte)
            row.update(tier=tier_name, model=mname, n_features=len(feats))
            rows.append(row)
            confusions[(tier_name, mname)] = cm
            joblib.dump(fitted, MODEL_DIR / f"state_clf_{mname}_{tier_name}.joblib")
            if tier_name == "full" and mname == "lda":
                saved_lda_full = (fitted, feats)
            if tier_name == "full" and mname == "dt":
                saved_dt_full = (fitted, feats)
        del X, Xtr, Xte

    metrics = pd.DataFrame(rows)[
        ["tier", "model", "n_features", "accuracy", "balanced_accuracy",
         "precision_braking", "recall_braking", "f1_braking", "roc_auc"]]
    metrics.to_csv(metrics_path, index=False)
    logger.info(f"metrics written:\n{metrics.to_string(index=False)}")

    # Tier A vs B contrast
    for m in ALL_MODELS:
        a = metrics[(metrics.tier == "auxonly") & (metrics.model == m)]["f1_braking"].iloc[0]
        b = metrics[(metrics.tier == "full") & (metrics.model == m)]["f1_braking"].iloc[0]
        logger.info(f"{m}: F1(brake) auxonly={a:.3f} -> full={b:.3f} (+{b - a:.3f})")

    # decision tree plot (tier B)
    dt, feats = saved_dt_full
    fig, ax = plt.subplots(figsize=(22, 11))
    plot_tree(dt, feature_names=feats, class_names=["not_braking", "braking"],
              filled=True, fontsize=7, max_depth=3, ax=ax, impurity=False)
    ax.set_title("State classifier decision tree (tier B: aux+actuation, top 3 levels)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "state_clf_tree.png", dpi=150)
    plt.close(fig)

    # LDA coefficients (tier B), top by |coef|
    lda_pipe, feats = saved_lda_full
    coef = lda_pipe.named_steps["lineardiscriminantanalysis"].coef_[0]
    order = np.argsort(np.abs(coef))[::-1][:20]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh([feats[i] for i in order][::-1], coef[order][::-1], color="darkorange")
    ax.set_xlabel("standardized LDA coefficient")
    ax.set_title("Top-20 LDA loadings — braking discriminant (tier B)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "state_clf_lda_coeffs.png", dpi=150)
    plt.close(fig)

    # confusion matrices (all tiers x all models: 4 rows x 3 cols)
    fig, axes = plt.subplots(4, 3, figsize=(13, 16))
    for i, tier_name in enumerate(["auxonly", "full"]):
        for j, mname in enumerate(ALL_MODELS):
            row = i * 2 + (j // 3)
            col = j % 3
            ax = axes[row, col]
            cm = confusions[(tier_name, mname)]
            cmn = cm / cm.sum(axis=1, keepdims=True)
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            for r in range(2):
                for c in range(2):
                    ax.text(c, r, f"{cm[r, c]:,}\n{cmn[r, c]:.2f}",
                            ha="center", va="center", fontsize=8)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["not", "brake"]); ax.set_yticklabels(["not", "brake"])
            ax.set_title(f"{tier_name} / {mname}")
            if col == 0:
                ax.set_ylabel("true")
            if row >= 2:
                ax.set_xlabel("predicted")
    fig.suptitle("State classification confusion matrices (held-out, row-normalized)\n"
                 "Rows 1–2: auxiliary-only tier  |  Rows 3–4: full tier (aux + actuation)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "state_clf_confusion.png", dpi=150)
    plt.close(fig)

    # model comparison bar chart (AUC and F1 for all models, both tiers)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(ALL_MODELS))
    for ax, metric in zip(axes, ["roc_auc", "f1_braking"]):
        for offset, tier_name in zip([-0.2, 0.2], ["auxonly", "full"]):
            vals = [metrics[(metrics.tier == tier_name) & (metrics.model == m)][metric].iloc[0]
                    for m in ALL_MODELS]
            ax.bar(x + offset, vals, width=0.35, label=tier_name, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(ALL_MODELS, rotation=15, ha="right")
        ax.set_ylabel(metric); ax.set_title(f"{metric} by model and predictor tier")
        ax.set_ylim(0, 1); ax.legend()
    fig.suptitle("Braking-state classifier comparison (held-out test set)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "state_clf_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("classifier plots written")


def main() -> None:
    schema = S.load_schema()
    ledger = build_ledger()
    leakage_audit(ledger)
    extract_events(schema)
    classify_states()
    logger.info("phase2 braking (part 1) complete")


if __name__ == "__main__":
    main()
