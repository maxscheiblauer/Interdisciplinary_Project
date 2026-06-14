"""Phase 2 — Task 2: predict braking-vs-not from pneumatics only.

A genuine, non-circular classifier: can the pneumatic sensors alone tell a
braking window from a non-braking one, with NO velocity/deceleration inputs?
Two predictor tiers, three interpretable models, all reported on a held-out split.

  Tier A (auxiliary only)        -- the real test (non-command health sensors)
  Tier B (auxiliary + actuation) -- expect a jump (actuation = brake command)

Models: DecisionTree(max_depth<=5), LDA, QDA. random_state=42.

Outputs:
  models/phase2/state_clf_{dt,lda,qda}_{auxonly,full}.joblib
  results/tables/state_classification_metrics.csv
  results/plots/phase2/{state_clf_tree,state_clf_lda_coeffs,state_clf_confusion}.png
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
from sklearn.discriminant_analysis import (  # noqa: E402
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
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

PROC = ROOT / "data" / "processed"
PLOT_DIR = ROOT / "results" / "plots" / "phase2"
TBL_DIR = ROOT / "results" / "tables"
MODEL_DIR = ROOT / "models" / "phase2"
LEDGER = TBL_DIR / "feature_roles.csv"
RNG = 42

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


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
    logger.info(f"[2] {name}: bal_acc={row['balanced_accuracy']:.3f} "
                f"F1(brake)={row['f1_braking']:.3f} AUC={row['roc_auc']:.3f}")
    return model, row, confusion_matrix(yte, pred)


def main() -> None:
    metrics_path = TBL_DIR / "state_classification_metrics.csv"
    if metrics_path.exists():
        logger.info("[2] metrics exist -- skipping (delete to recompute)")
        return

    ledger = pd.read_csv(LEDGER)
    df = pd.read_parquet(PROC / "train_windows_labeled.parquet")
    present = set(df.columns)

    aux = _tier_features(ledger, ["auxiliary"], present)
    full = _tier_features(ledger, ["auxiliary", "actuation"], present)
    # Leakage guard: no velocity feature may appear in either tier.
    vel = set(ledger[(ledger.level == "window") & (ledger.role == "velocity")]["feature"])
    assert not (set(full) & vel), "velocity feature leaked into predictors!"
    logger.info(f"[2] tier A (auxiliary)={len(aux)} feats | tier B (aux+actuation)={len(full)} feats")

    y = (df["state"] == "braking").astype(int).to_numpy()
    base_rate = y.mean()
    logger.info(f"[2] is_braking base rate = {base_rate:.3f} ({y.sum():,}/{len(y):,})")

    tiers = {"auxonly": aux, "full": full}
    rows = []
    confusions: dict[tuple[str, str], np.ndarray] = {}
    saved_lda_full = None
    saved_dt_full = None

    for tier_name, feats in tiers.items():
        X = df[feats].to_numpy("float32")
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=RNG, stratify=y)
        models = {
            "dt": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RNG),
            "lda": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
            "qda": make_pipeline(StandardScaler(),
                                 QuadraticDiscriminantAnalysis(reg_param=1e-3)),
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
    logger.info(f"[2] metrics written:\n{metrics.to_string(index=False)}")

    # --- Tier A vs B contrast (headline finding) ---
    for m in ["dt", "lda", "qda"]:
        a = metrics[(metrics.tier == "auxonly") & (metrics.model == m)]["f1_braking"].iloc[0]
        b = metrics[(metrics.tier == "full") & (metrics.model == m)]["f1_braking"].iloc[0]
        logger.info(f"[2] {m}: F1(brake) auxonly={a:.3f} -> full={b:.3f} (+{b - a:.3f})")

    # --- decision tree plot (tier B) ---
    dt, feats = saved_dt_full
    fig, ax = plt.subplots(figsize=(22, 11))
    plot_tree(dt, feature_names=feats, class_names=["not_braking", "braking"],
              filled=True, fontsize=7, max_depth=3, ax=ax, impurity=False)
    ax.set_title("State classifier decision tree (tier B: aux+actuation, top 3 levels)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "state_clf_tree.png", dpi=150)
    plt.close(fig)

    # --- LDA coefficients (tier B), top by |coef| ---
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

    # --- confusion matrices (tiers x models) ---
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for i, tier_name in enumerate(["auxonly", "full"]):
        for j, mname in enumerate(["dt", "lda", "qda"]):
            ax = axes[i, j]
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
            if j == 0:
                ax.set_ylabel("true")
            if i == 1:
                ax.set_xlabel("predicted")
    fig.suptitle("State classification confusion matrices (held-out, row-normalized)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "state_clf_confusion.png", dpi=150)
    plt.close(fig)
    logger.info("[2] plots written; Task 2 complete.")


if __name__ == "__main__":
    main()
