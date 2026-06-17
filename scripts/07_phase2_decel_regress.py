"""Phase 2 — Task 5: regress deceleration from non-velocity sensors (CO-PRIMARY).

The continuous, always-valid counterpart to Task 4: how much of braking INTENSITY
(peak & mean deceleration) can the NON-VELOCITY sensors explain? Honest metric
(held-out R^2), no class assumption -- this result stands even if Task 3 finds no
clusters.

Targets:    peak_deceleration, mean_deceleration  (velocity-role; the targets)
Predictors: tier A (auxiliary), tier B (+actuation); each run with and WITHOUT
            duration_seconds (kinematic-adjacent confound).
Models:     LinearRegression (standardized, interpretable coeffs) + a tree ceiling
            (HistGradientBoostingRegressor). Baseline = predict the train mean.

Outputs:
  models/phase2/decel_regress_{lin,tree}_{peak,mean}.joblib   (tier B, no-duration)
  results/tables/decel_regression_metrics.csv
  results/plots/phase2/decel_regression_coeffs.png
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
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Lasso, LinearRegression, Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PROC = ROOT / "data" / "processed"
PLOT_DIR = ROOT / "results" / "plots" / "phase2"
TBL_DIR = ROOT / "results" / "tables"
MODEL_DIR = ROOT / "models" / "phase2"
LEDGER = TBL_DIR / "feature_roles.csv"
RNG = 42
TARGETS = ["peak_deceleration", "mean_deceleration"]

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def _event_feats(ledger: pd.DataFrame, roles: list[str], present: set[str]) -> list[str]:
    sel = ledger[(ledger.level == "event") & (ledger.role.isin(roles))]
    return [f for f in sel["feature"] if f in present]


def main() -> None:
    metrics_path = TBL_DIR / "decel_regression_metrics.csv"
    if metrics_path.exists():
        logger.info("[5] metrics exist -- skipping (delete to recompute)")
        return

    ledger = pd.read_csv(LEDGER)
    ev = pd.read_parquet(PROC / "braking_state_events.parquet")
    present = set(ev.columns)

    aux = _event_feats(ledger, ["auxiliary"], present)
    full = _event_feats(ledger, ["auxiliary", "actuation"], present)
    vel = set(ledger[(ledger.level == "event") & (ledger.role == "velocity")]["feature"])
    assert not (set(full) & vel), "velocity feature leaked into predictors!"
    logger.info(f"[5] tier A={len(aux)} feats | tier B={len(full)} feats "
                f"(+duration variants)")

    # Scope: PRIMARY = real-deceleration events ("how hard a MOVING train brakes",
    # confirmed scope); SECONDARY = all braking-state events (incl. stationary holds).
    scopes = {
        "real_decel": ev[ev["is_real_deceleration"]].reset_index(drop=True),
        "all": ev,
    }
    logger.info(f"[5] scopes: real_decel={len(scopes['real_decel']):,} | all={len(scopes['all']):,}")

    tiers = {"auxonly": aux, "full": full}
    rows = []
    coef_store: dict[str, tuple[list[str], np.ndarray]] = {}
    scatter_store: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    lasso_coef_store: dict[str, tuple[list[str], np.ndarray]] = {}

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
                    for mname, model in (("lin", lin), ("ridge", ridge),
                                         ("lasso", lasso), ("tree", tree)):
                        pred = model.predict(Xte)
                        rows.append(dict(
                            scope=scope_name, target=target, tier=tier_name,
                            with_duration=dur, model=mname, n_features=len(cols),
                            r2=r2_score(yte, pred), mae=mean_absolute_error(yte, pred),
                            baseline_r2=base_r2, baseline_mae=base_mae))
                        logger.info(f"[5] {scope_name}/{target}/{tier_name}/dur={dur}/{mname}: "
                                    f"R2={r2_score(yte, pred):.3f} MAE={mean_absolute_error(yte, pred):.4f}")

                    # store coeffs/models for the canonical PRIMARY variant
                    if scope_name == "real_decel" and tier_name == "full" and not dur:
                        coef = lin.named_steps["linearregression"].coef_
                        coef_store[target] = (cols, coef)
                        scatter_store[target] = (yte, lin.predict(Xte))
                        lasso_coef_store[target] = (cols, lasso.named_steps["lasso"].coef_)
                        joblib.dump(lin, MODEL_DIR / f"decel_regress_lin_{target.split('_')[0]}.joblib")
                        joblib.dump(tree, MODEL_DIR / f"decel_regress_tree_{target.split('_')[0]}.joblib")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_path, index=False)
    logger.info(f"[5] metrics written:\n{metrics.round(3).to_string(index=False)}")

    # headline (PRIMARY scope = real-deceleration events, no duration)
    for target in TARGETS:
        best = metrics[(metrics.scope == "real_decel") & (metrics.target == target)
                       & (~metrics.with_duration)]
        b = best.loc[best.r2.idxmax()]
        logger.info(f"[5] HEADLINE {target} (real-decel): best held-out R2={b.r2:.3f} "
                    f"({b.tier}/{b.model}); non-velocity sensors explain "
                    f"{b.r2*100:.0f}% of variance out-of-sample")

    # --- coefficient plot: linear top-15 + Lasso (variable selection) ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, target in zip(axes[:2], TARGETS):
        cols, coef = coef_store[target]
        order = np.argsort(np.abs(coef))[::-1][:15]
        ax.barh([cols[i] for i in order][::-1], coef[order][::-1], color="teal")
        ax.set_xlabel("standardized coefficient")
        ax.set_title(f"{target}: top-15 linear drivers (tier B)")
    # Lasso variable selection: show all non-zero coefficients for peak_deceleration
    lasso_cols, lasso_coef = lasso_coef_store[TARGETS[0]]
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

    # --- scatter plot: predicted vs actual (primary linear model, tier B) ---
    rng_plot = np.random.default_rng(RNG)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, target in zip(axes, TARGETS):
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
                          & (metrics.model == "lin")]["r2"].iloc[0]
        ax.set_xlabel(f"actual {target}")
        ax.set_ylabel(f"predicted {target}")
        ax.set_title(f"{target}\nLinear regression, tier B — out-of-sample R²={best_r2:.2f}")
        ax.legend(fontsize=8)
    fig.suptitle("Regression: predicted vs actual deceleration (5 000 test-set points, primary model)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "decel_regression_scatter.png", dpi=150)
    plt.close(fig)
    logger.info("[5] Task 5 complete.")


if __name__ == "__main__":
    main()
