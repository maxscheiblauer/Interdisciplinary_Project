"""Phase 2 — Task 6: pre-failure braking behaviour (braking-only).

Replaces the fragile 30-day OLS-slope test with a DISTRIBUTIONAL comparison of
pre-failure braking behaviour vs a matched baseline, on a short (7 d, also 14 d)
window, with honest small-N (n=8 brake failures) framing.

  1. Per-failure event counts in the 7 d / 14 d window (sparsity is the main risk).
  2. Two-sample tests (Mann-Whitney + KS) of each key feature, pre-failure vs
     far-from-failure baseline, with Cliff's-delta effect size + bootstrap CI and
     Bonferroni correction across features.
  3. Exploratory coefficient-shift test: add a pre_failure indicator (+interaction)
     to the Task-5 deceleration regression (OLS) -- underpowered, framed as such.
  4. CUSUM on the weekly auxiliary wear-proxy series; change points within 0-7 d
     of a failure.
  5. Per-failure summary table + 7-day pre-failure trajectory small-multiples.

Outputs:
  results/tables/prefailure_tests.csv
  results/tables/cusum_changepoints.csv
  results/plots/phase2/{prefailure_case_studies,prefailure_cusum}.png
  logs/phase2/prefailure_summary.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

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

from metroat.validation import cusum  # noqa: E402

PROC = ROOT / "data" / "processed"
PLOT_DIR = ROOT / "results" / "plots" / "phase2"
TBL_DIR = ROOT / "results" / "tables"
LOG_DIR = ROOT / "logs" / "phase2"
RNG = 42

PRE_DAYS = 7
PRE_DAYS_ALT = 14
MIN_EVENTS = 20          # a pre-failure window with fewer events is excluded
CUSUM_THRESHOLD = 4.0    # std units; tuned for ~1 false alarm / 6 months (see note)

# Key features per role (the behaviour we expect to drift before a brake failure).
KEY_FEATURES = [
    # actuation wear / command
    "brake_cylinder_pressure_integral", "brake_cylinder_pressure_peak",
    "pneumatic_braking_force_mean", "spring_brake_pressure_mean",
    # kinematic outcome
    "jerk_rms", "peak_deceleration",
    # auxiliary health
    "main_reservoir_pressure_drop", "main_reservoir_pressure_rate_mean",
    "load_pressure_mean", "energy_braking_resistance_mean",
]
AUX_PROXY = "main_reservoir_pressure_drop"  # weekly wear-proxy for CUSUM

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


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


def main() -> None:
    out_tests = TBL_DIR / "prefailure_tests.csv"
    if out_tests.exists():
        logger.info("[6] prefailure tests exist -- skipping (delete to recompute)")
        return

    ev_all = pd.read_parquet(PROC / "braking_state_events.parquet")
    ev_all["start_timestamp"] = pd.to_datetime(ev_all["start_timestamp"])
    inv = pd.read_csv(ROOT / "logs" / "data_profiling" / "event_inventory.csv")
    inv = inv[(inv.event_class == "failure") & (inv.type == "Brake System Failure")]
    fail_starts = pd.to_datetime(inv["start"]).sort_values().to_numpy()

    # Scope to real-deceleration events (confirmed: "how a MOVING train brakes").
    # Stationary brake-holds have no deceleration signal and would dilute the tests.
    ev = ev_all[ev_all["is_real_deceleration"]].reset_index(drop=True)
    logger.info(f"[6] {len(fail_starts)} brake failures; {len(ev_all):,} braking events "
                f"({len(ev):,} real-deceleration, used for tests; "
                f"{len(ev_all) - len(ev):,} stationary holds excluded)")

    rng = np.random.default_rng(RNG)
    st = ev["start_timestamp"].to_numpy()
    st_all = ev_all["start_timestamp"].to_numpy()

    # --- 1. per-failure event counts (real-decel used for tests; total reported) ---
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
    logger.info(f"[6] per-failure 7d counts:\n{per_fail_df.to_string(index=False)}")
    n_incl = int(per_fail_df["included_7d"].sum())

    # --- 2. distributional comparison: pooled pre-failure vs baseline ---
    pre_mask = ev["failure_within_7_days"].to_numpy() == 1
    base_mask = ev["failure_within_30_days"].to_numpy() == 0  # far from any failure
    logger.info(f"[6] pooled pre-7d events={int(pre_mask.sum()):,} | "
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
        logger.info(f"[6] {feat}: delta={delta:+.3f} [{lo:+.2f},{hi:+.2f}] "
                    f"MWU p(bonf)={min(1.0, mwu_p*n_tests):.2e}")
    tests = pd.DataFrame(rows)
    tests.to_csv(out_tests, index=False)

    # --- 3. exploratory coefficient-shift (OLS with pre_failure interaction) ---
    coef_shift_lines = []
    try:
        sub_cols = ["peak_deceleration", "brake_cylinder_pressure_integral",
                    "main_reservoir_pressure_drop", "load_pressure_mean"]
        d = ev[sub_cols].copy()
        d["pre_failure"] = pre_mask.astype(int)
        # balance: sample baseline down to ~5x pre for a tractable, less-imbalanced fit
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

    # --- 4. CUSUM on weekly auxiliary wear-proxy ---
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
    logger.info(f"[6] CUSUM (thr={CUSUM_THRESHOLD}) on weekly {AUX_PROXY}: "
                f"{len(cp_dates)} change points, "
                f"{int(cusum_df['within_7d_of_failure'].sum()) if len(cusum_df) else 0} within 7d of a failure")

    # --- summary log ---
    n_sig = int(tests["sig_bonferroni"].sum())
    lines = ["Phase 2 -- Task 6 pre-failure braking summary", "=" * 55, ""]
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
    lines.append("")
    lines.append("FUTURE WORK (not implemented): brake-system failures may manifest during")
    lines.append("CHARGING/IDLE (compressor/reservoir/valve), not during deceleration. This")
    lines.append("braking-only analysis may be looking under the wrong lamp post. A follow-up")
    lines.append("should repeat it on state in {standing, cruising} windows using auxiliary")
    lines.append("sensors (compressor duty cycle, reservoir recharge time).")
    (LOG_DIR / "prefailure_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"[6] {n_sig}/{len(KEY_FEATURES)} features significant (Bonferroni)")

    # --- 5a. case-study small multiples (7-day pre-failure trajectories) ---
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

    # --- 5b. CUSUM plot ---
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
    logger.info("[6] Task 6 complete.")


if __name__ == "__main__":
    main()
