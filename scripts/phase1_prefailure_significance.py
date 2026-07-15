"""Re-analysis of the Phase-1 'pre-event state distribution shift' result.

Question under review: the original phase-1 report stated 61/62 chi-square tests
significant at p<0.05 for the pre-event operational-state distribution vs baseline.
Three doubts:
  (Q1) Is it significant only because N is huge (chi-square scales with N)?
  (Q2) Is the shift genuine?
  (Q3) Is it just the train sitting idle in the depot before maintenance
       (i.e. more 'standing' windows), not a change in braking dynamics?

Approach:
  - Cramer's V effect size for every (event, horizon) test  -> answers Q1.
  - Direction of shift: pre-event vs baseline standing fraction -> answers Q3.
  - Conditional test on MOVING windows only (drop 'standing')  -> answers Q2/Q3:
    if the shift survives after removing idle time, it is about how the train
    drives; if it collapses, the 'signal' was mostly depot idling.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "results" / "tables"
PLOT = ROOT / "results" / "plots" / "phase1"
PLOT.mkdir(parents=True, exist_ok=True)

STATES = ["standing", "accelerating", "cruising", "deceleration"]
MOVING = ["accelerating", "cruising", "deceleration"]


def cramers_v(obs, base):
    """Cramer's V for a 2xK table (obs vs base). For 2 rows, min(r-1,c-1)=1,
    so V = sqrt(chi2 / N_total)."""
    obs = np.asarray(obs, float); base = np.asarray(base, float)
    table = np.vstack([obs, base])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2 or obs.sum() < 5:
        return np.nan, np.nan, np.nan
    chi2, p, dof, _ = chi2_contingency(table)
    n = table.sum()
    v = np.sqrt(chi2 / n)  # k_eff rows=2 -> min(r-1,c-1)=1
    return float(chi2), float(p), float(v)


def analyze(w, inv, states):
    base = w["state"].value_counts().reindex(states).fillna(0)
    base_frac = base / base.sum()
    rows = []
    for _, e in inv.iterrows():
        for hours in (24, 48, 72):
            lo = e["start"] - pd.Timedelta(hours=hours)
            pre = w[(w["window_start"] >= lo) & (w["window_start"] < e["start"])]
            if len(pre) < 10:
                continue
            cnt = pre["state"].value_counts().reindex(states).fillna(0)
            frac = cnt / cnt.sum()
            chi2, p, v = cramers_v(cnt.to_numpy(), base.reindex(states).to_numpy())
            rec = dict(event_class=e["event_class"], type=e["type"],
                       start=e["start"], window_h=hours, n_pre=int(len(pre)),
                       chi2=chi2, p_value=p, cramers_v=v,
                       standing_frac=frac.get("standing", np.nan),
                       standing_delta=frac.get("standing", np.nan) - base_frac.get("standing", np.nan))
            rows.append(rec)
    return pd.DataFrame(rows), base_frac


def main():
    w = pd.read_parquet(ROOT / "data/processed/train_windows_labeled.parquet",
                        columns=["window_start", "state"])
    w["window_start"] = pd.to_datetime(w["window_start"])
    inv = pd.read_csv(ROOT / "inputs" / "event_inventory.csv")
    inv["start"] = pd.to_datetime(inv["start"])

    # ---- full 4-state test (reproduces original, adds effect size) ----
    full, base_frac = analyze(w, inv, STATES)
    # ---- conditional test: MOVING windows only (drop idle/depot standing) ----
    wm = w[w["state"].isin(MOVING)].copy()
    cond, base_frac_m = analyze(wm, inv, MOVING)
    cond = cond.rename(columns=lambda c: c if c in
                       ("event_class", "type", "start", "window_h") else c + "_moving")

    merged = full.merge(
        cond[["event_class", "type", "start", "window_h",
              "n_pre_moving", "chi2_moving", "p_value_moving", "cramers_v_moving"]],
        on=["event_class", "type", "start", "window_h"], how="left")
    merged.to_csv(TBL / "phase1_prefailure_significance.csv", index=False)

    sig = (full["p_value"] < 0.05).sum()
    print(f"baseline fractions: {base_frac.round(4).to_dict()}")
    print(f"full test: {len(full)} tests, {sig} sig at p<0.05")
    print(f"  Cramers V: median={full.cramers_v.median():.4f} "
          f"max={full.cramers_v.max():.4f} (0.1=small, 0.3=medium)")
    print(f"  standing_delta: median={full.standing_delta.median():+.3f} "
          f"mean={full.standing_delta.mean():+.3f} "
          f"share with standing UP={100*(full.standing_delta>0).mean():.0f}%")
    csig = (merged["p_value_moving"] < 0.05).sum()
    print(f"MOVING-only test: {merged.cramers_v_moving.notna().sum()} tests, "
          f"{csig} sig; Cramers V median={merged.cramers_v_moving.median():.4f} "
          f"max={merged.cramers_v_moving.max():.4f}")

    # focus: brake failures only
    bf = full[full["type"] == "Brake System Failure"]
    print(f"\nBrake failures only (n_tests={len(bf)}): "
          f"V median={bf.cramers_v.median():.4f}, "
          f"standing_delta median={bf.standing_delta.median():+.3f}")

    # ---- plot: effect size vs N, colored by standing direction ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    a.scatter(full.n_pre, full.cramers_v, c=(full.standing_delta > 0),
              cmap="coolwarm", alpha=0.75, edgecolor="k", linewidth=0.3)
    a.axhline(0.1, ls="--", c="grey"); a.text(full.n_pre.min(), 0.105, "V=0.1 (small)", fontsize=8)
    a.set_xlabel("# pre-event windows (N)"); a.set_ylabel("Cramer's V (effect size)")
    a.set_title("Effect size tiny despite p<0.05\n(red=standing UP before event)")
    b = ax[1]
    b.scatter(merged.cramers_v, merged.cramers_v_moving, alpha=0.7, edgecolor="k", linewidth=0.3)
    lim = max(merged.cramers_v.max(), merged.cramers_v_moving.max()) * 1.05
    b.plot([0, lim], [0, lim], ls="--", c="grey")
    b.set_xlabel("Cramer's V (all 4 states)")
    b.set_ylabel("Cramer's V (MOVING windows only)")
    b.set_title("Shift shrinks when idle 'standing' removed")
    fig.tight_layout(); fig.savefig(PLOT / "prefailure_significance.png", dpi=150)
    print(f"\nwrote {TBL/'phase1_prefailure_significance.csv'}")
    print(f"wrote {PLOT/'prefailure_significance.png'}")


if __name__ == "__main__":
    main()
