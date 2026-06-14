# Phase 2 — Braking Events, Honest Classification & Pre-Failure Behaviour

*MetroAT predictive-maintenance study — one Wiener Linien metro train, 1 Hz
pneumatic-system data, Jun 2024 – Jun 2025. Train-only analysis (test held out).
All stochastic operations use `random_state=42`.*

## Foundation: clean separation of roles

Phase 1 now defines operational states from **kinematics only** (velocity +
acceleration family). Pneumatic sensors are deliberately **excluded** from the state
definition, so they are fully independent evidence in Phase 2. Every feature has one
**role**:

| Role | Groups | Use |
|---|---|---|
| **velocity / kinematic** | speed, acceleration, jerk, deceleration, Δv | **targets only** |
| **actuation** | brake-cylinder, proportional-valve, spring-brake, pneumatic-braking-force | predictor (brake *command*) |
| **auxiliary** | main-reservoir, load-pressure, load-signal, energy-braking-resistance, ambient-temp, pressure-rate | predictor (independent health sensors) |

The leakage audit (`logs/phase2/leakage_audit.txt`) proves the predictor pools
(actuation + auxiliary) are disjoint from **both** the velocity targets **and** the 12
kinematic features that defined the states. So predicting braking from pneumatics and
recovering deceleration from pneumatics are **both fully cross-modal** — no predictor
helped define its own target. (This replaced an earlier version whose "intensity
classifier" was circular — it predicted deceleration-defined labels using deceleration
as an input, scoring a meaningless 100 %.)

## 1 — Braking events as regime occupancy

A braking event = one contiguous occupancy of the Phase-1 `braking` state (adjacent
10 s windows merged; a single missing window splits the event), so duration is a real
structural feature.

- **126,702 events** (0.80× the original count — within the sanity band).
- **99.9 % are real decelerations** (train moving and actually slowing). Because the
  states are now kinematic, the old problem — a parked train with the brake *held* being
  labelled "braking" — is gone: those windows are now `standing` (sub-labelled
  `standing_braked`). Median braking-state velocity 0.34, median acceleration **−0.033**.
- Duration: mean 23 s, median 20 s.

## 2 — Predict braking-vs-not from pneumatics only (honest, cross-modal)

Window-level `is_braking` target, **zero velocity predictors**. Two tiers × three
interpretable models, held-out 25 % split.

| Tier | model | balanced acc | F1(braking) | ROC-AUC |
|---|---|---|---|---|
| **A — auxiliary only** | DT | 0.80 | 0.58 | 0.88 |
| | LDA | 0.62 | 0.39 | 0.81 |
| | QDA | 0.69 | 0.45 | 0.84 |
| **B — + actuation** | DT | 0.83 | 0.61 | 0.91 |
| | LDA | 0.73 | 0.61 | 0.92 |
| | QDA | 0.75 | 0.60 | 0.90 |

**Finding:** pneumatic sensors recover the kinematically-defined braking state at
**AUC ≈ 0.88–0.92** — a genuine cross-modal result, with no sensor that defined the
state acting as a predictor. Auxiliary (non-command) sensors alone already reach
AUC 0.88; adding the actuation command lifts it modestly (AUC → ~0.91). The numbers are
lower than a naive circular setup would show — that is the honest cost of removing the
leakage. Interpretability: `state_clf_tree.png`, `state_clf_lda_coeffs.png`,
`state_clf_confusion.png`.

## 3 — Do braking events cluster into intensity groups? — CONTINUUM

A data-driven test (GMM BIC + silhouette + bootstrap-ARI) on the real-deceleration
events using intensity features (peak/mean deceleration, jerk). BIC decreases
**monotonically** (no preferred number of groups) and every k≥2 has a near-empty
cluster — i.e. k-means just peels slivers off a long tail. **Braking intensity is a
continuum, not discrete classes** (sharp mode ~0.046 normalized, thin tail). We did not
manufacture "light/heavy/emergency" classes. Consequently the recover-from-pneumatics
step is **skipped** and the regression (Step 5) is the primary intensity result. See
`cluster_selection.csv`, `cluster_model_selection.png`, `intensity_distribution.png`,
`logs/phase2/cluster_decision.txt`.

## 4 — Recover intensity clusters from pneumatics — not applicable

Skipped: Step 3 found a continuum, so there are no stable clusters to recover
(`logs/phase2/recover_skipped.txt`). The continuous regression carries the intensity
question instead.

## 5 — Regress deceleration from non-velocity sensors (primary intensity result)

How much of braking intensity (peak & mean deceleration) do the **non-velocity**
sensors explain? Held-out R², mean-predictor baseline (R²≈0), real-deceleration events.

| Target | Tier A (auxiliary) | Tier B (+actuation), linear | Tier B, tree |
|---|---|---|---|
| peak_deceleration | 0.16–0.24 | 0.24 | **0.43** |
| mean_deceleration | 0.09–0.23 | 0.20 | **0.47** |

**Headline:** non-velocity sensors explain **~43 % of peak** and **~47 % of mean**
deceleration variance out-of-sample — a real, fully cross-modal result that needs no
class assumption. Adding `duration_seconds` changes R² by ≤0.03 (not a confound).
Standardized linear drivers: `decel_regression_coeffs.png`.

## 6 — Pre-failure braking behaviour (braking-only, n = 8)

Distributional comparison (Mann–Whitney + Cliff's δ with bootstrap CI, Bonferroni)
of pre-failure braking vs a matched baseline, on real-deceleration events. n = 8 brake
failures — small N, interpret cautiously; all 8 had ample events in the 7-day window.

With ~17 k pooled events even negligible shifts are "significant", so read the **effect
size**: nearly all |δ| < 0.05. The only non-trivial shift is the **auxiliary**
`energy_braking_resistance` (δ ≈ −0.20) and `load_pressure` (δ ≈ −0.10), running lower
in the week before a brake failure. An exploratory coefficient-shift OLS is underpowered
(model R² ≈ 0.04). CUSUM on the weekly reservoir-drop proxy flags a few change points,
1 within 7 days of a failure. **The pre-failure braking signal is weak.**

**Future work (not implemented):** brake-system failures may manifest during
**charging/idle** (compressor/reservoir/valve), not during deceleration — so a
braking-only view may look under the wrong lamp post. The slightly larger auxiliary
shifts support this. A follow-up should repeat the analysis on `state ∈ {standing,
cruising}` windows using auxiliary sensors.

## Takeaways

1. Operational states are now **kinematic** — `braking` is genuine deceleration, and
   ~half of the old "braking" (parked-with-brake-held) correctly moved to `standing`;
   a real **accelerating** regime also emerged.
2. **Braking is recoverable from pneumatics with no leakage** (AUC ≈ 0.9), and
   auxiliary sensors alone carry most of that signal.
3. **Braking intensity is a continuum**, not discrete classes.
4. **Non-velocity sensors explain ~43 %/47 % of peak/mean deceleration** out-of-sample.
5. **Pre-failure braking signal is weak (n=8)**; auxiliary health sensors show the
   largest (still small) shifts, motivating an idle/charging follow-up.

*Artifacts: `results/tables/{feature_roles,event_summary,state_classification_metrics,
cluster_selection,decel_regression_metrics,prefailure_tests,cusum_changepoints}.csv`,
`results/plots/phase2/*.png`, `models/phase2/*.joblib`, `logs/phase2/*.txt`.
Reproduce with `scripts/04–08_phase2_*.py` (checkpoint-skippable).*
