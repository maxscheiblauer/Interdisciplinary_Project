# Phase 1 — Operational State Recognition: Findings

*Train set: Jun 2024 - 11 Feb 2025. Operational states from KINEMATIC-ONLY k-means
(k=4) on 10 s scaled window features (velocity + acceleration family). Pneumatic
sensors are deliberately excluded so they remain independent predictors in Phase 2.*

## States identified
standing, accelerating, cruising, braking.
Cluster->state mapping in `models/phase1/cluster_labels.json`; physical medians in
`results/tables/cluster_profiles.csv`. Per-state summary:

```
                   n    pct  median_velocity  median_accel
state                                                     
standing      717211  48.12           0.0000        0.0000
accelerating  213807  14.34           0.3046        0.0410
cruising      257471  17.27           0.8421        0.0019
braking       302008  20.26           0.3432       -0.0327
```

The `braking` state now has positive velocity and **negative** median acceleration —
i.e. it is genuine deceleration, not a parked train with the brake held (those are
now `standing` / `standing_braked`).

## Change vs the old (pneumatic-based) states
States are now defined from **kinematics only** (velocity + acceleration
family), so pneumatic sensors stay independent for Phase 2. The contingency
of old vs new states is in `phase1_new_vs_old_states.csv`. The main change:
stationary brake-holds (zero deceleration) now fall into **standing**
(sub-labelled `standing_braked`), and a genuine **accelerating** regime emerges.

## State transitions (RQ1)
See `transition_matrix.csv` / `transition_diagram.png` and dwell times below:

```
              mean    std   count
state                            
standing      54.2  641.0  132245
accelerating  17.1    5.0  124901
cruising      20.8   10.6  123545
braking       23.8    7.2  127055
```

## Feature importance (RQ2 / O2)
Random-Forest (state label as target, ALL window features as inputs — descriptive
only) test accuracy **0.994**. Top-10 features: acceleration__mean, TRAIN_SPEED_ACTUAL__mean, acceleration__max, TRAIN_SPEED_ACTUAL__std, acceleration__min, TRAIN_SPEED_ACTUAL__max, velocity_change_rate__mean, TRAIN_SPEED_ACTUAL__min, MW2_PNEUMATIC_BRAKING_FORCE_BOGIE2__mean, CW2_PROPORTIONAL_VALVE_PRESSURE_BOGIE2__mean.
Full ranking: `feature_importance_phase1.csv`.

## Validation against documented events (O1)
62 (event x window-size) chi-square tests; **61** with p<0.05.
Detail in `phase1_event_validation.csv` and `pre_event_cluster_distributions.png`.

## Notes
- Operational-state *variables* (`TRAIN_*_MODE` etc.) are validation-only; never inputs.
- States are defined by kinematics only (Option A); pneumatics are independent evidence.
- All sensor values are normalized ~[0,1]; physical medians reported in that space.
