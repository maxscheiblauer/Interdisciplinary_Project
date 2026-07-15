"""Shared operational-state definitions (Option A: kinematics only).

Phase 1 defines the states and Phase 2 audits them for leakage; both need the same
12 kinematic window features and the same physics-based cluster naming. Keeping one
copy here stops the two lists from drifting apart.
"""
from __future__ import annotations

import pandas as pd

STATES = ["standing", "accelerating", "cruising", "deceleration"]

# Velocity-role window features used as the clustering basis. Signed accel min/max
# are kept on purpose -- they are what could separate accelerate vs brake.
KINEMATIC_FEATS = [
    "TRAIN_SPEED_ACTUAL__mean", "TRAIN_SPEED_ACTUAL__std",
    "TRAIN_SPEED_ACTUAL__min", "TRAIN_SPEED_ACTUAL__max",
    "acceleration__mean", "acceleration__std",
    "acceleration__min", "acceleration__max",
    "jerk__mean", "jerk__std",
    "velocity_change_rate__mean", "velocity_change_rate__std",
]


def name_states_by_physics(profile: pd.DataFrame) -> dict:
    """Map cluster id -> state name from per-cluster median physics.

    ``profile`` is indexed by cluster and holds at least ``velocity_mean`` and
    ``accel_mean``. Lowest median velocity is standing, highest is cruising; of the
    remaining two the higher mean acceleration is accelerating and the other is
    deceleration (positive velocity with negative acceleration = genuine deceleration).
    """
    standing = profile["velocity_mean"].idxmin()
    cruising = profile["velocity_mean"].idxmax()
    rest = [c for c in profile.index if c not in (standing, cruising)]
    accelerating = profile.loc[rest, "accel_mean"].idxmax()
    deceleration = [c for c in rest if c != accelerating][0]
    labels = {standing: "standing", cruising: "cruising",
              accelerating: "accelerating", deceleration: "deceleration"}
    assert len(set(labels.values())) == 4, "label collision"
    return labels
