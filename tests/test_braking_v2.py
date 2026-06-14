import numpy as np
import pandas as pd

from metroat.braking_v2 import (
    EVENT_FEATURE_NAMES,
    base_name,
    braking_events_from_windows,
    build_role_ledger,
    event_features_v2,
    feature_role,
    is_real_deceleration,
)


def test_is_real_deceleration_flags_moving_stops_only():
    ev = pd.DataFrame({
        "delta_v": [0.0, 0.5, 0.001, 0.3],
        "velocity_at_start": [0.0, 0.6, 0.5, 0.001],
    })
    mask = is_real_deceleration(ev).tolist()
    # row0 stationary, row1 real, row2 negligible delta_v, row3 starts ~stopped
    assert mask == [False, True, False, False]


def _win(start_s, end_s, state):
    t0 = pd.Timestamp("2024-06-01 00:00:00")
    return dict(window_start=t0 + pd.Timedelta(seconds=start_s),
                window_end=t0 + pd.Timedelta(seconds=end_s), state=state)


def test_adjacent_braking_windows_form_one_event():
    w = pd.DataFrame([_win(0, 9, "braking"), _win(10, 19, "braking")])
    ev = braking_events_from_windows(w)
    assert len(ev) == 1
    assert ev.iloc[0]["n_windows"] == 2
    assert ev.iloc[0]["duration_s"] == 20.0  # 19s span + 1


def test_one_missing_window_splits_event():
    # gap from window_end=9 to window_start=20 is 11s > 10s -> split
    w = pd.DataFrame([_win(0, 9, "braking"), _win(20, 29, "braking")])
    ev = braking_events_from_windows(w)
    assert len(ev) == 2


def test_single_full_window_kept_partial_dropped():
    w = pd.DataFrame([_win(0, 9, "braking")])         # dur 10 -> kept
    assert len(braking_events_from_windows(w)) == 1
    w2 = pd.DataFrame([_win(0, 5, "braking")])        # dur 6 < 10 -> dropped
    assert len(braking_events_from_windows(w2)) == 0


def test_non_braking_ignored_v2():
    w = pd.DataFrame([_win(0, 9, "cruising"), _win(10, 19, "standing")])
    assert braking_events_from_windows(w).empty


def test_event_features_v2_roles_and_values():
    # synthetic 1 Hz day: one 11 s braking event, decel ramps speed down
    n = 20
    t0 = pd.Timestamp("2024-06-01 00:00:00")
    ts = [t0 + pd.Timedelta(seconds=i) for i in range(n)]
    df = pd.DataFrame({
        "TIMESTAMP": ts,
        "TRAIN_SPEED_ACTUAL": np.linspace(1.0, 0.0, n),
        "acceleration": np.full(n, -0.05),
        "jerk": np.zeros(n),
        "main_reservoir_pressure_rate": np.full(n, -0.01),
        "AMBIENT_TEMPERATURE": np.full(n, 0.5),
        "MW1_BRAKE_CYLINDER_PRESSURE_BOGIE1": np.full(n, 0.4),
        "CW1_MAIN_RESERVOIR_PRESSURE": np.linspace(0.9, 0.8, n),
        "MW1_LOAD_PRESSURE_BOGIE1": np.full(n, 0.3),
        "MW1_LOAD_SIGNAL": np.full(n, 0.2),
        "MW1_PROPORTIONAL_VALVE_PRESSURE_BOGIE1": np.full(n, 0.25),
        "MW1_SPRING_BRAKE_PRESSURE_BOGIE1": np.full(n, 0.1),
        "MW1_PNEUMATIC_BRAKING_FORCE_BOGIE1": np.full(n, 0.6),
        "MW1_ENERGY_BRAKING_RESISTANCE": np.full(n, 0.5),
    })
    intervals = pd.DataFrame([dict(start=ts[0], end=ts[10], duration_s=11.0,
                                   n_windows=2, is_long=False)])
    feats = event_features_v2(df, intervals)
    assert len(feats) == 1
    r = feats.iloc[0]
    assert abs(r["peak_deceleration"] - 0.05) < 1e-9      # -accel
    assert r["delta_v"] > 0                                 # speed dropped
    assert abs(r["brake_cylinder_pressure_mean"] - 0.4) < 1e-9
    assert r["main_reservoir_pressure_drop"] > 0           # reservoir fell
    # every produced feature is in the role ledger vocabulary
    led = {f for f in EVENT_FEATURE_NAMES}
    produced = set(feats.columns) - {"event_id", "start_timestamp", "end_timestamp",
                                     "n_windows", "is_long", "month"}
    assert produced <= led


def test_base_name_strips_window_suffixes():
    assert base_name("TRAIN_SPEED_ACTUAL__mean") == "TRAIN_SPEED_ACTUAL"
    assert base_name("MW4_BRAKE_CYLINDER_PRESSURE_BOGIE1__std") == "MW4_BRAKE_CYLINDER_PRESSURE_BOGIE1"
    assert base_name("acceleration__min") == "acceleration"
    assert base_name("duration_seconds") == "duration_seconds"


def test_velocity_features_are_targets_not_predictors():
    for f in ["TRAIN_SPEED_ACTUAL__mean", "acceleration__std", "jerk__mean",
              "velocity_change_rate__mean", "peak_deceleration", "mean_deceleration",
              "delta_v", "jerk_rms"]:
        grp, role = feature_role(f)
        assert role == "velocity", (f, grp, role)


def test_actuation_command_pneumatics():
    for f in ["MW4_BRAKE_CYLINDER_PRESSURE_BOGIE1__mean",
              "CW1_PROPORTIONAL_VALVE_PRESSURE_BOGIE2__std",
              "MW1_SPRING_BRAKE_PRESSURE_BOGIE2__mean",
              "MW3_PNEUMATIC_BRAKING_FORCE_BOGIE2__mean",
              "brake_cylinder_pressure_mean__mean",
              "CW1_PNEUMATIC_BRAKE_ACTIVE__maj"]:
        grp, role = feature_role(f)
        assert role == "actuation", (f, grp, role)


def test_auxiliary_health_pneumatics():
    for f in ["CW1_MAIN_RESERVOIR_PRESSURE__mean", "MW2_LOAD_PRESSURE_BOGIE2__std",
              "MW4_LOAD_SIGNAL__mean", "MW3_ENERGY_BRAKING_RESISTANCE__mean",
              "AMBIENT_TEMPERATURE__mean", "main_reservoir_pressure_rate__mean",
              "air_suspension_mean__mean", "CW2_COMPRESSOR_RUNNING__maj"]:
        grp, role = feature_role(f)
        assert role == "auxiliary", (f, grp, role)


def test_bookkeeping_is_other():
    for f in ["window_start", "n_rows", "failure_flag", "cluster", "state",
              "TRAIN_BRAKE_SIGNAL__mode", "event_id", "month"]:
        grp, role = feature_role(f)
        assert role == "other", (f, grp, role)


def test_duration_default_velocity_and_flagged():
    grp, role = feature_role("duration_seconds")
    assert role == "velocity"
    led = build_role_ledger(["duration_seconds"], level="event")
    assert led.loc[0, "note"] != ""


def test_predictor_target_pools_disjoint():
    feats = ["TRAIN_SPEED_ACTUAL__mean", "acceleration__mean",
             "MW4_BRAKE_CYLINDER_PRESSURE_BOGIE1__mean",
             "CW1_MAIN_RESERVOIR_PRESSURE__mean", "peak_deceleration"]
    led = build_role_ledger(feats)
    predictors = set(led[led.role.isin(["actuation", "auxiliary"])]["feature"])
    targets = set(led[led.role == "velocity"]["feature"])
    assert predictors.isdisjoint(targets)
    assert led["predictor_eligible"].sum() == 2


def test_event_feature_names_all_classified():
    led = build_role_ledger(EVENT_FEATURE_NAMES, level="event")
    # no event feature should be unknown OTHER except none expected here
    assert (led.sensor_group == "OTHER").sum() == 0
