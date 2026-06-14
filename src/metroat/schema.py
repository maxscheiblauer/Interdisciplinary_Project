"""Column-group helpers driven by ``logs/data_profiling/schema.json``.

Never hardcode column lists elsewhere — load them from here so the pipeline
stays in sync with the profiled schema.
"""
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _ROOT / "logs" / "data_profiling" / "schema.json"

VELOCITY_COL = "TRAIN_SPEED_ACTUAL"
FAILURE_COLS = [
    "TRAIN_IS_IN_FAILURE",
    "TRAIN_IS_IN_MAINTENANCE",
    "TRAIN_FAILURE_TYPE",
    "TRAIN_MAINTENANCE_TYPE",
]
TIMESTAMP_COLS = ["TIMESTAMP", "year", "month", "day"]


def load_schema(path: str | Path | None = None) -> dict:
    path = Path(path) if path else _SCHEMA_PATH
    return json.loads(path.read_text())


def columns_by_category(category: str, schema: dict | None = None) -> list[str]:
    schema = schema or load_schema()
    return [c for c, info in schema["columns"].items() if info["category"] == category]


def continuous_cols(schema: dict | None = None) -> list[str]:
    return columns_by_category("continuous_sensor", schema)


def binary_cols(schema: dict | None = None) -> list[str]:
    return columns_by_category("binary_sensor", schema)


def operational_cols(schema: dict | None = None) -> list[str]:
    return columns_by_category("operational_state", schema)


def brake_cylinder_cols(schema: dict | None = None) -> list[str]:
    return [c for c in continuous_cols(schema) if "BRAKE_CYLINDER_PRESSURE" in c]


def main_reservoir_cols(schema: dict | None = None) -> list[str]:
    return [c for c in continuous_cols(schema) if "MAIN_RESERVOIR_PRESSURE" in c]


def air_suspension_cols(schema: dict | None = None) -> list[str]:
    """Air-suspension / leveling proxy: the per-bogie air-spring LOAD_PRESSURE
    sensors (the dataset has no column literally named 'suspension'/'leveling';
    load pressure is the air-spring pressure that reflects leveling state)."""
    return [c for c in continuous_cols(schema) if "LOAD_PRESSURE" in c]
