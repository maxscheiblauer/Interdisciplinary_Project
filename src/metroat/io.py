"""I/O utilities for the MetroAT dataset.

The MetroAT data is delivered as Hive-partitioned Parquet:
``<root>/year=YYYY/month=MM/day=DD/day.parquet`` (one file per day, 1 Hz).

Core rule from the execution plan: **stream daily files one at a time**;
never concatenate the whole dataset into memory.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import pandas as pd

# Matches the Hive partition components in a daily-file path.
_PART_RE = re.compile(r"year=(\d+).*?month=(\d+).*?day=(\d+)", re.DOTALL)


def discover_files(root: str | Path) -> list[Path]:
    """Return all daily Parquet files under ``root`` in chronological order.

    Sort key is parsed from the ``year=/month=/day=`` partition components so
    ordering is independent of filesystem glob order.
    """
    root = Path(root)
    files = list(root.glob("year=*/month=*/day=*/*.parquet"))

    def _key(p: Path) -> tuple[int, int, int]:
        m = _PART_RE.search(str(p))
        if not m:
            return (0, 0, 0)
        return tuple(int(g) for g in m.groups())  # type: ignore[return-value]

    return sorted(files, key=_key)


def partition_date(path: str | Path) -> tuple[int, int, int]:
    """Extract ``(year, month, day)`` from a daily-file path."""
    m = _PART_RE.search(str(path))
    if not m:
        raise ValueError(f"No year/month/day partition in path: {path}")
    return tuple(int(g) for g in m.groups())  # type: ignore[return-value]


def iter_daily(
    root: str | Path, columns: list[str] | None = None
) -> Iterator[tuple[Path, pd.DataFrame]]:
    """Yield ``(path, dataframe)`` for each daily file in chronological order.

    ``columns`` enables column projection (predicate/column pushdown) so only
    needed columns are read off disk.
    """
    for path in discover_files(root):
        df = pd.read_parquet(path, columns=columns, engine="pyarrow")
        yield path, df
