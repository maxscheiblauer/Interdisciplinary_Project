"""Marker-delimited report-section writer.

Phase 3 scripts each own one section of ``03_phase3_findings.md``. Each section
is wrapped in ``<!-- BEGIN <id> -->`` / ``<!-- END <id> -->`` markers so a script
can idempotently (re)write its own section without clobbering the others,
regardless of the order the scripts are run.
"""
from __future__ import annotations

from pathlib import Path

_HEADER = "# MetroAT Phase 3 — Deeper Analysis & Failure-Driver Discovery\n"

# Canonical section order in the assembled report.
SECTION_ORDER = ["A", "B", "C", "D", "E", "F"]


def _begin(sid: str) -> str:
    return f"<!-- BEGIN {sid} -->"


def _end(sid: str) -> str:
    return f"<!-- END {sid} -->"


def df_to_md(df, index: bool = True, floatfmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub markdown table (no tabulate dependency)."""
    import pandas as pd  # local import keeps module import cheap

    df = df.copy()
    if index:
        df = df.reset_index()

    def fmt(v):
        if isinstance(v, float):
            if v.is_integer():
                return str(int(v))
            return floatfmt.format(v)
        return str(v)

    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def write_section(report_path: str | Path, section_id: str, content: str) -> None:
    """Insert/replace ``section_id``'s block in the report, keeping canonical order."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    blocks: dict[str, str] = {}
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        for sid in SECTION_ORDER:
            b, e = _begin(sid), _end(sid)
            if b in text and e in text:
                inner = text.split(b, 1)[1].split(e, 1)[0].strip("\n")
                blocks[sid] = inner

    block = content.strip("\n")
    blocks[section_id] = block

    parts = [_HEADER]
    for sid in SECTION_ORDER:
        if sid in blocks:
            parts.append(f"{_begin(sid)}\n{blocks[sid]}\n{_end(sid)}\n")
    report_path.write_text("\n".join(parts), encoding="utf-8")
