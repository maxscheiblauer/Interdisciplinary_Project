"""Unit tests for metroat.report (markdown table + section writer)."""
import pandas as pd

from metroat import report as R


def test_df_to_md_formats_ints_and_floats():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [0.12345, 3.5]})
    md = R.df_to_md(df, index=False)
    lines = md.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "| --- | --- |"
    assert "| 1 | 0.1235 |" in md   # int-valued float -> "1"; float -> 4dp
    assert "| 2 | 3.5000 |" in md


def test_write_section_roundtrip_and_order(tmp_path):
    p = tmp_path / "report.md"
    R.write_section(p, "C", "C body")
    R.write_section(p, "A", "A body")
    text = p.read_text(encoding="utf-8")
    # header present, and A appears before C regardless of write order
    assert text.startswith("# MetroAT Phase 3")
    assert text.index("A body") < text.index("C body")
    assert "<!-- BEGIN A -->" in text and "<!-- END C -->" in text


def test_write_section_replaces_in_place(tmp_path):
    p = tmp_path / "report.md"
    R.write_section(p, "B", "first")
    R.write_section(p, "B", "second")
    text = p.read_text(encoding="utf-8")
    assert "second" in text
    assert "first" not in text
    assert text.count("<!-- BEGIN B -->") == 1
