import numpy as np
import pandas as pd

from metroat.io import discover_files, partition_date
from metroat.stats import Welford


def _make_tree(root, dates):
    for y, m, d in dates:
        p = root / f"year={y}" / f"month={m:02d}" / f"day={d:02d}"
        p.mkdir(parents=True)
        pd.DataFrame({"x": [1, 2, 3]}).to_parquet(p / "day.parquet")


def test_discover_files_chronological(tmp_path):
    _make_tree(tmp_path, [(2025, 1, 3), (2024, 12, 31), (2025, 1, 1), (2024, 6, 5)])
    files = discover_files(tmp_path)
    dates = [partition_date(f) for f in files]
    assert dates == sorted(dates)
    assert dates[0] == (2024, 6, 5)
    assert dates[-1] == (2025, 1, 3)


def test_parquet_roundtrip_preserves_dtypes(tmp_path):
    df = pd.DataFrame({
        "f": np.array([1.5, 2.5], dtype="float32"),
        "i": np.array([1, 2], dtype="int64"),
        "b": np.array([True, False]),
        "t": pd.to_datetime(["2024-06-01", "2024-06-02"]),
    })
    p = tmp_path / "x.parquet"
    df.to_parquet(p, engine="pyarrow")
    back = pd.read_parquet(p, engine="pyarrow")
    assert dict(back.dtypes.astype(str)) == dict(df.dtypes.astype(str))


def test_welford_matches_single_pass():
    rng = np.random.default_rng(0)
    data = rng.normal(5, 3, size=1000)
    w = Welford()
    for chunk in np.array_split(data, 7):
        w.update(chunk, total_len=len(chunk))
    assert np.isclose(w.mean, data.mean())
    assert np.isclose(w.std, data.std())
    assert w.min == data.min() and w.max == data.max()
    assert w.nnull == 0


def test_welford_counts_nulls():
    w = Welford()
    w.update(np.array([1.0, np.nan, 3.0]), total_len=3)
    assert w.nnull == 1
    assert np.isclose(w.mean, 2.0)
