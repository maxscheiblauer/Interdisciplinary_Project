"""Phase 1.1 — Preprocessing.

Reads the partitioned train Parquet, and for each day: time-sorts, dedups,
imputes (continuous have 0% nulls so this is a safety no-op), clips continuous
to global [1st, 99th] percentiles, adds derived features, and writes a per-day
cleaned+derived Parquet under data/processed/train_features/.

A StandardScaler is fit from streaming statistics of the clipped continuous +
derived features (never on binary/operational/failure) and saved.

Outputs:
  data/processed/train_features/year=*/month=*/day=*/day.parquet
  models/phase1/standard_scaler.joblib
  models/phase1/scaled_feature_names.json
  logs/phase1/timestamp_gaps.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psutil  # noqa: E402
from loguru import logger  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from metroat import features as F  # noqa: E402
from metroat import schema as S  # noqa: E402
from metroat.io import discover_files, partition_date  # noqa: E402

TRAIN_ROOT = ROOT / "train"
OUT_ROOT = ROOT / "data" / "processed" / "train_features"
MODEL_DIR = ROOT / "models" / "phase1"
LOG_DIR = ROOT / "logs" / "phase1"
for d in (OUT_ROOT, MODEL_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add(LOG_DIR / "preprocess.log", level="INFO")
proc = psutil.Process()
SAMPLE_FRAC = 0.05
RNG = np.random.default_rng(42)


class Welford:
    __slots__ = ("n", "mean", "M2")

    def __init__(self):
        self.n, self.mean, self.M2 = 0, 0.0, 0.0

    def update(self, arr):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        bn, bm = arr.size, float(arr.mean())
        bM2 = float(((arr - bm) ** 2).sum())
        delta = bm - self.mean
        tot = self.n + bn
        self.mean += delta * bn / tot
        self.M2 += bM2 + delta**2 * self.n * bn / tot
        self.n = tot

    @property
    def std(self):
        return float(np.sqrt(self.M2 / self.n)) if self.n > 1 else 1.0


def compute_clip_bounds(files, cont):
    """1st/99th percentile per continuous col from a 5% sample."""
    buf = {c: [] for c in cont}
    for f in files:
        df = pd.read_parquet(f, columns=cont, engine="pyarrow")
        k = max(1, int(len(df) * SAMPLE_FRAC))
        idx = RNG.choice(len(df), size=k, replace=False)
        s = df.iloc[idx]
        for c in cont:
            buf[c].append(s[c].to_numpy("float64"))
    lo, hi = {}, {}
    for c in cont:
        v = np.concatenate(buf[c])
        v = v[np.isfinite(v)]
        lo[c], hi[c] = np.percentile(v, [1, 99])
    logger.info(f"[1.1] clip bounds computed from {sum(len(x) for x in buf[cont[0]]):,} sampled rows")
    return lo, hi


def main():
    schema = S.load_schema()
    cont = S.continuous_cols(schema)
    binr = S.binary_cols(schema)
    oper = S.operational_cols(schema)
    files = discover_files(TRAIN_ROOT)

    done = list(OUT_ROOT.glob("year=*/month=*/day=*/day.parquet"))
    scaler_path = MODEL_DIR / "standard_scaler.joblib"
    if len(done) >= len(files) and scaler_path.exists():
        logger.info(f"[1.1] checkpoint: {len(done)} day files + scaler exist — skipping")
        return

    logger.info(f"[1.1] {len(files)} daily files | RSS={proc.memory_info().rss/1e9:.2f}GB")
    lo, hi = compute_clip_bounds(files, cont)

    scaled_feats = cont + F.DERIVED_COLS
    wel = {c: Welford() for c in scaled_feats}
    gap_rows = []
    total_dups = 0
    total_rows = 0

    for i, f in enumerate(files):
        y, m, d = partition_date(f)
        out = OUT_ROOT / f"year={y}" / f"month={m:02d}" / f"day={d:02d}" / "day.parquet"
        df = pd.read_parquet(f, engine="pyarrow").sort_values("TIMESTAMP").reset_index(drop=True)
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        total_dups += before - len(df)

        # impute: continuous ffill (safety), binary ffill; operational/failure untouched
        df[cont] = df[cont].ffill()
        df[binr] = df[binr].ffill()
        # clip continuous to global percentiles
        for c in cont:
            df[c] = df[c].clip(lo[c], hi[c])

        gaps = F.count_gaps(df)
        if not gaps.empty:
            gaps.insert(0, "date", f"{y}-{m:02d}-{d:02d}")
            gap_rows.append(gaps)

        df = F.add_derived_features(df, schema)
        for c in scaled_feats:
            wel[c].update(df[c].to_numpy("float64"))

        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, engine="pyarrow", compression="snappy", index=False)
        total_rows += len(df)
        if i % 30 == 0 or i == len(files) - 1:
            logger.info(f"[1.1] {i+1}/{len(files)} {y}-{m:02d}-{d:02d} rows={len(df)} "
                        f"RSS={proc.memory_info().rss/1e9:.2f}GB")

    # Build scaler from streaming stats.
    means = np.array([wel[c].mean for c in scaled_feats])
    stds = np.array([wel[c].std if wel[c].std > 0 else 1.0 for c in scaled_feats])
    scaler = StandardScaler()
    scaler.mean_ = means
    scaler.scale_ = stds
    scaler.var_ = stds**2
    scaler.n_features_in_ = len(scaled_feats)
    scaler.feature_names_in_ = np.array(scaled_feats)
    scaler.n_samples_seen_ = wel[scaled_feats[0]].n
    joblib.dump(scaler, scaler_path)
    (MODEL_DIR / "scaled_feature_names.json").write_text(json.dumps(scaled_feats, indent=2))

    if gap_rows:
        pd.concat(gap_rows, ignore_index=True).to_csv(LOG_DIR / "timestamp_gaps.csv", index=False)
    n_gaps = sum(len(g) for g in gap_rows)
    logger.info(f"[1.1] DONE rows={total_rows:,} dups_removed={total_dups} "
                f"gaps>{F.GAP_S}s={n_gaps} scaler+features saved "
                f"RSS={proc.memory_info().rss/1e9:.2f}GB")


if __name__ == "__main__":
    main()
