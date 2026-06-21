# Step 0 — Data Decisions & Profiling Report (MetroAT)

*Generated 2026-06-04. Train set = Jun 2024 – 11 Feb 2025 (246 daily files, 14,342,361 rows @ 1 Hz). Test set held out.*

## 1. Deviations from the execution plan

| Plan assumption | Reality | Action taken |
|---|---|---|
| Data delivered as `train.zip` / `test.zip` CSV | Already **Hive-partitioned Parquet** (`year=/month=/day=/day.parquet`); no zips | Skipped MD5 (0.1) & CSV→Parquet conversion (0.8); read Parquet directly |
| Raw physical units (bar, m/s², °C; pressure ≥ 0, % in [0,100]) | **Min-max normalized ~[0,1]** (speed max 1.03, MR pressure max 1.15, ambient temp [−0.02, 0.19]) | Physical-plausibility checks dropped; Phase 2 thresholds will be **empirical quantile-based** (confirmed) |
| Rows chronologically ordered | Rows **not time-sorted within a daily file** | Every daily file is `sort_values("TIMESTAMP")` before processing |
| 72 continuous + 2 aggregated | 74 continuous (the 2 "aggregated" are folded in) | Redundancy handled generically via |r|>0.98 detection (see §4) |
| Full year, ~25M rows, 11 failures / 18 maintenance | Train is an **8.5-month subset**: 14.3M rows; 10 logical failures / 11 maintenance | Documented; full-year totals belong to train+test combined |
| Python 3.11 | 3.11 incompatible with default project metadata; 3.12.10 installed | Pinned **3.12** (compatible) |

## 2. Confirmed schema (109 columns)

74 continuous · 20 binary · **7 operational-state** · 4 failure/maintenance · 4 timestamp.

- **Velocity:** `TRAIN_SPEED_ACTUAL` (continuous, normalized).
- **7 operational-state vars:** `TRAIN_MANUAL_MODE`, `TRAIN_AUTOMATIC_MODE`, `TRAIN_EMERGENCY_MODE`, `TRAIN_BRAKE_SIGNAL`, `TRAIN_LINE` (constant = 3), `TRAIN_CURRENT_SECTION` (route section id), `TRAIN_IS_SPECIAL_SECTION`.
- **4 failure/maintenance:** `TRAIN_IS_IN_FAILURE` (bool), `TRAIN_IS_IN_MAINTENANCE` (bool), `TRAIN_FAILURE_TYPE` (str), `TRAIN_MAINTENANCE_TYPE` (str).
- Per-wagon naming: `{CW1,CW2,MW1..MW4}_<signal>_{BOGIE1,BOGIE2}` — 6 wagons × 2 bogies.

## 3. Failure / maintenance inventory (train period)

**Failure types present:** `Brake System Failure`, `Leveling System Failure`, `Compressor Module Failure`.
**Brake-related failure type code = `"Brake System Failure"`.**

Events are defined by merging same-type flag activations across gaps ≤ 24 h (a single
documented incident toggles the flag many times — 35 raw failure activations → 10 logical
failures). Fine activations are in `event_activations.csv`; logical events in `event_inventory.csv`.

| Class | Logical events (train) | Breakdown |
|---|---|---|
| Failures | **10** | 8 Brake · 1 Leveling · 1 Compressor |
| Maintenance | **11** | 4-/8-/48-week revisions, Door, Train Control, Air Conditioning, Spot Inspection, Revision QC |

> **⚠ Discrepancy flagged for review:** the exposé documents 7 brake failures for the *full
> year*; the train subset alone yields 8 brake-failure flag clusters. Sensor-flag activations
> need not map 1:1 to documented incidents. If a ground-truth event list exists, it should be
> used to reconcile; otherwise downstream Phase 2 uses the flag-derived brake failures.

## 4. Data quality

- **Continuous (74):** null %, min/max/mean/std in `full_stats.csv`; ±5σ outlier fraction and
  negative-value flag in `data_quality.csv`.
- **Binary (20):** confirmed {0,1,NaN}; duty cycles in `binary_sensor_stats.csv`.
- **Redundancy:** 214 continuous pairs with |r| > 0.98 (`high_correlation_pairs.csv`) — expected
  given per-wagon sensor replication. PCA in Phase 1 absorbs this; no columns dropped at Step 0.
- `TRAIN_EMERGENCY_MODE` is effectively constant (0) in the train sample — low information.

## 5. Outputs

`logs/data_profiling/`: schema.json, full_stats.csv, operational_state_analysis.csv,
event_inventory.csv, event_activations.csv, label_vocabularies.json, data_quality.csv,
binary_sensor_stats.csv, high_correlation_pairs.csv, profiling.log.
`results/tables/sensor_summary.csv`. `results/plots/eda/` — 7 plots (timeline, nulls,
continuous distributions, duty cycles, operational-state distributions, failure-vs-normal
comparison, seasonal patterns).

## 6. Quality Check 0 status

- [n/a] MD5 — no zips delivered
- [x] schema.json — 109 cols categorized
- [x] 7 operational-state names confirmed
- [x] event inventory (10 failures / 11 maintenance, train subset; discrepancy flagged)
- [x] brake failure type code = `Brake System Failure`
- [x] file-by-file streaming only (peak RSS 1.5 GB)
- [x] all 7 EDA plots saved
- [n/a] interim Parquet — source is already Parquet
- [x] this report written
