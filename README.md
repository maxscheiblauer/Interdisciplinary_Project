# MetroAT — Operational State Recognition & Braking Behaviour Analysis

Data-driven predictive-maintenance indicators from a one-year, 1 Hz pneumatic-system
sensor dataset of a Wiener Linien metro train (TU Wien, *Instandhaltungsmanagement*).

- **Phase 1** — unsupervised recognition of operational regimes from **kinematics only**
  (K-means, k=4: standing / accelerating / cruising / braking) and validation against
  documented maintenance/failure events.
- **Phase 2** — braking-event extraction, an honest (leakage-free) braking classifier from
  the air-system sensors, a data-driven test of braking-intensity structure (a continuum,
  not classes), deceleration regression, and pre-failure analysis.

> **Design principle:** operational states are defined from **motion only**; the
> pneumatic (air-system) sensors are kept out of that definition so they serve as
> *independent* predictors in Phase 2 (no circular "predict a definition with itself").

## Dataset

| | |
|---|---|
| Name | MetroAT |
| DOI | [10.48436/9ja0q-bq581](https://doi.org/10.48436/9ja0q-bq581) |
| License | CC BY-NC-ND 4.0 |
| Period | Jun 2024 – Jun 2025, 1 Hz |
| Schema | 109 cols: 74 continuous · 20 binary · 7 operational · 4 failure/maintenance · 4 timestamp |

The data is delivered as **Hive-partitioned Parquet** (`train/year=*/month=*/day=*/day.parquet`).
**Sensor values are min-max normalized to ~[0,1]** (not physical units). Train = Jun 2024 –
11 Feb 2025; test (held out) = 12 Feb – Jun 2025. See
[results/reports/00_data_decisions.md](results/reports/00_data_decisions.md) for the full
profiling report and how the data differs from the original execution plan.

## Repository structure

```
src/metroat/        io, schema, features, windowing, braking_v2, validation, stats
scripts/            00_profile_data … 08_phase2_prefailure
tests/              pytest suite (io, features, windowing, braking_v2, validation)
data/processed/     cleaned per-day features + windowed datasets (gitignored; regenerable)
models/{phase1,2}/  scalers, K-means, classifiers/regressors (joblib)
results/{plots,tables,reports}/   all figures, tables, and the phase reports
logs/               profiling / phase logs
```

## Environment

```bash
uv sync                      # Python 3.12, pinned in .python-version
uv run python -c "import pandas, sklearn, pyarrow; print('OK')"
```

## Reproducing the analysis

```bash
# place the dataset under ./train and ./test (year=/month=/day=/day.parquet)
uv run python scripts/00_profile_data.py        # Step 0: profiling + EDA
uv run python scripts/01_preprocess.py          # 1.1 clean/derive/scale
uv run python scripts/02_phase1_cluster.py      # 1.2-1.4 kinematic clustering (k=3,4 candidates)
uv run python scripts/03_phase1_validate.py     # 1.5-1.7 finalize k=4 labels + validation
uv run python scripts/04_phase2_events.py       # 2.1 role ledger + leakage audit + braking events
uv run python scripts/05_phase2_state_classify.py  # 2.2 braking-from-pneumatics (DT/LDA/QDA)
uv run python scripts/06_phase2_intensity.py    # 2.3 continuum-vs-clusters (+recover if stable)
uv run python scripts/07_phase2_decel_regress.py   # 2.4 deceleration regression (primary)
uv run python scripts/08_phase2_prefailure.py   # 2.5 pre-failure braking behaviour
uv run pytest tests/ -q --cov=metroat
```

Phase 1/2 findings: [01_phase1_findings.md](results/reports/01_phase1_findings.md),
[02_phase2_findings.md](results/reports/02_phase2_findings.md). Supervisor-facing
walkthroughs (plain language + a techniques glossary) are in
[notebooks/phase1_operational_states.ipynb](notebooks/phase1_operational_states.ipynb)
and [notebooks/phase2_braking_analysis.ipynb](notebooks/phase2_braking_analysis.ipynb).

## Reproducibility

All stochastic operations use `random_state=42`. Streaming, one daily file at a time
(peak RSS < 4 GB). Models saved with `joblib`; plots ≥ 150 dpi.

## Citation

```
Steiner, A., Abdelkader, O., & Solovastru, I.-R. (2026). MetroAT [Dataset].
TU Wien Research Data. https://doi.org/10.48436/9ja0q-bq581
```
