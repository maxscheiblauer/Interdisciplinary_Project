# MetroAT - Operational State Recognition & Braking Behaviour Analysis

Data-driven predictive-maintenance indicators from a one-year, 1 Hz pneumatic-system
sensor dataset of a Wiener Linien metro train (TU Wien, *Instandhaltungsmanagement*).

- **Phase 1** - unsupervised recognition of operational regimes from **kinematics only**
  (K-means, k=4: standing / accelerating / cruising / braking) and validation against
  documented maintenance/failure events.
- **Phase 2** - braking-event extraction, an honest (leakage-free) braking classifier from
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
[results/reports/data_decisions.md](results/reports/data_decisions.md) for the full
profiling report and how the data differs from the original execution plan.

## Repository structure

```
src/metroat/        io, schema, features, windowing, braking, validation, stats
scripts/            profile_data, preprocess, phase1_states, phase2_braking, phase2_intensity
tests/              pytest suite (braking, features, core)
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
# run the scripts in this order (each reads the previous one's artifacts):
uv run python scripts/profile_data.py      # profiling + EDA
uv run python scripts/preprocess.py        # clean / derive / fit scaler
uv run python scripts/phase1_states.py     # kinematic clustering + k=4 state labels + validation
uv run python scripts/phase2_braking.py    # role ledger, leakage audit, events, braking classifier
uv run python scripts/phase2_intensity.py  # intensity clusters/continuum, decel regression, pre-failure
uv run pytest tests/ -q --cov=metroat
```

Phase 1/2 findings: [phase1_findings.md](results/reports/phase1_findings.md),
[phase2_findings.md](results/reports/phase2_findings.md). Supervisor-facing
walkthroughs are in
[notebooks/phase1_operational_states.ipynb](notebooks/phase1_operational_states.ipynb)
and [notebooks/phase2_braking_analysis.ipynb](notebooks/phase2_braking_analysis.ipynb).

## Reproducibility

All stochastic operations use `random_state=42`. Streaming, one daily file at a time
(peak RSS < 4 GB). Models saved with `joblib`; plots ≥ 150 dpi.

## Glossary

Plain-language definitions of every method used in the analysis.

**Normalization / standardization** - Sensors come in different units and ranges. Each sensor is
rescaled to be comparable (converting everyone's height and weight to "how many standard
deviations above/below average" so neither dominates just because its numbers are bigger).

**Clustering** - Letting the data sort itself into natural groups, without specifying the answer in
advance (like sorting a pile of mixed coins into piles by size without being told the denominations).

**k-means** - A clustering method: the number of groups *k* is specified; *k* "centres" are placed
and each point is assigned to its nearest centre, then the centres are nudged until they settle.
Fast and simple.

**Hierarchical clustering + dendrogram** - Builds a family tree of the data by repeatedly merging
the two closest groups. The tree (dendrogram) shows at what "distance" groups merge, which hints at
how many real groups there are. Run on a sample because it is memory-heavy on millions of rows.

**Choosing the number of groups (k):** the number of groups is not guessed - it is measured.

- **Silhouette** - how tightly packed and well-separated the groups are (higher = cleaner; ~1 is
  perfect, ~0 means overlapping).
- **Davies–Bouldin** - similar idea, lower = better.
- **BIC / AIC** - score a statistical model that rewards fitting the data but penalises needless
  complexity. A genuine "best k" shows up as a low point; if the score just keeps improving with
  more groups, there is no natural number of groups (a *continuum*).
- **Bootstrap + Adjusted Rand Index (ARI)** - the data is redrawn many times, re-clustered, and
  the groupings checked for agreement (ARI ≈ 1 = stable, ≈ 0 = random). Stability is the decisive
  test: pretty groups that don't survive resampling aren't real.

**PCA (Principal Component Analysis)** - Squeezes many correlated sensors into a few summary axes
that capture most of the variation (like summarising a detailed survey by its two or three main
themes). Useful for plotting; PCA axes maximise variance, not class separation, so some
discriminative signal can be lost.

**Train / held-out split** - Models are fit on one part of the data and scored on a *different,
unseen* part. Scoring on data the model already saw would be like grading an exam with the answer
key taped to it - meaningless.

**Decision tree / LDA / QDA** - Three transparent ways to draw the boundary between classes: a
decision tree asks a sequence of yes/no questions ("is pressure > x?"); LDA draws a straight
dividing line; QDA allows a curved one. All three are reported so the result doesn't depend on one
method's quirks.

**Scoring a classifier:**

- **Accuracy** - fraction correct. Misleading when one class is rare.
- **Balanced accuracy** - accuracy averaged per class, so a rare class still counts.
- **Precision** - of the cases flagged positive, how many really were.
- **Recall** - of the real positives, how many were caught.
- **F1** - the balance of precision and recall (one number).
- **ROC-AUC** - probability the model ranks a real positive above a real negative; 0.5 = coin-flip,  1.0 = perfect.
- **Confusion matrix** - the table of right/wrong calls per class.

**Regression & R²** - Predicting a number (not a class). **R²** is the fraction of the real
variation the model explains: 0 = no better than always guessing the average, 1 = perfect. **MAE**
is the average size of the error. Comparison is always made against a **baseline**
(guess-the-average) so the number means something.

**Comparing two groups of measurements:**

- **Mann–Whitney U / Kolmogorov–Smirnov** - tests for whether two groups differ.
- **p-value** - the chance of seeing a difference this big if there were truly none. With huge
  samples, *even trivial differences get tiny p-values*, so a small p alone is not "important".
- **Cliff's delta** - the *effect size*: how large the difference actually is (0 = none, ±1 = total
  separation). This is what is read, not the p-value alone.
- **Bonferroni correction** - when many tests are run, the bar is tightened to avoid being fooled
  by chance.

**CUSUM** - A cumulative drift alarm: it adds up small deviations from normal and raises a flag
once they pile up, good at catching slow trends. **Lead time** = how far ahead of a failure it
fired; **false-alarm rate** = how often it cries wolf.

**Leakage / circularity (the key idea behind this whole study)** - A predictor must never have
helped define the thing it's predicting. If you label "hard braking" using deceleration and then
"predict" that label using deceleration, you've just memorised your own definition - you'll score
~100% and learn nothing. This is avoided by defining operational states from **motion only** and
predicting them from the **air-system sensors only**, which never touched the definition.

## Citation

```
Steiner, A., Abdelkader, O., & Solovastru, I.-R. (2026). MetroAT [Dataset].
TU Wien Research Data. https://doi.org/10.48436/9ja0q-bq581
```
