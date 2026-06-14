"""Build the two supervisor-facing notebooks (Phase 1 and Phase 2).

Audience: an engineering supervisor who is smart but not a data-science specialist.
Every section says WHAT we did, WHAT it means, and HOW confident we are; every
technique is explained in plain language with an everyday analogy in a shared glossary.
Notebooks load artifacts from disk (no heavy recompute).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": t.splitlines(keepends=True)}


SETUP = """import pandas as pd
from pathlib import Path
from IPython.display import Image, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
TABLES = ROOT / "results" / "tables"
P1 = ROOT / "results" / "plots" / "phase1"
P2 = ROOT / "results" / "plots" / "phase2"
LOGS = ROOT / "logs"
pd.set_option("display.max_columns", 40)
print("Project root:", ROOT)"""

# ---- shared "how to read this" + glossary -------------------------------------

HOW_TO_READ = """> **How to read this notebook.** Each step has three short notes:
> **What we did** (the action), **What it means** (the plain-English result), and
> **How confident** (how much to trust it). Technical terms are explained at the very
> bottom, in *"Techniques explained"* — every method used here is in that glossary with
> an everyday analogy. Nothing assumes a data-science background."""

GLOSSARY = """## Techniques explained (plain language)

Each method we used, with an everyday analogy.

**Normalization / standardization** — Sensors come in different units and ranges. We
rescale each one so they're comparable (think: converting everyone's height and weight
to "how many standard deviations above/below average" so neither dominates just because
its numbers are bigger).

**Clustering** — Letting the data sort itself into natural groups, without telling it
the answer in advance (like sorting a pile of mixed coins into piles by size without
being told the denominations).

**k-means** — A clustering method: you pick how many groups *k* you want; it places *k*
"centres" and assigns each point to its nearest centre, then nudges the centres until
they settle. Fast and simple.

**Hierarchical clustering + dendrogram** — Builds a family tree of the data by repeatedly
merging the two closest groups. The tree (dendrogram) shows at what "distance" groups
merge, which hints at how many real groups there are. We run it on a sample because it's
memory-heavy on millions of rows.

**Choosing the number of groups (k):** we don't guess — we measure.
- **Silhouette** — how tightly packed and well-separated the groups are (higher = cleaner;
  ~1 is perfect, ~0 means overlapping).
- **Davies–Bouldin** — similar idea, lower = better.
- **BIC / AIC** — score a statistical model that rewards fitting the data but penalises
  needless complexity. A genuine "best k" shows up as a low point; if the score just keeps
  improving with more groups, there is no natural number of groups (a *continuum*).
- **Bootstrap + Adjusted Rand Index (ARI)** — re-draw the data many times, re-cluster, and
  check the groupings still agree (ARI ≈ 1 = stable, ≈ 0 = random). Stability is the
  decisive test: pretty groups that don't survive resampling aren't real.

**PCA (Principal Component Analysis)** — Squeezes many correlated sensors into a few
summary axes that capture most of the variation (like summarising a detailed survey by
its two or three main themes). Useful for plotting and for an *anomaly score*: if a
reading can't be rebuilt from those summary axes, it's unusual.

**Train / held-out split** — We fit models on one part of the data and score them on a
*different, unseen* part. Scoring on data the model already saw would be like grading an
exam with the answer key taped to it — meaningless.

**Decision tree / LDA / QDA** — Three transparent ways to draw the boundary between
classes: a decision tree asks a sequence of yes/no questions ("is pressure > x?"); LDA
draws a straight dividing line; QDA allows a curved one. We report all three so the result
doesn't depend on one method's quirks.

**Scoring a classifier:**
- **Accuracy** — fraction correct. Misleading when one class is rare.
- **Balanced accuracy** — accuracy averaged per class, so a rare class still counts.
- **Precision** — of the cases flagged positive, how many really were.
- **Recall** — of the real positives, how many we caught.
- **F1** — the balance of precision and recall (one number).
- **ROC-AUC** — probability the model ranks a real positive above a real negative; 0.5 =
  coin-flip, 1.0 = perfect.
- **Confusion matrix** — the table of right/wrong calls per class.

**Regression & R²** — Predicting a number (not a class). **R²** is the fraction of the
real variation the model explains: 0 = no better than always guessing the average, 1 =
perfect. **MAE** is the average size of the error. We always compare against a
**baseline** (guess-the-average) so the number means something.

**Comparing two groups of measurements:**
- **Mann–Whitney U / Kolmogorov–Smirnov** — tests for whether two groups differ.
- **p-value** — the chance of seeing a difference this big if there were truly none. With
  huge samples, *even trivial differences get tiny p-values*, so a small p alone is not
  "important".
- **Cliff's delta** — the *effect size*: how large the difference actually is (0 = none,
  ±1 = total separation). This is what we read, not the p-value alone.
- **Bonferroni correction** — when you run many tests, you tighten the bar to avoid being
  fooled by chance.

**CUSUM** — A cumulative drift alarm: it adds up small deviations from normal and raises a
flag once they pile up, good at catching slow trends. **Lead time** = how far ahead of a
failure it fired; **false-alarm rate** = how often it cries wolf.

**SHAP** *(used in Phase 3)* — Explains a prediction by how much each sensor pushed it up
or down, like an itemised receipt for the model's decision.

**Leakage / circularity (the key idea behind this whole study)** — A predictor must never
have helped define the thing it's predicting. If you label "hard braking" using
deceleration and then "predict" that label using deceleration, you've just memorised your
own definition — you'll score ~100 % and learn nothing. We avoid this by defining
operational states from **motion only** and predicting them from the **air-system sensors
only**, which never touched the definition."""


# ============================ PHASE 1 NOTEBOOK ================================
def phase1():
    c = []
    c.append(md("""# Phase 1 — What is the train doing? (Operational states)

This notebook finds the train's basic **operational states** — *standing,
accelerating, cruising, braking* — directly from the data, and checks they make
physical sense.

The important design choice: we define these states from the train's **motion only**
(speed and acceleration), and deliberately keep the air-brake/pressure sensors out of
that definition. That keeps those sensors as *independent evidence* for Phase 2 — see
**leakage** in the glossary for why this matters."""))
    c.append(md(HOW_TO_READ))
    c.append(code(SETUP))

    c.append(md("""## 1. From raw sensors to 10-second windows

**What we did:** the train logs ~100 sensors once per second for a year. We summarise
each non-overlapping **10-second window** (its average speed, how much it sped up or
slowed down, etc.). The 10-second window is our unit of analysis.

**What it means:** instead of 30 million raw rows we work with ~1.5 million tidy windows,
each describing a short slice of driving.

**How confident:** this is a standard, lossless-enough summarisation; window length
(10 s) was fixed earlier in the project."""))

    c.append(md("""## 2. How many states are there? (we measure, not guess)

**What we did:** we grouped the windows by motion using two independent methods
(**k-means** and **hierarchical clustering**) and scored candidate group-counts with
**silhouette**, **Davies–Bouldin**, and **stability (bootstrap ARI)**. See glossary."""))
    c.append(code("""display(pd.read_csv(TABLES / "phase1_kselection.csv").round(3))
display(Image(filename=str(P1 / "kinematic_kselection.png")))
display(Image(filename=str(P1 / "dendrogram.png")))"""))
    c.append(md("""**What it means:** four groups (**k = 4**) give the cleanest, most
physically meaningful split, and the two methods agree (high ARI).

**How confident:** good — the four groups are stable under resampling. (Three groups also
works statistically but is physically muddier: it has no steady "cruising" state.)"""))

    c.append(md("""## 3. The four operational states

**What we did:** we named each group from its physical profile (median speed and
acceleration)."""))
    c.append(code("""display(pd.read_csv(TABLES / "phase1_state_summary.csv", index_col=0).round(4))
display(Image(filename=str(P1 / "state_profiles.png")))"""))
    c.append(md("""**What it means:**
- **standing** — not moving (~48% of the time).
- **accelerating** — speed rising (positive acceleration, ~14%).
- **cruising** — high steady speed (~17%).
- **braking** — speed falling (negative acceleration, ~20%).

Crucially, **braking now means *actually slowing down*** (negative acceleration), not a
parked train with the brake held.

**How confident:** high — the profiles are clean and physically sensible."""))

    c.append(md("""## 4. The "held brake" discovery (why we re-did Phase 1)

**What we did:** an earlier version included the brake-pressure sensors when defining the
states. That made a *stationary train holding its brake* (at a station) look like
"braking", because the brake pressure is high even though the train isn't moving. Defining
states from motion only fixes this. We also keep the useful information as a **sub-label**
of standing: `standing_braked` vs `standing_idle`."""))
    c.append(code("""print("Old-state vs new-state cross-tab (how windows were relabelled):")
display(pd.read_csv(TABLES / "phase1_new_vs_old_states.csv", index_col=0))"""))
    c.append(md("""**What it means:** the old "braking" group was contaminated — a large
chunk of it was really *standing* (brake held) or even *accelerating*. The new motion-based
states separate these correctly.

**How confident:** high, and it materially improves everything downstream — Phase 2's
braking analysis is now about genuine decelerations."""))

    c.append(md("""## 5. How the train moves between states

**What we did:** we measured how often each state follows another (transition
probabilities) and how long the train typically stays in each (dwell times)."""))
    c.append(code("""display(pd.read_csv(TABLES / "dwell_times.csv", index_col=0).round(1))
display(Image(filename=str(P1 / "transition_diagram.png")))
display(Image(filename=str(P1 / "state_distribution_by_month.png")))"""))
    c.append(md("""**What it means:** the sequence matches normal metro operation
(stop → accelerate → cruise → brake → stop), and the mix of states is stable month to
month.

**How confident:** high — this is descriptive and consistent across the year."""))

    c.append(md(GLOSSARY))
    return {"cells": c, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


# ============================ PHASE 2 NOTEBOOK ================================
def phase2():
    c = []
    c.append(md("""# Phase 2 — Braking: can the air system tell the story?

Building on the Phase 1 **braking** state, this notebook asks three honest questions:
1. Can the **air-system sensors alone** tell when the train is braking — without using
   speed at all?
2. Are there distinct **braking-intensity classes** (gentle / hard / emergency), or is
   intensity a smooth continuum?
3. Does braking behaviour **change before a brake failure**?

The golden rule throughout (see **leakage** in the glossary): the sensors that *defined*
the braking state (motion) are never used to *predict* it. The air-system sensors, which
never touched the definition, are the predictors."""))
    c.append(md(HOW_TO_READ))
    c.append(code(SETUP))

    c.append(md("""## 0. Keeping predictors and targets separate (no cheating)

**What we did:** we sorted every sensor into a role — *velocity* (motion; targets only),
*actuation* (the brake command), *auxiliary* (independent health sensors like reservoir
and load pressure) — and proved the predictor sensors never overlap with what defined the
target."""))
    c.append(code("""print((LOGS / "phase2" / "leakage_audit.txt").read_text(encoding="utf-8")[:900])"""))
    c.append(md("""**What it means:** any predictive success below is *real*, not an artefact
of predicting a definition with itself.

**How confident:** this is a structural guarantee, checked automatically."""))

    c.append(md("""## 1. Braking events (and a reality check)

**What we did:** a braking event = one continuous stretch of the braking state. We also
flag whether the train was genuinely decelerating."""))
    c.append(code("""display(pd.read_csv(TABLES / "event_summary.csv"))
display(Image(filename=str(P2 / "event_duration_distribution.png")))"""))
    c.append(md("""**What it means:** ~127,000 braking events, and **99.9% are genuine
decelerations** (the old "parked with brake held" contamination is gone, thanks to the
Phase 1 fix).

**How confident:** high."""))

    c.append(md("""## 2. Can the air sensors alone detect braking?

**What we did:** predict "braking vs not" for each window using only air-system sensors
(no speed). Two tiers — *auxiliary only*, then *+ the brake command* — and three
transparent models (decision tree, LDA, QDA), scored on unseen data."""))
    c.append(code("""display(pd.read_csv(TABLES / "state_classification_metrics.csv").round(3))
display(Image(filename=str(P2 / "state_clf_confusion.png")))
display(Image(filename=str(P2 / "state_clf_lda_coeffs.png")))"""))
    c.append(md("""**What it means:** the air sensors recover braking well — **ROC-AUC ≈ 0.9**
— and the *auxiliary* (non-command) sensors alone already reach ~0.88. So braking leaves a
real fingerprint across the air system, not just in the brake command.

**How confident:** solid and honest. These numbers are lower than a naive "predict the
definition with itself" setup would show — that drop *is* the value of doing it cleanly."""))

    c.append(md("""## 3. Are there braking-intensity classes? → No, it's a continuum

**What we did:** we tested whether deceleration falls into distinct classes
(gentle/hard/emergency) using BIC + silhouette + stability — letting the data decide."""))
    c.append(code("""display(pd.read_csv(TABLES / "cluster_selection.csv").round(3))
display(Image(filename=str(P2 / "intensity_distribution.png")))
display(Image(filename=str(P2 / "cluster_model_selection.png")))"""))
    c.append(md("""**What it means:** there is **no natural number of classes** — the score
keeps "improving" with more groups and the groups aren't stable. Deceleration is a smooth
**continuum** (one common level with a long tail of harder stops). So we do **not** invent
"light/hard/emergency" classes.

**How confident:** high — this is a clear, honest negative result."""))

    c.append(md("""## 4. How much of braking *intensity* do the air sensors explain?

**What we did:** since intensity is continuous, we *predict the deceleration value* from
the non-speed sensors (held-out R²; baseline = guessing the average)."""))
    c.append(code("""r = pd.read_csv(TABLES / "decel_regression_metrics.csv")
display(r[r.scope == "real_decel"].round(3).reset_index(drop=True))
display(Image(filename=str(P2 / "decel_regression_coeffs.png")))"""))
    c.append(md("""**What it means:** the air sensors explain roughly **43% of peak** and
**47% of mean** deceleration on unseen data — substantial, given speed is never used.
Auxiliary sensors alone explain a meaningful share; the brake command adds the rest.

**How confident:** this is the headline result and the cleanest one — fully cross-modal,
no class assumptions."""))

    c.append(md("""## 5. Does braking change before a brake failure? (8 failures — small)

**What we did:** compared braking in the 7 days before each of the 8 brake failures
against normal periods, reading the **effect size** (Cliff's delta), not just p-values."""))
    c.append(code("""t = pd.read_csv(TABLES / "prefailure_tests.csv")
t["abs_delta"] = t["cliffs_delta"].abs()
display(t.sort_values("abs_delta", ascending=False)[
    ["feature", "cliffs_delta", "cliffs_ci_lo", "cliffs_ci_hi", "sig_bonferroni"]].round(3))
display(Image(filename=str(P2 / "prefailure_cusum.png")))"""))
    c.append(md("""**What it means:** the pre-failure signal in *braking* is **weak**. With so
many events, almost everything is "statistically significant", but the actual effect sizes
are tiny — the only mild exceptions are two **auxiliary health sensors**
(energy-braking-resistance, load-pressure) which drift slightly before a failure.

**How confident:** low statistically (only 8 failures) — we report this as a genuine weak
/ near-null result, not a detector. It hints that failures may show up more during
*charging/idle* than during braking — a clear next step."""))

    c.append(md("""## Takeaways

1. Braking is **recoverable from the air sensors with no cheating** (AUC ≈ 0.9); the
   independent health sensors carry most of that signal.
2. Braking **intensity is a continuum**, not distinct classes.
3. Non-speed sensors explain **~43–47%** of how hard the train brakes (out-of-sample).
4. The **pre-failure braking signal is weak** (8 failures) — pointing toward an
   idle/charging follow-up."""))

    c.append(md(GLOSSARY))
    return {"cells": c, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


(NB / "phase1_operational_states.ipynb").write_text(json.dumps(phase1(), indent=1), encoding="utf-8")
(NB / "phase2_braking_analysis.ipynb").write_text(json.dumps(phase2(), indent=1), encoding="utf-8")
print("wrote phase1_operational_states.ipynb and phase2_braking_analysis.ipynb")
