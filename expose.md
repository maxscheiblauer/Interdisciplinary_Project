# Operational State Recognition and Braking Behavior Analysis for Predictive Maintenance on Metro Trains

**Maximilian Scheiblauer (11776651)**  
Domain Lecture: 330.331 Instandhaltungsmanagement (TU Wien)  
Domain Specific Supervisor: Dr. Andreas Steiner  
Data Science Supervisor: Dr. Florina Piroi

---

## Abstract

Urban metro systems face a persistent challenge: maintenance is scheduled at fixed intervals regardless of actual operational wear patterns. This inefficiency leads to both premature component replacement and unexpected failures that cause costly service disruptions. Current practice does not account for how different operational regimes — heavy braking days, varied route profiles, or seasonal load changes — actually stress rolling stock components.

This project addresses this gap by developing data-driven maintenance indicators from a one-year, 1 Hz sensor dataset (MetroAT) from a Wiener Linien metro train. The dataset comprises 105 sensor variables covering the pneumatic system, enriched with 11 documented service-affecting failures and 18 maintenance events. Pneumatic systems account for approximately 20% of metro failures and include safety-critical components (brakes, leveling, air production units), making them an ideal target for predictive maintenance research.

The analysis follows a two-phase structure: Phase 1 establishes distinct operational regimes through unsupervised learning and validates them against documented maintenance patterns; Phase 2 performs fine-grained braking event analysis to detect degradation trends and correlate intensity patterns with actual failure occurrences. The full MetroAT dataset will be available via GitHub within two weeks.

---

## Problems, Research Questions & Expected Outcomes

| Problem | Research Question | Expected Outcome |
|---|---|---|
| **P1 Maintenance Impact Assessment:** Operational variability (different driving profiles, braking intensities, route characteristics) affects component wear rates, but current maintenance schedules do not differentiate between high-wear and normal operation periods. | RQ1: Can distinct operational regimes be identified from sensor data that correlate with different wear patterns? RQ2: Which operational features have the strongest association with documented maintenance and failure events? | O1: Labeled operational state sequences with transition diagram showing dwell times and transition frequencies, validated against maintenance patterns. O2: Ranked feature importance analysis identifying sensors most relevant to predicting wear and maintenance needs. |
| **P2 Braking System Degradation Patterns:** Braking is the primary contributor to mechanical wear on metro trains. The MetroAT dataset documents 7 brake-related failures out of 11 total pneumatic failures. Quantifying braking intensity and detecting degradation trends could enable targeted maintenance before failure occurs. | RQ3: Are there temporal patterns in braking intensity that correlate with documented brake system failures and maintenance needs? RQ4: Can gradual degradation trends be detected in braking behavior before documented failures occur? | O3: Labeled braking event dataset with intensity categories, validated against actual brake system failures in the dataset. O4: Temporal pattern analysis identifying high-stress periods and degradation trends, with evidence of early warning indicators for documented failures. |

---

## Methodology

The analysis follows the CRISP-DM framework, adapted for exploratory time-series sensor data with explicit validation against documented maintenance outcomes. Phase 2 takes the labeled output of Phase 1 as direct input, establishing sequential dependency between the phases.

### Phase 1 — Operational State Recognition

**Data Preparation:** The MetroAT dataset provides 105 sensor variables sampled at 1 Hz over one year, covering brake control computers, pneumatic valve blocks, brake cylinders, air-suspension bellows, and air production units across six wagons. Each sensor reading includes contextual metadata (train line, track section, operational mode). Standard preprocessing includes deduplication, handling missing values, outlier treatment, and timestamp synchronization. Derived features (acceleration, jerk, velocity change rates) will be computed as these carry discriminative signal for state separation. The dataset structure will be clarified by documenting what each sensor provides: single time series (e.g., main reservoir pressure) versus multi-axis measurements (e.g., forces on multiple brake cylinders per wagon).

**Clustering & Validation:** K-means clustering (k=4) will partition sensor windows into four operational states: standing, accelerating, cruising, and braking. This choice is motivated by domain knowledge of distinct physical regimes. Cluster quality will be evaluated using silhouette scores and Davies-Bouldin indices. Crucially, the identified states will be validated against the 11 documented failures and 18 maintenance events in the dataset to assess whether specific operational regimes correlate with actual maintenance needs. State transitions will be visualized to reveal operational patterns such as typical trip structure and high-transition periods that may indicate operational stress.

**Feature Importance:** Sensors and derived quantities will be ranked by contribution to cluster separation using methods such as feature importance from Random Forest classifiers or permutation importance. This ranking serves dual purposes: informing Phase 2 feature selection and providing interpretability for domain experts assessing which signals define each operational state and which correlate most strongly with maintenance events.

### Phase 2 — Braking Event Analysis

**Event Extraction:** All windows labeled "braking" in Phase 1 output will be isolated and segmented into individual events. Per-event features will be extracted: peak deceleration, mean deceleration, total duration, and jerk profile. These features capture both severity and smoothness of braking action — the physically relevant dimensions for estimating mechanical stress on brake components.

**Intensity Classification:** A lightweight, interpretable classifier (k-NN or decision tree) will assign each braking event to an intensity category: light, heavy, or emergency. The focus on interpretable models is deliberate for maintenance contexts where category boundaries must be explainable in physical terms (e.g., peak deceleration thresholds) rather than only statistically validated. Class separability will be assessed, and each category characterized by typical deceleration profiles.

**Temporal Analysis & Validation Against Failures:** The full-year distribution of braking frequency and intensity will be analyzed to identify patterns relevant to maintenance planning. This analysis will be validated against the 7 documented brake-related failures in the dataset, examining whether:
1. Periods preceding failures show increased braking intensity or frequency
2. Degradation trends are detectable in braking smoothness or deceleration profiles
3. Seasonal or route-based load variations correlate with maintenance timing

Findings will be presented as visualizations and an analytical summary, framed as preliminary evidence for maintenance scheduling rather than definitive predictions.

**Connecting to Maintenance Reality:** The MetroAT dataset includes binary indicators (`TRAIN_IS_IN_FAILURE`, `TRAIN_IS_IN_MAINTENANCE`) and categorical fields (`TRAIN_FAILURE_TYPE`, `TRAIN_MAINTENANCE_TYPE`) that mark exact timestamps of maintenance activities and failures. This enables direct correlation between operational patterns identified in the analysis and actual maintenance outcomes. Additional maintenance data (duration, extent, material use, deferred actions) will be requested from Wiener Linien to strengthen the connection between operational indicators and maintenance reality. If unavailable, analysis will proceed with documented failure and maintenance timestamps as ground truth for validation.

---

## Documentation & Reproducibility

- **Code Repository:** GitHub repository with complete analysis pipeline (data loading, preprocessing, clustering, classification, visualization). README documenting environment setup, data access, reproduction steps, and expected outputs.
- **Reproducibility Standards:** Fixed random seeds for all stochastic operations (clustering initialization, train-test splits). Versioned dependencies (requirements.txt with its own environment). Documented train-test split strategy with consistent fold definitions. Containerization (Docker) if required by supervisor.
- **Documentation:** Inline code comments explaining non-obvious implementation choices. Jupyter notebooks for exploratory analysis with markdown explanations. Final project report documenting methodology, validation, findings, and limitations.
- **Data Access:** The MetroAT dataset will be publicly available via GitHub (https://github.com/steiner-andreas/metroAT) within two weeks.

---

## Process Overview

**Phase 1:** P1 → RQ1/RQ2 → O1/O2  
**Phase 2:** P2 → RQ3/RQ4 → O3/O4

---

## Expected Contributions

- Methodology for identifying maintenance-relevant operational patterns from multivariate sensor data
- Validation framework connecting operational indicators to documented maintenance outcomes
- Braking intensity classification system with evidence of degradation detection before failure
- Open-source implementation serving as foundation for future predictive maintenance research on metro systems