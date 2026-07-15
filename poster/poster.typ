// Project poster, DIN A0 landscape (1189 x 841 mm), 12.7 mm margins.
// Header fonts and styles follow the TU Wien Informatics poster template
// (Roboto Light/Medium at template sizes), adapted to landscape.

#let accent = rgb("#006699")
#let ink = rgb("#1a1a1a")
#let softgray = rgb("#f2f2f2")
#let midgray = rgb("#666666")

#set page(width: 1189mm, height: 841mm, margin: 0mm)
#set text(font: "Roboto", weight: 300, size: 20pt, fill: ink)
#set par(leading: 0.55em, justify: false)

// ---------------------------------------------------------------- header ---
#let header-height = 148mm

// TU Wien Informatics logo, top left
#place(top + left, dx: 25.9mm, dy: 24mm, image("assets/tu_informatics_logo.svg", width: 235mm))

// TU Wien logo, top right (Institute of Management Science, Faculty of
// Mechanical and Industrial Engineering, uses the central TU Wien mark)
#place(top + right, dx: -25.9mm, dy: 24mm, image("assets/tu_wien_cube.svg", height: 60mm))

// Title (template style: Roboto Medium)
#place(top + left, dx: 25.9mm, dy: 78mm, box(width: 760mm,
  text(font: "Roboto", weight: 500, size: 44pt, fill: ink,
    par(leading: 0.35em)[Operational State Recognition and Braking Behaviour Analysis \ for Predictive Maintenance on Metro Trains])
))

// Name and Master's program (template style: Roboto Light)
#place(top + left, dx: 25.9mm, dy: 117mm,
  text(size: 32pt, weight: 300)[Maximilian Scheiblauer, BSc])
#place(top + left, dx: 25.9mm, dy: 131mm,
  text(size: 32pt, weight: 300)[Master Programme Data Science])

// Institute / supervisor block, right-aligned (template style: 28 pt)
#place(top + right, dx: -25.9mm, dy: 94mm, box(width: 560mm,
  align(right, par(leading: 0.62em)[
    #text(size: 26pt, weight: 500)[Institute of Management Science] \
    #text(size: 26pt, weight: 300)[Institute of Information Systems Engineering] \
    #text(size: 26pt, weight: 300)[Supervisors: Dr. Andreas Steiner, Dr. Florina Piroi] \
    #text(size: 26pt, weight: 300)[Contact: e11776651\@student.tuwien.ac.at]
  ])
))

// --------------------------------------------------------------- helpers ---

#let sect(title, body) = block(width: 100%, breakable: false, {
  block(width: 100%, below: 4.5mm, {
    text(font: "Roboto", weight: 500, size: 28pt, fill: accent, title)
    v(2mm)
    line(length: 100%, stroke: 1.6pt + accent)
  })
  body
})

#let fig(path, caption, width: 100%) = block(width: 100%, above: 4.5mm, below: 5mm, {
  align(center, image(path, width: width))
  v(1.2mm)
  align(center, box(width: 96%, text(size: 15.5pt, fill: midgray, style: "italic", caption)))
})

#let kpi(value, label) = box(width: 100%, fill: softgray, inset: 5mm, radius: 2mm, {
  align(center)[
    #text(font: "Roboto", weight: 500, size: 30pt, fill: accent, value) \
    #v(1mm)
    #text(size: 15.5pt, fill: midgray, label)
  ]
})

#set list(marker: text(fill: accent)[•], indent: 0mm, body-indent: 2.5mm, spacing: 0.72em)

// --------------------------------------------------------------- content ---

#let column-height = 841mm - header-height - 2 * 12.7mm  // 667.0mm

#place(top + left, dx: 12.7mm, dy: header-height + 12.7mm, box(width: 1163.6mm, height: column-height,
  grid(columns: (1fr, 1fr, 1fr, 1fr), column-gutter: 12.7mm, row-gutter: 0mm,

  // ------------------------------------------------------------ column 1 ---
  box(height: column-height)[
    #sect[Context & Problem][
      - Much rail maintenance runs on fixed intervals, regardless of how hard components were actually used. Predictive maintenance ties each intervention to measured condition instead.
      - Metro friction brakes are a natural test case: the train brakes at almost every stop, hundreds of times a day, so wear should leave a large, repeating signal.
      - *Data:* MetroAT, one year of 1 Hz recordings from a single Wiener Linien metro train (109 channels: speed, pneumatic system, mode signals), annotated with documented failure and maintenance events. All sensor values are min-max normalised, so the analysis works in relative units.
    ]

    #v(1fr)

    #sect[Research Questions][
      + Do distinct operational regimes emerge from the motion data alone, without fixing their number in advance?
      + Can the air-system sensors recover the deceleration regime using only signals that did not help define it?
      + Is braking intensity a set of discrete classes or a continuum, and how much of it do the air-system sensors explain?
      + Does braking behaviour change detectably before documented brake-system failures?
    ]

    #v(1fr)

    #sect[Method][
      - 30 M raw rows → 1.49 M non-overlapping 10 s windows with per-channel summary statistics.
      - Time-based split: training June 2024 to February 2025, held-out window February to June 2025. Every reported score is out-of-sample.
      - *Leakage control by feature roles:* motion channels (speed, acceleration, jerk, deceleration energy) define the targets and are never predictors; only air-system channels (actuation + auxiliary) predict.
      - 214 near-duplicate per-wagon sensor pairs (|r| > 0.98) absorbed by PCA instead of manual deletion.
    ]

    #v(1fr)

    #sect[Operational States][
      Motion-only k-means; the cluster count was selected by internal validation, not fixed in advance.

      #fig("../results/plots/report/kinematic_kselection.png")[Cluster-count selection: the largest BIC improvement is the step k = 3 → 4; methods agree at k = 4 (ARI 0.77), stable under resampling (bootstrap ARI 0.87).]

      #table(
        columns: (1.5fr, 1fr, 1fr, 1fr),
        stroke: 0.5pt + rgb("#cccccc"),
        inset: 3mm,
        align: (left, right, right, right),
        table.header(
          text(weight: 500)[State], text(weight: 500)[Share], text(weight: 500)[Med. vel.], text(weight: 500)[Med. acc.]
        ),
        [standing], [48.1 %], [0.000], [0.000],
        [deceleration], [20.3 %], [0.343], [−0.033],
        [cruising], [17.3 %], [0.842], [0.002],
        [accelerating], [14.3 %], [0.305], [0.041],
      )
    ]
  ],

  // ------------------------------------------------------------ column 2 ---
  box(height: column-height)[
    #sect[Plausibility of the Four States][
      - An earlier state definition that included brake pressures mislabelled a standing train holding its brake as "braking". Under the motion-only definition the deceleration state is 99.9 % genuine deceleration.
      - A Random Forest given all features (0.994 accuracy) still ranks the kinematic ones first, which confirms the regimes are motion-defined rather than a pneumatic artefact.

      #fig("../results/plots/report/kinematic_vel_accel_scatter.png", width: 64%)[The velocity-acceleration cycle, coloured by state: the train moves anticlockwise through stop, accelerate, cruise, decelerate. A minority of off-curve points motivate future work.]
    ]

    #v(1fr)

    #sect[Transitions & Dwell][
      #fig("../results/plots/report/transition_diagram.png", width: 62%)[The transition matrix is dominated by the operating cycle stop → accelerate → cruise → decelerate → stop.]
      Mean dwell times match the physical picture: standing ≈ 54 s, accelerating ≈ 17 s, cruising ≈ 21 s, deceleration ≈ 24 s. The state mix is stable from month to month, so the regimes reflect how the train is operated rather than any particular period.
    ]

    #v(1fr)

    #sect[Window Sensitivity][
      #fig("../results/plots/report/window_sensitivity.png")[Re-clustering at 5, 10, 15 and 20 s windows: state fractions are stable (deceleration ≈ 20 % throughout); label chatter rises with window length (0.19 → 0.54).]
      The 10 s window is a sound compromise between resolution and stability, and short windows are not the source of label churn.
    ]
  ],

  // ------------------------------------------------------------ column 3 ---
  box(height: column-height)[
    #sect[Recovering Deceleration from Air Sensors][
      Can the pneumatic sensors, which never touched the state definition, recover the deceleration state on held-out data?

      #grid(columns: (1fr, 1fr), column-gutter: 6mm,
        kpi[AUC 0.954][full tier random forest (all air-system channels)],
        kpi[AUC 0.921][auxiliary only (brake command withheld)],
      )
      #v(3.5mm)
      - Balanced accuracy 0.869, recall 0.956 for the deceleration class (full tier).
      - Interpretable models agree on where the signal sits: pneumatic braking force and proportional-valve pressure dominate the decision tree and LDA.
      - Deceleration leaves an independent fingerprint across the pneumatic system. The braking analysis rests on this premise.
    ]

    #v(1fr)

    #sect[Braking Events: Classes or Continuum?][
      A deceleration event is one continuous stretch of the deceleration state.

      #grid(columns: (1fr, 1fr, 1fr), column-gutter: 5mm,
        kpi[126,700][deceleration events in the training window],
        kpi[23.4 s][mean event duration (median 20 s)],
        kpi[99.9 %][genuine deceleration after removing held-brake windows],
      )

      #fig("../results/plots/report/cluster_model_selection.png")[Model selection on intensity features: BIC keeps improving without settling, silhouette singles out no k, and the k = 5 stability bump only peels off a negligibly small outlier cluster.]

      - The intensity histogram shows a dominant mild mode and a broader firmer-braking hump, but they overlap without a clean gap and no clustering confirms the visual split.
      - *Result:* braking intensity is a continuum, varying by degree rather than by kind. No light / hard / emergency labels are supported by the data; any such split is an operational threshold requiring domain input.
    ]

    #v(1fr)

    #sect[Intensity Regression & Feature Importance][
      Deceleration magnitude is predicted from the non-speed sensors on held-out events (baseline R² ≈ 0).

      #fig("../results/plots/report/decel_regression_scatter.png")[Predicted vs. actual intensity (random forest, full tier): positively associated, but compressed toward the mean. The pneumatic signal is real yet partial.]

      #grid(columns: (1fr, 1fr, 1fr), column-gutter: 5mm,
        kpi[R² 0.57][mean deceleration (RF); linear model: 0.20],
        kpi[R² 0.43][peak deceleration (RF); linear model: 0.24],
        kpi[85 %][of specific braking energy ½(v₀² − v₁²) explained],
      )
      #v(3.5mm)
      - A cross-validated Lasso keeps 19 of 22 air-system features across subsystems: the intensity signal is distributed, not carried by the brake command alone.
    ]
  ],

  // ------------------------------------------------------------ column 4 ---
  box(height: column-height)[
    #sect[Maintenance & Failure Analysis][
      *Braking energy vs. the maintenance calendar.*
      #fig("../results/plots/report/maintenance_interval_energy.png", width: 98%)[Deceleration energy summed per between-maintenance interval. ≈ 96 % of all real-deceleration energy falls inside the intervals. Every interval longer than ≈ 3 weeks contained a brake failure; every shorter one was failure-free. The split is suggestive, but it is confounded with exposure time and rests on ten intervals.]

      *Change-point detection.*
      #fig("../results/plots/report/prefailure_cusum.png")[CUSUM on the weekly mean of energy-braking-resistance (strongest pre-failure channel), reset at each maintenance date: four change-points fire, only one within seven days of a documented failure.]

      *Statistical null results.*
      - Pre-failure vs. baseline deceleration events (Mann-Whitney U, Bonferroni-corrected): p-values are significant only through sample size (> 23,000 events). Effect sizes are negligible: all braking-kinematic and brake-command features sit at Cliff's |δ| < 0.06, and only energy-braking-resistance reaches |δ| ≈ 0.20.
      - Pre-event operational-state mix: 61 of 62 chi-square tests significant, but Cramér's V ≤ 0.07 (typically 0.018), and ≈ 0.003 once standing windows are excluded. The shift only reflects depot idling before scheduled work.
      - With eight brake failures, no reliable pre-failure braking indicator exists in this data; the result is reported as a genuine negative.

      #v(3mm)
      #table(
        columns: (2fr, 1fr, 1.2fr),
        stroke: 0.5pt + rgb("#cccccc"),
        inset: 3mm,
        align: (left, right, center),
        table.header(
          text(weight: 500)[Feature (pre-failure vs. baseline)], text(weight: 500)[Cliff's δ], text(weight: 500)[Above negligible?]
        ),
        [energy-braking-resistance (mean)], [−0.201], [yes],
        [load pressure (mean)], [−0.103], [no],
        [jerk RMS], [+0.054], [no],
        [brake-cylinder pressure (integral)], [−0.040], [no],
        [peak deceleration], [−0.005], [no],
      )
    ]

    #v(1fr)

    #sect[Conclusions][
      - Motion alone yields four stable, physically meaningful operational states, robust to the window-length choice.
      - The air system independently recovers deceleration (AUC 0.95; 0.92 without the brake command) and explains a moderate share of intensity and 85 % of dissipated energy.
      - The link to failures is an honest null: no braking indicator separates the pre-failure week from baseline.
      - *Next:* characterise the off-cycle outliers, compare wagons instead of collapsing them, and probe idle/charging behaviour, ideally on data with more documented failures.
    ]

    #v(3.5mm)
    #text(size: 14.5pt, fill: midgray)[Data: MetroAT (TU Wien Research Data, DOI 10.48436/9ja0q-bq581). Full reproducible pipeline: github.com/maxscheiblauer/Interdisciplinary_Project]
  ],
  )
))
