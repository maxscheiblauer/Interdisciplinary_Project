"""Data profiling and EDA for the MetroAT dataset.

The data is already Hive-partitioned Parquet (no zip / CSV stage). Streams the
train set one daily file at a time (never loads everything), accumulating exact
global statistics and a 5% row sample used for distribution plots and
correlations. Produces the schema, summary stats, operational-state breakdown,
failure/maintenance inventory, data-quality report and EDA plots under
logs/data_profiling/ and results/.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import psutil  # noqa: E402
from loguru import logger  # noqa: E402

from metroat.io import discover_files  # noqa: E402
from metroat import schema as S  # noqa: E402

TRAIN_ROOT = ROOT / "train"
LOG_DIR = ROOT / "logs" / "data_profiling"
TBL_DIR = ROOT / "results" / "tables"
EDA_DIR = ROOT / "results" / "plots" / "eda"
for d in (LOG_DIR, TBL_DIR, EDA_DIR):
    d.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add(LOG_DIR / "profiling.log", level="INFO")

SAMPLE_FRAC = 0.05
RNG = np.random.default_rng(42)
proc = psutil.Process()


def rss_gb() -> float:
    return proc.memory_info().rss / 1e9


# --------------------------------------------------------------------------- #
# Schema discovery
# --------------------------------------------------------------------------- #
def _jsonsafe(v):
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if hasattr(v, "item"):
        return v.item()
    return v


def step_schema(files: list[Path]) -> dict:
    out = LOG_DIR / "schema.json"
    sample = pd.concat(
        [pd.read_parquet(f, engine="pyarrow") for f in files[:3]], ignore_index=True
    )
    info: dict[str, dict] = {}
    op_candidates = [
        c
        for c in sample.columns
        if c.startswith("TRAIN_")
        and c not in S.FAILURE_COLS
        and c != S.VELOCITY_COL
    ]
    for col in sample.columns:
        s = sample[col]
        uniques = s.dropna().unique()
        is_binary = False
        if pd.api.types.is_numeric_dtype(s) and len(uniques):
            is_binary = set(np.asarray(uniques).ravel().tolist()).issubset({0.0, 1.0})
        if col in S.TIMESTAMP_COLS:
            cat = "timestamp"
        elif col in S.FAILURE_COLS:
            cat = "failure_maintenance"
        elif col in op_candidates:
            cat = "operational_state"
        elif is_binary:
            cat = "binary_sensor"
        else:
            cat = "continuous_sensor"
        info[col] = {
            "dtype": str(s.dtype),
            "n_unique_nonnull": int(len(uniques)),
            "is_binary_0_1": bool(is_binary),
            "sample_values": [_jsonsafe(v) for v in uniques[:8]],
            "category": cat,
        }
    counts = Counter(v["category"] for v in info.values())
    schema = {
        "n_files_train": len(files),
        "sample_rows": int(sample.shape[0]),
        "category_counts": dict(counts),
        "columns": info,
    }
    out.write_text(json.dumps(schema, indent=2))
    logger.info(f"schema.json written | categories={dict(counts)}")
    return schema


# --------------------------------------------------------------------------- #
# Streaming pass: exact stats + event inventory + 5% sample
# --------------------------------------------------------------------------- #
class Welford:
    __slots__ = ("n", "mean", "M2", "min", "max", "nnull")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = np.inf
        self.max = -np.inf
        self.nnull = 0

    def update(self, arr: np.ndarray, total_len: int):
        self.nnull += total_len - arr.size
        if arr.size == 0:
            return
        self.min = min(self.min, float(arr.min()))
        self.max = max(self.max, float(arr.max()))
        # batched Welford
        b_n = arr.size
        b_mean = float(arr.mean())
        b_M2 = float(((arr - b_mean) ** 2).sum())
        delta = b_mean - self.mean
        tot = self.n + b_n
        self.mean += delta * b_n / tot
        self.M2 += b_M2 + delta**2 * self.n * b_n / tot
        self.n = tot

    @property
    def std(self):
        return float(np.sqrt(self.M2 / self.n)) if self.n > 1 else 0.0


def step_stream(files: list[Path], schema: dict):
    cont = S.continuous_cols(schema)
    binr = S.binary_cols(schema)
    oper = S.operational_cols(schema)
    bc_cols = S.brake_cylinder_cols(schema)

    stats = {c: Welford() for c in cont}
    binsum = {c: 0.0 for c in binr}
    bincount = {c: 0 for c in binr}
    # operational-state value accumulation handled via sample (for medians)
    fail_rows = []  # (timestamp, failure_type)
    maint_rows = []  # (timestamp, maint_type)
    fail_vocab = Counter()
    maint_vocab = Counter()
    monthly = defaultdict(lambda: {"mr": [], "bc": []})  # month -> sampled values
    samples = []
    total_rows = 0

    for i, f in enumerate(files):
        df = pd.read_parquet(f, engine="pyarrow")
        df = df.sort_values("TIMESTAMP").reset_index(drop=True)
        n = len(df)
        total_rows += n

        for c in cont:
            col = df[c].to_numpy(dtype="float64")
            stats[c].update(col[np.isfinite(col)], n)
        for c in binr:
            col = df[c].to_numpy(dtype="float64")
            v = col[np.isfinite(col)]
            binsum[c] += float(v.sum())
            bincount[c] += int(v.size)

        fv = df["TRAIN_FAILURE_TYPE"].value_counts()
        for k, val in fv.items():
            fail_vocab[k] += int(val)
        mv = df["TRAIN_MAINTENANCE_TYPE"].value_counts()
        for k, val in mv.items():
            maint_vocab[k] += int(val)

        fmask = df["TRAIN_IS_IN_FAILURE"].fillna(False).to_numpy(bool)
        if fmask.any():
            sub = df.loc[fmask, ["TIMESTAMP", "TRAIN_FAILURE_TYPE"]]
            fail_rows.extend(zip(sub["TIMESTAMP"], sub["TRAIN_FAILURE_TYPE"]))
        mmask = df["TRAIN_IS_IN_MAINTENANCE"].fillna(False).to_numpy(bool)
        if mmask.any():
            sub = df.loc[mmask, ["TIMESTAMP", "TRAIN_MAINTENANCE_TYPE"]]
            maint_rows.extend(zip(sub["TIMESTAMP"], sub["TRAIN_MAINTENANCE_TYPE"]))

        # 5% sample
        k = max(1, int(n * SAMPLE_FRAC))
        idx = RNG.choice(n, size=k, replace=False)
        samples.append(df.iloc[np.sort(idx)])

        if i % 30 == 0 or i == len(files) - 1:
            logger.info(
                f"{i+1}/{len(files)} {f.parent.name} rows={n} "
                f"cumrows={total_rows:,} RSS={rss_gb():.2f}GB"
            )

    sample = pd.concat(samples, ignore_index=True)
    logger.info(f"total_rows={total_rows:,} sample_rows={len(sample):,}")
    return dict(
        cont=cont, binr=binr, oper=oper, bc_cols=bc_cols,
        stats=stats, binsum=binsum, bincount=bincount,
        fail_rows=fail_rows, maint_rows=maint_rows,
        fail_vocab=fail_vocab, maint_vocab=maint_vocab,
        sample=sample, total_rows=total_rows,
    )


# --------------------------------------------------------------------------- #
# full stats
# --------------------------------------------------------------------------- #
def step_full_stats(R):
    rows = []
    for c, w in R["stats"].items():
        rows.append(dict(
            column=c, n=w.n, n_null=w.nnull,
            null_pct=100 * w.nnull / (w.n + w.nnull) if (w.n + w.nnull) else 0,
            min=w.min, max=w.max, mean=w.mean, std=w.std,
        ))
    fs = pd.DataFrame(rows).sort_values("column")
    fs.to_csv(LOG_DIR / "full_stats.csv", index=False)
    fs.to_csv(TBL_DIR / "sensor_summary.csv", index=False)
    logger.info(f"full_stats.csv + sensor_summary.csv ({len(fs)} continuous cols)")
    return fs


# --------------------------------------------------------------------------- #
# operational state analysis
# --------------------------------------------------------------------------- #
def step_operational(R):
    sample = R["sample"]
    vel = S.VELOCITY_COL
    bc = R["bc_cols"]
    sample = sample.copy()
    sample["_bc_mean"] = sample[bc].mean(axis=1)
    rows = []
    for var in R["oper"]:
        vc = sample[var].value_counts(dropna=False)
        for val, cnt in vc.items():
            m = sample[sample[var] == val] if pd.notna(val) else sample[sample[var].isna()]
            rows.append(dict(
                variable=var, value=_jsonsafe(val), count=int(cnt),
                pct=100 * cnt / len(sample),
                median_velocity=float(m[vel].median()),
                median_brake_pressure=float(m["_bc_mean"].median()),
            ))
    osa = pd.DataFrame(rows)
    osa.to_csv(LOG_DIR / "operational_state_analysis.csv", index=False)

    # plot: distribution of each operational var (top values)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, var in zip(axes.ravel(), R["oper"]):
        vc = sample[var].value_counts().head(15)
        ax.bar([str(x) for x in vc.index], vc.values, color="steelblue")
        ax.set_title(var, fontsize=9)
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        ax.set_ylabel("count (5% sample)")
    for ax in axes.ravel()[len(R["oper"]):]:
        ax.axis("off")
    fig.suptitle("Operational State Variable Distributions (5% train sample)")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "operational_state_distributions.png", dpi=150)
    plt.close(fig)
    logger.info("operational_state_analysis.csv + distribution plot")
    return osa


# --------------------------------------------------------------------------- #
# failure/maintenance inventory
# --------------------------------------------------------------------------- #
def _events_from_rows(rows, merge_gap_s=60):
    """rows: list of (timestamp, type). Returns fine activation windows: same
    type, consecutive 1 Hz samples merged across gaps <= merge_gap_s (handles
    sub-minute flicker). Type changes always split."""
    if not rows:
        return []
    df = pd.DataFrame(rows, columns=["ts", "type"]).dropna(subset=["ts"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["ts"]).reset_index(drop=True)
    gap = df["ts"].diff().dt.total_seconds().fillna(0)
    type_change = df["type"].ne(df["type"].shift()).fillna(True)
    new_event = ((gap > merge_gap_s) | type_change).cumsum()
    events = []
    for _, g in df.groupby(new_event):
        events.append(dict(
            start=g["ts"].iloc[0], end=g["ts"].iloc[-1],
            duration_s=(g["ts"].iloc[-1] - g["ts"].iloc[0]).total_seconds(),
            n_active_rows=len(g),
            type=g["type"].iloc[0],
        ))
    return events


def _merge_logical(events, gap_h=24):
    """Collapse fine activation windows of the SAME type into logical events
    (one maintenance visit / failure incident may toggle the flag many times).
    Two same-type windows merge if separated by <= gap_h hours."""
    if not events:
        return []
    ev = sorted(events, key=lambda e: (str(e["type"]), e["start"]))
    out = []
    cur = None
    for e in ev:
        if cur is None:
            cur = dict(e, n_activations=1)
        elif e["type"] == cur["type"] and (
            e["start"] - cur["end"]
        ).total_seconds() <= gap_h * 3600:
            cur["end"] = max(cur["end"], e["end"])
            cur["n_active_rows"] += e["n_active_rows"]
            cur["n_activations"] += 1
            cur["duration_s"] = (cur["end"] - cur["start"]).total_seconds()
        else:
            out.append(cur)
            cur = dict(e, n_activations=1)
    if cur is not None:
        out.append(cur)
    return sorted(out, key=lambda e: e["start"])


def step_events(R):
    vocab = {
        "TRAIN_FAILURE_TYPE": {str(k): int(v) for k, v in R["fail_vocab"].items()},
        "TRAIN_MAINTENANCE_TYPE": {str(k): int(v) for k, v in R["maint_vocab"].items()},
    }
    (LOG_DIR / "label_vocabularies.json").write_text(json.dumps(vocab, indent=2))

    # Fine activation windows (sub-minute flicker merged), then logical events.
    fail_act = _events_from_rows(R["fail_rows"], merge_gap_s=60)
    maint_act = _events_from_rows(R["maint_rows"], merge_gap_s=60)
    act_rows = (
        [{"event_class": "failure", **e} for e in fail_act]
        + [{"event_class": "maintenance", **e} for e in maint_act]
    )
    pd.DataFrame(act_rows).to_csv(LOG_DIR / "event_activations.csv", index=False)

    GAP_H = 24
    fail_events = _merge_logical(fail_act, gap_h=GAP_H)
    maint_events = _merge_logical(maint_act, gap_h=GAP_H)
    rows = []
    for e in fail_events:
        rows.append({"event_class": "failure", **e})
    for e in maint_events:
        rows.append({"event_class": "maintenance", **e})
    inv = pd.DataFrame(rows)
    inv.to_csv(LOG_DIR / "event_inventory.csv", index=False)
    n_fail = len(fail_events)
    n_maint = len(maint_events)
    n_brake = sum(1 for e in fail_events if e["type"] == "Brake System Failure")
    by_type = Counter(e["type"] for e in fail_events)
    logger.info(
        f"logical events (same-type, {GAP_H}h merge): {n_fail} failures "
        f"({n_brake} brake) + {n_maint} maintenance | from "
        f"{len(fail_act)}+{len(maint_act)} fine activations"
    )
    logger.info(f"failure types: {dict(by_type)}")
    logger.info(
        "NOTE train is an 8.5-month subset; exposé full-year totals are "
        "11 failures (7 brake) / 18 maintenance — flagged for review"
    )

    # timeline plot
    fig, ax = plt.subplots(figsize=(16, 4))
    fcolors = {"Brake System Failure": "red", "Leveling System Failure": "orange",
               "Compressor Module Failure": "purple"}
    for e in fail_events:
        ax.axvspan(e["start"], e["end"] if e["end"] > e["start"] else e["start"] + pd.Timedelta(hours=2),
                   color=fcolors.get(e["type"], "red"), alpha=0.6)
    for e in maint_events:
        ax.axvspan(e["start"], e["end"], color="green", alpha=0.25)
    handles = [plt.Line2D([0], [0], color=c, lw=6, label=t) for t, c in fcolors.items()]
    handles.append(plt.Line2D([0], [0], color="green", lw=6, alpha=0.4, label="Maintenance"))
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.set_title("Failure & Maintenance Timeline (train: Jun 2024 – Feb 2025)")
    ax.set_xlabel("date")
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(EDA_DIR / "eda_failure_timeline.png", dpi=150)
    plt.close(fig)
    logger.info("event_inventory.csv + label_vocabularies.json + timeline plot")
    return inv, fail_events, maint_events


# --------------------------------------------------------------------------- #
# data quality
# --------------------------------------------------------------------------- #
def step_quality(R, fs):
    sample = R["sample"]
    cont = R["cont"]
    # continuous quality
    rows = []
    for c in cont:
        w = R["stats"][c]
        col = sample[c].to_numpy("float64")
        col = col[np.isfinite(col)]
        if col.size and w.std > 0:
            out5 = int(np.sum(np.abs(col - w.mean) > 5 * w.std))
            out_pct = 100 * out5 / col.size
        else:
            out_pct = 0.0
        rows.append(dict(column=c, null_pct=w.nnull / (w.n + w.nnull) * 100 if (w.n + w.nnull) else 0,
                         min=w.min, max=w.max, mean=w.mean, std=w.std,
                         outlier_5sigma_pct_sample=out_pct,
                         negative=bool(w.min < 0)))
    dq = pd.DataFrame(rows).sort_values("null_pct", ascending=False)
    dq.to_csv(LOG_DIR / "data_quality.csv", index=False)

    # binary stats
    brows = []
    for c in R["binr"]:
        duty = R["binsum"][c] / R["bincount"][c] if R["bincount"][c] else np.nan
        brows.append(dict(column=c, n_nonnull=R["bincount"][c], duty_cycle=duty))
    bdf = pd.DataFrame(brows).sort_values("duty_cycle", ascending=False)
    bdf.to_csv(LOG_DIR / "binary_sensor_stats.csv", index=False)

    # aggregated-sensor detection: continuous cols with r>0.98 vs another continuous
    corr = sample[cont].corr()
    redundant = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) > 0.98:
                redundant.append((cols[i], cols[j], float(r)))
    pd.DataFrame(redundant, columns=["col_a", "col_b", "pearson_r"]).to_csv(
        LOG_DIR / "high_correlation_pairs.csv", index=False)
    logger.info(
        f"data_quality.csv + binary_sensor_stats.csv | "
        f"{len(redundant)} continuous pairs with |r|>0.98"
    )
    return dq, bdf


# --------------------------------------------------------------------------- #
# EDA plots
# --------------------------------------------------------------------------- #
def step_eda_plots(R, fs, dq, bdf):
    sample = R["sample"]
    cont = R["cont"]

    # 2. null % bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    d = dq.sort_values("null_pct", ascending=False)
    ax.bar(range(len(d)), d["null_pct"], color="indianred")
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["column"], rotation=90, fontsize=5)
    ax.set_ylabel("null %")
    ax.set_title("Continuous Sensor Null Percentage")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "eda_sensor_nulls.png", dpi=150)
    plt.close(fig)

    # 3. 72+ sensor histogram grid
    ncol = 9
    nrow = int(np.ceil(len(cont) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.2, nrow * 1.8))
    for ax, c in zip(axes.ravel(), cont):
        ax.hist(sample[c].dropna(), bins=40, color="steelblue")
        ax.set_title(c, fontsize=5)
        ax.tick_params(labelsize=4)
    for ax in axes.ravel()[len(cont):]:
        ax.axis("off")
    fig.suptitle("Continuous Sensor Distributions (5% train sample)")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "eda_continuous_sensor_distributions.png", dpi=150)
    plt.close(fig)

    # 4. binary duty cycles
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(bdf)), bdf["duty_cycle"], color="darkorange")
    ax.set_xticks(range(len(bdf)))
    ax.set_xticklabels(bdf["column"], rotation=90, fontsize=6)
    ax.set_ylabel("duty cycle (mean)")
    ax.set_title("Binary Sensor Duty Cycles")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "eda_binary_sensor_duty_cycles.png", dpi=150)
    plt.close(fig)

    # 6. top-10 variance sensors: during-failure vs normal density
    var_rank = fs.assign(v=fs["std"]).sort_values("v", ascending=False)
    top10 = var_rank["column"].head(10).tolist()
    in_fail = sample["TRAIN_IS_IN_FAILURE"].fillna(False).to_numpy(bool)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, c in zip(axes.ravel(), top10):
        a = sample.loc[~in_fail, c].dropna()
        b = sample.loc[in_fail, c].dropna()
        ax.hist(a, bins=40, density=True, alpha=0.5, label="normal", color="steelblue")
        if len(b) > 5:
            ax.hist(b, bins=40, density=True, alpha=0.5, label="in failure", color="red")
        ax.set_title(c, fontsize=7)
        ax.legend(fontsize=6)
    fig.suptitle("Top-10 Variance Sensors: Normal vs During-Failure (5% sample)")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "eda_failure_sensor_comparison.png", dpi=150)
    plt.close(fig)

    # 7. seasonal: monthly boxplots of main reservoir + brake cylinder pressure
    mr_cols = S.main_reservoir_cols()
    bc_cols = R["bc_cols"]
    s = sample.copy()
    s["_mr"] = s[mr_cols].mean(axis=1)
    s["_bc"] = s[bc_cols].mean(axis=1)
    s["_ym"] = s["year"].astype(int).astype(str) + "-" + s["month"].astype(int).map("{:02d}".format)
    order = sorted(s["_ym"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].boxplot([s.loc[s["_ym"] == m, "_mr"].dropna() for m in order], labels=order, showfliers=False)
    axes[0].set_ylabel("main reservoir pressure (norm.)")
    axes[0].set_title("Seasonal Patterns — Main Reservoir Pressure")
    axes[1].boxplot([s.loc[s["_ym"] == m, "_bc"].dropna() for m in order], labels=order, showfliers=False)
    axes[1].set_ylabel("brake cylinder pressure (norm.)")
    axes[1].set_title("Seasonal Patterns — Brake Cylinder Pressure")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "eda_seasonal_patterns.png", dpi=150)
    plt.close(fig)
    logger.info("EDA plots written (nulls, distributions, duty cycles, failure comparison, seasonal)")


def main():
    logger.info(f"START profiling | RSS={rss_gb():.2f}GB")
    files = discover_files(TRAIN_ROOT)
    schema = step_schema(files)
    R = step_stream(files, schema)
    fs = step_full_stats(R)
    osa = step_operational(R)
    inv, fe, me = step_events(R)
    dq, bdf = step_quality(R, fs)
    step_eda_plots(R, fs, dq, bdf)
    logger.info(f"DONE profiling | peak RSS={rss_gb():.2f}GB | total_rows={R['total_rows']:,}")


if __name__ == "__main__":
    main()
