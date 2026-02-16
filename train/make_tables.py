# train/make_tables.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    # Overall performance
    "episode_reward",
    "avg_latency_ms",
    "p95_latency_ms",
    "avg_loss",
    # Attack-only effectiveness
    "attack_avg_suppression",
    "attack_avg_retention",
    "mitigation_time_s",
    # “Detection / FP proxies”
    "attack_detection_rate_0p90",
    "benign_non_aggressive_rate",
    # Behavior
    "aggressive_rate",
    "block_rate",
]


def fmt_mean_sd(mean: float, sd: float, decimals: int = 3) -> str:
    if np.isnan(mean):
        return "NA"
    if np.isnan(sd):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for two independent samples (pooled SD)."""
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    if sp == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / sp


def mannwhitney_u_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    """
    Mann–Whitney U test (two-sided) p-value.
    Uses scipy if available; otherwise returns NA.
    """
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    try:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(x, y, alternative="two-sided")
        return float(p)
    except Exception:
        return np.nan


def main():
    parser = argparse.ArgumentParser(description="Create thesis-ready summary tables from results.csv")
    parser.add_argument("--in_csv", type=str, default="experiments/results.csv")
    parser.add_argument("--out_dir", type=str, default="experiments")
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--metrics", type=str, nargs="*", default=DEFAULT_METRICS)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    in_csv = Path(args.in_csv)
    if not in_csv.is_absolute():
        in_csv = project_root / in_csv
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)

    if "policy" not in df.columns:
        raise ValueError("results.csv must include a 'policy' column.")

    policies = sorted(df["policy"].unique())
    if "baseline" not in policies or "ppo" not in policies:
        # Still works, but effect sizes assume PPO vs baseline.
        pass

    # ---- Summary Mean ± SD table ----
    rows: List[Dict[str, str]] = []
    for metric in args.metrics:
        if metric not in df.columns:
            continue

        row = {"Metric": metric}
        for pol in policies:
            vals = df.loc[df["policy"] == pol, metric].to_numpy(dtype=float)
            mean = float(np.nanmean(vals))
            sd = float(np.nanstd(vals, ddof=1)) if np.sum(~np.isnan(vals)) >= 2 else np.nan
            row[pol] = fmt_mean_sd(mean, sd, decimals=args.decimals)
        rows.append(row)

    table = pd.DataFrame(rows)

    # Save CSV
    out_csv = out_dir / "table_summary_mean_sd.csv"
    table.to_csv(out_csv, index=False)

    # Save Markdown
    out_md = out_dir / "table_summary_mean_sd.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(table.to_markdown(index=False))

    # Save LaTeX
    out_tex = out_dir / "table_summary_mean_sd.tex"
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(
            table.to_latex(
                index=False,
                escape=True,
                caption="Summary results (Mean ± SD) by policy.",
                label="tab:summary_mean_sd",
            )
        )

    # ---- Effect sizes + significance (PPO vs Baseline) ----
    eff_rows = []
    sig_rows = []

    if "ppo" in policies and "baseline" in policies:
        for metric in args.metrics:
            if metric not in df.columns:
                continue

            x = df.loc[df["policy"] == "ppo", metric].to_numpy(dtype=float)
            y = df.loc[df["policy"] == "baseline", metric].to_numpy(dtype=float)

            d = cohen_d(x, y)
            p = mannwhitney_u_pvalue(x, y)

            eff_rows.append({"Metric": metric, "Cohens_d (PPO - Baseline)": d})
            sig_rows.append({"Metric": metric, "MannWhitneyU_pvalue": p})

    eff_df = pd.DataFrame(eff_rows)
    sig_df = pd.DataFrame(sig_rows)

    out_eff = out_dir / "table_effect_sizes.csv"
    out_sig = out_dir / "table_significance_mannwhitney.csv"
    eff_df.to_csv(out_eff, index=False)
    sig_df.to_csv(out_sig, index=False)

    print("\n[INFO] Tables created:")
    print(f" - {out_csv}")
    print(f" - {out_md}")
    print(f" - {out_tex}")
    if len(eff_df) > 0:
        print(f" - {out_eff}")
        print(f" - {out_sig}")
        if sig_df["MannWhitneyU_pvalue"].isna().all():
            print("\n[NOTE] SciPy not found; p-values are NA. Install with: pip install scipy")

    print("\n[INFO] Tip: Use the .md table directly in Word (paste), or .tex in Overleaf.")


if __name__ == "__main__":
    main()
