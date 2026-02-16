# train/make_improvement_table.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# Metric direction: True = higher is better, False = lower is better
METRIC_DIRECTION = {
    "episode_reward": True,

    "avg_latency_ms": False,
    "p95_latency_ms": False,
    "avg_loss": False,

    "attack_avg_suppression": True,
    "attack_avg_retention": True,
    "mitigation_time_s": False,

    "attack_detection_rate_0p90": True,
    "benign_non_aggressive_rate": True,

    "aggressive_rate": False,
    "block_rate": False,
}


def delta_percent(ppo: float, base: float) -> float:
    if np.isnan(ppo) or np.isnan(base) or base == 0:
        return np.nan
    return ((ppo - base) / abs(base)) * 100.0


def label_improvement(delta: float, higher_is_better: bool) -> str:
    if np.isnan(delta):
        return "NA"
    if higher_is_better:
        return "Improved" if delta > 0 else "Degraded" if delta < 0 else "No Change"
    else:
        return "Improved" if delta < 0 else "Degraded" if delta > 0 else "No Change"


def main():
    parser = argparse.ArgumentParser(description="Create PPO vs Baseline improvement (Δ%) table.")
    parser.add_argument("--in_csv", type=str, default="experiments/results.csv")
    parser.add_argument("--out_dir", type=str, default="experiments")
    parser.add_argument("--decimals", type=int, default=2)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    in_csv = Path(args.in_csv)
    if not in_csv.is_absolute():
        in_csv = project_root / in_csv
    if not in_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {in_csv}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)

    if not {"policy"}.issubset(df.columns):
        raise ValueError("results.csv must include a 'policy' column.")

    summary = df.groupby("policy").mean(numeric_only=True)

    if "ppo" not in summary.index or "baseline" not in summary.index:
        raise ValueError("Both 'ppo' and 'baseline' must be present in results.csv")

    rows: List[Dict[str, str]] = []

    for metric, higher_is_better in METRIC_DIRECTION.items():
        if metric not in summary.columns:
            continue

        ppo_mean = float(summary.loc["ppo", metric])
        base_mean = float(summary.loc["baseline", metric])
        delta = delta_percent(ppo_mean, base_mean)
        label = label_improvement(delta, higher_is_better)

        rows.append({
            "Metric": metric,
            "Baseline Mean": round(base_mean, args.decimals),
            "PPO Mean": round(ppo_mean, args.decimals),
            "Δ% (PPO vs Baseline)": f"{delta:.{args.decimals}f}" if not np.isnan(delta) else "NA",
            "Outcome": label,
        })

    table = pd.DataFrame(rows)

    # Save CSV
    out_csv = out_dir / "table_improvement_delta.csv"
    table.to_csv(out_csv, index=False)

    # Save Markdown
    out_md = out_dir / "table_improvement_delta.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(table.to_markdown(index=False))

    # Save LaTeX
    out_tex = out_dir / "table_improvement_delta.tex"
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(
            table.to_latex(
                index=False,
                escape=True,
                caption="Percentage improvement (Δ%) of PPO over static baseline.",
                label="tab:ppo_improvement_delta",
            )
        )

    print("\n[INFO] Improvement tables created:")
    print(f" - {out_csv}")
    print(f" - {out_md}")
    print(f" - {out_tex}")
    print("\n[INFO] Use the Markdown table directly in Word for best formatting.")


if __name__ == "__main__":
    main()
