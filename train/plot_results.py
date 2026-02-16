# train/plot_results.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(fig, out_path: Path, dpi: int = 300) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def load_results(project_root: Path, csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.is_absolute():
        p = project_root / p
    if not p.exists():
        raise FileNotFoundError(f"results.csv not found at: {p}")
    df = pd.read_csv(p)
    if "policy" not in df.columns:
        raise ValueError("CSV must contain a 'policy' column.")
    return df


def plot_episode_scatter(df: pd.DataFrame, metric: str, title: str, ylabel: str, out_file: Path) -> None:
    fig = plt.figure()
    for policy in sorted(df["policy"].unique()):
        d = df[df["policy"] == policy].sort_values("episode")
        plt.scatter(d["episode"], d[metric], label=policy, alpha=0.8)

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    save_fig(fig, out_file)


def plot_box(df: pd.DataFrame, metric: str, title: str, ylabel: str, out_file: Path) -> None:
    fig = plt.figure()
    policies = [p for p in sorted(df["policy"].unique())]
    data = [df[df["policy"] == p][metric].dropna().values for p in policies]
    plt.boxplot(data, labels=policies, showmeans=True)
    plt.title(title)
    plt.ylabel(ylabel)
    save_fig(fig, out_file)


def plot_bar_mean_std(df: pd.DataFrame, metric: str, title: str, ylabel: str, out_file: Path) -> None:
    fig = plt.figure()
    g = df.groupby("policy")[metric]
    means = g.mean()
    stds = g.std()

    x = np.arange(len(means.index))
    plt.bar(x, means.values, yerr=stds.values, capsize=6)
    plt.xticks(x, means.index.tolist())
    plt.title(title)
    plt.ylabel(ylabel)
    save_fig(fig, out_file)


def main():
    parser = argparse.ArgumentParser(description="Generate thesis figures from experiments/results.csv")
    parser.add_argument("--in_csv", type=str, default="experiments/results.csv")
    parser.add_argument("--out_dir", type=str, default="figures")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    ensure_dir(out_dir)

    df = load_results(project_root, args.in_csv)

    # ---- Sanity checks ----
    required_cols = [
        "episode_reward",
        "avg_latency_ms",
        "avg_loss",
        "attack_avg_suppression",
        "attack_avg_retention",
        "mitigation_time_s",
        "attack_detection_rate_0p90",
        "benign_non_aggressive_rate",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results.csv: {missing}")

    # ---- Figures (no subplots, thesis-clean) ----

    # 1) Episode reward (scatter + box)
    plot_episode_scatter(
        df,
        metric="episode_reward",
        title="Episode Reward by Policy",
        ylabel="Total Episode Reward",
        out_file=out_dir / "fig_reward_scatter.png",
    )
    plot_box(
        df,
        metric="episode_reward",
        title="Episode Reward Distribution by Policy",
        ylabel="Total Episode Reward",
        out_file=out_dir / "fig_reward_box.png",
    )

    # 2) Average latency (scatter + box)
    plot_episode_scatter(
        df,
        metric="avg_latency_ms",
        title="Average Latency by Policy",
        ylabel="Average Latency (ms)",
        out_file=out_dir / "fig_latency_avg_scatter.png",
    )
    plot_box(
        df,
        metric="avg_latency_ms",
        title="Average Latency Distribution by Policy",
        ylabel="Average Latency (ms)",
        out_file=out_dir / "fig_latency_avg_box.png",
    )

    # 3) Average loss (scatter + box)
    plot_episode_scatter(
        df,
        metric="avg_loss",
        title="Average Packet Loss by Policy",
        ylabel="Average Loss (0–1)",
        out_file=out_dir / "fig_loss_avg_scatter.png",
    )
    plot_box(
        df,
        metric="avg_loss",
        title="Average Packet Loss Distribution by Policy",
        ylabel="Average Loss (0–1)",
        out_file=out_dir / "fig_loss_avg_box.png",
    )

    # 4) Attack-only suppression (mean/std bar + box)
    plot_bar_mean_std(
        df,
        metric="attack_avg_suppression",
        title="Attack Suppression (Mean ± SD) by Policy",
        ylabel="Average Suppression During Attack (0–1)",
        out_file=out_dir / "fig_attack_suppression_bar.png",
    )
    plot_box(
        df,
        metric="attack_avg_suppression",
        title="Attack Suppression Distribution by Policy",
        ylabel="Average Suppression During Attack (0–1)",
        out_file=out_dir / "fig_attack_suppression_box.png",
    )

    # 5) Attack-only legit retention (mean/std bar + box)
    plot_bar_mean_std(
        df,
        metric="attack_avg_retention",
        title="Legitimate Traffic Retention During Attack (Mean ± SD)",
        ylabel="Average Retention During Attack (0–1)",
        out_file=out_dir / "fig_attack_retention_bar.png",
    )
    plot_box(
        df,
        metric="attack_avg_retention",
        title="Legitimate Traffic Retention Distribution (During Attack)",
        ylabel="Average Retention During Attack (0–1)",
        out_file=out_dir / "fig_attack_retention_box.png",
    )

    # 6) Mitigation time (bar mean/std + box)
    # mitigation_time_s may contain NaN (if never mitigated), so keep dropna internally
    plot_bar_mean_std(
        df.dropna(subset=["mitigation_time_s"]),
        metric="mitigation_time_s",
        title="Mitigation Time (Mean ± SD) by Policy",
        ylabel="Mitigation Time (seconds)",
        out_file=out_dir / "fig_mitigation_time_bar.png",
    )
    plot_box(
        df.dropna(subset=["mitigation_time_s"]),
        metric="mitigation_time_s",
        title="Mitigation Time Distribution by Policy",
        ylabel="Mitigation Time (seconds)",
        out_file=out_dir / "fig_mitigation_time_box.png",
    )

    # 7) Detection proxy (attack_detection_rate_0p90)
    plot_bar_mean_std(
        df,
        metric="attack_detection_rate_0p90",
        title="Attack Detection Proxy (Suppression ≥ 0.90) Mean ± SD",
        ylabel="Rate (0–1)",
        out_file=out_dir / "fig_detection_proxy_bar.png",
    )
    plot_box(
        df,
        metric="attack_detection_rate_0p90",
        title="Attack Detection Proxy Distribution (Suppression ≥ 0.90)",
        ylabel="Rate (0–1)",
        out_file=out_dir / "fig_detection_proxy_box.png",
    )

    # 8) Benign non-aggressive rate (false-positive style proxy)
    plot_bar_mean_std(
        df,
        metric="benign_non_aggressive_rate",
        title="Benign Non-Aggressive Rate (Higher is Better) Mean ± SD",
        ylabel="Rate (0–1)",
        out_file=out_dir / "fig_benign_non_aggressive_bar.png",
    )
    plot_box(
        df,
        metric="benign_non_aggressive_rate",
        title="Benign Non-Aggressive Rate Distribution",
        ylabel="Rate (0–1)",
        out_file=out_dir / "fig_benign_non_aggressive_box.png",
    )

    print("\n[INFO] Figures saved to:", out_dir)
    print("[INFO] Generated files:")
    for p in sorted(out_dir.glob("fig_*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
