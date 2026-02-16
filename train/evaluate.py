# train/evaluate.py
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from env.sim_env import AutoPPOFirewallEnv, EnvConfig


# ------------------------------
# Fair static baseline policy
# ------------------------------
@dataclass
class FairBaselineConfig:
    syn_ack_ratio_thresh: float = 3.5
    pkt_rate_thresh: float = 9000.0
    udp_share_thresh: float = 0.55
    icmp_share_thresh: float = 0.20
    prefer_block: bool = False
    meltdown_load_thresh: float = 1.2
    meltdown_latency_ms: float = 200.0


class FairStaticFirewallPolicy:
    def __init__(self, env_cfg: EnvConfig, cfg: Optional[FairBaselineConfig] = None):
        self.env_cfg = env_cfg
        self.cfg = cfg or FairBaselineConfig()

    def act(self, derived: Dict[str, float], info: Dict[str, Any]) -> int:
        c = self.cfg

        load = float(info.get("load", 0.0))
        latency = float(info.get("latency_ms", 0.0))
        if load > c.meltdown_load_thresh or latency > c.meltdown_latency_ms:
            return 7  # global rate-limit

        pkt_rate = derived["pkt_rate_pre"]
        syn_ack_ratio = derived["syn_ack_ratio_pre"]
        tcp_share = derived["tcp_share"]
        udp_share = derived["udp_share"]
        icmp_share = derived["icmp_share"]

        suspicious = pkt_rate > c.pkt_rate_thresh

        if suspicious and syn_ack_ratio > c.syn_ack_ratio_thresh and tcp_share > 0.5:
            return 4 if c.prefer_block else 1  # TCP block or rate-limit
        if suspicious and udp_share > c.udp_share_thresh:
            return 5 if c.prefer_block else 2  # UDP block or rate-limit
        if suspicious and icmp_share > c.icmp_share_thresh:
            return 6 if c.prefer_block else 3  # ICMP block or rate-limit

        if suspicious:
            return 7  # unsure -> global RL

        return 0


# ------------------------------
# Feature derivation (observable)
# ------------------------------
def derive_observables(env_cfg: EnvConfig, info: Dict[str, Any]) -> Dict[str, float]:
    L_total = float(info.get("L_total", 0.0))
    A_tcp = float(info.get("A_tcp", 0.0))
    A_udp = float(info.get("A_udp", 0.0))
    A_icmp = float(info.get("A_icmp", 0.0))

    L_tcp = env_cfg.p_tcp * L_total
    L_udp = env_cfg.p_udp * L_total
    L_icmp = env_cfg.p_icmp * L_total

    tcp_rate_pre = L_tcp + A_tcp
    udp_rate_pre = L_udp + A_udp
    icmp_rate_pre = L_icmp + A_icmp
    pkt_rate_pre = tcp_rate_pre + udp_rate_pre + icmp_rate_pre
    bytes_rate_pre = pkt_rate_pre * env_cfg.avg_packet_size_bytes

    syn_legit = env_cfg.p_syn_legit * L_tcp
    ack_legit = env_cfg.p_ack_legit * L_tcp
    syn_attack = env_cfg.syn_attack_share * A_tcp
    ack_attack = env_cfg.ack_attack_share * A_tcp

    syn_rate_pre = syn_legit + syn_attack
    ack_rate_pre = ack_legit + ack_attack
    syn_ack_ratio_pre = syn_rate_pre / (ack_rate_pre + 1.0)

    denom = pkt_rate_pre + 1e-9
    tcp_share = tcp_rate_pre / denom
    udp_share = udp_rate_pre / denom
    icmp_share = icmp_rate_pre / denom

    return {
        "pkt_rate_pre": pkt_rate_pre,
        "bytes_rate_pre": bytes_rate_pre,
        "tcp_rate_pre": tcp_rate_pre,
        "udp_rate_pre": udp_rate_pre,
        "icmp_rate_pre": icmp_rate_pre,
        "syn_rate_pre": syn_rate_pre,
        "ack_rate_pre": ack_rate_pre,
        "syn_ack_ratio_pre": syn_ack_ratio_pre,
        "tcp_share": tcp_share,
        "udp_share": udp_share,
        "icmp_share": icmp_share,
    }


# ------------------------------
# Metrics
# ------------------------------
def compute_mitigation_time(step_infos: List[Dict[str, Any]], threshold: float = 0.90, consecutive: int = 5) -> float:
    attack_idxs = [i for i, inf in enumerate(step_infos) if bool(inf.get("attack_present", False))]
    if not attack_idxs:
        return np.nan
    start = attack_idxs[0]
    for i in range(start, len(step_infos) - consecutive + 1):
        window = step_infos[i : i + consecutive]
        if all(float(w.get("suppression", 0.0)) >= threshold for w in window):
            return float(i - start)  # seconds (dt=1)
    return np.nan


def action_aggressiveness(action_id: int) -> int:
    """
    Simple severity score for reporting:
    0 allow  -> 0
    1-3 rate -> 1
    4-6 block-> 2
    7 global rate -> 1
    """
    if action_id == 0:
        return 0
    if action_id in (1, 2, 3, 7):
        return 1
    if action_id in (4, 5, 6):
        return 2
    return 0


def run_episode(env: AutoPPOFirewallEnv, policy_name: str, ppo: Optional[PPO], baseline: Optional[FairStaticFirewallPolicy],
                deterministic: bool) -> Dict[str, Any]:
    obs, _ = env.reset()
    done = False

    step_infos: List[Dict[str, Any]] = []
    rewards: List[float] = []
    actions: List[int] = []

    while not done:
        if policy_name == "baseline":
            if env.last_info:
                derived = derive_observables(env.cfg, env.last_info)
                action = int(baseline.act(derived, env.last_info))  # type: ignore
            else:
                action = 0
        else:
            action, _ = ppo.predict(obs, deterministic=deterministic)  # type: ignore
            action = int(action)

        obs, r, terminated, truncated, info = env.step(action)
        rewards.append(float(r))
        actions.append(action)
        step_infos.append(info)
        done = terminated or truncated

    # Arrays
    lat = np.array([float(i["latency_ms"]) for i in step_infos], dtype=float)
    loss = np.array([float(i["loss"]) for i in step_infos], dtype=float)
    sup = np.array([float(i["suppression"]) for i in step_infos], dtype=float)
    ret = np.array([float(i["retention"]) for i in step_infos], dtype=float)
    fp = np.array([float(i["fp_penalty"]) for i in step_infos], dtype=float)
    atk = np.array([bool(i["attack_present"]) for i in step_infos], dtype=bool)

    # Proxy "TPR/FPR-like" metrics (episode-level, clear + defensible):
    # - During attack: how often suppression is "good"
    # - During no-attack: how often policy stays non-aggressive (doesn't rate-limit/block)
    sup_good = sup >= 0.90
    benign = ~atk
    non_aggressive = np.array([a == 0 for a in actions], dtype=bool)

    attack_detection_rate = float(np.mean(sup_good[atk])) if atk.any() else np.nan
    benign_non_aggressive_rate = float(np.mean(non_aggressive[benign])) if benign.any() else np.nan

    # Aggressiveness rate
    aggress = np.array([action_aggressiveness(a) for a in actions], dtype=int)
    aggressive_rate = float(np.mean(aggress >= 1))
    block_rate = float(np.mean(aggress == 2))

    return {
        # identity
        "policy": policy_name,
        "episode_reward": float(np.sum(rewards)),

        # overall QoS/security
        "avg_latency_ms": float(np.mean(lat)),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "avg_loss": float(np.mean(loss)),
        "avg_suppression": float(np.mean(sup)),
        "avg_retention": float(np.mean(ret)),
        "avg_fp_penalty": float(np.mean(fp)),

        # attack-only
        "attack_steps": int(np.sum(atk)),
        "attack_avg_latency_ms": float(np.mean(lat[atk])) if atk.any() else np.nan,
        "attack_avg_loss": float(np.mean(loss[atk])) if atk.any() else np.nan,
        "attack_avg_suppression": float(np.mean(sup[atk])) if atk.any() else np.nan,
        "attack_avg_retention": float(np.mean(ret[atk])) if atk.any() else np.nan,
        "mitigation_time_s": compute_mitigation_time(step_infos),

        # baseline-style “detection” proxies (easy to explain in thesis)
        "attack_detection_rate_0p90": attack_detection_rate,
        "benign_non_aggressive_rate": benign_non_aggressive_rate,

        # behavior
        "aggressive_rate": aggressive_rate,
        "block_rate": block_rate,
    }


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO vs Baseline and output clean results.csv + summary.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--model_path", type=str, default="models/ppo/best/best_model.zip")
    parser.add_argument("--out_csv", type=str, default="experiments/results.csv")
    parser.add_argument("--out_summary_csv", type=str, default="experiments/results_summary.csv")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"PPO model not found at: {model_path}\n"
            f"Train first or point --model_path to models/ppo/final_model.zip"
        )

    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = project_root / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_summary = Path(args.out_summary_csv)
    if not out_summary.is_absolute():
        out_summary = project_root / out_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = EnvConfig()
    baseline = FairStaticFirewallPolicy(env_cfg)
    ppo = PPO.load(str(model_path))

    rows: List[Dict[str, Any]] = []

    # Run baseline and PPO with matched seeds per episode (fairness)
    for policy in ["baseline", "ppo"]:
        for ep in range(args.episodes):
            ep_seed = args.seed + ep
            env = AutoPPOFirewallEnv(config=env_cfg, seed=ep_seed)

            res = run_episode(
                env=env,
                policy_name=policy,
                ppo=ppo if policy == "ppo" else None,
                baseline=baseline if policy == "baseline" else None,
                deterministic=args.deterministic,
            )
            res["episode"] = ep
            res["seed"] = ep_seed
            rows.append(res)

    df = pd.DataFrame(rows)

    # Fixed, thesis-friendly column order
    cols = [
        "policy", "episode", "seed",
        "episode_reward",

        "avg_latency_ms", "p95_latency_ms", "avg_loss",
        "avg_suppression", "avg_retention", "avg_fp_penalty",

        "attack_steps",
        "attack_avg_latency_ms", "attack_avg_loss",
        "attack_avg_suppression", "attack_avg_retention",
        "mitigation_time_s",

        "attack_detection_rate_0p90",
        "benign_non_aggressive_rate",

        "aggressive_rate", "block_rate",
    ]
    df = df[cols]
    df.to_csv(out_csv, index=False)

    # Summary (mean/std) by policy
    numeric_cols = [c for c in cols if c not in ("policy", "episode", "seed")]
    summary_mean = df.groupby("policy")[numeric_cols].mean(numeric_only=True)
    summary_std = df.groupby("policy")[numeric_cols].std(numeric_only=True)

    summary = summary_mean.add_suffix("_mean").join(summary_std.add_suffix("_std"))
    summary.to_csv(out_summary)

    print("\n=== Saved ===")
    print(f"- Episode results: {out_csv}")
    print(f"- Summary results: {out_summary}\n")

    print("=== Quick Summary (means) ===")
    print(summary_mean.round(3))


if __name__ == "__main__":
    main()
