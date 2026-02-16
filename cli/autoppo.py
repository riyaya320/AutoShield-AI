# cli/autoppo.py
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from cli.runtime import start_lab, stop_lab, ensure_forward_hook, start_traffic, stop_traffic
from cli.metrics import sample_metrics
from cli.drl import apply_action, load_policy, choose_action

from env.mininet_env import MininetFirewallEnv  # for consistent obs format


def run_once(mode: str, model_path: str | None, duration_s: int, action: int, out_csv: Path) -> None:
    handles = start_lab()
    try:
        ensure_forward_hook(handles)
        start_traffic(handles, attack=True)
        time.sleep(2.0)

        # Create env object for obs format (even if not training)
        env = MininetFirewallEnv(fw=handles.fw, hleg1=handles.hleg1, episode_len_s=duration_s)

        # Ensure PPO_SYN exists and is readable (depends on your updated env)
        if hasattr(env, "_ensure_chain"):
            env._ensure_chain()  # type: ignore[attr-defined]

        # Baseline should still have ACCEPT+DROP so counters are stable
        if mode == "baseline":
            apply_action(handles.fw, 0)
        elif mode == "static":
            apply_action(handles.fw, action)
        elif mode == "drl":
            if not model_path:
                raise ValueError("--model is required for mode=drl")
            model = load_policy(model_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Initialize counters + baseline throughput
        prev_a, prev_d, prev_tx = 0, 0, 0
        m0, prev_a, prev_d, prev_tx = sample_metrics(handles.fw, handles.hleg1, prev_a, prev_d, prev_tx)

        # baseline throughput should be measured AFTER traffic stabilizes
        baseline_mbps = max(1.0, m0.legit_mbps)


        # Start obs (same shape as env)
        obs = env.reset()[0]  # (obs, info)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "t_s", "mode", "action",
                "syn_accept_rate", "syn_drop_rate", "syn_total",
                "rtt_ms", "legit_mbps", "thr_norm"
            ])

            t0 = time.time()
            for _ in range(duration_s):
                # Choose/apply action
                if mode == "drl":
                    a = choose_action(model, obs)
                    apply_action(handles.fw, a)
                elif mode == "static":
                    a = int(action)
                    apply_action(handles.fw, a)
                else:
                    a = 0
                    apply_action(handles.fw, a)

                time.sleep(1.0)

                
                m, prev_a, prev_d, prev_tx = sample_metrics(handles.fw, handles.hleg1, prev_a, prev_d, prev_tx)
                syn_total = m.syn_accept_rate + m.syn_drop_rate
                thr_norm = (m.legit_mbps / baseline_mbps) if baseline_mbps > 0 else 0.0

                # Build an obs consistent with your env (so DRL can run)
                syn_norm = min(1.0, syn_total / env.syn_max)
                drop_norm = min(1.0, m.syn_drop_rate / env.syn_max)
                rtt_norm = min(1.0, m.rtt_ms / env.rtt_max)
                thr_norm_clip = min(1.0, thr_norm)
                act_norm = float(a) / 4.0
                obs = [syn_norm, drop_norm, rtt_norm, thr_norm_clip, act_norm]

                t_s = int(time.time() - t0)
                print(
                    f"[t={t_s:>3}s] mode={mode:<8} action={a} "
                    f"syn_total={syn_total:<10} drop={m.syn_drop_rate:<10} "
                    f"rtt={m.rtt_ms:>6.2f}ms  legit={m.legit_mbps:>8.2f} Mbps"
                )

                w.writerow([t_s, mode, a, m.syn_accept_rate, m.syn_drop_rate, syn_total, m.rtt_ms, m.legit_mbps, thr_norm])

    finally:
        # Always cleanup
        try:
            stop_traffic(handles)
        except Exception:
            pass
        stop_lab(handles)


def main():
    parser = argparse.ArgumentParser(prog="autoppo", description="AutoPPO-FW CLI (Mininet runtime controller)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a live experiment (baseline/static/drl) and log CSV")
    run.add_argument("--mode", choices=["baseline", "static", "drl"], required=True)
    run.add_argument("--model", type=str, default=None, help="Path to SB3 PPO model zip (required for drl mode)")
    run.add_argument("--duration", type=int, default=60, help="Run length in seconds")
    run.add_argument("--action", type=int, default=4, help="Static mode action 0..4")
    run.add_argument("--out", type=str, default="logs/ppo/cli_run.csv", help="Output CSV path")

    args = parser.parse_args()

    if args.cmd == "run":
        out_csv = Path(args.out)
        run_once(
            mode=args.mode,
            model_path=args.model,
            duration_s=args.duration,
            action=args.action,
            out_csv=out_csv,
        )


if __name__ == "__main__":
    main()
