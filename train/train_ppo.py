# train/train_ppo.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

# SIM env
from env.sim_env import AutoPPOFirewallEnv, EnvConfig

def make_sim_env(seed: int, config: EnvConfig, rank: int = 0) -> Callable[[], AutoPPOFirewallEnv]:
    def _init():
        env = AutoPPOFirewallEnv(config=config, seed=seed + rank)
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO on AutoPPO-FW (sim or mininet).")

    parser.add_argument("--backend", choices=["sim", "mininet"], default="sim",
                        help="sim = synthetic env, mininet = real lab env")

    # Core training params
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--use_subproc", action="store_true")

    # PPO hyperparams
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    # Eval/logging
    parser.add_argument("--eval_freq", type=int, default=20_000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=50_000)

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    logs_dir = project_root / "logs" / "ppo"
    models_dir = project_root / "models" / "ppo"
    tb_dir = logs_dir / "tb"
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)

    net = None  # Only used in mininet mode

    try:
        # ---------------- Create Env(s) ----------------
        if args.backend == "sim":
            env_cfg = EnvConfig()

            env_fns = [make_sim_env(seed=args.seed, config=env_cfg, rank=i) for i in range(args.n_envs)]
            if args.use_subproc and args.n_envs > 1:
                vec_env = SubprocVecEnv(env_fns)
            else:
                vec_env = DummyVecEnv(env_fns)

            vec_env = VecMonitor(vec_env, filename=str(logs_dir / "monitor.csv"))

            eval_env = DummyVecEnv([make_sim_env(seed=args.seed + 10_000, config=env_cfg, rank=0)])
            eval_env = VecMonitor(eval_env)

        else:
            # MININET mode: force single env, no subproc
            if args.n_envs != 1:
                print("[WARN] Mininet backend requires --n_envs 1. Forcing n_envs=1.")
            if args.use_subproc:
                raise ValueError("Mininet backend cannot use --use_subproc (only one live network).")

            from mnlab.topo import build_net
            from env.mininet_env import MininetFirewallEnv

            net = build_net(start=True, do_sanity=False)

            fw = net.get("fw")
            srv = net.get("srv")
            hleg1 = net.get("hleg1")
            hatk1 = net.get("hatk1")

            # Ensure PPO_SYN hook exists (your manual setup in Python)
            fw.cmd("sysctl -w net.ipv4.ip_forward=1 >/dev/null")
            fw.cmd("iptables -F; iptables -X; iptables -t nat -F; iptables -t mangle -F")

            fw.cmd("iptables -X PPO_SYN 2>/dev/null || true; iptables -N PPO_SYN")
            fw.cmd("iptables -A FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT")
            fw.cmd("iptables -A FORWARD -d 10.0.2.10 -p tcp --syn -j PPO_SYN")
            fw.cmd("iptables -A FORWARD -j ACCEPT")

            # Start iperf server + legit flow + attack
            srv.cmd("pkill -f iperf3 || true")
            srv.cmd("iperf3 -s -p 5201 -D")

            hleg1.cmd('pkill -f "iperf3 -c" || true')
            hleg1.cmd('sh -c "iperf3 -c 10.0.2.10 -p 5201 -t 9999 -i 1 > /tmp/leg_hleg1.log 2>&1 &"')

            hatk1.cmd("pkill -f hping3 || true")
            hatk1.cmd('sh -c "hping3 -S -p 5201 --flood 10.0.2.10 > /tmp/atk_hatk1.log 2>&1 &"')

            # Wrap in SB3 VecEnv (still needed, even for one env)
            def _init():
                env = MininetFirewallEnv(fw=fw, hleg1=hleg1, episode_len_s=60)
                env = Monitor(env)
                return env

            vec_env = DummyVecEnv([_init])
            vec_env = VecMonitor(vec_env, filename=str(logs_dir / "monitor.csv"))

            eval_env = None  # Optional: you can add later once stable

        # ---------------- Callbacks ----------------
        checkpoint_cb = CheckpointCallback(
            save_freq=args.save_freq // max(1, args.n_envs),
            save_path=str(models_dir / "checkpoints"),
            name_prefix=f"ppo_autoppo_fw_{args.backend}",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        callbacks = [checkpoint_cb]

        if eval_env is not None:
            eval_cb = EvalCallback(
                eval_env=eval_env,
                best_model_save_path=str(models_dir / "best"),
                log_path=str(logs_dir / "eval"),
                eval_freq=args.eval_freq // max(1, args.n_envs),
                n_eval_episodes=args.n_eval_episodes,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_cb)

        # ---------------- Model ----------------
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            verbose=1,
            tensorboard_log=str(tb_dir),
            device="auto",
        )

        print("\n[INFO] Training started...")
        print(f"[INFO] Backend: {args.backend}")
        print(f"[INFO] Logs:   {logs_dir}")
        print(f"[INFO] Models: {models_dir}\n")

        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
        )

        final_path = models_dir / f"final_model_{args.backend}.zip"
        model.save(str(final_path))
        print(f"\n[INFO] Training finished. Final model saved to: {final_path}")

        vec_env.close()
        if eval_env is not None:
            eval_env.close()

    finally:
        # Mininet cleanup
        if net is not None:
            try:
                net.get("hatk1").cmd("pkill -f hping3 || true")
                net.get("hleg1").cmd('pkill -f "iperf3 -c" || true')
                net.get("srv").cmd("pkill -f iperf3 || true")
            except Exception:
                pass
            net.stop()


if __name__ == "__main__":
    main()

