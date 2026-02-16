# run_demo.py
from env.autoppo_fw_env import AutoPPOFirewallEnv, EnvConfig
from policies.baseline_policy import StaticFirewallPolicy

def main():
    env = AutoPPOFirewallEnv(EnvConfig(), seed=42)
    policy = StaticFirewallPolicy()

    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = policy.act(env.last_info) if env.last_info else 0
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        print(
            f"t={info['t']:3d} | "
            f"action={info['action']} | "
            f"lat={info['latency_ms']:.1f}ms | "
            f"loss={info['loss']:.2f} | "
            f"reward={reward:.2f}"
        )

    print("\nTotal episode reward:", total_reward)

if __name__ == "__main__":
    main()
