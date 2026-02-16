# cli/drl.py
from __future__ import annotations

from stable_baselines3 import PPO

from env.mininet_env import ACTIONS  # your existing ladder

def apply_action(fw, action: int) -> None:
    fw.cmd(ACTIONS[int(action)])

def load_policy(model_path: str) -> PPO:
    return PPO.load(model_path)

def choose_action(model: PPO, obs) -> int:
    action, _ = model.predict(obs, deterministic=True)
    return int(action)
