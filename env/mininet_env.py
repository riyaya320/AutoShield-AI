import time

import re
import gymnasium as gym
from gymnasium import spaces
import numpy as np

SRV_IP = "10.0.2.10"

ACTIONS = {
    0: 'iptables -F PPO_SYN; iptables -A PPO_SYN -j ACCEPT; iptables -A PPO_SYN -j DROP',
    1: ('iptables -F PPO_SYN; '
        'iptables -A PPO_SYN -m hashlimit --hashlimit-name syn_a1 '
        '--hashlimit 10000/second --hashlimit-burst 20000 '
        '--hashlimit-mode srcip --hashlimit-htable-expire 10000 -j ACCEPT; '
        'iptables -A PPO_SYN -j DROP'),
    2: ('iptables -F PPO_SYN; '
        'iptables -A PPO_SYN -m hashlimit --hashlimit-name syn_a2 '
        '--hashlimit 2000/second --hashlimit-burst 4000 '
        '--hashlimit-mode srcip --hashlimit-htable-expire 10000 -j ACCEPT; '
        'iptables -A PPO_SYN -j DROP'),
    3: ('iptables -F PPO_SYN; '
        'iptables -A PPO_SYN -m hashlimit --hashlimit-name syn_a3 '
        '--hashlimit 300/second --hashlimit-burst 600 '
        '--hashlimit-mode srcip --hashlimit-htable-expire 10000 -j ACCEPT; '
        'iptables -A PPO_SYN -j DROP'),
    4: ('iptables -F PPO_SYN; '
        'iptables -A PPO_SYN -m hashlimit --hashlimit-name syn_a4 '
        '--hashlimit 50/second --hashlimit-burst 100 '
        '--hashlimit-mode srcip --hashlimit-htable-expire 10000 -j ACCEPT; '
        'iptables -A PPO_SYN -j DROP'),
}

class MininetFirewallEnv(gym.Env):
    """
    RL environment backed by your Mininet lab.
    Requires references to Mininet nodes: fw, hleg1.
    """
    metadata = {"render_modes": []}

    def __init__(self, fw, hleg1, episode_len_s=60):
        super().__init__()
        self.fw = fw
        self.hleg1 = hleg1
        self.episode_len_s = episode_len_s

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        self.syn_max = 1_000_000.0
        self.rtt_max = 200.0

        self.t = 0
        self.prev_tx = 0
        self.prev_accept = None
        self.prev_drop = None
        self.baseline_mbps = None

    def _ensure_chain(self):
        """
        Ensure PPO_SYN chain exists and is hooked into FORWARD.
        Safe to call repeatedly.
        """
        self.fw.cmd("iptables -N PPO_SYN 2>/dev/null || true")
        self.fw.cmd("iptables -F PPO_SYN")

        # Ensure FORWARD hook exists
        self.fw.cmd(
            "iptables -C FORWARD -d 10.0.2.10 -p tcp --syn -j PPO_SYN 2>/dev/null || "
            "iptables -A FORWARD -d 10.0.2.10 -p tcp --syn -j PPO_SYN"
        )
 
    # ---------- helpers ----------
    def _apply_action(self, action: int):
        self.fw.cmd(ACTIONS[int(action)])

    def _read_accept_drop_pkts(self):
        out = self.fw.cmd("iptables -L PPO_SYN -n -v -x --line-numbers")
        lines = [ln for ln in out.strip().splitlines() if ln.strip()]
        if len(lines) < 3:
            raise RuntimeError("Unexpected PPO_SYN output:\n" + out)

    # Find rule lines (they start with a number)
        rule_lines = [ln for ln in lines if re.match(r"^\s*\d+\s+", ln)]
        if len(rule_lines) < 1:
            raise RuntimeError("No rules found in PPO_SYN:\n" + out)

    # Rule 1 is ACCEPT counter
        r1 = rule_lines[0].split()
        accept_pkts = int(r1[1])

    # Rule 2 might not exist yet; treat as 0
        drop_pkts = 0
        if len(rule_lines) >= 2:
            r2 = rule_lines[1].split()
            drop_pkts = int(r2[1])

        return accept_pkts, drop_pkts

    def _ping_rtt_ms(self):
        out = self.hleg1.cmd(f'ping -c 1 -W 1 {SRV_IP}')
        m = re.search(r'time=([\d\.]+)\s*ms', out)
        if not m:
            return self.rtt_max  # treat as bad
        return float(m.group(1))

    def _read_tx_bytes(self, intf: str = "hleg1-eth0") -> int:
        out = self.hleg1.cmd(f"cat /proc/net/dev | grep '{intf}:'")
        m = re.search(
            rf"{intf}:\s*(\d+)\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)", 
            out
        )
        if not m:
            return 0
        return int(m.group(2))

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0

        # Ensure PPO_SYN exists nefore reading the counters
        self._ensure_chain()

        # Start with Action 0 (or 2) depending on your training strategy
        self._apply_action(0)

        # Initialize counters
        a, d = self._read_accept_drop_pkts()
        self.prev_accept, self.prev_drop = a, d

        # Baseline throughput estimate (take current)
        self.prev_tx = self._read_tx_bytes()
        time.sleep(1.0)
        tx2 = self._read_tx_bytes()
        mbps = max(1.0, (tx2 - self.prev_tx) * 8.0 / 1_000_000.0)
        self.baseline_mbps = mbps
        self.prev_tx = tx2


        obs = np.zeros((5,), dtype=np.float32)
        return obs, {}

    def step(self, action: int):
        self._apply_action(action)

        time.sleep(1.0)

        a, d = self._read_accept_drop_pkts()
        accept_rate = max(0, a - self.prev_accept)
        drop_rate = max(0, d - self.prev_drop)
        self.prev_accept, self.prev_drop = a, d

        syn_total = accept_rate + drop_rate
        rtt = self._ping_rtt_ms()
        tx_now = self._read_tx_bytes()
        mbps = max(0.0, (tx_now - self.prev_tx) * 8.0 / 1_000_000.0)
        self.prev_tx = tx_now

        syn_norm = min(1.0, syn_total / self.syn_max)
        drop_norm = min(1.0, drop_rate / self.syn_max)
        rtt_norm = min(1.0, rtt / self.rtt_max)
        thr_norm = min(1.0, mbps / max(1.0, self.baseline_mbps))
        act_norm = float(action) / 4.0

        obs = np.array([syn_norm, drop_norm, rtt_norm, thr_norm, act_norm], dtype=np.float32)

        reward = (2.0 * thr_norm) - (2.0 * syn_norm) - (1.0 * rtt_norm) - (0.15 * act_norm)

        self.t += 1
        terminated = False
        truncated = self.t >= self.episode_len_s

        info = {
            "accept_rate": accept_rate,
            "drop_rate": drop_rate,
            "rtt_ms": rtt,
            "legit_mbps": mbps,
        }
        return obs, reward, terminated, truncated, info
