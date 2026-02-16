# autoppo_fw_env.py
from __future__ import annotations


ACTION_MEANINGS = {
    0: "ALLOW_ALL",
    1: "RATE_LIMIT_TCP",
    2: "RATE_LIMIT_UDP",
    3: "RATE_LIMIT_ICMP",
    4: "BLOCK_TCP",
    5: "BLOCK_UDP",
    6: "BLOCK_ICMP",
    7: "GLOBAL_RATE_LIMIT",
}


from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class EnvConfig:
    # Timing
    episode_horizon: int = 300
    dt_seconds: int = 1

    # Legit traffic
    mu_levels: Tuple[float, float, float] = (600.0, 1500.0, 3500.0)  # low/med/high pkt/s
    noise_frac: float = 0.08  # std = noise_frac * mu
    burst_prob: float = 0.05
    burst_amp_range: Tuple[float, float] = (800.0, 2500.0)
    burst_dur_range: Tuple[int, int] = (5, 20)

    # Legit protocol mix
    p_tcp: float = 0.70
    p_udp: float = 0.28
    p_icmp: float = 0.02

    # Legit TCP flags
    p_syn_legit: float = 0.10
    p_ack_legit: float = 0.45

    # Attack peaks (pkt/s)
    amax_syn: float = 30000.0
    amax_udp: float = 35000.0
    amax_icmp: float = 20000.0

    # SYN flood signature
    syn_attack_share: float = 0.95
    ack_attack_share: float = 0.02

    # Attack curve
    ramp_seconds: int = 20

    # Network capacity & QoS
    capacity_pktps: float = 12000.0
    base_latency_ms: float = 20.0
    latency_k: float = 120.0
    loss_m: float = 0.6
    latency_norm_max_ms: float = 300.0

    # Bytes model (for a simple bytes/sec feature)
    avg_packet_size_bytes: float = 800.0

    # Reward weights
    w_suppression: float = 2.0
    w_retention: float = 1.2
    w_latency: float = 1.0
    w_loss: float = 1.0
    w_fp: float = 2.5

    # FP retention threshold during attack
    retention_soft_floor: float = 0.9

    # Episode schedule (seconds)
    # 0-60 normal; 60-140 SYN; 140-220 SYN+UDP; 220-260 UDP+ICMP; 260-300 normal
    t_normal_1_end: int = 60
    t_syn_end: int = 140
    t_syn_udp_end: int = 220
    t_udp_icmp_end: int = 260


class AutoPPOFirewallEnv(gym.Env):
    """
    Observation vector (13 dims):
    0  pkt_rate_total_pre
    1  bytes_rate_total_pre
    2  tcp_rate_pre
    3  udp_rate_pre
    4  icmp_rate_pre
    5  syn_rate_pre
    6  ack_rate_pre
    7  syn_ack_ratio_pre
    8  latency_ms_post
    9  loss_post (0-1)
    10 queue_proxy_post (max(0, load-1))
    11 firewall_mode (0 allow, 1 rate, 2 block, 3 global_rate)
    12 rate_limit_level (0..1)
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None, seed: Optional[int] = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(seed)

        # 8 discrete actions as designed
        self.action_space = spaces.Discrete(8)

        # Normalized observation space in [0,1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(13,), dtype=np.float32
        )

        # Internal episode state
        self.t: int = 0
        self._burst_remaining: int = 0
        self._burst_amp: float = 0.0

        # For dashboard/logging
        self.last_info: Dict[str, Any] = {}

    # ------------------------ helpers ------------------------

    def _attack_curve(self, t_in_phase: int, phase_len: int) -> float:
        """Smooth-ish ramp-up/hold/ramp-down curve g(t) in [0,1]."""
        r = self.cfg.ramp_seconds
        if phase_len <= 0:
            return 0.0
        if t_in_phase < 0:
            return 0.0
        if t_in_phase < r:
            return t_in_phase / float(max(1, r))
        if t_in_phase > phase_len - r:
            return max(0.0, (phase_len - t_in_phase) / float(max(1, r)))
        return 1.0

    def _active_attacks(self, t: int) -> Tuple[bool, bool, bool]:
        """Return (syn, udp, icmp) booleans based on fixed schedule."""
        c = self.cfg
        if t < c.t_normal_1_end:
            return (False, False, False)
        if t < c.t_syn_end:
            return (True, False, False)
        if t < c.t_syn_udp_end:
            return (True, True, False)
        if t < c.t_udp_icmp_end:
            return (False, True, True)
        return (False, False, False)

    def _get_mu(self, t: int) -> float:
        """Pick baseline traffic level (simple segmentation)."""
        # Simple: low early, medium mid, high later (can randomize later)
        if t < self.cfg.episode_horizon * 0.33:
            return self.cfg.mu_levels[0]
        if t < self.cfg.episode_horizon * 0.66:
            return self.cfg.mu_levels[1]
        return self.cfg.mu_levels[2]

    def _apply_action_multipliers(self, action: int) -> Dict[str, Dict[str, float]]:
        """
        Returns multipliers dict:
        multipliers["attack"][protocol], multipliers["legit"][protocol]
        plus meta: mode, rate_level
        """
        # Defaults (allow all)
        a = {"tcp": 1.0, "udp": 1.0, "icmp": 1.0}
        l = {"tcp": 1.0, "udp": 1.0, "icmp": 1.0}
        mode = 0
        rate_level = 0.0

        if action == 0:  # allow
            mode = 0
            rate_level = 0.0

        elif action == 1:  # rate-limit tcp
            a["tcp"] = 0.20
            l["tcp"] = 0.75
            mode = 1
            rate_level = 0.75

        elif action == 2:  # rate-limit udp
            a["udp"] = 0.20
            l["udp"] = 0.75
            mode = 1
            rate_level = 0.75

        elif action == 3:  # rate-limit icmp
            a["icmp"] = 0.20
            l["icmp"] = 0.80
            mode = 1
            rate_level = 0.80

        elif action == 4:  # block tcp
            a["tcp"] = 0.00
            l["tcp"] = 0.45
            mode = 2
            rate_level = 0.0

        elif action == 5:  # block udp
            a["udp"] = 0.00
            l["udp"] = 0.55
            mode = 2
            rate_level = 0.0

        elif action == 6:  # block icmp
            a["icmp"] = 0.00
            l["icmp"] = 0.90
            mode = 2
            rate_level = 0.0

        elif action == 7:  # global rate-limit
            a = {"tcp": 0.25, "udp": 0.25, "icmp": 0.25}
            l = {"tcp": 0.70, "udp": 0.70, "icmp": 0.70}
            mode = 3
            rate_level = 0.70

        return {"attack": a, "legit": l, "mode": mode, "rate_level": rate_level}

    def _normalize_obs(self, raw: np.ndarray) -> np.ndarray:
        """
        Min-max normalize to [0,1] using reasonable bounds.
        These are engineering bounds; tune later if needed.
        """
        # Bounds (rough but stable)
        # rates: up to ~ (legit+attack) peaks
        max_pkt = 100000.0
        max_bytes = max_pkt * self.cfg.avg_packet_size_bytes
        max_rate = 100000.0
        max_syn = 100000.0
        max_ack = 100000.0
        max_ratio = 50.0  # syn/ack can spike; clamp
        max_latency = self.cfg.latency_norm_max_ms
        max_loss = 1.0
        max_queue = 10.0  # proxy
        max_mode = 3.0
        max_rl = 1.0

        mins = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        maxs = np.array(
            [
                max_pkt,
                max_bytes,
                max_rate,
                max_rate,
                max_rate,
                max_syn,
                max_ack,
                max_ratio,
                max_latency,
                max_loss,
                max_queue,
                max_mode,
                max_rl,
            ],
            dtype=np.float32,
        )
        x = np.clip(raw.astype(np.float32), mins, maxs)
        return (x - mins) / (maxs - mins + 1e-9)

    # ------------------------ Gym API ------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        self._burst_remaining = 0
        self._burst_amp = 0.0
        self.last_info = {}

        # Start with a neutral observation (no previous QoS yet)
        raw = np.zeros(13, dtype=np.float32)
        obs = self._normalize_obs(raw)
        return obs, {}

    def step(self, action: int):
        c = self.cfg
        t = self.t

        # -------- 1) Legit traffic --------
        mu = self._get_mu(t)
        noise = self.rng.normal(0.0, c.noise_frac * mu)

        # burst management
        if self._burst_remaining <= 0:
            if self.rng.random() < c.burst_prob:
                self._burst_remaining = int(self.rng.integers(c.burst_dur_range[0], c.burst_dur_range[1] + 1))
                self._burst_amp = float(self.rng.uniform(c.burst_amp_range[0], c.burst_amp_range[1]))
        burst = self._burst_amp if self._burst_remaining > 0 else 0.0
        if self._burst_remaining > 0:
            self._burst_remaining -= 1

        L_total = max(0.0, mu + noise + burst)
        L_tcp = c.p_tcp * L_total
        L_udp = c.p_udp * L_total
        L_icmp = c.p_icmp * L_total

        # -------- 2) Attack traffic (schedule + curve) --------
        syn_on, udp_on, icmp_on = self._active_attacks(t)

        # Compute per-phase curve g(t)
        A_tcp = 0.0
        A_udp = 0.0
        A_icmp = 0.0

        if syn_on:
            phase_start = c.t_normal_1_end
            phase_end = c.t_syn_udp_end  # SYN present through SYN+UDP phase
            g = self._attack_curve(t - phase_start, phase_end - phase_start)
            A_tcp = c.amax_syn * g

        if udp_on:
            # UDP active in SYN+UDP and UDP+ICMP phases
            if t < c.t_syn_udp_end:
                phase_start, phase_end = c.t_syn_end, c.t_syn_udp_end
            else:
                phase_start, phase_end = c.t_syn_udp_end, c.t_udp_icmp_end
            g = self._attack_curve(t - phase_start, phase_end - phase_start)
            A_udp = c.amax_udp * g

        if icmp_on:
            phase_start, phase_end = c.t_syn_udp_end, c.t_udp_icmp_end
            g = self._attack_curve(t - phase_start, phase_end - phase_start)
            A_icmp = c.amax_icmp * g

        # -------- 3) Flags (SYN/ACK) --------
        syn_legit = c.p_syn_legit * L_tcp
        ack_legit = c.p_ack_legit * L_tcp

        syn_attack = c.syn_attack_share * A_tcp
        ack_attack = c.ack_attack_share * A_tcp

        syn_total = syn_legit + syn_attack
        ack_total = ack_legit + ack_attack
        syn_ack_ratio = syn_total / (ack_total + 1.0)

        # -------- 4) Apply firewall action --------
        mult = self._apply_action_multipliers(int(action))
        a_mul = mult["attack"]
        l_mul = mult["legit"]

        A_tcp_pass = a_mul["tcp"] * A_tcp
        A_udp_pass = a_mul["udp"] * A_udp
        A_icmp_pass = a_mul["icmp"] * A_icmp

        L_tcp_pass = l_mul["tcp"] * L_tcp
        L_udp_pass = l_mul["udp"] * L_udp
        L_icmp_pass = l_mul["icmp"] * L_icmp

        T_pass = (A_tcp_pass + A_udp_pass + A_icmp_pass) + (L_tcp_pass + L_udp_pass + L_icmp_pass)

        # -------- 5) Congestion -> QoS --------
        load = T_pass / c.capacity_pktps
        queue_proxy = max(0.0, load - 1.0)

        latency = c.base_latency_ms + c.latency_k * (max(0.0, load - 1.0) ** 2)
        loss = min(1.0, c.loss_m * max(0.0, load - 1.0))

        # -------- 6) Reward --------
        A_sum = A_tcp + A_udp + A_icmp
        A_pass_sum = A_tcp_pass + A_udp_pass + A_icmp_pass
        L_sum = L_tcp + L_udp + L_icmp
        L_pass_sum = L_tcp_pass + L_udp_pass + L_icmp_pass

        suppression = 1.0 - (A_pass_sum / (A_sum + 1e-9))
        retention = (L_pass_sum / (L_sum + 1e-9))

        lat_n = min(1.0, latency / c.latency_norm_max_ms)
        loss_n = loss

        attack_present = A_sum > 0.0
        if not attack_present:
            fp = (1.0 - retention)
        else:
            fp = max(0.0, c.retention_soft_floor - retention)

        reward = (
            c.w_suppression * suppression
            + c.w_retention * retention
            - c.w_latency * lat_n
            - c.w_loss * loss_n
            - c.w_fp * fp
        )

        # -------- 7) Next state (pre-firewall traffic + post-firewall QoS + context) --------
        pkt_rate_pre = (L_tcp + L_udp + L_icmp) + (A_tcp + A_udp + A_icmp)
        bytes_rate_pre = pkt_rate_pre * c.avg_packet_size_bytes

        tcp_rate_pre = L_tcp + A_tcp
        udp_rate_pre = L_udp + A_udp
        icmp_rate_pre = L_icmp + A_icmp

        mode = float(mult["mode"])
        rate_level = float(mult["rate_level"])

        raw_obs = np.array(
            [
                pkt_rate_pre,
                bytes_rate_pre,
                tcp_rate_pre,
                udp_rate_pre,
                icmp_rate_pre,
                syn_total,
                ack_total,
                syn_ack_ratio,
                latency,
                loss,
                queue_proxy,
                mode,
                rate_level,
            ],
            dtype=np.float32,
        )
        obs = self._normalize_obs(raw_obs)

        # -------- 8) Termination/truncation --------
        self.t += 1
        terminated = False
        truncated = (self.t >= c.episode_horizon)

        # -------- 9) info for logs/dashboard --------
        info = {
            "t": t,
            "attack_present": attack_present,
            "attack_syn": syn_on,
            "attack_udp": udp_on,
            "attack_icmp": icmp_on,
            "L_total": L_total,
            "A_tcp": A_tcp,
            "A_udp": A_udp,
            "A_icmp": A_icmp,
            "A_pass": A_pass_sum,
            "L_pass": L_pass_sum,
            "T_pass": T_pass,
            "load": load,
            "latency_ms": latency,
            "loss": loss,
            "suppression": suppression,
            "retention": retention,
            "fp_penalty": fp,
            "action": int(action),
            "mode": int(mult["mode"]),
            "rate_level": rate_level,
            "reward": float(reward),
        }
        self.last_info = info

        return obs, float(reward), terminated, truncated, info
