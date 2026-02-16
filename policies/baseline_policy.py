# baseline_policy.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class BaselineConfig:
    # Thresholds (tune to match your environment scale)
    pkt_rate_attack: float = 9000.0      # if total pkt/s above this -> suspicious
    syn_ack_ratio_attack: float = 3.5    # above this -> SYN flood likely

    udp_share_attack: float = 0.55       # if UDP dominates heavily -> UDP flood
    icmp_share_attack: float = 0.20      # if ICMP unusually high -> ICMP flood

    # Conservativeness: prefer rate-limit before block
    prefer_block: bool = False


class StaticFirewallPolicy:
    """
    Uses RAW (unnormalized) logic assumptions, but we only have normalized obs in env.
    So this expects you to pass `info` from env.step() OR keep a parallel unnormalized state.
    The simplest: decide using env.info fields (recommended).
    """

    def __init__(self, cfg: BaselineConfig | None = None):
        self.cfg = cfg or BaselineConfig()

    def act(self, info: dict) -> int:
        """
        Returns one of the 8 actions:
        0 allow
        1 rate-limit TCP
        2 rate-limit UDP
        3 rate-limit ICMP
        4 block TCP
        5 block UDP
        6 block ICMP
        7 global rate-limit
        """
        c = self.cfg

        # Use info that has protocol rates pre-firewall:
        # We stored A_* and L_total, but not explicit tcp/udp/icmp pre totals in info.
        # We'll infer using attack flags + expected legit split.
        # Better approach: extend env.info to include tcp_rate_pre etc if you want.
        # For now: use total load + which attacks are active (info gives ground-truth).
        # That still counts as a baseline because it doesn't "learn".
        attack_present = info.get("attack_present", False)
        syn_on = info.get("attack_syn", False)
        udp_on = info.get("attack_udp", False)
        icmp_on = info.get("attack_icmp", False)

        load = info.get("load", 0.0)
        latency = info.get("latency_ms", 0.0)

        # If the network is melting, apply global rate-limit (simple safety control)
        if load > 1.2 or latency > 200:
            return 7  # global rate-limit

        if not attack_present:
            return 0  # allow all

        # Vector-specific response
        if syn_on:
            return 4 if c.prefer_block else 1

        if udp_on and not syn_on:
            return 5 if c.prefer_block else 2

        if icmp_on and not syn_on and not udp_on:
            return 6 if c.prefer_block else 3

        # If multiple vectors suspected, global rate-limit
        if (syn_on and udp_on) or (udp_on and icmp_on):
            return 7

        return 0
