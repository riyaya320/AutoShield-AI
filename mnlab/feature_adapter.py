from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np


MininetHost = object


@dataclass
class LinkConfig:
    # Used for queue/load proxy (rough capacity model)
    wan_capacity_mbit: float = 100.0
    interval_s: float = 1.0


@dataclass
class FlowCounters:
    # Packet counters sampled each interval (deltas become rates)
    total_pkts: int = 0
    total_bytes: int = 0
    tcp_pkts: int = 0
    udp_pkts: int = 0
    icmp_pkts: int = 0
    syn_pkts: int = 0
    ack_pkts: int = 0
    t: float = 0.0


def _run(node: MininetHost, cmd: str) -> str:
    return node.cmd(cmd)


def install_counting_rules(fw: MininetHost, lan_iface: str, wan_iface: str) -> None:
    """
    Count offered load arriving at the firewall (before it is forwarded/blocked).
    Hook into mangle/PREROUTING 
    IMPORTANT:
    We use MARK (not ACCEPT/RETURN) so counters increment WITHOUT terminating rule traversal.
    That way MN_TOTAL, MN_TCP, MN_SYN, etc. can all increment on the same packet.
    """
    _run(fw, "iptables -t mangle -N MNCOUNT 2>/dev/null || true")
    _run(fw, "iptables -t mangle -F MNCOUNT")

    # Ensure PREROUTING jumps to MNCOUNT at the top (exactly once)
    _run(fw, "iptables -t mangle -D PREROUTING -j MNCOUNT 2>/dev/null || true")
    _run(fw, "iptables -t mangle -I PREROUTING 1 -j MNCOUNT")

    # MARK target: counts + continues (does NOT accept/drop)
    mark = "MARK --set-mark 1"

    # Count everything (total offered load)
    _run(fw, f"iptables -t mangle -A MNCOUNT -m comment --comment MN_TOTAL -j {mark}")

    # Protocol mix
    _run(fw, f"iptables -t mangle -A MNCOUNT -p tcp  -m comment --comment MN_TCP  -j {mark}")
    _run(fw, f"iptables -t mangle -A MNCOUNT -p udp  -m comment --comment MN_UDP  -j {mark}")
    _run(fw, f"iptables -t mangle -A MNCOUNT -p icmp -m comment --comment MN_ICMP -j {mark}")

    # SYN only (SYN=1, ACK=0)
    _run(
    fw,
    f"iptables -t mangle -A MNCOUNT -p tcp --tcp-flags SYN,ACK SYN "
    f"-m comment --comment MN_SYN -j {mark}",
    )

    # ACK only (ACK=1, SYN=0)
    _run(
    fw,
    f"iptables -t mangle -A MNCOUNT -p tcp --tcp-flags SYN,ACK ACK "
    f"-m comment --comment MN_ACK -j {mark}",
    )

def read_iptables_counters(fw: MininetHost) -> Tuple[int, int, int, int, int, int, int]:
    """
    Read counters from mangle/MNCOUNT based on rule comments.
    Returns: total_pkts, total_bytes, tcp_pkts, udp_pkts, icmp_pkts, syn_pkts, ack_pkts
    """
    out = _run(fw, "iptables -t mangle -L MNCOUNT -v -x -n")

    def find_counter(tag: str) -> Tuple[int, int]:
        # iptables -L output ends the rule line with: /* TAG */ ... maybe followed by extra text
        # We match pkts+bytes at line start, then find the comment anywhere on the line.
        m = re.search(
            rf"^\s*(\d+)\s+(\d+)\s+.*?/\*\s*{re.escape(tag)}\s*\*/.*$",
            out,
            re.MULTILINE,
        )
        if not m:
            return 0, 0
        return int(m.group(1)), int(m.group(2))

    total_pkts, total_bytes = find_counter("MN_TOTAL")
    tcp_pkts, _ = find_counter("MN_TCP")
    udp_pkts, _ = find_counter("MN_UDP")
    icmp_pkts, _ = find_counter("MN_ICMP")
    syn_pkts, _ = find_counter("MN_SYN")
    ack_pkts, _ = find_counter("MN_ACK")

    return total_pkts, total_bytes, tcp_pkts, udp_pkts, icmp_pkts, syn_pkts, ack_pkts

def ping_latency_and_loss(src: MininetHost, dst_ip: str) -> Tuple[float, float]:
    """
    Returns (latency_ms, loss) from one ping.
    If ping fails: latency_ms=1000.0, loss=1.0
    """
    out = _run(src, f"ping -c 1 -W 1 {dst_ip}")
    if "1 received" not in out and "1 packets received" not in out:
        return 1000.0, 1.0
    # Extract time=XX ms
    m = re.search(r"time=([\d\.]+)\s*ms", out)
    latency = float(m.group(1)) if m else 1000.0
    return latency, 0.0


def action_to_mode_level(action: int) -> Tuple[float, float]:
    """
    Mirrors your env meaning:
    firewall_mode (0 allow, 1 rate, 2 block, 3 global_rate)
    rate_limit_level (0..1)
    """
    if action == 0:
        return 0.0, 0.0
    if action in (1, 2):
        return 1.0, 0.75
    if action == 3:
        return 1.0, 0.80
    if action in (4, 5, 6):
        return 2.0, 0.0
    if action == 7:
        return 3.0, 0.70
    return 0.0, 0.0


class MininetFeatureAdapter:
    """
    Produces your 13-dim observation using real Mininet stats.
    """

    def __init__(self, cfg: LinkConfig):
        self.cfg = cfg
        self.prev = FlowCounters(t=time.time())

    def sample_pre(self, fw: MininetHost) -> FlowCounters:
        total_pkts, total_bytes, tcp_pkts, udp_pkts, icmp_pkts, syn_pkts, ack_pkts = read_iptables_counters(fw)
        return FlowCounters(
            total_pkts=total_pkts,
            total_bytes=total_bytes,
            tcp_pkts=tcp_pkts,
            udp_pkts=udp_pkts,
            icmp_pkts=icmp_pkts,
            syn_pkts=syn_pkts,
            ack_pkts=ack_pkts,
            t=time.time(),
        )

    def rates_from_delta(self, cur: FlowCounters, prev: FlowCounters) -> Dict[str, float]:
        dt = max(1e-6, cur.t - prev.t)

        def safe_delta(cur_val, prev_val):
            delta = cur_val -prev_val
            # If negative (counter reser), use current value as delta
            return max(0, delta) if delta >= 0 else cur_val

        pkt_rate_total = safe_delta(cur.total_pkts, prev.total_pkts) / dt
        bytes_rate_total = safe_delta(cur.total_bytes, prev.total_bytes) / dt

        tcp_rate = safe_delta(cur.tcp_pkts, prev.tcp_pkts) / dt
        udp_rate = safe_delta(cur.udp_pkts, prev.udp_pkts) / dt
        icmp_rate = safe_delta(cur.icmp_pkts,  prev.icmp_pkts) / dt
        syn_rate = safe_delta(cur.syn_pkts,  prev.syn_pkts) / dt
        ack_rate = safe_delta (cur.ack_pkts,  prev.ack_pkts) / dt

        ack = max(ack_rate, 1.0)  # 1 packet/s floor
        syn_ack_ratio = syn_rate / ack
        syn_ack_ratio = min(syn_ack_ratio, 50.0)   # tighter cap for training/demo
  
        # queue/load proxy: how much offered load vs link capacity
        cap_bytes_s = (self.cfg.wan_capacity_mbit * 1_000_000) / 8.0
        load = bytes_rate_total / max(1e-6, cap_bytes_s)
        queue_proxy = max(0.0, load - 1.0)

        return {
            "pkt_rate_total_pre": pkt_rate_total,
            "bytes_rate_total_pre": bytes_rate_total,
            "tcp_rate_pre": tcp_rate,
            "udp_rate_pre": udp_rate,
            "icmp_rate_pre": icmp_rate,
            "syn_rate_pre": syn_rate,
            "ack_rate_pre": ack_rate,
            "syn_ack_ratio_pre": syn_ack_ratio,
            "queue_proxy_post": queue_proxy,
        }

    def build_obs(
        self,
        rates: Dict[str, float],
        latency_ms_post: float,
        loss_post: float,
        firewall_mode: float,
        rate_limit_level: float,
    ) -> np.ndarray:
        obs = np.array(
            [
                rates["pkt_rate_total_pre"],
                rates["bytes_rate_total_pre"],
                rates["tcp_rate_pre"],
                rates["udp_rate_pre"],
                rates["icmp_rate_pre"],
                rates["syn_rate_pre"],
                rates["ack_rate_pre"],
                rates["syn_ack_ratio_pre"],
                latency_ms_post,
                loss_post,
                rates["queue_proxy_post"],
                firewall_mode,
                rate_limit_level,
            ],
            dtype=np.float32,
        )
        return obs
