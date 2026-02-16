# cli/metrics.py
from __future__ import annotations

import re
from dataclasses import dataclass

SRV_IP = "10.0.2.10"

@dataclass
class Metrics:
    syn_accept_pkts: int
    syn_drop_pkts: int
    syn_accept_rate: int
    syn_drop_rate: int
    rtt_ms: float
    legit_mbps: float

_rule_re = re.compile(r"^\s*\d+\s+")

def read_accept_drop_totals(fw) -> tuple[int, int]:
    out = fw.cmd("iptables -L PPO_SYN -n -v -x --line-numbers")
    lines = [ln for ln in out.splitlines() if ln.strip()]
    rule_lines = [ln for ln in lines if _rule_re.match(ln)]
    if not rule_lines:
        return 0, 0

    # Rule1: ACCEPT, Rule2: DROP (if exists)
    r1 = rule_lines[0].split()
    accept_pkts = int(r1[1])
    drop_pkts = 0
    if len(rule_lines) >= 2:
        r2 = rule_lines[1].split()
        drop_pkts = int(r2[1])
    return accept_pkts, drop_pkts

def ping_rtt_ms(hleg1, fallback: float = 200.0) -> float:
    out = hleg1.cmd(f"ping -c 1 -W 1 {SRV_IP}")
    m = re.search(r"time=([\d\.]+)\s*ms", out)
    return float(m.group(1)) if m else fallback

def read_legit_mbps_from_log(hleg1) -> float:
    out = hleg1.cmd("tail -n 3 /tmp/leg_hleg1.log")
    m = re.search(r"([\d\.]+)\s+Gbits/sec", out)
    if m:
        return float(m.group(1)) * 1000.0
    m = re.search(r"([\d\.]+)\s+Mbits/sec", out)
    if m:
        return float(m.group(1))
    return 0.0

def _read_iface_bytes(node, intf: str, which: str = "rx") -> int:
    out = node.cmd(f"cat /proc/net/dev | grep '{intf}:'")
    m = re.search(
        rf"{intf}:\s*(\d+)\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)",
        out
    )
    if not m:
        return 0
    rx_bytes = int(m.group(1))
    tx_bytes = int(m.group(2))
    return rx_bytes if which == "rx" else tx_bytes


def throughput_mbps_from_tx_delta(tx_now: int, tx_prev: int, interval_s: float = 1.0) -> float:
    if tx_now <= tx_prev:
        return 0.0
    delta_bytes = tx_now - tx_prev
    return (delta_bytes * 8.0) / (1_000_000.0 * interval_s)


def sample_metrics(fw, hleg1, prev_accept: int, prev_drop: int, prev_tx: int):
    a, d = read_accept_drop_totals(fw)
    accept_rate = max(0, a - prev_accept)
    drop_rate = max(0, d - prev_drop)

    rtt = ping_rtt_ms(hleg1)

    tx_now = _read_iface_bytes(hleg1, "hleg1-eth0", which="tx")
    mbps = throughput_mbps_from_tx_delta(tx_now, prev_tx, 1.0)

    m = Metrics(
        syn_accept_pkts=a,
        syn_drop_pkts=d,
        syn_accept_rate=accept_rate,
        syn_drop_rate=drop_rate,
        rtt_ms=rtt,
        legit_mbps=mbps,
    )
    return m, a, d, tx_now



