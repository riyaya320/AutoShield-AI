# cli/runtime.py
from __future__ import annotations
from dataclass import dataclass
from mininet.net import Mininet

import time
from dataclasses import dataclass
from typing import Optional

SRV_IP = "10.0.2.10"
IPERF_PORT = 5201

@dataclass
class LabHandles:
    net: object
    fw: object
    srv: object
    hleg1: object
    hatk1: object

def start_lab() -> LabHandles:
    # Import here so CLI loads fast & avoids Mininet import unless needed
    from mnlab.topo import build_net

    net = build_net(start=True, do_sanity=False)
    fw = net.get("fw")
    srv = net.get("srv")
    hleg1 = net.get("hleg1")
    hatk1 = net.get("hatk1")

    return LabHandles(
        net=net, 
        fw=net.get("fw"), 
        srv=net.get("srv"),
        hleg1=net.get("hleg1"), 
        hatk1=net.get("hatk1"),
        )

def stop_lab(handles: LabHandles) -> None:
    # Best-effort cleanup
    try:
        handles.hatk1.cmd("pkill -f hping3 || true")
        handles.hleg1.cmd('pkill -f "iperf3 -c" || true')
        handles.srv.cmd("pkill -f iperf3 || true")
    except Exception:
        pass
    try:
        handles.net.stop()
    except Exception:
        pass

def ensure_forward_hook(handles: LabHandles) -> None:
    """
    Ensures the FORWARD -> PPO_SYN hook exists on fw.
    Keeps this deterministic for CLI runs.
    """
    fw = handles.fw
    fw.cmd("sysctl -w net.ipv4.ip_forward=1 >/dev/null")

    # Minimal forward policy:
    fw.cmd("iptables -N PPO_SYN 2>/dev/null || true")
    fw.cmd("iptables -C FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT 2>/dev/null || "
           "iptables -A FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT")
    fw.cmd("iptables -C FORWARD -d 10.0.2.10 -p tcp --syn -j PPO_SYN 2>/dev/null || "
           "iptables -A FORWARD -d 10.0.2.10 -p tcp --syn -j PPO_SYN")
    fw.cmd("iptables -C FORWARD -j ACCEPT 2>/dev/null || iptables -A FORWARD -j ACCEPT")

def start_traffic(handles: LabHandles, attack: bool = True) -> None:
    """
    Starts:
      - iperf3 server on srv
      - long-running iperf3 client on hleg1 -> srv writing /tmp/leg_hleg1.log
      - optional SYN flood on hatk1 -> srv
    """
    srv = handles.srv
    hleg1 = handles.hleg1
    hatk1 = handles.hatk1

    srv.cmd("pkill -f iperf3 || true")
    srv.cmd(f"iperf3 -s -p {IPERF_PORT} -D")

    hleg1.cmd('pkill -f "iperf3 -c" || true')
    hleg1.cmd(
        f'sh -c "iperf3 -c {SRV_IP} -p {IPERF_PORT} -t 9999 -i 1 > /tmp/leg_hleg1.log 2>&1 &"'
    )

    if attack:
        hatk1.cmd("pkill -f hping3 || true")
        hatk1.cmd(
            f'sh -c "hping3 -S -p {IPERF_PORT} --flood {SRV_IP} > /tmp/atk_hatk1.log 2>&1 &"'
        )

def stop_traffic(handles: LabHandles) -> None:
    try:
        handles.hatk1.cmd("pkill -f hping3 || true")
        handles.hleg1.cmd('pkill -f "iperf3 -c" || true')
        handles.srv.cmd("pkill -f iperf3 || true")
    except Exception:
        pass
