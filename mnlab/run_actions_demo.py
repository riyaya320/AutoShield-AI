from __future__ import annotations

import os
import time
from dataclasses import dataclass

from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info

from mnlab.topo import DDoSFirewallTopo, setup_ip_routing
from mnlab.firewall_actions import apply_action, firewall_status, detect_fw_ifaces


@dataclass
class DemoConfig:
    ip_srv: str = "10.0.2.10"
    ip_fw_lan: str = "10.0.1.1"
    warmup_s: float = 0.5
    iperf_time_s: int = 2


def start_iperf_server(srv) -> None:
    # Kill any old iperf3 servers and start fresh in daemon mode
    srv.cmd("pkill -f 'iperf3 -s' 2>/dev/null || true")
    srv.cmd("iperf3 -s -D")


def stop_iperf_server(srv) -> None:
    srv.cmd("pkill -f 'iperf3 -s' 2>/dev/null || true")


def test_ping(host, dst_ip: str, count: int = 2) -> str:
    return host.cmd(f"ping -c {count} -W 1 {dst_ip}")


def test_iperf_tcp(client, dst_ip: str, seconds: int = 2) -> str:
    # single TCP stream
    return client.cmd(f"iperf3 -c {dst_ip} -t {seconds} --connect-timeout 1000")


def test_iperf_udp(client, dst_ip: str, seconds: int = 2, mbit: int = 50) -> str:
    # UDP test at target rate
    return client.cmd(f"iperf3 -u -c {dst_ip} -t {seconds} -b {mbit}M --connect-timeout 1000")


def run_demo() -> None:
    setLogLevel("info")
    cfg = DemoConfig()

    topo = DDoSFirewallTopo()
    info("\n*** Cleaning up old Mininet state\n")
    os.system("mn -c > /dev/null 2>&1")

    net = Mininet(
        topo=topo,
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True,
        build=False,
    )

    info("\n*** Building & starting Mininet\n")
    net.build()
    net.start()

    try:
        setup_ip_routing(net)

        # Grab nodes
        fw = net.get("fw")
        srv = net.get("srv")
        hleg1 = net.get("hleg1")
        hatk1 = net.get("hatk1")

        ifaces = detect_fw_ifaces(fw)

        info("\n*** Starting iperf3 server on srv\n")
        start_iperf_server(srv)
        time.sleep(cfg.warmup_s)

        info("\n*** Baseline sanity checks (before actions)\n")
        info(test_ping(hleg1, cfg.ip_fw_lan))
        info(test_ping(hleg1, cfg.ip_srv))
        info(test_iperf_tcp(hleg1, cfg.ip_srv, cfg.iperf_time_s))

        # Apply all actions 0..7 and test
        for action in range(8):
            info("\n" + "=" * 70 + "\n")
            info(f"*** Applying action {action}\n")
            name = apply_action(fw, action, ifaces)

            # Small pause so rules/qdisc settle
            time.sleep(cfg.warmup_s)

            info(f"*** Action name: {name}\n")

            # Basic connectivity
            info("\n--- ping: hleg1 -> fw (LAN gw)\n")
            info(test_ping(hleg1, cfg.ip_fw_lan))

            info("\n--- ping: hleg1 -> srv (routed)\n")
            info(test_ping(hleg1, cfg.ip_srv))

            # TCP throughput (legit)
            info("\n--- iperf3 TCP: hleg1 -> srv\n")
            info(test_iperf_tcp(hleg1, cfg.ip_srv, cfg.iperf_time_s))

            # UDP throughput (legit) - useful to see UDP shaping/blocking
            info("\n--- iperf3 UDP: hleg1 -> srv (50 Mbit/s target)\n")
            info(test_iperf_udp(hleg1, cfg.ip_srv, cfg.iperf_time_s, mbit=50))

            # Optional: show what happens if attacker tries UDP flood
            info("\n--- iperf3 UDP: hatk1 -> srv (attacker-ish 100 Mbit/s target)\n")
            info(test_iperf_udp(hatk1, cfg.ip_srv, cfg.iperf_time_s, mbit=100))

            # Print firewall state for screenshots
            info("\n--- Firewall status snapshot (iptables + tc)\n")
            info(firewall_status(fw, ifaces))

        info("\n*** Demo complete.\n")

    finally:
        info("\n*** Stopping iperf3 server\n")
        try:
            stop_iperf_server(net.get("srv"))
        except Exception:
            pass

        info("\n*** Stopping Mininet\n")
        net.stop()


if __name__ == "__main__":
    run_demo()
