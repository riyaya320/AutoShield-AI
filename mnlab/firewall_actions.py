# mnlab/firewall_actions.py
from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Optional, Dict

# Mininet "Host" type is not importable cleanly for type checking everywhere;
# we just treat it as an object exposing .cmd(str)->str and .name
# (which Mininet Host does).
MininetHost = object


@dataclass(frozen=True)
class FirewallIfaces:
    lan: str  # fw-eth0
    wan: str  # fw-eth1


# Your action mapping (locked)
ACTION_MEANINGS: Dict[int, str] = {
    0: "ALLOW_ALL",
    1: "RATE_LIMIT_TCP",
    2: "RATE_LIMIT_UDP",
    3: "RATE_LIMIT_ICMP",
    4: "BLOCK_TCP",
    5: "BLOCK_UDP",
    6: "BLOCK_ICMP",
    7: "GLOBAL_RATE_LIMIT",
}


def _run(fw: MininetHost, cmd: str) -> str:
    """Run a command on the firewall node."""
    return fw.cmd(cmd)


def detect_fw_ifaces(fw: MininetHost) -> FirewallIfaces:
    """
    Detect LAN/WAN interfaces by IP address (robust to Mininet eth ordering).
    Assumes:
      LAN side: 10.0.1.1/24
      WAN side: 10.0.2.1/24
    """
    out = _run(fw, "ip -o -4 addr show | awk '{print $2, $4}'")

    lan = None
    wan = None

    for line in out.splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        iface, cidr = parts
        ip = cidr.split("/")[0]
        iface = iface.split("@", 1)[0]  # strip @ifXX if present

        if ip == "10.0.1.1":
            lan = iface
        elif ip == "10.0.2.1":
            wan = iface

    if not lan or not wan:
        raise RuntimeError(f"Could not detect LAN/WAN by IP. Got:\n{out}")

    return FirewallIfaces(lan=lan, wan=wan)


def reset_firewall(fw: MininetHost, ifaces: Optional[FirewallIfaces] = None) -> None:
    """
    Clear iptables rules and tc qdiscs so each experiment starts clean.
    """
    if ifaces is None:
        ifaces = detect_fw_ifaces(fw)
    # Flush filter rules (IPv4)
    _run(fw, "iptables -F")
    _run(fw, "iptables -X")
    _run(fw, "iptables -t nat -F")
    # Set default policies to ACCEPT (we control via explicit rules)
    _run(fw, "iptables -P INPUT ACCEPT")
    _run(fw, "iptables -P OUTPUT ACCEPT")
    _run(fw, "iptables -P FORWARD ACCEPT")
    # Clear any existing traffic control shaping on WAN interface
    _run(fw, f"tc qdisc del dev {ifaces.wan} root 2>/dev/null || true")
    _run(fw, f"tc qdisc del dev {ifaces.wan} ingress 2>/dev/null || true")
    
    # === ADD THESE LINES ===
    # Re-attach MNCOUNT to PREROUTING (in case it was removed)
   # _run(fw, "iptables -t mangle -D PREROUTING -j MNCOUNT 2>/dev/null || true")
   # _run(fw, "iptables -t mangle -I PREROUTING 1 -j MNCOUNT")
    # === END NEW LINES ===
    fw.cmd("iptables -t mangle -D PREROUTING -j MNCOUNT 2>/dev/null || true")
    fw.cmd("iptables -t mangle -I PREROUTING 1 -j MNCOUNT")

def _apply_block_proto(fw: MininetHost, proto: str, ifaces: FirewallIfaces) -> None:
    """
    Block a protocol across forwarding path (LAN->WAN and WAN->LAN).
    We block in FORWARD chain so fw can still be managed.
    """
    # Forward path blocks
    _run(fw, f"iptables -A FORWARD -i {ifaces.lan} -o {ifaces.wan} -p {proto} -j DROP")
    _run(fw, f"iptables -A FORWARD -i {ifaces.wan} -o {ifaces.lan} -p {proto} -j DROP")


def _apply_global_rate_limit(fw: MininetHost, ifaces: FirewallIfaces, rate_mbit: int) -> None:
    """
    Apply a global egress rate limit on the WAN interface using TBF.
    This is simple and stable for demos.
    """
    # root qdisc TBF: rate, burst, latency
    _run(
        fw,
        f"tc qdisc replace dev {ifaces.wan} root tbf rate {rate_mbit}mbit burst 32kbit latency 400ms",
    )


def _apply_protocol_rate_limit(
    fw: MininetHost,
    ifaces: FirewallIfaces,
    proto: str,
    rate_mbit: int,
) -> None:
    """
    Rate-limit a specific protocol using:
    - iptables mangle MARK packets
    - tc HTB classes + fw filter (handle mark)
    This is closer to “real firewall shaping” than global-only.
    """
    # Clear any existing shaping first
    _run(fw, f"tc qdisc del dev {ifaces.wan} root 2>/dev/null || true")
    _run(fw, "iptables -t mangle -F")

    # 1) Mark protocol packets leaving LAN towards WAN
    mark = {"tcp": 10, "udp": 20, "icmp": 30}[proto]
    _run(
        fw,
        f"iptables -t mangle -A FORWARD -i {ifaces.lan} -o {ifaces.wan} -p {proto} -j MARK --set-mark {mark}",
    )

    # 2) HTB root with default class for everything else
    #    We set a generous default for non-target traffic.
    _run(fw, f"tc qdisc add dev {ifaces.wan} root handle 1: htb default 30")
    _run(fw, f"tc class add dev {ifaces.wan} parent 1: classid 1:1 htb rate 1000mbit")

    # Target class (limited)
    _run(fw, f"tc class add dev {ifaces.wan} parent 1:1 classid 1:{mark} htb rate {rate_mbit}mbit ceil {rate_mbit}mbit")

    # Default class (not limited much)
    _run(fw, f"tc class add dev {ifaces.wan} parent 1:1 classid 1:30 htb rate 1000mbit ceil 1000mbit")

    # 3) Attach filter: fw mark -> class
    _run(fw, f"tc filter add dev {ifaces.wan} parent 1: protocol ip handle {mark} fw flowid 1:{mark}")


def apply_action(fw: MininetHost, action: int, ifaces: Optional[FirewallIfaces] = None) -> str:
    """
    Apply one of the 8 actions in your RL environment to the real firewall node.
    Returns the action name (string) for logging.
    """
    if ifaces is None:
        ifaces = detect_fw_ifaces(fw)

    if action not in ACTION_MEANINGS:
        raise ValueError(f"Invalid action {action}. Expected 0..7")

    # Always start from clean state for deterministic behavior
    reset_firewall(fw, ifaces)

    name = ACTION_MEANINGS[action]

    # Rate limit values (Mb/s): tune later.
    # Keep them conservative so effects are visible in metrics.
    RATE_TCP = 20
    RATE_UDP = 20
    RATE_ICMP = 10
    RATE_GLOBAL = 30

    if action == 0:
        # allow all: already clean
        return name

    if action == 1:
        _apply_protocol_rate_limit(fw, ifaces, "tcp", RATE_TCP)
        return name

    if action == 2:
        _apply_protocol_rate_limit(fw, ifaces, "udp", RATE_UDP)
        return name

    if action == 3:
        _apply_protocol_rate_limit(fw, ifaces, "icmp", RATE_ICMP)
        return name

    if action == 4:
        _apply_block_proto(fw, "tcp", ifaces)
        return name

    if action == 5:
        _apply_block_proto(fw, "udp", ifaces)
        return name

    if action == 6:
        _apply_block_proto(fw, "icmp", ifaces)
        return name

    if action == 7:
        _apply_global_rate_limit(fw, ifaces, RATE_GLOBAL)
        return name

    return name  # unreachable


def firewall_status(fw: MininetHost, ifaces: Optional[FirewallIfaces] = None) -> str:
    """Return a compact status snapshot for debugging/logging."""
    if ifaces is None:
        ifaces = detect_fw_ifaces(fw)

    ipt = _run(fw, "iptables -S")
    tc = _run(fw, f"tc -s qdisc show dev {ifaces.wan} || true")
    return f"=== iptables ===\n{ipt}\n\n=== tc ({ifaces.wan}) ===\n{tc}\n"
