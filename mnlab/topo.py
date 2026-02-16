# mnlab/topo.py
from __future__ import annotations

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.net import Controller
from mininet.node import OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI


class DDoSFirewallTopo(Topo):
    """
    LAN side: hleg1, hleg2, hatk1, hatk2 -> s1 -> fw -> s2 -> srv
    fw is a Linux host acting as a router/firewall (IP forwarding enabled).
    """

    def build(self):
        # Switches
        s1 = self.addSwitch("s1", failMode="standalone")
        s2 = self.addSwitch("s2", failMode="standalone")

        # Hosts (legitimate clients + attackers)
        hleg1 = self.addHost("hleg1")
        hleg2 = self.addHost("hleg2")
        hatk1 = self.addHost("hatk1")
        hatk2 = self.addHost("hatk2")

        # Server + Firewall
        srv = self.addHost("srv")
        fw = self.addHost("fw")

        # hosts -> s1 (LAN)
        self.addLink(hleg1, s1, cls=TCLink, bw=100, delay="1ms", use_htb=False)
        self.addLink(hleg2, s1, cls=TCLink, bw=100, delay="1ms", use_htb=False)
        self.addLink(hatk1, s1, cls=TCLink, bw=100, delay="1ms", use_htb=False)
        self.addLink(hatk2, s1, cls=TCLink, bw=100, delay="1ms", use_htb=False)

        # s1 <-> fw (LAN uplink)
        self.addLink(s1, fw, cls=TCLink, bw=100, delay="1ms", use_htb=False)

        # fw <-> s2 (WAN/uplink)
        self.addLink(fw, s2, cls=TCLink, bw=50, delay="2ms", use_htb=False)

        #  s2 <-> srv (server access)
        self.addLink(s2, srv, cls=TCLink, bw=50, delay="2ms", r2q=1000)

def _flush_and_set_ip(host, intf: str, ip_cidr: str) -> None:
    host.cmd(f"ip addr flush dev {intf}")
    host.cmd(f"ip addr add {ip_cidr} dev {intf}")
    host.cmd(f"ip link set {intf} up")


def _fw_intf_names(fw) -> list[str]:
    # Mininet host has intfNames()
    return [i for i in fw.intfNames() if i != "lo"]


def _auto_assign_fw_ips(fw) -> tuple[str, str]:
    ifnames = _fw_intf_names(fw)
    if len(ifnames) < 2:
        raise RuntimeError(f"Expected fw to have 2 interfaces, found: {ifnames}")

    # Prefer explicit names if present
    lan = next((i for i in ifnames if i.startswith("fw-eth0")), None)
    wan = next((i for i in ifnames if i.startswith("fw-eth1")), None)

    # Fallback: just take first two
    if lan is None or wan is None:
        lan, wan = ifnames[0], ifnames[1]

    for intf in (lan, wan):
        fw.cmd(f"ip addr flush dev {intf}")
        fw.cmd(f"ip link set {intf} up")

    fw.cmd(f"ip addr add 10.0.1.1/24 dev {lan}")
    fw.cmd(f"ip addr add 10.0.2.1/24 dev {wan}")

    return lan, wan


def install_mncount_rules(fw) -> None:
    """
    Create MNCOUNT chain and attach it to mangle/PREROUTING.
    Uses MARK target so counters increment without terminating rule traversal.
    """
    fw.cmd("iptables -t mangle -N MNCOUNT 2>/dev/null || true")
    fw.cmd("iptables -t mangle -F MNCOUNT")
    fw.cmd("iptables -t mangle -D PREROUTING -j MNCOUNT 2>/dev/null || true")
    fw.cmd("iptables -t mangle -I PREROUTING 1 -j MNCOUNT")

    # Count everything first
    fw.cmd("iptables -t mangle -A MNCOUNT -m comment --comment MN_TOTAL -j MARK --set-mark 1")

    # Protocol mix
    fw.cmd("iptables -t mangle -A MNCOUNT -p tcp  -m comment --comment MN_TCP  -j MARK --set-mark 1")
    fw.cmd("iptables -t mangle -A MNCOUNT -p udp  -m comment --comment MN_UDP  -j MARK --set-mark 1")
    fw.cmd("iptables -t mangle -A MNCOUNT -p icmp -m comment --comment MN_ICMP -j MARK --set-mark 1")

    # SYN/ACK (DDoS indicators)
    fw.cmd("iptables -t mangle -A MNCOUNT -p tcp --tcp-flags SYN SYN -m comment --comment MN_SYN -j MARK --set-mark 1")
    fw.cmd("iptables -t mangle -A MNCOUNT -p tcp --tcp-flags ACK ACK -m comment --comment MN_ACK -j MARK --set-mark 1")

def setup_ip_routing(net):
    """
    Deterministic IP addressing + routing.
    Prevents accidental IP conflicts.
    """
    fw = net.get("fw")
    srv = net.get("srv")
    hleg1 = net.get("hleg1")
    hleg2 = net.get("hleg2")
    hatk1 = net.get("hatk1")
    hatk2 = net.get("hatk2")

    # ---- Firewall interfaces (auto-detect, no eth assumptions) ----
    fw_lan, fw_wan = _auto_assign_fw_ips(fw)

    # ---- LAN hosts (s1 side) ----
    _flush_and_set_ip(hleg1, "hleg1-eth0", "10.0.1.11/24")
    _flush_and_set_ip(hleg2, "hleg2-eth0", "10.0.1.12/24")
    _flush_and_set_ip(hatk1, "hatk1-eth0", "10.0.1.21/24")
    _flush_and_set_ip(hatk2, "hatk2-eth0", "10.0.1.22/24")

    # Default route via firewall LAN IP
    for h in [hleg1, hleg2, hatk1, hatk2]:
        h.cmd("ip route del default 2>/dev/null || true")
        h.cmd("ip route add default via 10.0.1.1")

    # ---- Server (s2 side) ----
    _flush_and_set_ip(srv, "srv-eth0", "10.0.2.10/24")
    srv.cmd("ip route del default 2>/dev/null || true")
    srv.cmd("ip route add default via 10.0.2.1")

    # Enable forwarding on firewall
    fw.cmd("sysctl -w net.ipv4.ip_forward=1 >/dev/null")


    # Clear ARP/neighbor caches (avoids stale entries after previous runs)
    for h in [fw, srv, hleg1, hleg2, hatk1, hatk2]:
        h.cmd("ip neigh flush all >/dev/null 2>&1 || true")

    #Now install counting rules (starts from clean slate)
    install_mncount_rules(fw)

    info("\n *** DEBUG: MNCOUNT inside fw namespace\n")
    info(fw.cmd("iptables -t mangle -L MNCOUNT -v -x -n | head -n 20") + "\n")

    # Helpful debug (leave it, it’s good for thesis + sanity)
    info("\n*** DEBUG: fw interfaces after IP assignment\n")
    info(fw.cmd("ip -br addr") + "\n")
    info(f"*** fw_lan={fw_lan}, fw_wan={fw_wan}\n")


def sanity_test(net: Mininet) -> None:
    """
    Quick ping tests to confirm routing through firewall works.
    """
    hleg1 = net.get("hleg1")
    srv = net.get("srv")
    info("\n*** Sanity test: hleg1 -> srv ping\n")
    info(hleg1.cmd("ping -c 2 10.0.2.10"))


def build_net(start: bool = True, do_sanity: bool = True) -> Mininet:
    """
    Build the Mininet topology and (optionally) start it, configure routing, and run sanity checks.
    Returns the live Mininet object so other modules (e.g., PPO training) can control it.
    """
    topo = DDoSFirewallTopo()
    net = Mininet(
        topo=topo,
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True,
        build=False,
    )

    info("\n*** Building network\n")
    net.build()

    if start:
        info("\n*** Starting network\n")
        net.start()
        setup_ip_routing(net)
        if do_sanity:
            sanity_test(net)

    return net

def run_cli() -> None:
    link = TCLink
    setLogLevel("info")

    info("\n*** Building and starting network\n")
    net = build_net(start=True, do_sanity=True)

    info("\n*** Network is up. Entering Mininet CLI\n")
    CLI(net)

    info("\n*** Stopping network\n")
    net.stop()


if __name__ == "__main__":
    run_cli()
