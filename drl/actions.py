ACTIONS = {
    0: {
        "name": "ALLOW_ALL",
        "cmd": (
            "iptables -F PPO_SYN; "
            "iptables -A PPO_SYN -j ACCEPT"
        )
    },

    1: {
        "name": "MILD",
        "cmd": (
            "iptables -F PPO_SYN; "
            "iptables -A PPO_SYN "
            "-m hashlimit --hashlimit-name syn_a1 "
            "--hashlimit 10000/second --hashlimit-burst 20000 "
            "--hashlimit-mode srcip --hashlimit-htable-expire 10000 "
            "-j ACCEPT; "
            "iptables -A PPO_SYN -j DROP"
        )
    },

    2: {
        "name": "MEDIUM",
        "cmd": (
            "iptables -F PPO_SYN; "
            "iptables -A PPO_SYN "
            "-m hashlimit --hashlimit-name syn_a2 "
            "--hashlimit 2000/second --hashlimit-burst 4000 "
            "--hashlimit-mode srcip --hashlimit-htable-expire 10000 "
            "-j ACCEPT; "
            "iptables -A PPO_SYN -j DROP"
        )
    },

    3: {
        "name": "STRICT",
        "cmd": (
            "iptables -F PPO_SYN; "
            "iptables -A PPO_SYN "
            "-m hashlimit --hashlimit-name syn_a3 "
            "--hashlimit 300/second --hashlimit-burst 600 "
            "--hashlimit-mode srcip --hashlimit-htable-expire 10000 "
            "-j ACCEPT; "
            "iptables -A PPO_SYN -j DROP"
        )
    },

    4: {
        "name": "VERY_STRICT",
        "cmd": (
            "iptables -F PPO_SYN; "
            "iptables -A PPO_SYN "
            "-m hashlimit --hashlimit-name syn_a4 "
            "--hashlimit 50/second --hashlimit-burst 100 "
            "--hashlimit-mode srcip --hashlimit-htable-expire 10000 "
            "-j ACCEPT; "
            "iptables -A PPO_SYN -j DROP"
        )
    },
}
