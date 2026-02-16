# cli/loadgen.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import ipaddress
import time


SRV_IP_DEFAULT = "10.0.2.10"


@dataclass
class LoadgenConfig:
    srv_ip: str = SRV_IP_DEFAULT
    port: int = 8080
    duration_s: int = 60

    # Conservative caps (keep thesis-safe + reproducible)
    conn_events_per_sec: int = 50          # connection churn events/sec
    http_concurrency: int = 20             # max simultaneous HTTP workers
    http_target_rps: int = 50              # total target requests/sec (approx)
    max_total_events: int = 10_000         # hard stop safety cap


def _guard_lab_ip(ip: str) -> None:
    """
    Safety: refuse to run unless target is in RFC1918 space and specifically inside 10.0.0.0/8.
    Also refuses non-IPv4.
    """
    addr = ipaddress.ip_address(ip)
    if addr.version != 4:
        raise ValueError("Loadgen only supports IPv4 targets.")
    if not addr.is_private:
        raise ValueError("Refusing to run loadgen against non-private IP.")
    if not ipaddress.ip_network("10.0.0.0/8").supernet_of(ipaddress.ip_network(f"{ip}/32")):
        raise ValueError("Refusing to run loadgen outside 10.0.0.0/8 lab range.")


class MininetLoadGenerator:
    """
    Starts bounded, auditable load generators INSIDE a Mininet host namespace (e.g., hatk1).
    This is intended for lab-only demonstration/testing.
    """

    def __init__(self, hatk_host, cfg: LoadgenConfig):
        _guard_lab_ip(cfg.srv_ip)
        self.h = hatk_host
        self.cfg = cfg

        self._conn_pidfile = "/tmp/autoppo_conn.pid"
        self._http_pidfile = "/tmp/autoppo_http.pid"
        self._conn_log = "/tmp/autoppo_conn.log"
        self._http_log = "/tmp/autoppo_http.log"

    def stop_all(self) -> None:
        # Best-effort kill by pidfile, then by pattern
        self.h.cmd(f"sh -c 'test -f {self._conn_pidfile} && kill $(cat {self._conn_pidfile}) 2>/dev/null || true'")
        self.h.cmd(f"sh -c 'test -f {self._http_pidfile} && kill $(cat {self._http_pidfile}) 2>/dev/null || true'")
        self.h.cmd("pkill -f autoppo_conn_worker.py 2>/dev/null || true")
        self.h.cmd("pkill -f autoppo_http_worker.py 2>/dev/null || true")
        self.h.cmd(f"rm -f {self._conn_pidfile} {self._http_pidfile} 2>/dev/null || true")

    def start(self, mode: Literal["none", "conn", "http", "both"]) -> None:
        self.stop_all()

        if mode == "none":
            return

        if mode in ("conn", "both"):
            self._start_conn_churn()

        if mode in ("http", "both"):
            self._start_http_surge()

    def _start_conn_churn(self) -> None:
        """
        Connection-churn surge: bounded number of short TCP connection attempts/sec.
        This simulates "new-connection pressure" in a controlled way.
        """
        cfg = self.cfg
        # Write a small python worker to /tmp to keep the command short and auditable
        self.h.cmd(
            "sh -c 'cat > /tmp/autoppo_conn_worker.py <<\"PY\"\n"
            "import socket, time\n"
            "SRV_IP = \"" + cfg.srv_ip + "\"\n"
            "PORT = " + str(int(cfg.port)) + "\n"
            "DURATION = " + str(int(cfg.duration_s)) + "\n"
            "EPS = " + str(int(cfg.conn_events_per_sec)) + "  # events per second\n"
            "MAX_TOTAL = " + str(int(cfg.max_total_events)) + "\n"
            "\n"
            "# Token bucket pacing\n"
            "tokens = 0.0\n"
            "rate = float(EPS)\n"
            "last = time.time()\n"
            "start = last\n"
            "done = 0\n"
            "\n"
            "def attempt():\n"
            "    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "    s.settimeout(0.2)\n"
            "    try:\n"
            "        s.connect((SRV_IP, PORT))\n"
            "    except Exception:\n"
            "        pass\n"
            "    try:\n"
            "        s.close()\n"
            "    except Exception:\n"
            "        pass\n"
            "\n"
            "while True:\n"
            "    now = time.time()\n"
            "    if now - start >= DURATION:\n"
            "        break\n"
            "    if done >= MAX_TOTAL:\n"
            "        break\n"
            "    dt = now - last\n"
            "    last = now\n"
            "    tokens += dt * rate\n"
            "    if tokens < 1.0:\n"
            "        time.sleep(0.01)\n"
            "        continue\n"
            "    # spend one token\n"
            "    tokens -= 1.0\n"
            "    attempt()\n"
            "    done += 1\n"
            "\n"
            "print(f\"conn_churn_done={done}\")\n"
            "PY\n"
            "python3 /tmp/autoppo_conn_worker.py > " + self._conn_log + " 2>&1 & echo $! > " + self._conn_pidfile + "'"
        )

    def _start_http_surge(self) -> None:
        """
        HTTP request surge: bounded concurrency & bounded approximate RPS.
        Uses urllib (no external deps). Includes strict timeouts.
        """
        cfg = self.cfg
        # Derive per-worker pacing (approx). Keep it bounded.
        conc = max(1, int(cfg.http_concurrency))
        total_rps = max(1, int(cfg.http_target_rps))
        per_worker_rps = max(1, total_rps // conc)

        self.h.cmd(
            "sh -c 'cat > /tmp/autoppo_http_worker.py <<\"PY\"\n"
            "import time, threading\n"
            "from urllib.request import urlopen, Request\n"
            "\n"
            "SRV_IP = \"" + cfg.srv_ip + "\"\n"
            "PORT = " + str(int(cfg.port)) + "\n"
            "DURATION = " + str(int(cfg.duration_s)) + "\n"
            "CONC = " + str(int(conc)) + "\n"
            "RPS = " + str(int(per_worker_rps)) + "  # per worker approx\n"
            "MAX_TOTAL = " + str(int(cfg.max_total_events)) + "\n"
            "\n"
            "URL = f\"http://{SRV_IP}:{PORT}/\"\n"
            "stop_at = time.time() + DURATION\n"
            "lock = threading.Lock()\n"
            "done = 0\n"
            "ok = 0\n"
            "\n"
            "def worker():\n"
            "    global done, ok\n"
            "    interval = 1.0 / float(RPS)\n"
            "    while time.time() < stop_at:\n"
            "        with lock:\n"
            "            if done >= MAX_TOTAL:\n"
            "                return\n"
            "            done += 1\n"
            "        t0 = time.time()\n"
            "        try:\n"
            "            req = Request(URL, headers={\"User-Agent\": \"autoppo-lab\"})\n"
            "            with urlopen(req, timeout=0.5) as r:\n"
            "                if 200 <= int(r.status) < 300:\n"
            "                    ok += 1\n"
            "        except Exception:\n"
            "            pass\n"
            "        # pace\n"
            "        dt = time.time() - t0\n"
            "        sleep_for = interval - dt\n"
            "        if sleep_for > 0:\n"
            "            time.sleep(sleep_for)\n"
            "\n"
            "threads = [threading.Thread(target=worker, daemon=True) for _ in range(CONC)]\n"
            "for t in threads: t.start()\n"
            "for t in threads: t.join()\n"
            "\n"
            "print(f\"http_done={done} http_ok={ok}\")\n"
            "PY\n"
            "python3 /tmp/autoppo_http_worker.py > " + self._http_log + " 2>&1 & echo $! > " + self._http_pidfile + "'"
        )

    def status(self) -> dict:
        # Best-effort; doesn't fail the run if missing
        conn_tail = self.h.cmd(f"tail -n 1 {self._conn_log} 2>/dev/null || true").strip()
        http_tail = self.h.cmd(f"tail -n 1 {self._http_log} 2>/dev/null || true").strip()
        return {"conn_log_last": conn_tail, "http_log_last": http_tail}
