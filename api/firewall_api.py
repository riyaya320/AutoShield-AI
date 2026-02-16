# api/firewall_api.py
"""
Flask API Bridge for AutoShield AI
Connects Laravel frontend to Python/Mininet backend
"""
from __future__ import annotations

import os
import sys
import json
import time
import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from flask import Flask, jsonify, request
from flask_cors import CORS

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(__name__)
CORS(app)  # Allow Laravel to call this API

# ============================================================================
# Global State Management
# ============================================================================

class ExperimentState:
    """Tracks current experiment status"""
    def __init__(self):
        self.running = False
        self.mode = "idle"  # 'idle', 'automated', 'live'
        self.process: Optional[subprocess.Popen] = None
        self.current_csv = "experiments/current_run.csv"
        self.start_time = 0.0
        self.mininet_net = None  # Will be set if running in-process
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "mode": self.mode,
            "uptime_seconds": time.time() - self.start_time if self.running else 0,
        }

STATE = ExperimentState()

# ============================================================================
# Status & Monitoring Endpoints
# ============================================================================

@app.route('/api/status', methods=['GET'])
def get_status():
    """
    Laravel polls this for dashboard status updates
    
    Returns:
        {
            "running": bool,
            "mode": "idle" | "automated" | "live",
            "uptime_seconds": float
        }
    """
    return jsonify(STATE.to_dict())


@app.route('/api/metrics/latest', methods=['GET'])
def get_latest_metrics():
    """
    Read last line from CSV for real-time dashboard updates
    
    Returns:
        {
            "time": float,
            "phase": str,
            "action": int,
            "action_name": str,
            "latency_ms": float,
            "syn_ack_ratio": float,
            "legit_mbps": float,
            "loss": float,
            "tcp_rate": float,
            "udp_rate": float,
            "icmp_rate": float
        }
    """
    csv_path = Path(STATE.current_csv)
    
    if not csv_path.exists():
        return jsonify({
            "error": "No data yet",
            "time": 0,
            "phase": "waiting",
            "action": 0,
            "action_name": "ALLOW_ALL",
            "latency_ms": 0,
            "syn_ack_ratio": 0,
            "legit_mbps": 0,
            "loss": 0,
            "tcp_rate": 0,
            "udp_rate": 0,
            "icmp_rate": 0
        })
    
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            
        if len(lines) < 2:  # Need header + at least 1 data row
            return jsonify({"error": "Waiting for data..."})
        
        # Parse last line
        last_line = lines[-1].strip()
        values = last_line.split(',')
        
        # CSV columns from controller_ppo.py:
        # 0:t, 1:phase, 2:action, 3:action_name, 4:pkt_rate_total_pre,
        # 5:bytes_rate_total_pre, 6:tcp_rate_pre, 7:udp_rate_pre,
        # 8:icmp_rate_pre, 9:syn_rate_pre, 10:ack_rate_pre,
        # 11:syn_ack_ratio_pre, 12:latency_ms_post, 13:loss_post,
        # 14:queue_proxy_post, 15:firewall_mode, 16:rate_limit_level,
        # 17:legit_mbps
        
        return jsonify({
            "time": float(values[0]),
            "phase": values[1],
            "action": int(values[2]),
            "action_name": values[3],
            "pkt_rate_total": float(values[4]),
            "tcp_rate": float(values[6]),
            "udp_rate": float(values[7]),
            "icmp_rate": float(values[8]),
            "syn_rate": float(values[9]),
            "ack_rate": float(values[10]),
            "syn_ack_ratio": float(values[11]),
            "latency_ms": float(values[12]),
            "loss": float(values[13]),
            "queue_proxy": float(values[14]),
            "legit_mbps": float(values[17]) if len(values) > 17 else 0.0,
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/metrics/history', methods=['GET'])
def get_metrics_history():
    """
    Get full CSV data for plotting (optional - can be slow for large files)
    
    Query params:
        limit: int = number of recent rows to return (default: 300)
    
    Returns:
        {
            "data": [
                {"time": float, "latency_ms": float, ...},
                ...
            ]
        }
    """
    limit = int(request.args.get('limit', 300))
    csv_path = Path(STATE.current_csv)
    
    if not csv_path.exists():
        return jsonify({"data": []})
    
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return jsonify({"data": []})
        
        # Get last N lines (excluding header)
        data_lines = lines[1:]  # Skip header
        recent_lines = data_lines[-limit:] if len(data_lines) > limit else data_lines
        
        result = []
        for line in recent_lines:
            values = line.strip().split(',')
            if len(values) < 17:
                continue
                
            result.append({
                "time": float(values[0]),
                "phase": values[1],
                "action": int(values[2]),
                "latency_ms": float(values[12]),
                "syn_ack_ratio": float(values[11]),
                "legit_mbps": float(values[17]) if len(values) > 17 else 0.0,
            })
        
        return jsonify({"data": result})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================================
# Experiment Control Endpoints
# ============================================================================

@app.route('/api/experiment/start', methods=['POST'])
def start_experiment():
    """
    Start PPO controller experiment
    
    Request body:
        {
            "mode": "automated" | "live",
            "model_path": "models/ppo/best/best_model.zip" (optional)
        }
    
    Returns:
        {
            "status": "started" | "error",
            "message": str
        }
    """
    # Force reset state (in case of stale state)
    STATE.running = False
    STATE.process = None
    
    print(f"DEBUG: STATE.running = {STATE.running}")
    
    if STATE.running:
        return jsonify({
            "status": "error",
            "message": "Experiment already running"
        }), 400
    
    data = request.json or {}
    mode = data.get('mode', 'automated')
    model_path = data.get('model_path', 'models/ppo/best/best_model.zip')
    
    if mode not in ('automated', 'live'):
        return jsonify({
            "status": "error",
            "message": f"Invalid mode: {mode}"
        }), 400
    
    # Ensure output directory exists
    os.makedirs('experiments', exist_ok=True)
    
    # Build command (use venv_mn for Mininet)
    cmd = [
        'sudo', '-E',
        '/home/yayari/Downloads/autoppo-fw/venv_mn/bin/python', 
        '-m', 
        'mnlab.controller_ppo',
        '--model', model_path,
        '--out', STATE.current_csv,
        '--interval', '1.0',
    ]
    
    if mode == 'live':
        cmd.extend([
            '--live',
            '--live_silent',  # No terminal spam
            '--live_log', 'logs/live_ppo.log',  # Log to file instead
        ])
    
    try:
        # Set PYTHONPATH environment variable
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PROJECT_ROOT)
        
        # Start controller in background
        STATE.process = subprocess.Popen(
            cmd,
            stdout=open('/tmp/controller_stdout.log', 'w'),
            stderr=open('/tmp/controller_stderr.log', 'w'),
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        
        print(f"DEBUG: Started process PID={STATE.process.pid}")
        print(f"DEBUG: Logs at /tmp/controller_std*.log")
        
        STATE.running = True
        STATE.mode = mode
        STATE.start_time = time.time()
        
        return jsonify({
            "status": "started",
            "message": f"Experiment started in {mode} mode",
            "pid": STATE.process.pid
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to start: {str(e)}"
        }), 500


@app.route('/api/experiment/stop', methods=['POST'])
def stop_experiment():
    """
    Stop running experiment
    
    Returns:
        {
            "status": "stopped" | "error",
            "message": str
        }
    """
    if not STATE.running:
        return jsonify({
            "status": "error",
            "message": "No experiment running"
        }), 400
    
    try:
        # Kill the process gracefully
        if STATE.process:
            STATE.process.terminate()
            STATE.process.wait(timeout=5)
        
        # Fallback: kill all controller processes
        os.system('pkill -f controller_ppo 2>/dev/null || true')
        
        # Also stop Mininet if running
        os.system('sudo mn -c > /dev/null 2>&1 || true')
        
        STATE.running = False
        STATE.mode = "idle"
        STATE.process = None
        
        return jsonify({
            "status": "stopped",
            "message": "Experiment stopped successfully"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to stop: {str(e)}"
        }), 500

# ============================================================================
# Attack Control Endpoints (Live Mode Only)
# ============================================================================

@app.route('/api/attack/launch', methods=['POST'])
def launch_attack():
    """
    Launch manual attack (live mode only)
    
    Request body:
        {
            "type": "syn" | "udp" | "icmp" | "multi",
            "target": "10.0.2.10" (optional),
            "intensity": "low" | "medium" | "high" (optional)
        }
    
    Returns:
        {
            "status": "launched" | "error",
            "message": str
        }
    """
    if STATE.mode != 'live':
        return jsonify({
            "status": "error",
            "message": "Manual attacks only available in live mode"
        }), 400
    
    data = request.json or {}
    attack_type = data.get('type', 'syn')
    target = data.get('target', '10.0.2.10')
    intensity = data.get('intensity', 'medium')
    
    # Attack intensity mappings
    intensity_map = {
        'low': {'syn': 10000, 'udp': 50, 'icmp': 100},
        'medium': {'syn': 50000, 'udp': 150, 'icmp': 500},
        'high': {'syn': 'flood', 'udp': 300, 'icmp': 'flood'},
    }
    
    params = intensity_map.get(intensity, intensity_map['medium'])
    
    # Build attack commands (these run in Mininet namespace)
    attack_cmds = {
        'syn': f'mininet> hatk1 hping3 -S -p 80 --flood --rand-source {target} > /tmp/syn_attack.log 2>&1 &',
        'udp': f'mininet> hatk1 iperf3 -u -c {target} -b {params["udp"]}M -t 3600 > /tmp/udp_attack.log 2>&1 &',
        'icmp': f'mininet> hatk1 ping -f {target} > /tmp/icmp_attack.log 2>&1 &',
        'multi': [
            f'mininet> hatk1 hping3 -S -p 80 --flood {target} > /tmp/multi_syn.log 2>&1 &',
            f'mininet> hatk2 iperf3 -u -c {target} -b {params["udp"]}M -t 3600 > /tmp/multi_udp.log 2>&1 &',
        ]
    }
    
    try:
        cmd_file = Path('logs/attack_commands.txt')
        cmd_file.parent.mkdir(exist_ok=True)
        
        with open(cmd_file, 'a') as f:
            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ")
            f.write(f"Attack: {attack_type} (intensity: {intensity})\n")
            if isinstance(attack_cmds.get(attack_type), list):
                for cmd in attack_cmds[attack_type]:
                    f.write(f"  {cmd}\n")
            else:
                f.write(f"  {attack_cmds.get(attack_type, 'unknown')}\n")
        
        return jsonify({
            "status": "launched",
            "message": f"{attack_type.upper()} attack launched at {intensity} intensity",
            "instructions": "Execute these commands in Mininet CLI (see logs/attack_commands.txt)"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to launch attack: {str(e)}"
        }), 500


@app.route('/api/attack/stop', methods=['POST'])
def stop_attacks():
    """
    Stop all running attacks
    
    Returns:
        {
            "status": "stopped" | "error",
            "message": str
        }
    """
    try:
        # Kill common attack processes
        commands = [
            'pkill -f hping3 2>/dev/null || true',
            'pkill -f "iperf3 -u -c" 2>/dev/null || true',
            'pkill -f "ping -f" 2>/dev/null || true',
        ]
        
        for cmd in commands:
            os.system(cmd)
        
        return jsonify({
            "status": "stopped",
            "message": "All attacks stopped"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to stop attacks: {str(e)}"
        }), 500

# ============================================================================
# System & Health Endpoints
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring
    
    Returns:
        {
            "status": "healthy" | "unhealthy",
            "api_version": str,
            "timestamp": float
        }
    """
    return jsonify({
        "status": "healthy",
        "api_version": "1.0.0",
        "timestamp": time.time(),
        "experiment_running": STATE.running,
    })


@app.route('/api/models/list', methods=['GET'])
def list_models():
    """
    List available trained models
    
    Returns:
        {
            "models": [
                {
                    "name": str,
                    "path": str,
                    "size_mb": float,
                    "modified": float
                },
                ...
            ]
        }
    """
    models_dir = Path('models/ppo')
    
    if not models_dir.exists():
        return jsonify({"models": []})
    
    models = []
    for model_file in models_dir.rglob('*.zip'):
        try:
            stat = model_file.stat()
            models.append({
                "name": model_file.stem,
                "path": str(model_file.relative_to(PROJECT_ROOT)),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": stat.st_mtime,
            })
        except Exception:
            continue
    
    return jsonify({"models": models})

# ============================================================================
# Main
# ============================================================================

def main():
    """Run Flask API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AutoShield AI Flask API")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    print("=" * 70)
    print("AutoShield AI - Flask API Server")
    print("=" * 70)
    print(f"API running at: http://{args.host}:{args.port}")
    print(f"Health check:   http://{args.host}:{args.port}/api/health")
    print(f"Debug mode:     {args.debug}")
    print("=" * 70)
    print("\nEndpoints:")
    print("  GET  /api/health              - Health check")
    print("  GET  /api/status              - Experiment status")
    print("  GET  /api/metrics/latest      - Latest metrics")
    print("  GET  /api/metrics/history     - Historical metrics")
    print("  POST /api/experiment/start    - Start experiment")
    print("  POST /api/experiment/stop     - Stop experiment")
    print("  POST /api/attack/launch       - Launch attack (live mode)")
    print("  POST /api/attack/stop         - Stop attacks")
    print("  GET  /api/models/list         - List trained models")
    print("=" * 70)
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
    )


if __name__ == '__main__':
    main()
