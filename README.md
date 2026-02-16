
# AutoShield AI

**Autonomous DDoS Mitigation Using Proximal Policy Optimization**

A reinforcement learning-based firewall system that autonomously detects and mitigates DDoS attacks in real-time for SME networks.

---

## 🎯 Overview

AutoShield AI uses PPO (Proximal Policy Optimization) to train an autonomous agent that:
- Detects DDoS attacks (SYN flood, UDP flood, ICMP flood, multi-vector)
- Selects appropriate mitigation actions in real-time (<1s detection)
- Maintains network performance during attacks
- Provides complete transparency through audit logging and dashboard

---

## 🏗️ Architecture
```
┌─────────────┐
│   Mininet   │  6-host network topology
│   Network   │  (2 legit, 2 attack, 1 fw, 1 srv)
└──────┬──────┘
       │
┌──────▼──────┐
│  PPO Agent  │  13-dim observation space
│  (Stable-   │  8 bounded firewall actions
│  Baselines3)│  Multi-objective reward
└──────┬──────┘
       │
┌──────▼──────┐
│  iptables + │  Firewall enforcement
│  tc (Linux) │  Rate limiting & blocking
└──────┬──────┘
       │
┌──────▼──────┐
│  Dashboard  │  Laravel + Flask API
│  (Laravel)  │  Real-time visualization
└─────────────┘
```

---

## 📊 Key Results

- **Detection Latency:** <1 second
- **Attack Mitigation:** SYN (148K pkt/s), UDP (89K pkt/s), ICMP (52K pkt/s)
- **Multi-Vector:** 186K pkt/s combined attack handled
- **Recovery Time:** <2 seconds autonomous recovery
- **False Positives:** 0 during legitimate traffic phases
- **Performance:** 58% throughput maintained during worst-case multi-vector attack

---

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y mininet python3-pip hping3 iperf3

# Install Python packages
pip install stable-baselines3 torch mininet flask flask-cors
```

### Running the System

**Terminal 1: Flask API**
```bash
cd autoppo-fw
sudo ./venv_mn/bin/python api/firewall_api.py
```

**Terminal 2: Laravel Dashboard**
```bash
cd ebuzzlive/accounting
php artisan serve --port=8000
```

**Terminal 3: Run Experiment**
```bash
cd autoppo-fw
sudo -E ./venv_mn/bin/python -m mnlab.controller_ppo
```

**Browser:**
```
http://localhost:8000/dashboard
```

---

## 📁 Project Structure
```
autoppo-fw/
├── mnlab/
│   ├── topo.py                    # Mininet network topology
│   ├── controller_ppo.py          # Main PPO controller
│   ├── feature_adapter.py         # Network metric extraction
│   ├── firewall_actions.py        # Action enforcement (iptables/tc)
│   └── attack_scenarios.py        # Attack generation
├── models/
│   └── ppo/
│       └── final_model.zip        # Trained PPO model
├── api/
│   └── firewall_api.py            # Flask REST API
├── experiments/
│   └── *.csv                      # Experiment logs
├── train/
│   ├── train_ppo.py               # Training script
│   └── evaluate.py                # Evaluation script
└── README.md
```

---

## 🎓 System Design

### Observation Space (13 dimensions)
- Total packet/byte rates
- TCP/UDP/ICMP packet rates  
- SYN/ACK packet rates & ratio
- Network latency & packet loss
- Queue depth
- Current firewall mode & rate limit level

### Action Space (8 bounded actions)
0. ALLOW_ALL
1. RATE_LIMIT_TCP (20 Mbps)
2. RATE_LIMIT_UDP (20 Mbps)
3. RATE_LIMIT_ICMP (10 Mbps)
4. BLOCK_TCP
5. BLOCK_UDP
6. BLOCK_ICMP
7. GLOBAL_RATE_LIMIT (30 Mbps)

### Reward Function
```python
reward = +1  # Correct action with low latency/loss
reward = -5  # False positive (blocking legitimate traffic)
reward = -10 # False negative (missing attack)
reward = -1  # High latency/loss
```

---

## 🧪 Experimental Scenarios

The system is evaluated through 6 phases (60 seconds total):

1. **Warmup (8s)** - Baseline legitimate traffic
2. **SYN Flood (10s)** - 148,000 pkt/s attack
3. **UDP Flood (10s)** - 89,000 pkt/s attack
4. **ICMP Flood (10s)** - 52,000 pkt/s attack
5. **Multi-Vector (12s)** - All three simultaneously (186K pkt/s)
6. **Cooldown (6s)** - Return to baseline

---

## 📈 Training
```bash
cd train
python train_ppo.py --timesteps 50000 --env mininet
```

Training parameters:
- Algorithm: PPO (Stable-Baselines3)
- Timesteps: 50,000
- Learning rate: 3e-4
- Batch size: 64
- Network: 2-layer MLP [64, 64]

---

## 🔒 Ethical Considerations

AutoShield AI implements responsible autonomous security:

- **Bounded Actions:** Only pre-approved firewall policies
- **Audit Logging:** Complete CSV logs of all decisions
- **Emergency Override:** Manual control & emergency stop
- **Transparency:** Real-time dashboard showing agent reasoning
- **Privacy:** No payload inspection, aggregate metrics only

---

## 📝 Citation
```bibtex
@mastersthesis{autoshield2026,
  title={AutoShield AI: PPO-Based Autonomous DDoS Mitigation for SME Networks},
  author={Your Name},
  year={2026},
  school={Your University}
}
```

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- Stable-Baselines3 for RL implementation
- Mininet for network emulation
- Laravel & Flask for dashboard/API

---

## 📧 Contact

For questions or collaboration: your.email@example.com
EOF
