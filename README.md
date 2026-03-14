# 📦 Inventory Management Simulation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![SimPy](https://img.shields.io/badge/SimPy-Discrete%20Event%20Simulation-green.svg)](https://simpy.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL%20Environment-orange.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Advanced inventory management system for warehouse optimization using reinforcement learning (PPO) and classical (s,S) policies.

## 🎯 Overview

This system leverages **SimPy** for discrete-event simulation and **Gymnasium** for reinforcement learning compatibility. It simulates a warehouse managing two products with independent demand processes, comparing the performance of:

- 🤖 **Reinforcement Learning** (PPO - Proximal Policy Optimization)
- 📊 **Classical Inventory Theory** ((s,S) policy optimization)

### Key Features
- ⚡ Real-time discrete-event simulation
- 🧠 Deep RL agent with neural network architecture
- 📈 Comprehensive performance evaluation
- 🌐 Interactive web interface
- 📊 Rich visualizations and analytics

## 🏗️ Project Architecture

```
Inventory_System/
├── 📁 Core Simulation
│   ├── simpyy.py              # SimPy simulation engine
│   ├── inventory_env.py       # Gymnasium RL environment
│   └── ss_policy.py           # Classical (s,S) implementation
├── 🤖 Machine Learning
│   ├── rl_training.py         # PPO agent training
│   └── models/                # Trained model weights
├── 📊 Analysis & Visualization
│   ├── evaluation.py          # Policy comparison framework
│   └── visualization.py       # Plotting & analytics
├── 🌐 Web Interface
│   └── app.py                 # Flask dashboard
└── 📁 Output
    ├── plots/                 # Generated visualizations
    └── reports/               # Analysis results
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/shazimjaved/RL_Agent_Simulation.git
cd Inventory_System

# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the System

#### 🖥️ Command Line Interface
```bash
# Run complete evaluation and comparison
python evaluation.py

# Train RL agent only
python rl_training.py

# Optimize (s,S) policy only
python ss_policy.py
```

#### 🌐 Web Dashboard
```bash
# Launch Flask application
python app.py
```
Then navigate to **http://localhost:5000** in your browser.

## 📋 Simulation Model

### Demand Processes
| Product | Demand Values | Probabilities | Inter-arrival Time |
|---------|---------------|---------------|-------------------|
| **Product 1** | {1, 2, 3, 4} | {1/6, 1/3, 1/3, 1/6} | Exponential(λ=0.1) |
| **Product 2** | {5, 4, 3, 2} | {1/8, 1/2, 1/4, 1/8} | Exponential(λ=0.1) |

### Supply Chain Parameters
```python
# Lead Times (in days)
Product_1: Uniform(0.5, 1.0)
Product_2: Uniform(0.2, 0.7)

# Cost Structure
Fixed_Order_Cost (K):     10
Variable_Cost (i):       3 per unit
Holding_Cost (h):        1 per unit/day
Penalty_Cost (π):        7 per unit shortage
```

## 🧠 Implementation Details

### Reinforcement Learning Agent
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Architecture**: Neural Network (128-128-64 layers)
- **Training**: 300,000 timesteps
- **Environment**: Custom Gymnasium wrapper

### Classical Policy
- **Method**: (s,S) inventory policy
- **Optimization**: Grid search
  - s ∈ [0, 10] (reorder point)
  - S ∈ [1, 50] (order-up-to level)
- **Approach**: Independent optimization per product

### Evaluation Framework
- **Simulation Horizon**: 1,000 days per policy
- **Metrics**:
  - 💰 Total cost and average daily cost
  - 📈 Service level
  - 📦 Total shortages and orders
  - 📊 Per-product performance analysis

## 📊 Results & Outputs

The system generates comprehensive analysis including:

- 📈 **Performance comparison plots**
- 📋 **Detailed evaluation reports** (`evaluation_report.txt`)
- 📄 **Executive summary** (`final_report.md`)
- 🎯 **Policy recommendations**

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Simulation** | SimPy | Discrete-event modeling |
| **RL Framework** | Gymnasium | Environment interface |
| **ML Library** | Stable-Baselines3 | PPO implementation |
| **Web Framework** | Flask | Dashboard interface |
| **Data Science** | NumPy, Pandas | Data processing |
| **Visualization** | Matplotlib, Seaborn | Plotting & charts |

## 📈 Performance Metrics

### Key Performance Indicators
- **Cost Efficiency**: Total inventory cost minimization
- **Service Level**: Demand fulfillment rate
- **Inventory Turns**: Stock rotation efficiency
- **Stockout Rate**: Shortage frequency

### Comparative Analysis
The system provides detailed comparison between:
- RL agent adaptability vs. classical policy stability
- Computational requirements
- Convergence characteristics
- Robustness to demand variations

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SimPy** community for excellent discrete-event simulation framework
- **Stable-Baselines3** for robust RL implementations
- **Gymnasium** for standardized RL environments

## 📞 Contact

[Shazim Javed] - [@shazimjaved] - shazimjaved448@gmail.com

---

**⭐ Star this repository if it helped you!** 

