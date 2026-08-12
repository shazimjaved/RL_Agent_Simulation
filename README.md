# 📦 Reinforcement Learning & Classical Policy Inventory Management Simulation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![SimPy](https://img.shields.io/badge/SimPy-Discrete%20Event%20Simulation-green.svg)](https://simpy.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL%20Environment-orange.svg)](https://gymnasium.farama.org/)
[![Stable--Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-brightgreen.svg)](https://stable-baselines3.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Dashboard-red.svg)](https://flask.palletsprojects.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, multi-product warehouse inventory optimization framework using **Reinforcement Learning (Proximal Policy Optimization - PPO)**, **Discrete-Event Simulation (SimPy)**, **Custom Gymnasium Environments**, and **Classical $(s, S)$ Inventory Control Theory**.

The system features multi-seed statistical validation (Welch's t-test), rich visualization dashboards, an exploratory Jupyter Notebook, and an interactive Flask Web Dashboard.

---

## 🚀 Key Highlights

- 🤖 **Deep Reinforcement Learning**: Custom PPO agent trained on Gymnasium environment to dynamic order placement.
- 📊 **Classical Inventory Control**: Grid-search optimized $(s, S)$ continuous-review inventory policy.
- ⚡ **Discrete-Event Engine**: Built on `SimPy` for modeling stochastic customer demand arrivals and supplier lead times.
- 🔬 **Rigorous Statistical Verification**: Evaluates performance across 5 random seeds (123, 234, 345, 456, 567) using **Welch's t-test** for statistical significance.
- 🌐 **Interactive Web Dashboard**: Flask UI to execute simulations, visualize real-time policy comparisons, and download reports.
- 📈 **Visualization Suite**: Generates 10+ analytical plots (cost breakdown, service levels, Welch moving averages, per-product dynamics).

---

## 🏗️ Project Architecture

```
Inventory_System/
├── 📁 Core Simulation & Environment
│   ├── simpyy.py              # SimPy discrete-event simulation core engine
│   ├── inventory_env.py       # Custom Gymnasium environment wrapper
│   └── ss_policy.py           # Classical (s,S) policy implementation & optimizer
│
├── 🤖 Machine Learning & Training
│   ├── rl_training.py         # PPO agent multi-seed training pipeline
│   └── models/                # Saved PPO model weights and best checkpoints
│
├── 📊 Evaluation & Analytics
│   ├── evaluation.py          # Multi-seed evaluation framework & Welch t-test
│   ├── visualization.py       # Matplotlib & Seaborn analytics dashboard generator
│   └── Analysis .ipynb        # Exploratory analysis notebook
│
├── 🌐 Interactive Web Interface
│   ├── app.py                 # Flask web server & route handlers
│   ├── templates/             # Jinja2 HTML templates (index.html, results.html)
│   └── static/                # CSS stylesheet and UI branding assets
│
└── 📁 Generated Outputs
    ├── plots/                 # Saved visualization figures & dashboard images
    └── requirements.txt       # Project dependency specifications
```

---

## 📋 Mathematical Simulation Model

### 1. Products & Stochastic Demand

The warehouse manages **2 products** sourced from distinct suppliers with stochastic discrete demand distributions and exponential customer inter-arrival times ($\lambda = 0.1$ arrivals/day):

| Product | Demand Values ($D$) | Probability Mass Function $P(D)$ | Inter-Arrival Time |
| :--- | :--- | :--- | :--- |
| **Product 1** | $\{1, 2, 3, 4\}$ | $\{1/6, 1/3, 1/3, 1/6\}$ | $\text{Exponential}(\lambda = 0.1)$ |
| **Product 2** | $\{2, 3, 4, 5\}$ | $\{1/8, 1/4, 1/2, 1/8\}$ | $\text{Exponential}(\lambda = 0.1)$ |

### 2. Stochastic Supplier Lead Times

When an order is placed, delivery to the warehouse is subject to random lead times:
- **Product 1 Lead Time**: $L_1 \sim \text{Uniform}(0.5, 1.0)$ days
- **Product 2 Lead Time**: $L_2 \sim \text{Uniform}(0.2, 0.7)$ days

### 3. Inventory Cost Structure

The overall goal is to **minimize total operational inventory cost**:

$$\text{Total Cost} = \text{Holding Cost} + \text{Ordering Cost} + \text{Shortage Penalty Cost}$$

- **Fixed Order Cost ($K$)**: $\$10$ per order batch
- **Unit Order Cost ($i$)**: $\$3$ per ordered unit
- **Unit Holding Cost ($h$)**: $\$1$ per unit/day held in inventory
- **Shortage Penalty Cost ($\pi$)**: $\$7$ per unfulfilled unit (backorder/stockout penalty)

---

## 🧠 Control Policies & Optimization

### 1. Reinforcement Learning Agent (PPO)
- **Algorithm**: Proximal Policy Optimization (PPO via `Stable-Baselines3`)
- **State Space**: Current inventory level, inventory position (inventory + on-order), and active pending orders for each product.
- **Action Space**: Order quantities for Product 1 and Product 2.
- **Network Architecture**: MLP Policy (Layer sizes: 128 - 128 - 64).
- **Training Horizon**: 300,000 timesteps per seed.

### 2. Classical $(s, S)$ Inventory Policy
- **Logic**: Continuous-review inventory policy. Reorder when inventory position drops to or below reorder level $s$, ordering up to target level $S$.
- **Optimization**: Multi-dimensional grid search evaluating candidate pairs $(s_1, S_1)$ and $(s_2, S_2)$ over:
  - Reorder point $s \in [0, 10]$
  - Order-up-to level $S \in [1, 50]$

### 3. Statistical Testing (Welch's t-Test)
- Evaluates both policies over **1,000 simulation days** across 5 distinct random seeds (`123`, `234`, `345`, `456`, `567`).
- Performs **Welch's unequal variances t-test** to confirm whether difference in mean daily operational costs is statistically significant ($p < 0.05$).

---

## ⚙️ Quick Start Guide

### Prerequisites
- Python 3.8+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/shazimjaved/RL_Agent_Simulation.git
cd Inventory_System

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows (PowerShell / CMD):
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🖥️ Running the System

### Option A: Launch Interactive Web Dashboard 🌐
Run the Flask server:
```bash
python app.py
```
Open your browser and navigate to **`http://localhost:5000`** (or `http://127.0.0.1:5000`).
From the dashboard, click **Run Simulation** to trigger full RL training, $(s,S)$ optimization, comparative evaluation, and live chart generation.

### Option B: Run Full Evaluation via CLI 📊
Execute multi-seed evaluation, Welch's t-test, and plot generation:
```bash
python evaluation.py
```

### Option C: Train RL Agents Only 🤖
Train PPO models across multiple seeds:
```bash
python rl_training.py
```

### Option D: Optimize Classical Policy Only 📈
Run grid search for optimal $(s,S)$ parameters:
```bash
python ss_policy.py
```

### Option E: Interactive Notebook 📓
Explore detailed data analysis and policy comparisons in Jupyter:
```bash
jupyter notebook "Analysis .ipynb"
```

---

## 📊 Analytics & Visualizations

Running the evaluation framework automatically generates figures inside the [`plots/`](file:///d:/Projects/Inventory_System/plots) directory:

- 📊 `summary_dashboard.png`: Unified overview comparing total cost, daily cost, and service levels.
- 💰 `cost_comparison.png`: Holding vs Ordering vs Shortage cost breakdown.
- 📈 `daily_performance.png`: Time-series inventory trajectory across 1,000 simulation days.
- 📉 `rigorous_welch_moving_average.png`: Welch moving average chart showing steady-state convergence.
- 🎯 `per_product_service_levels.png`: Service level & stockout performance per product.
- 📦 `per_product_comparison.png` & `per_product_daily_performance.png`: Detailed itemized dynamics.

---

## 🛠️ Technology Stack

| Component | Library / Framework | Description |
| :--- | :--- | :--- |
| **Simulation Core** | `SimPy` | Discrete-event simulation of events, queues, and delays |
| **RL Environment** | `Gymnasium` | Standardized Gym environment API |
| **RL Algorithm** | `Stable-Baselines3` | PPO implementation |
| **Web Dashboard** | `Flask`, `HTML5`, `CSS3` | Interactive frontend UI |
| **Data Processing** | `NumPy`, `Pandas`, `SciPy` | Numerical processing & Welch t-test |
| **Visualization** | `Matplotlib`, `Seaborn` | Automated chart and plot generation |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the project repository.
2. Create your feature branch (`git checkout -b feature/OptimizationFeature`).
3. Commit your changes (`git commit -m 'Add OptimizationFeature'`).
4. Push to the branch (`git push origin feature/OptimizationFeature`).
5. Open a Pull Request.

---

## 📝 License

Distributed under the **MIT License**. See `LICENSE` for details.

---

## 👤 Author & Contact

**Shazim Javed**
- 📧 Email: shazimjaved448@gmail.com
- 💻 GitHub: [@shazimjaved](https://github.com/shazimjaved)
- 🚀 Repository: [shazimjaved/RL_Agent_Simulation](https://github.com/shazimjaved/RL_Agent_Simulation)
