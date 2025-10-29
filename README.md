# Inventory Management

This project implements a complete inventory management simulation using reinforcement learning and classical (s, S) policies, featuring a modern Flask-based web interface for easy interaction and visualization.

## 🚀 Features

- **Gym-compatible inventory environment** with two products
- **Classical (s, S) policy benchmark** with 5 different parameter sets
- **Reinforcement learning agents** (PPO)
- **Flask web interface** with Bootstrap styling
- **Real-time progress tracking** with loading animations
- **Interactive visualizations** and comprehensive reports
- **Downloadable evaluation reports** (TXT and Markdown)

## 📁 Project Structure

### Core Simulation Files
- `inventory_env.py` - Main environment class extending gym.Env
- `ss_policy.py` - Classical (s, S) policy implementation
- `rl_training.py` - RL agent training script (PPO and A2C)
- `evaluation.py` - Comparison and evaluation system
- `visualization.py` - Plotting and reporting functions
- `main.py` - Command-line execution script

### Flask Web Interface
- `app.py` - Flask server with routes and simulation orchestration
- `templates/` - HTML templates for web interface
  - `index.html` - Home page with simulation controls
  - `results.html` - Results display with metrics and plots
- `static/css/` - Custom CSS styling
  - `style.css` - Bootstrap enhancements and animations

### Generated Outputs
- `plots/` - Visualization images (cost comparison, service levels, etc.)
- `models/` - Trained RL models
- `evaluation_report.txt` - Detailed text analysis
- `final_report.md` - Comprehensive markdown report

## 🛠️ Installation

1. **Clone the repository** (if applicable)
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Web Interface
```bash
python app.py
```
Then open your browser and go to: `http://localhost:5000`

### Terminal Interface
```bash
python main.py
```

## 🌐 Web Interface Features

### Home Page (`/`)
- **Project overview** and simulation description
- **"Run Complete Simulation"** button with loading animation
- **Real-time progress tracking** with step indicators:
  - Environment testing
  - (s, S) policy evaluation
  - RL agent training
  - Comprehensive evaluation
- **Estimated time display** (5-10 minutes)

### Results Page (`/results`)
- **Winner announcement** with key metrics
- **Performance comparison table** for all policies
- **Interactive visualizations**:
  - Cost comparison charts
  - Service level analysis
  - Daily performance trends
  - Summary dashboard
- **Downloadable reports** (TXT and Markdown formats)
- **Detailed analysis** section with full evaluation report

## 📊 Simulation Details

### Environment As Per Your University Requirement
- **Products**: 2 products from different suppliers
- **Demand Distributions**:
  - Product 1: 1,2,3,4 with probabilities 1/6, 1/3, 1/3, 1/6
  - Product 2: 2,3,4,5 with probabilities 1/8, 1/4, 1/2, 1/8
- **Lead Times**: Product 1 ~ Uniform(0.5, 1), Product 2 ~ Uniform(0.2, 0.7)
- **Cost Structure**: Holding=1, Ordering=3+10, Penalty=7

### Policies Evaluated
- **5 (s, S) policies** with different parameter combinations
- **2 RL agents**: PPO (Proximal Policy Optimization) and A2C (Advantage Actor-Critic)
- **1000-day simulation** period for each policy
- **Comprehensive metrics**: Total cost, service level, shortages, orders

## 🔧 Technical Requirements

- Python 3.8+
- Flask 2.3.0+
- Stable-Baselines3 2.7.0+
- Gym 0.21.0
- NumPy, Pandas, Matplotlib, Seaborn
- TensorBoard for RL training logs

## 📈 Sample Results

The simulation typically shows:
- **Best performing policy**: Usually a classical (s, S) policy
- **Cost improvements**: 30-40% better than baseline RL agents
- **Service levels**: 55-95% depending on policy parameters
- **Training time**: 5-10 minutes for complete evaluation

## 🎓 Educational Value

This project demonstrates:
- **Inventory management theory** and classical policies
- **Reinforcement learning** in supply chain optimization
- **Web application development** with Flask
- **Data visualization** and reporting
- **Performance comparison** methodologies

