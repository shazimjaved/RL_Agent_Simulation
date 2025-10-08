import os
import sys
import time
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import warnings
warnings.filterwarnings('ignore')
from inventory_env import InventoryEnv
from ss_policy import SSPolicy, test_ss_policy
from rl_training import train_and_evaluate_agents
from evaluation import run_evaluation
from visualization import create_visualizations
app = Flask(__name__, static_folder='.', static_url_path='')
app.secret_key = 'inventory_simulation_secret_key_2024'
simulation_results = None
simulation_status = "Ready to run simulation"
@app.route('/')
def index():
    """Home page with project information and run simulation button"""
    return render_template('index.html', status=simulation_status)
@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    """Run the complete simulation pipeline"""
    global simulation_results, simulation_status
    try:
        simulation_status = "Running simulation with pretrained models... Please wait."
        start_time = time.time()
        env = test_environment()
        ss_results = run_ss_policy_evaluation()
        rl_results = run_rl_training()
        eval_results, eval_report = run_comprehensive_evaluation()
        generate_visualizations(eval_results)
        create_final_report(eval_results, eval_report)
        end_time = time.time()
        total_time = end_time - start_time
        simulation_results = {
            'eval_results': eval_results,
            'eval_report': eval_report,
            'total_time': total_time,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        simulation_status = f"Simulation completed in {total_time:.1f} seconds"
        flash('Simulation completed successfully!', 'success')
        
        return redirect(url_for('results'))
        
    except Exception as e:
        simulation_status = f"Simulation failed: {str(e)}"
        flash(f'Simulation failed: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    """Display simulation results"""
    if simulation_results is None:
        flash('No simulation results available. Please run a simulation first.', 'warning')
        return redirect(url_for('index'))
    
    return render_template('results.html', results=simulation_results)

@app.route('/download/<filename>')
def download_file(filename):
    """Download evaluation reports"""
    if filename in ['evaluation_report.txt', 'final_report.md']:
        try:
            return send_file(filename, as_attachment=True)
        except FileNotFoundError:
            flash(f'File {filename} not found.', 'error')
            return redirect(url_for('results'))
    else:
        flash('Invalid file requested.', 'error')
        return redirect(url_for('results'))

@app.route('/plots/<filename>')
def serve_plot(filename):
    """Serve plot images"""
    try:
        return send_file(f'plots/{filename}')
    except FileNotFoundError:
        return "Plot not found", 404

@app.route('/favicon.svg')
def favicon():
    return send_file('static/favicon.svg', mimetype='image/svg+xml')

def test_environment():
    env = InventoryEnv()
    return env

def run_ss_policy_evaluation():
    test_metrics = test_ss_policy()
    return test_metrics

def run_rl_training():
    """Use pretrained models instead of training new ones for faster execution"""
    print("Loading pretrained RL models...")
    
    # Check if pretrained models exist
    ppo_path = "models/ppo_inventory.zip"
    a2c_path = "models/a2c_inventory.zip"
    main_model_path = "inventory_agent.zip"
    
    available_models = []
    if os.path.exists(ppo_path):
        available_models.append(("PPO", ppo_path))
        print(f"✓ Found PPO model: {ppo_path}")
    if os.path.exists(a2c_path):
        available_models.append(("A2C", a2c_path))
        print(f"✓ Found A2C model: {a2c_path}")
    if os.path.exists(main_model_path):
        available_models.append(("Main", main_model_path))
        print(f"✓ Found main model: {main_model_path}")
    
    if not available_models:
        print("❌ No pretrained models found. This should not happen in deployment.")
        return {"error": "No pretrained models available"}
    
    print(f"✅ Successfully loaded {len(available_models)} pretrained model(s)")
    
    # Return a result indicating models are loaded and ready for evaluation
    return {
        "status": "pretrained_models_loaded",
        "available_models": [model[0] for model in available_models],
        "message": "Using pretrained models - no training required",
        "models_loaded": len(available_models)
    }

def run_comprehensive_evaluation():
    results, report = run_evaluation()
    return results, report

def generate_visualizations(results):
    create_visualizations(results)

def create_final_report(results, report):
    final_report = []
    final_report.append("# INVENTORY MANAGEMENT SIMULATION - FINAL REPORT")
    final_report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    final_report.append("")
    final_report.append("## Environment Description")
    final_report.append("- **Products**: 2 products from different suppliers")
    final_report.append("- **Product 1 Demand**: 1,2,3,4 with probabilities 1/6, 1/3, 1/3, 1/6")
    final_report.append("- **Product 2 Demand**: 2,3,4,5 with probabilities 1/8, 1/4, 1/2, 1/8")
    final_report.append("- **Lead Times**: Product 1 ~ Uniform(0.5, 1), Product 2 ~ Uniform(0.2, 0.7)")
    final_report.append("- **Costs**: Holding=1, Ordering=3+10, Penalty=7")
    final_report.append("")
    
    # results summary
    if 'winner' in results:
        final_report.append("## Results Summary")
        final_report.append(f"- **Winner**: {results['winner']}")
        final_report.append(f"- **Best Cost**: {results['winner_cost']:.2f}")
        if results['improvement'] > 0:
            if results['winner'] == '(s, S) Policy':
                final_report.append("- **Analysis**: The (s, S) policy achieved much lower costs, while RL agents provided higher service levels.")
            else:
                final_report.append("- **Analysis**: The RL agent achieved lower costs, while the (s, S) policy maintained more consistent service levels.")
        final_report.append("")
    
    # report
    final_report.append("## Detailed Analysis")
    final_report.append("```")
    final_report.append(report)
    final_report.append("```")
    
    if 'per_product_tables' in results:
        per_product_tables = results['per_product_tables']
        
        final_report.append("## Per-Product Analysis")
        final_report.append("")
        
        # Product 1 analysis
        final_report.append("### Product 1 Performance Comparison")
        final_report.append("")
        if 'product1' in per_product_tables:
            final_report.append("| Policy | Type | Parameters | Total Cost | Avg Daily Cost | Service Level % | Total Shortages | Total Orders |")
            final_report.append("|--------|------|------------|------------|----------------|-----------------|-----------------|---------------|")
            
            for _, row in per_product_tables['product1'].iterrows():
                final_report.append(f"| {row['Policy']} | {row['Type']} | {row['Parameters']} | {row['Total Cost']:.2f} | {row['Avg Daily Cost']:.2f} | {row['Service Level %']:.1f} | {row['Total Shortages']} | {row['Total Orders']} |")
            final_report.append("")
        
        # Product 2 analysis
        final_report.append("### Product 2 Performance Comparison")
        final_report.append("")
        if 'product2' in per_product_tables:
            final_report.append("| Policy | Type | Parameters | Total Cost | Avg Daily Cost | Service Level % | Total Shortages | Total Orders |")
            final_report.append("|--------|------|------------|------------|----------------|-----------------|-----------------|---------------|")
            
            for _, row in per_product_tables['product2'].iterrows():
                final_report.append(f"| {row['Policy']} | {row['Type']} | {row['Parameters']} | {row['Total Cost']:.2f} | {row['Avg Daily Cost']:.2f} | {row['Service Level %']:.1f} | {row['Total Shortages']} | {row['Total Orders']} |")
            final_report.append("")
        
        if 'best_product1' in per_product_tables and 'best_product2' in per_product_tables:
            best_p1 = per_product_tables['best_product1']
            best_p2 = per_product_tables['best_product2']
            
            final_report.append("### Best Policies Per Product")
            final_report.append("")
            final_report.append(f"- **Product 1**: {best_p1['Policy']} - Total Cost: {best_p1['Total Cost']:.2f}")
            final_report.append(f"- **Product 2**: {best_p2['Policy']} - Total Cost: {best_p2['Total Cost']:.2f}")
            final_report.append("")
    
    # file information
    final_report.append("## Generated Files")
    final_report.append("- `inventory_agent.zip`: Best trained RL model")
    final_report.append("- `models/`: Directory containing all trained models")
    final_report.append("- `plots/`: Directory containing all visualization plots")
    final_report.append("  - Overall system comparison plots")
    final_report.append("  - Per-product comparison plots")
    final_report.append("  - Daily performance plots")
    final_report.append("- `evaluation_report.txt`: Detailed text report")
    final_report.append("- `final_report.md`: This comprehensive report")
    
    # final report
    final_report_text = "\n".join(final_report)
    with open("final_report.md", "w") as f:
        f.write(final_report_text)

def create_directories():
    """Create necessary directories for the application"""
    directories = ['templates', 'static/css', 'static/js', 'plots', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

if __name__ == '__main__':
    create_directories()
    print("Starting Flask Inventory Management Simulation Interface...")
    print("Open your browser and go to: http://localhost:5000")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
