import os
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from inventory_env import InventoryEnv
from ss_policy import SSPolicy, test_ss_policy
from rl_training import train_and_evaluate_agents
from evaluation import run_evaluation
from visualization import create_visualizations


def print_banner():
    """Print project banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    TERMINAL RUN:                             ║
    ║          INVENTORY MANAGEMENT SIMULATION                     ║
    ║                                                              ║
    ║        RL Agents vs Classical (s, S) Policies                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_section(title):
    """Print section header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def test_environment():
    """Test the inventory environment"""
    print_section("TESTING INVENTORY ENVIRONMENT")
    
    print("Creating and testing inventory environment...")
    env = InventoryEnv()
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    obs = env.reset()
    print(f"Initial observation: {obs}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Obs={obs}, Reward={reward:.2f}")
    
    print("✓ Environment test completed successfully!")
    return env


def run_ss_policy_evaluation():
    """Run (s, S) policy evaluation"""
    print_section("EVALUATING (s, S) POLICIES")
    
    print("Testing (s, S) policy implementation...")
    test_metrics = test_ss_policy()
    
    print("✓ (s, S) policy evaluation completed!")
    return test_metrics


def run_rl_training():
    """Run RL agent training"""
    print_section("TRAINING RL AGENTS")
    
    print("Starting RL agent training...")
    print("This may take several minutes...")
    
    start_time = time.time()
    results = train_and_evaluate_agents()
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"✓ RL training completed in {training_time:.1f} seconds!")
    
    return results


def run_comprehensive_evaluation():
    """Run comprehensive evaluation"""
    print_section("COMPREHENSIVE EVALUATION")
    
    print("Running comprehensive evaluation...")
    results, report = run_evaluation()
    
    print("✓ Comprehensive evaluation completed!")
    return results, report


def generate_visualizations(results):
    """Generate visualizations"""
    print_section("GENERATING VISUALIZATIONS")
    
    print("Creating plots and visualizations...")
    create_visualizations(results)
    
    print("✓ Visualizations generated successfully!")
    print("Check the 'plots/' directory for generated images.")


def create_final_report(results, report):
    """Create final summary report"""
    print_section("FINAL SUMMARY REPORT")
    
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

    if 'winner' in results:
        final_report.append("## Results Summary")
        final_report.append(f"- **Winner**: {results['winner']}")
        final_report.append(f"- **Best Cost**: {results['winner_cost']:.2f}")
        if results['improvement'] > 0:
            final_report.append(f"- **Improvement**: {results['improvement']:.1f}%")
        final_report.append("")
    
    final_report.append("## Detailed Analysis")
    final_report.append("```")
    final_report.append(report)
    final_report.append("```")
    
    final_report.append("## Generated Files")
    final_report.append("- `inventory_agent.zip`: Best trained RL model")
    final_report.append("- `models/`: Directory containing all trained models")
    final_report.append("- `plots/`: Directory containing all visualization plots")
    final_report.append("- `evaluation_report.txt`: Detailed text report")
    final_report.append("- `final_report.md`: This comprehensive report")
    
    final_report_text = "\n".join(final_report)
    with open("final_report.md", "w") as f:
        f.write(final_report_text)
    
    print("Final report created: 'final_report.md'")
    print("\n" + "="*60)
    print("PROJECT COMPLETION SUMMARY")
    print("="*60)
    
    if 'winner' in results:
        print(f"🏆 WINNER: {results['winner']}")
        print(f"💰 Best Cost: {results['winner_cost']:.2f}")
        if results['improvement'] > 0:
            print(f"📈 Improvement: {results['improvement']:.1f}%")
    
    print("\n📁 Generated Files:")
    print("   • inventory_agent.zip - Best RL model")
    print("   • models/ - All trained models")
    print("   • plots/ - Visualization plots")
    print("   • evaluation_report.txt - Detailed analysis")
    print("   • final_report.md - Comprehensive report")
    
    print("\n✅ Project completed successfully!")
    print("   All requirements have been fulfilled:")
    print("   ✓ Gym-compatible environment implemented")
    print("   ✓ (s, S) policy benchmark created")
    print("   ✓ RL agents trained (PPO and A2C)")
    print("   ✓ Comprehensive evaluation performed")
    print("   ✓ Visualizations generated")
    print("   ✓ Documentation and reporting completed")


def main():
    """Main execution function"""
    print_banner()
    
    start_time = time.time()
    
    try:
      
        env = test_environment()  
        ss_results = run_ss_policy_evaluation()
        rl_results = run_rl_training()
        eval_results, eval_report = run_comprehensive_evaluation()
        generate_visualizations(eval_results)
        create_final_report(eval_results, eval_report)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Please check the error and try again.")
        sys.exit(1)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n⏱️  Total execution time: {total_time:.1f} seconds")
    print(" Thank you for using the Inventory Management Simulation!")


if __name__ == "__main__":
    main()
