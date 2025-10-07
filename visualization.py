import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for deployment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import seaborn as sns
from datetime import datetime
import os


class InventoryVisualizer:
    """
    Comprehensive visualizer for inventory management results
    """
    
    def __init__(self, results: Dict[str, Any]):
        """
        Initialize the visualizer
        
        Args:
            results: Results from evaluation
        """
        self.results = results
        self.setup_style()
        
    def setup_style(self):
        """Setup matplotlib style"""
        plt.style.use('default')
        sns.set_palette("husl")

        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
    def plot_cost_comparison(self, save_path: str = "plots/cost_comparison.png"):
        """
        Plot cost comparison between policies
        
        Args:
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        policies = []
        total_costs = []
        avg_costs = []
        
        if 'ss_policies' in self.results:
            for i, policy in enumerate(self.results['ss_policies']):
                policies.append(f'(s,S) {i+1}')
                total_costs.append(policy['total_cost'])
                avg_costs.append(policy['average_daily_cost'])
        
        if 'rl_agents' in self.results:
            for agent in self.results['rl_agents']:
                policies.append(agent['policy_type'])
                total_costs.append(agent['total_cost'])
                avg_costs.append(agent['average_daily_cost'])
        
        colors = ['skyblue' if '(s,S)' in p else 'lightcoral' for p in policies]
        bars1 = ax1.bar(policies, total_costs, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Total Cost Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Total Cost')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, cost in zip(bars1, total_costs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(total_costs)*0.01,
                    f'{cost:.0f}', ha='center', va='bottom', fontweight='bold')
        
        bars2 = ax2.bar(policies, avg_costs, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Average Daily Cost Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Daily Cost')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, cost in zip(bars2, avg_costs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(avg_costs)*0.01,
                    f'{cost:.1f}', ha='center', va='bottom', fontweight='bold')
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='skyblue', alpha=0.7, label='(s, S) Policies'),
                          Patch(facecolor='lightcoral', alpha=0.7, label='RL Agents')]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"Cost comparison plot saved to {save_path}")
    
    def plot_service_level_comparison(self, save_path: str = "plots/service_level_comparison.png"):
        """
        Plot service level comparison
        
        Args:
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        policies = []
        service_levels = []
        if 'ss_policies' in self.results:
            for i, policy in enumerate(self.results['ss_policies']):
                policies.append(f'(s,S) {i+1}')
                service_levels.append(policy['service_level'])
        
        if 'rl_agents' in self.results:
            for agent in self.results['rl_agents']:
                policies.append(agent['policy_type'])
                service_levels.append(agent['service_level'])
        colors = ['skyblue' if '(s,S)' in p else 'lightcoral' for p in policies]
        bars = ax.bar(policies, service_levels, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Service Level Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Service Level (%)')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=45)
        for bar, level in zip(bars, service_levels):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{level:.1f}%', ha='center', va='bottom', fontweight='bold')
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='skyblue', alpha=0.7, label='(s, S) Policies'),
                          Patch(facecolor='lightcoral', alpha=0.7, label='RL Agents')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"Service level comparison plot saved to {save_path}")
    
    def plot_daily_performance(self, save_path: str = "plots/daily_performance.png"):
        """
        Plot daily performance metrics for best policies
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        best_ss = self.results.get('best_ss')
        best_rl = self.results.get('best_rl')
        
        if best_ss and best_rl:
            axes[0, 0].plot(best_ss['daily_costs'], label='Best (s, S)', alpha=0.7, linewidth=2)
            axes[0, 0].plot(best_rl['daily_costs'], label='Best RL', alpha=0.7, linewidth=2)
            axes[0, 0].set_title('Daily Costs Over Time')
            axes[0, 0].set_xlabel('Day')
            axes[0, 0].set_ylabel('Daily Cost')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot daily shortages
            axes[0, 1].plot(best_ss['daily_shortages'], label='Best (s, S)', alpha=0.7, linewidth=2)
            axes[0, 1].plot(best_rl['daily_shortages'], label='Best RL', alpha=0.7, linewidth=2)
            axes[0, 1].set_title('Daily Shortages Over Time')
            axes[0, 1].set_xlabel('Day')
            axes[0, 1].set_ylabel('Daily Shortages')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot daily orders
            axes[1, 0].plot(best_ss['daily_orders'], label='Best (s, S)', alpha=0.7, linewidth=2)
            axes[1, 0].plot(best_rl['daily_orders'], label='Best RL', alpha=0.7, linewidth=2)
            axes[1, 0].set_title('Daily Orders Over Time')
            axes[1, 0].set_xlabel('Day')
            axes[1, 0].set_ylabel('Daily Orders')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot stock levels (first 100 days)
            days_to_plot = min(100, len(best_ss['daily_stock_levels']))
            stock1_ss = [s[0] for s in best_ss['daily_stock_levels'][:days_to_plot]]
            stock2_ss = [s[1] for s in best_ss['daily_stock_levels'][:days_to_plot]]
            stock1_rl = [s[0] for s in best_rl['daily_stock_levels'][:days_to_plot]]
            stock2_rl = [s[1] for s in best_rl['daily_stock_levels'][:days_to_plot]]
            
            axes[1, 1].plot(stock1_ss, label='(s, S) Product 1', alpha=0.7, linewidth=2)
            axes[1, 1].plot(stock2_ss, label='(s, S) Product 2', alpha=0.7, linewidth=2)
            axes[1, 1].plot(stock1_rl, label='RL Product 1', alpha=0.7, linewidth=2, linestyle='--')
            axes[1, 1].plot(stock2_rl, label='RL Product 2', alpha=0.7, linewidth=2, linestyle='--')
            axes[1, 1].set_title('Stock Levels Over Time (First 100 Days)')
            axes[1, 1].set_xlabel('Day')
            axes[1, 1].set_ylabel('Stock Level')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"Daily performance plot saved to {save_path}")
    
    def plot_summary_dashboard(self, save_path: str = "plots/summary_dashboard.png"):
        """
        Create a comprehensive summary dashboard
        
        Args:
            save_path: Path to save the plot
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        policies = []
        total_costs = []
        
        if 'ss_policies' in self.results:
            for i, policy in enumerate(self.results['ss_policies']):
                policies.append(f'(s,S) {i+1}')
                total_costs.append(policy['total_cost'])
        
        if 'rl_agents' in self.results:
            for agent in self.results['rl_agents']:
                policies.append(agent['policy_type'])
                total_costs.append(agent['total_cost'])
        
        colors = ['skyblue' if '(s,S)' in p else 'lightcoral' for p in policies]
        bars = ax1.bar(policies, total_costs, color=colors, alpha=0.7)
        ax1.set_title('Total Cost Comparison', fontweight='bold')
        ax1.set_ylabel('Total Cost')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2 = fig.add_subplot(gs[0, 1])
        service_levels = []
        
        if 'ss_policies' in self.results:
            for policy in self.results['ss_policies']:
                service_levels.append(policy['service_level'])
        
        if 'rl_agents' in self.results:
            for agent in self.results['rl_agents']:
                service_levels.append(agent['service_level'])
        
        bars2 = ax2.bar(policies, service_levels, color=colors, alpha=0.7)
        ax2.set_title('Service Level Comparison', fontweight='bold')
        ax2.set_ylabel('Service Level (%)')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='x', rotation=45)
    
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        if 'winner' in self.results:
            winner_text = f"WINNER:\n{self.results['winner']}\n\nCost: {self.results['winner_cost']:.2f}"
            
            ax3.text(0.5, 0.5, winner_text, ha='center', va='center', 
                    fontsize=14, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')
        
        if 'summary_table' in self.results:
            table_data = self.results['summary_table'].round(2)
            table = ax4.table(cellText=table_data.values,
                            colLabels=table_data.columns,
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            ax4.set_title('Detailed Performance Metrics', fontweight='bold', pad=20)
    
        if 'best_ss' in self.results and 'best_rl' in self.results and self.results['best_rl']:
            best_ss = self.results['best_ss']
            best_rl = self.results['best_rl']
        
            ax5 = fig.add_subplot(gs[2, 0])
            ax5.plot(best_ss['daily_costs'], label='Best (s, S)', alpha=0.7)
            ax5.plot(best_rl['daily_costs'], label='Best RL', alpha=0.7)
            ax5.set_title('Daily Costs')
            ax5.set_xlabel('Day')
            ax5.set_ylabel('Cost')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
              
            ax6 = fig.add_subplot(gs[2, 1])
            ax6.plot(best_ss['daily_shortages'], label='Best (s, S)', alpha=0.7)
            ax6.plot(best_rl['daily_shortages'], label='Best RL', alpha=0.7)
            ax6.set_title('Daily Shortages')
            ax6.set_xlabel('Day')
            ax6.set_ylabel('Shortages')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            
            ax7 = fig.add_subplot(gs[2, 2])
            days_to_plot = min(50, len(best_ss['daily_stock_levels']))
            stock1_ss = [s[0] for s in best_ss['daily_stock_levels'][:days_to_plot]]
            stock2_ss = [s[1] for s in best_ss['daily_stock_levels'][:days_to_plot]]
            stock1_rl = [s[0] for s in best_rl['daily_stock_levels'][:days_to_plot]]
            stock2_rl = [s[1] for s in best_rl['daily_stock_levels'][:days_to_plot]]
            
            ax7.plot(stock1_ss, label='(s, S) P1', alpha=0.7)
            ax7.plot(stock2_ss, label='(s, S) P2', alpha=0.7)
            ax7.plot(stock1_rl, label='RL P1', alpha=0.7, linestyle='--')
            ax7.plot(stock2_rl, label='RL P2', alpha=0.7, linestyle='--')
            ax7.set_title('Stock Levels (50 days)')
            ax7.set_xlabel('Day')
            ax7.set_ylabel('Stock')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        
        fig.suptitle('Inventory Management Simulation - Summary Dashboard', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"Summary dashboard saved to {save_path}")
    
    def plot_per_product_comparison(self, save_path: str = "plots/per_product_comparison.png"):
        """
        Plot per-product cost comparison
        
        Args:
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        policies = []
        product1_costs = []
        product2_costs = []
        
        if 'ss_policies' in self.results:
            for i, policy in enumerate(self.results['ss_policies']):
                policies.append(f'(s,S) {i+1}')
                product1_costs.append(policy['total_cost_product1'])
                product2_costs.append(policy['total_cost_product2'])
        
        if 'rl_agents' in self.results:
            for agent in self.results['rl_agents']:
                policies.append(agent['policy_type'])
                product1_costs.append(agent['total_cost_product1'])
                product2_costs.append(agent['total_cost_product2'])
        
        colors = ['skyblue' if '(s,S)' in p else 'lightcoral' for p in policies]
        
        # Product 1 costs
        bars1 = ax1.bar(policies, product1_costs, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Product 1 Cost Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Total Cost')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, cost in zip(bars1, product1_costs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(product1_costs)*0.01,
                    f'{cost:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Product 2 costs
        bars2 = ax2.bar(policies, product2_costs, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Product 2 Cost Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Total Cost')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, cost in zip(bars2, product2_costs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(product2_costs)*0.01,
                    f'{cost:.0f}', ha='center', va='bottom', fontweight='bold')
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='skyblue', alpha=0.7, label='(s, S) Policies'),
                          Patch(facecolor='lightcoral', alpha=0.7, label='RL Agents')]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"Per-product comparison plot saved to {save_path}")
    
    def plot_per_product_service_levels(self, save_path: str = "plots/per_product_service_levels.png"):
        """
        Plot per-product service level comparison
        
        Args:
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        policies = []
        product1_service = []
        product2_service = []
        
        if 'ss_policies' in self.results:
            for i, policy in enumerate(self.results['ss_policies']):
                policies.append(f'(s,S) {i+1}')
                product1_service.append(policy['service_level_product1'])
                product2_service.append(policy['service_level_product2'])
        
        if 'rl_agents' in self.results:
            for agent in self.results['rl_agents']:
                policies.append(agent['policy_type'])
                product1_service.append(agent['service_level_product1'])
                product2_service.append(agent['service_level_product2'])
        
        colors = ['skyblue' if '(s,S)' in p else 'lightcoral' for p in policies]
        
        # Product 1 service levels
        bars1 = ax1.bar(policies, product1_service, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Product 1 Service Level Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Service Level (%)')
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, level in zip(bars1, product1_service):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{level:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Product 2 service levels
        bars2 = ax2.bar(policies, product2_service, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Product 2 Service Level Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Service Level (%)')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, level in zip(bars2, product2_service):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{level:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='skyblue', alpha=0.7, label='(s, S) Policies'),
                          Patch(facecolor='lightcoral', alpha=0.7, label='RL Agents')]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"Per-product service levels plot saved to {save_path}")
    
    def plot_per_product_daily_performance(self, save_path: str = "plots/per_product_daily_performance.png"):
        """
        Plot per-product daily performance metrics
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        best_ss = self.results.get('best_ss')
        best_rl = self.results.get('best_rl')
        
        if best_ss and best_rl:
            # Product 1 daily costs
            axes[0, 0].plot(best_ss['daily_costs_product1'], label='Best (s, S)', alpha=0.7, linewidth=2)
            axes[0, 0].plot(best_rl['daily_costs_product1'], label='Best RL', alpha=0.7, linewidth=2)
            axes[0, 0].set_title('Product 1 Daily Costs Over Time')
            axes[0, 0].set_xlabel('Day')
            axes[0, 0].set_ylabel('Daily Cost')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Product 2 daily costs
            axes[0, 1].plot(best_ss['daily_costs_product2'], label='Best (s, S)', alpha=0.7, linewidth=2)
            axes[0, 1].plot(best_rl['daily_costs_product2'], label='Best RL', alpha=0.7, linewidth=2)
            axes[0, 1].set_title('Product 2 Daily Costs Over Time')
            axes[0, 1].set_xlabel('Day')
            axes[0, 1].set_ylabel('Daily Cost')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Product 1 daily shortages
            axes[1, 0].plot(best_ss['daily_shortages_product1'], label='Best (s, S)', alpha=0.7, linewidth=2)
            axes[1, 0].plot(best_rl['daily_shortages_product1'], label='Best RL', alpha=0.7, linewidth=2)
            axes[1, 0].set_title('Product 1 Daily Shortages Over Time')
            axes[1, 0].set_xlabel('Day')
            axes[1, 0].set_ylabel('Daily Shortages')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Product 2 daily shortages
            axes[1, 1].plot(best_ss['daily_shortages_product2'], label='Best (s, S)', alpha=0.7, linewidth=2)
            axes[1, 1].plot(best_rl['daily_shortages_product2'], label='Best RL', alpha=0.7, linewidth=2)
            axes[1, 1].set_title('Product 2 Daily Shortages Over Time')
            axes[1, 1].set_xlabel('Day')
            axes[1, 1].set_ylabel('Daily Shortages')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"Per-product daily performance plot saved to {save_path}")
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("Generating all visualization plots...")

        os.makedirs("plots", exist_ok=True)
        self.plot_cost_comparison()
        self.plot_service_level_comparison()
        self.plot_daily_performance()
        self.plot_summary_dashboard()
        
        # Generate per-product plots
        self.plot_per_product_comparison()
        self.plot_per_product_service_levels()
        self.plot_per_product_daily_performance()
        
        print("All plots generated successfully!")
        print("Plots saved in 'plots/' directory")


def create_visualizations(results: Dict[str, Any]):
    """
    Create all visualizations for the results
    
    Args:
        results: Evaluation results
    """
    visualizer = InventoryVisualizer(results)
    visualizer.generate_all_plots()

if __name__ == "__main__":
    print("Visualization module loaded successfully!")
    print("Use create_visualizations(results) to generate plots")
