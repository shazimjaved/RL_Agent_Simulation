import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from inventory_env import InventoryEnv
from ss_policy import SSPolicy
from rl_training import evaluate_agent
import os

class InventoryEvaluator:   
    def __init__(self, env: InventoryEnv):
        self.env = env
        self.results = {}
        
    def evaluate_ss_policy(self, s1: int, S1: int, s2: int, S2: int, 
                         days: int = 1000) -> Dict[str, Any]:
        policy = SSPolicy(s1, S1, s2, S2)
        metrics = policy.simulate(self.env, days)
        
        metrics['policy_type'] = 'SS'
        metrics['policy_params'] = f's1={s1}, S1={S1}, s2={s2}, S2={S2}'
        
        return metrics
    
    def evaluate_rl_agent(self, model_path: str, days: int = 1000, agent_type: str = None) -> Dict[str, Any]:
        from stable_baselines3 import PPO
        
        if agent_type is None:
            if 'ppo' in model_path.lower():
                agent_type = 'PPO'
            else:
                agent_type = 'PPO' 
        
        # Load model 
        try:
            if agent_type == 'PPO':
                model = PPO.load(model_path)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}. Only PPO is supported.")
        except Exception as e:
            raise ValueError(f"Could not load {agent_type} model from {model_path}: {e}")
        
        # Evaluate agent
        metrics = evaluate_agent(model, self.env, days, agent_name=agent_type)
        
        # Add policy type
        metrics['policy_type'] = agent_type
        metrics['policy_params'] = f'Model: {os.path.basename(model_path)}'
        
        return metrics
    
    def run_comprehensive_evaluation(self, days: int = 1000, optimize_ss: bool = True) -> Dict[str, Any]:
        results = {}
        
        # Optimize (s, S) 
        if optimize_ss:
            from ss_policy import optimize_ss_policy_per_product
            print("Optimizing (s, S) policy parameters")
            best_ss_params = optimize_ss_policy_per_product(self.env, days=days, seed=42)
            ss_policies = [best_ss_params]
            print(f"Using optimized (s, S) parameters: {best_ss_params}")
        else:
            print("Error")
        
        ss_results = []
        for i, (s1, S1, s2, S2) in enumerate(ss_policies):
            metrics = self.evaluate_ss_policy(s1, S1, s2, S2, days)
            ss_results.append(metrics)
        
        best_ss = min(ss_results, key=lambda x: x['total_cost'])
        results['ss_policies'] = ss_results
        results['best_ss'] = best_ss
        
        # Evaluate RL agent
        rl_results = []
        rl_agent_paths = [
            ("models/ppo_inventory", "PPO"),
        ]
        
        for model_path, agent_type in rl_agent_paths:
            if os.path.exists(model_path + ".zip"):
                try:
                    metrics = self.evaluate_rl_agent(model_path, days, agent_type=agent_type)
                    rl_results.append(metrics)
                    print(f"Evaluated {agent_type}: Cost={metrics['total_cost']:.2f}, Service Level={metrics['service_level']:.1f}%")
                except Exception as e:
                    print(f"Error evaluating {agent_type}: {e}")
        
        if rl_results:
            best_rl = min(rl_results, key=lambda x: x['total_cost'])
            print(f"Best RL Agent: {best_rl['policy_type']} - Cost={best_rl['total_cost']:.2f}")
        else:
            best_rl = None
            print("No RL agent was evaluated.")
        
        results['rl_agents'] = rl_results
        results['best_rl'] = best_rl
        
        if best_rl:
            if best_rl['total_cost'] < best_ss['total_cost']:
                winner = "PPO Agent"
                winner_cost = best_rl['total_cost']
                improvement = ((best_ss['total_cost'] - best_rl['total_cost']) / 
                              best_ss['total_cost'] * 100)
            else:
                winner = "(s, S) Policy"
                winner_cost = best_ss['total_cost']
                improvement = ((best_rl['total_cost'] - best_ss['total_cost']) / 
                              best_rl['total_cost'] * 100)
        else:
            winner = "(s, S) Policy"
            winner_cost = best_ss['total_cost']
            improvement = 0
        
        results['winner'] = winner
        results['winner_cost'] = winner_cost
        results['improvement'] = improvement
        
        summary_data = []
        
        for i, policy in enumerate(ss_results):
            summary_data.append({
                'Policy': f'(s,S) {i+1}',
                'Type': 'Classical',
                'Parameters': policy['policy_params'],
                'Total Cost': policy['total_cost'],
                'Avg Daily Cost': policy['average_daily_cost'],
                'Service Level %': policy['service_level'],
                'Total Shortages': policy['total_shortages'],
                'Total Orders': policy['total_orders']
            })
        
        
        for agent in rl_results:
            summary_data.append({
                'Policy': agent['policy_type'],
                'Type': 'RL',
                'Parameters': agent['policy_params'],
                'Total Cost': agent['total_cost'],
                'Avg Daily Cost': agent['average_daily_cost'],
                'Service Level %': agent['service_level'],
                'Total Shortages': agent['total_shortages'],
                'Total Orders': agent['total_orders']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Total Cost')
        results['summary_table'] = summary_df
        
        results['per_product_tables'] = self._generate_per_product_tables(results)
        
        return results
    
    def _generate_per_product_tables(self, results: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
       
        per_product_tables = {}
        
        # Product 1 comparison table
        product1_data = []
        if 'ss_policies' in results:
            for i, policy in enumerate(results['ss_policies']):
                if len(results['ss_policies']) == 1:
                    policy_name = 'Optimized (s,S)'
                else:
                    policy_name = f'(s,S) {i+1}'
                product1_data.append({
                    'Policy': policy_name,
                    'Type': 'Classical',
                    'Parameters': policy['policy_params'],
                    'Total Cost': policy['total_cost_product1'],
                    'Avg Daily Cost': policy['average_daily_cost_product1'],
                    'Service Level %': policy['service_level_product1'],
                    'Total Shortages': policy['total_shortages_product1'],
                    'Total Orders': policy['total_orders_product1']
                })
        
        if 'rl_agents' in results:
            for agent in results['rl_agents']:
                product1_data.append({
                    'Policy': agent['policy_type'],
                    'Type': 'RL',
                    'Parameters': agent['policy_params'],
                    'Total Cost': agent['total_cost_product1'],
                    'Avg Daily Cost': agent['average_daily_cost_product1'],
                    'Service Level %': agent['service_level_product1'],
                    'Total Shortages': agent['total_shortages_product1'],
                    'Total Orders': agent['total_orders_product1']
                })
        
        product1_df = pd.DataFrame(product1_data)
        product1_df = product1_df.sort_values('Total Cost')
        per_product_tables['product1'] = product1_df
        
        
        product2_data = []
        if 'ss_policies' in results:
            for i, policy in enumerate(results['ss_policies']):
                # Use Optimized policy
                if len(results['ss_policies']) == 1:
                    policy_name = 'Optimized (s,S)'
                else:
                    policy_name = f'(s,S) {i+1}'
                product2_data.append({
                    'Policy': policy_name,
                    'Type': 'Classical',
                    'Parameters': policy['policy_params'],
                    'Total Cost': policy['total_cost_product2'],
                    'Avg Daily Cost': policy['average_daily_cost_product2'],
                    'Service Level %': policy['service_level_product2'],
                    'Total Shortages': policy['total_shortages_product2'],
                    'Total Orders': policy['total_orders_product2']
                })
        
        if 'rl_agents' in results:
            for agent in results['rl_agents']:
                product2_data.append({
                    'Policy': agent['policy_type'],
                    'Type': 'RL',
                    'Parameters': agent['policy_params'],
                    'Total Cost': agent['total_cost_product2'],
                    'Avg Daily Cost': agent['average_daily_cost_product2'],
                    'Service Level %': agent['service_level_product2'],
                    'Total Shortages': agent['total_shortages_product2'],
                    'Total Orders': agent['total_orders_product2']
                })
        
        product2_df = pd.DataFrame(product2_data)
        product2_df = product2_df.sort_values('Total Cost')
        per_product_tables['product2'] = product2_df
        
        best_product1 = product1_df.iloc[0]
        best_product2 = product2_df.iloc[0]
        per_product_tables['best_product1'] = best_product1
        per_product_tables['best_product2'] = best_product2
        
        return per_product_tables
    
    def generate_detailed_report(self, results: Dict[str, Any]) -> str:
       
        report = []
        report.append("DETAILED REPORT")
        report.append("=" * 50)
        report.append("")
        
        report.append("ENVIRONMENT DESCRIPTION")
        report.append("- Target: Minimize overall cost")
        report.append("- Two products with stochastic demand and lead times")
        report.append("- Product 1: Demand D = {1,2,3,4} with probabilities {1/6, 1/3, 1/3, 1/6}")
        report.append("- Product 2: Demand D = {5,4,3,2} with probabilities {1/8, 1/2, 1/4, 1/8}")
        report.append("- Lead times: Product 1 ~ Uniform(0.5, 1), Product 2 ~ Uniform(0.2, 0.7)")
        report.append("- Cost structure: K=10 (fixed), i=3 (variable), h=1 (holding), pi=7 (penalty)")
        report.append("- Demand inter-arrival: Exponential random variable, lambda=0.1")
        report.append("")
        
        report.append("POLICY COMPARISON:")
        report.append("-" * 30)
        
        if 'best_ss' in results:
            best_ss = results['best_ss']
            report.append(f"Best (s, S) Policy: {best_ss['policy_params']}")
            report.append(f"  Total Cost: {best_ss['total_cost']:.2f}")
            report.append(f"  Average Daily Cost: {best_ss['average_daily_cost']:.2f}")
            report.append(f"  Service Level: {best_ss['service_level']:.1f}%")
            report.append(f"  Total Shortages: {best_ss['total_shortages']}")
            report.append(f"  Total Orders: {best_ss['total_orders']}")
            report.append("")
        
        if 'best_rl' in results and results['best_rl']:
            best_rl = results['best_rl']
            report.append(f"Best RL Agent: {best_rl['policy_type']}")
            report.append(f"  Total Cost: {best_rl['total_cost']:.2f}")
            report.append(f"  Average Daily Cost: {best_rl['average_daily_cost']:.2f}")
            report.append(f"  Service Level: {best_rl['service_level']:.1f}%")
            report.append(f"  Total Shortages: {best_rl['total_shortages']}")
            report.append(f"  Total Orders: {best_rl['total_orders']}")
            report.append("")
        
        if 'winner' in results:
            report.append(f"OVERALL WINNER: {results['winner']}")
            report.append(f"Best Cost: {results['winner_cost']:.2f}")
            if results['improvement'] > 0:
                report.append(f"Improvement: {results['improvement']:.1f}%")
            report.append("")
        
        # Detailed results table
        if 'summary_table' in results:
            report.append("DETAILED RESULTS TABLE:")
            report.append("-" * 30)
            report.append(results['summary_table'].to_string(index=False, float_format='%.2f'))
            report.append("")
        
        # Per-product analysis
        if 'per_product_tables' in results:
            per_product_tables = results['per_product_tables']
            
            report.append("PER-PRODUCT ANALYSIS:")
            report.append("-" * 20)
            
            # Product 1 analysis
            report.append("PRODUCT 1 PERFORMANCE:")
            report.append("-" * 20)
            if 'product1' in per_product_tables:
                report.append(per_product_tables['product1'].to_string(index=False, float_format='%.2f'))
                report.append("")
            
            # Product 2 analysis
            report.append("PRODUCT 2 PERFORMANCE:")
            report.append("-" * 20)
            if 'product2' in per_product_tables:
                report.append(per_product_tables['product2'].to_string(index=False, float_format='%.2f'))
                report.append("")
            
            # Best policies per product
            if 'best_product1' in per_product_tables and 'best_product2' in per_product_tables:
                best_p1 = per_product_tables['best_product1']
                best_p2 = per_product_tables['best_product2']
                report.append("BEST POLICIES PER PRODUCT:")
                report.append("-" * 20)
                report.append(f"Product 1: {best_p1['Policy']} - Cost: {best_p1['Total Cost']:.2f}")
                report.append(f"Product 2: {best_p2['Policy']} - Cost: {best_p2['Total Cost']:.2f}")
                report.append("")
        
        report.append("CONCLUSIONS:")
        report.append("-" * 20)
        
        if 'best_ss' in results and 'best_rl' in results and results['best_rl']:
            ss_cost = results['best_ss']['total_cost']
            rl_cost = results['best_rl']['total_cost']
            
            if rl_cost < ss_cost:
                improvement = ((ss_cost - rl_cost) / ss_cost) * 100
                report.append(f"• RL agent outperformed classical (s, S) policy by {improvement:.1f}%")
                report.append("• RL agent achieved lower total cost")
            else:
                improvement = ((rl_cost - ss_cost) / rl_cost) * 100
                report.append(f"• Classical (s, S) policy outperformed RL agent by {improvement:.1f}%")
                report.append("• Classical policy achieved lower total cost")
            
            ss_service = results['best_ss']['service_level']
            rl_service = results['best_rl']['service_level']
            
            if abs(ss_service - rl_service) < 5:
                report.append("• Both policies achieved similar service levels")
            elif rl_service > ss_service:
                report.append(f"• RL agent achieved higher service level ({rl_service:.1f}% vs {ss_service:.1f}%)")
            else:
                report.append(f"• Classical policy achieved higher service level ({ss_service:.1f}% vs {rl_service:.1f}%)")
        
        return "\n".join(report)


def run_evaluation():
    env = InventoryEnv()
    evaluator = InventoryEvaluator(env)
    results = evaluator.run_comprehensive_evaluation(days=1000)
    report = evaluator.generate_detailed_report(results)
    with open("evaluation_report.txt", "w") as f:
        f.write(report)
    return results, report


if __name__ == "__main__":
    results, report = run_evaluation()
    if 'winner' in results:
        print(f"Winner: {results['winner']} (Cost: {results['winner_cost']:.2f})")
