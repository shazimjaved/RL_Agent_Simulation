import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import os
from typing import Dict, Any, List
from inventory_env import InventoryEnv

class CostTrackingCallback(BaseCallback):
   
    def __init__(self, verbose=0):
        super(CostTrackingCallback, self).__init__(verbose)
        self.costs = []  # Store all costs
        self.episode_costs = []
        
    def _on_step(self) -> bool:
        """Called at each step and it collects cost information"""
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if isinstance(info, dict) and 'daily_cost' in info:
                    self.costs.append(info['daily_cost'])
        elif 'info' in self.locals:
            info = self.locals['info']
            if isinstance(info, dict) and 'daily_cost' in info:
                self.costs.append(info['daily_cost'])
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at end of each rollout and it print average cost"""
        if len(self.costs) > 0:
            avg_cost = np.mean(self.costs[-100:])  # Last 100 steps
            if self.verbose > 0 and len(self.costs) % 100 == 0:
                print(f"  Average cost (last 100 steps): {avg_cost:.2f}")

def train_ppo_agent(env: InventoryEnv, total_timesteps: int = 50000, 
                   save_path: str = "models/ppo_inventory", 
                   eval_env: InventoryEnv = None,
                   eval_freq: int = 10000) -> PPO: 
    if eval_env is None:
        eval_env = InventoryEnv(episode_length=1000)
    
    # Initialize PPO 
    model = PPO(
        "MlpPolicy",  
        env,
        verbose=1,
        learning_rate=2e-4,  
        n_steps=2048,  
        batch_size=64,  
        n_epochs=10, 
        gamma=0.99,  
        gae_lambda=0.95, 
        clip_range=0.2, 
        ent_coef=0.1,  
        vf_coef=0.5, 
        max_grad_norm=0.5,  
        policy_kwargs=dict(
            net_arch=[128, 128, 64]  # Neural network
        ),
        tensorboard_log="./tensorboard_logs/ppo/"  # Logs for TensorBoard
    )
    
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # EvalCallback Validates agent during training
    eval_callback = EvalCallback(
        eval_env,  
        best_model_save_path=save_path + "_best",  # Save best model here
        log_path="./logs/ppo_eval/",
        eval_freq=eval_freq,  # Evaluate every 10,000 steps
        deterministic=True,  
        render=False,
        n_eval_episodes=5 
    )
    
    # Cost tracking callback 
    cost_callback = CostTrackingCallback(verbose=1)
    callbacks = [eval_callback, cost_callback]
    
    print(f"Training PPO agent for {total_timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except (TypeError, ImportError, ValueError) as e:
        error_str = str(e).lower()
        if "progress_bar" in error_str or "tqdm" in error_str or "rich" in error_str:
            print(" tqdm/rich not installed")
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks
            )
        else:
            raise
    model.save(save_path)
    print(f"PPO model saved to {save_path}")
    return model

def evaluate_agent(model, env: InventoryEnv, days: int = 1000, agent_name: str = "Agent") -> Dict[str, Any]:

    obs, _ = env.reset()  # Reset to initial state
    
    # Lists to track metrics
    daily_costs = []
    daily_shortages = []
    daily_orders = []
    daily_stock_levels = []
    total_reward = 0
    
    # Per-product tracking
    daily_costs_product1 = []
    daily_costs_product2 = []
    daily_shortages_product1 = []
    daily_shortages_product2 = []
    daily_orders_product1 = []
    daily_orders_product2 = []
    
    # Run simulation for specified days
    for day in range(days):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        # Track total metrics
        daily_costs.append(info['daily_cost'])
        daily_shortages.append(info['shortage1'] + info['shortage2'])
        num_orders = info.get('num_orders1', 1 if info['order1'] > 0 else 0) + info.get('num_orders2', 1 if info['order2'] > 0 else 0)
        daily_orders.append(num_orders)
        daily_stock_levels.append(obs.copy())
        # Track per-product metrics
        daily_costs_product1.append(info['daily_cost_product1'])
        daily_costs_product2.append(info['daily_cost_product2'])
        daily_shortages_product1.append(info['shortage1'])
        daily_shortages_product2.append(info['shortage2'])
        daily_orders_product1.append(info['order1'])
        daily_orders_product2.append(info['order2'])
        
        if done or truncated:
            break
    # summary metrics
    total_cost = sum(daily_costs)
    average_daily_cost = np.mean(daily_costs)
    total_shortages = sum(daily_shortages)
    average_shortages = np.mean(daily_shortages)
    total_orders = sum(daily_orders)
    average_orders = np.mean(daily_orders)
    
    # Calculate per-product metrics
    total_cost_product1 = sum(daily_costs_product1)
    total_cost_product2 = sum(daily_costs_product2)
    average_daily_cost_product1 = np.mean(daily_costs_product1)
    average_daily_cost_product2 = np.mean(daily_costs_product2)
    total_shortages_product1 = sum(daily_shortages_product1)
    total_shortages_product2 = sum(daily_shortages_product2)
    total_orders_product1 = sum(daily_orders_product1)
    total_orders_product2 = sum(daily_orders_product2)
    
    # Calculate service level
    service_level = (days - sum(1 for s in daily_shortages if s > 0)) / days * 100 if days > 0 else 0
    service_level_product1 = (days - sum(1 for s in daily_shortages_product1 if s > 0)) / days * 100 if days > 0 else 0
    service_level_product2 = (days - sum(1 for s in daily_shortages_product2 if s > 0)) / days * 100 if days > 0 else 0
    
    return {
        'total_cost': total_cost,
        'average_daily_cost': average_daily_cost,
        'total_shortages': total_shortages,
        'average_shortages': average_shortages,
        'total_orders': total_orders,
        'average_orders': average_orders,
        'service_level': service_level,
        'total_reward': total_reward,
        'days_simulated': days,
        'daily_costs': daily_costs,
        'daily_shortages': daily_shortages,
        'daily_orders': daily_orders,
        'daily_stock_levels': daily_stock_levels,
        # Per-product metrics
        'total_cost_product1': total_cost_product1,
        'total_cost_product2': total_cost_product2,
        'average_daily_cost_product1': average_daily_cost_product1,
        'average_daily_cost_product2': average_daily_cost_product2,
        'total_shortages_product1': total_shortages_product1,
        'total_shortages_product2': total_shortages_product2,
        'total_orders_product1': total_orders_product1,
        'total_orders_product2': total_orders_product2,
        'service_level_product1': service_level_product1,
        'service_level_product2': service_level_product2,
        'daily_costs_product1': daily_costs_product1,
        'daily_costs_product2': daily_costs_product2,
        'daily_shortages_product1': daily_shortages_product1,
        'daily_shortages_product2': daily_shortages_product2,
        'daily_orders_product1': daily_orders_product1,
        'daily_orders_product2': daily_orders_product2
    }

def train_and_evaluate_agents(training_timesteps: int = 50000) -> Dict[str, Any]:
    env = InventoryEnv(episode_length=1000)
    eval_env = InventoryEnv(episode_length=1000) 
    
    results = {
        'ppo_model': None,
        'ppo_metrics': None,
    }
    
    # PPO agent training
    try:
        ppo_model = train_ppo_agent(env, training_timesteps, "models/ppo_inventory", eval_env=eval_env)
        ppo_metrics = evaluate_agent(ppo_model, eval_env, days=1000, agent_name="PPO")
        results['ppo_model'] = ppo_model
        results['ppo_metrics'] = ppo_metrics
        print(f"PPO - Total Cost: {ppo_metrics['total_cost']:.2f}, Service Level: {ppo_metrics['service_level']:.1f}%")
        
        results['best_name'] = 'PPO'
        results['best_model'] = ppo_model
        results['best_metrics'] = ppo_metrics
        print(f"\n RL Agent: PPO - Total Cost: {ppo_metrics['total_cost']:.2f}")
    except Exception as e:
        print(f"Error training PPO: {e}")
        results['ppo_metrics'] = None
        results['best_name'] = None
        results['best_model'] = None
        results['best_metrics'] = None
        print("\nNo agent trained.")
    
    return results

if __name__ == "__main__":
    results = train_and_evaluate_agents()
