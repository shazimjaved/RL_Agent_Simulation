import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import os
from typing import Dict, Any
from inventory_env import InventoryEnv


def train_ppo_agent(env: InventoryEnv, total_timesteps: int = 10000, 
                   save_path: str = "models/ppo_inventory") -> PPO:
    """
    Train a PPO agent for inventory management
    
    Args:
        env: Inventory environment
        total_timesteps: Number of training timesteps
        save_path: Path to save the trained model
        
    Returns:
        Trained PPO agent
    """
    print("Training PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=4096,         
        batch_size=128,  
        n_epochs=20,          
        gamma=0.95,       
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,       
        vf_coef=0.5,
        max_grad_norm=0.5, 
    )
    
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"PPO model saved to {save_path}")
    
    return model


def train_a2c_agent(env: InventoryEnv, total_timesteps: int = 10000,
                   save_path: str = "models/a2c_inventory") -> A2C:
    """
    Train an A2C agent for inventory management
    
    Args:
        env: Inventory environment
        total_timesteps: Number of training timesteps
        save_path: Path to save the trained model
        
    Returns:
        Trained A2C agent
    """
    print("Training A2C agent...")
    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,   
        n_steps=20,           
        gamma=0.95,           
        gae_lambda=0.95,      
        ent_coef=0.01,       
        vf_coef=0.5,         
        max_grad_norm=0.5,
        
    )
    
    
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    model.learn(total_timesteps=total_timesteps)
    
    model.save(save_path)
    print(f"A2C model saved to {save_path}")
    
    return model


def evaluate_agent(model, env: InventoryEnv, days: int = 1000) -> Dict[str, Any]:
    """
    Evaluate a trained agent
    
    Args:
        model: Trained RL model
        env: Inventory environment
        days: Number of days to evaluate
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"Evaluating agent for {days} days...")
    obs, _ = env.reset()
    
    # Track metrics
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
    
    for day in range(days):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Execute action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Track overall metrics
        daily_costs.append(info['daily_cost'])
        daily_shortages.append(info['shortage1'] + info['shortage2'])
        daily_orders.append(info['order1'] + info['order2'])
        daily_stock_levels.append(obs.copy())
        
        # Track per-product metrics
        daily_costs_product1.append(info['daily_cost_product1'])
        daily_costs_product2.append(info['daily_cost_product2'])
        daily_shortages_product1.append(info['shortage1'])
        daily_shortages_product2.append(info['shortage2'])
        daily_orders_product1.append(info['order1'])
        daily_orders_product2.append(info['order2'])
        
        if (day + 1) % 100 == 0:
            print(f"Day {day + 1}: Stock={obs}, Action={action}, "
                  f"Cost={info['daily_cost']:.2f}")
    
    # Calculate summary metrics
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
    service_level = (days - sum(1 for s in daily_shortages if s > 0)) / days * 100
    service_level_product1 = (days - sum(1 for s in daily_shortages_product1 if s > 0)) / days * 100
    service_level_product2 = (days - sum(1 for s in daily_shortages_product2 if s > 0)) / days * 100
    
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


def train_and_evaluate_agents():
    """
    Train both PPO and A2C agents and evaluate their performance
    """
    print("Starting RL Agent Training and Evaluation")
    print("="*50)
    
    # environment
    env = InventoryEnv()
    
    # Training parameters
    training_timesteps = 15000 
    
    # PPO agent
    print("\n1. Training PPO Agent")
    print("-" * 30)
    ppo_model = train_ppo_agent(env, training_timesteps)
    
    # A2C agent
    print("\n2. Training A2C Agent")
    print("-" * 30)
    a2c_model = train_a2c_agent(env, training_timesteps)
    
    # Evaluate both agents
    print("\n3. Evaluating Agents")
    print("-" * 30)
    
    # Evaluate PPO
    print("\nEvaluating PPO Agent:")
    ppo_metrics = evaluate_agent(ppo_model, env, days=1000)
    
    # Evaluate A2C
    print("\nEvaluating A2C Agent:")
    a2c_metrics = evaluate_agent(a2c_model, env, days=1000)
    
    # comparison
    print("\n4. Performance Comparison")
    print("-" * 30)
    print(f"{'Metric':<20} {'PPO':<15} {'A2C':<15}")
    print("-" * 50)
    print(f"{'Total Cost':<20} {ppo_metrics['total_cost']:<15.2f} {a2c_metrics['total_cost']:<15.2f}")
    print(f"{'Avg Daily Cost':<20} {ppo_metrics['average_daily_cost']:<15.2f} {a2c_metrics['average_daily_cost']:<15.2f}")
    print(f"{'Total Shortages':<20} {ppo_metrics['total_shortages']:<15} {a2c_metrics['total_shortages']:<15}")
    print(f"{'Service Level %':<20} {ppo_metrics['service_level']:<15.1f} {a2c_metrics['service_level']:<15.1f}")
    print(f"{'Total Orders':<20} {ppo_metrics['total_orders']:<15} {a2c_metrics['total_orders']:<15}")
    
    # best agent
    if ppo_metrics['total_cost'] < a2c_metrics['total_cost']:
        best_model = ppo_model
        best_metrics = ppo_metrics
        best_name = "PPO"
    else:
        best_model = a2c_model
        best_metrics = a2c_metrics
        best_name = "A2C"
    
    print(f"\nBest performing agent: {best_name}")
    
    # Save the best model
    best_model.save("inventory_agent")
    print("Best model saved as 'inventory_agent.zip'")
    
    return {
        'ppo_model': ppo_model,
        'a2c_model': a2c_model,
        'ppo_metrics': ppo_metrics,
        'a2c_metrics': a2c_metrics,
        'best_model': best_model,
        'best_metrics': best_metrics,
        'best_name': best_name
    }


def test_trained_agent():
    """Test a trained agent with a short simulation"""
    print("Testing Trained Agent...")
    
    env = InventoryEnv()
    
    # Load the best model
    try:
        model = PPO.load("models/ppo_inventory")
        print("Loaded PPO model")
    except:
        try:
            model = A2C.load("models/a2c_inventory")
            print("Loaded A2C model")
        except:
            print("No trained model found. Training a quick model...")
            model = train_ppo_agent(env, total_timesteps=1000)
    
    # Run short test
    obs, _ = env.reset()
    print(f"Initial state: {obs}")
    
    for i in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, State={obs}, Reward={reward:.2f}")
    
    print("Agent test completed!")


if __name__ == "__main__":
    results = train_and_evaluate_agents()
    print("\n" + "="*50)
    print("Training and evaluation completed!")
    print("Models saved in 'models/' directory")
    print("Best model saved as 'inventory_agent.zip'")
