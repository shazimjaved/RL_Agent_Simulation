import numpy as np
from typing import Tuple, Dict, Any
from inventory_env import InventoryEnv

class SSPolicy:
    def __init__(self, s1: int, S1: int, s2: int, S2: int):
        
        self.s1 = s1  # Reorder point for Product 1
        self.S1 = S1  # Order-up-to level for Product 1
        self.s2 = s2  # Reorder point for Product 2
        self.S2 = S2  # Order-up-to level for Product 2
        
    def get_action(self, state: np.ndarray) -> Tuple[int, int]:
        # current state
        stock1 = int(state[0])  # On-hand stock Product 1
        stock2 = int(state[1])  # On-hand stock Product 2
        pending1 = int(state[2]) if len(state) > 2 else 0  # Pending orders Product 1
        pending2 = int(state[3]) if len(state) > 3 else 0  # Pending orders Product 2
        
        # Inventory position
        inventory_pos1 = stock1 + pending1
        inventory_pos2 = stock2 + pending2
    
        # (s,S) rule for Product 1
        if inventory_pos1 <= self.s1:
            # Order up to S1
            order1 = max(0, self.S1 - inventory_pos1)
        else:
            order1 = 0
            
        # (s,S) rule for Product 2
        if inventory_pos2 <= self.s2:
            order2 = max(0, self.S2 - inventory_pos2)
        else:
            order2 = 0
            
        return (order1, order2)
    
    def simulate(self, env: InventoryEnv, days: int = 1000) -> Dict[str, Any]:
        #Run (s,S) policy simulation 
        obs, _ = env.reset()
        
        # Lists to track daily metrics
        daily_costs = []
        daily_shortages = []
        daily_orders = []
        daily_stock_levels = []
        
        # Per-product tracking
        daily_costs_product1 = []
        daily_costs_product2 = []
        daily_shortages_product1 = []
        daily_shortages_product2 = []
        daily_orders_product1 = []
        daily_orders_product2 = []
        

        for day in range(days):
            # Get action from (s,S) policy
            action = self.get_action(obs)
            
            # Execute action in environment
            obs, reward, done, truncated, info = env.step(action)
            
            
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
        
        # Calculate service levels
        service_level = (days - sum(1 for s in daily_shortages if s > 0)) / days * 100
        service_level_product1 = (days - sum(1 for s in daily_shortages_product1 if s > 0)) / days * 100
        service_level_product2 = (days - sum(1 for s in daily_shortages_product2 if s > 0)) / days * 100
        
        return {
            'policy_name': f'(s,S) Policy: s1={self.s1}, S1={self.S1}, s2={self.s2}, S2={self.S2}',
            'total_cost': total_cost,
            'average_daily_cost': average_daily_cost,
            'total_shortages': total_shortages,
            'average_shortages': average_shortages,
            'total_orders': total_orders,
            'average_orders': average_orders,
            'service_level': service_level,
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

def optimize_ss_policy_per_product(env: InventoryEnv, days: int = 1000, 
                                   s_range: range = None, S_range: range = None,
                                   seed: int = None) -> Tuple[int, int, int, int]:
    #policy parameters using grid search
    if s_range is None:
        s_range = range(0, 11)  # s from 0 to 10
    if S_range is None:
        S_range = range(1, 51)  # S from 1 to 50
    
    print("(s, S) grid search")
    print(f"Searching s in {list(s_range)[:5]}...{list(s_range)[-1]}, S in {list(S_range)[:5]}...{list(S_range)[-1]}")
    
    #  Product 1
    print("Optimizing for Product 1")
    best_cost_p1 = float('inf')
    best_params_p1 = None
    temp_s2, temp_S2 = 3, 10
    
    for s1 in s_range:
        for S1 in S_range:
            if S1 <= s1: 
                continue
            
            # Create policy 
            policy = SSPolicy(s1, S1, temp_s2, temp_S2)
            if seed is not None:
                obs, _ = env.reset(seed=seed)  
            else:
                obs, _ = env.reset()
            metrics = policy.simulate(env, days)
            
            # Use Product 1 cost to find best Product 1 parameters
            if metrics['total_cost_product1'] < best_cost_p1:
                best_cost_p1 = metrics['total_cost_product1']
                best_params_p1 = (s1, S1)
    
    print(f"Best Product 1: s1={best_params_p1[0]}, S1={best_params_p1[1]}, cost={best_cost_p1:.2f}")
    
    # Product 2
    print("Optimizing for Product 2")
    best_cost_p2 = float('inf')
    best_params_p2 = None
    
    # Grid search over all s2, S2 combinations
    for s2 in s_range:
        for S2 in S_range:
            if S2 <= s2:  # Skip invalid
                continue
            
            # Use best Product 1 parameters, optimize Product 2
            policy = SSPolicy(best_params_p1[0], best_params_p1[1], s2, S2)
            if seed is not None:
                obs, _ = env.reset(seed=seed)  # Same seed 
            else:
                obs, _ = env.reset()
            metrics = policy.simulate(env, days)
          
            # Use Product 2 cost to find best Product 2 parameters
            if metrics['total_cost_product2'] < best_cost_p2:
                best_cost_p2 = metrics['total_cost_product2']
                best_params_p2 = (s2, S2)
    
    print(f"Best Product 2: s2={best_params_p2[0]}, S2={best_params_p2[1]}, cost={best_cost_p2:.2f}")
    
    best_params = (best_params_p1[0], best_params_p1[1], best_params_p2[0], best_params_p2[1])
    print(f"Combined best parameters: s1={best_params[0]}, S1={best_params[1]}, s2={best_params[2]}, S2={best_params[3]}")

    return best_params

def test_ss_policy():
    env = InventoryEnv()
    policy = SSPolicy(s1=3, S1=10, s2=4, S2=12)
    metrics = policy.simulate(env, days=100)
    return metrics


