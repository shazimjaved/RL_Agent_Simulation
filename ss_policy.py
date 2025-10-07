import numpy as np
from typing import Tuple, Dict, Any
from inventory_env import InventoryEnv


class SSPolicy:
    """
    Classical (s, S) Policy Implementation
    
    This class implements the traditional inventory management policy where:
    - s is the reorder point (when to order)
    - S is the target stock level (how much to order up to)
    """
    
    def __init__(self, s1: int, S1: int, s2: int, S2: int):
        """
        Initialize the (s, S) policy
        
        Args:
            s1: Reorder point for product 1
            S1: Target level for product 1
            s2: Reorder point for product 2
            S2: Target level for product 2
        """
        self.s1 = s1
        self.S1 = S1
        self.s2 = s2
        self.S2 = S2
        
    def get_action(self, state: np.ndarray) -> Tuple[int, int]:
        """
        Get action based on current state using (s, S) policy
        
        Args:
            state: Current stock levels [stock1, stock2]
            
        Returns:
            Action tuple (order1, order2)
        """
        stock1, stock2 = state
        
        if stock1 < self.s1:
            order1 = self.S1 - stock1
        else:
            order1 = 0
            
        if stock2 < self.s2:
            order2 = self.S2 - stock2
        else:
            order2 = 0
            
        return (order1, order2)
    
    def simulate(self, env: InventoryEnv, days: int = 1000) -> Dict[str, Any]:
        """
        Simulate the (s, S) policy for specified number of days
        
        Args:
            env: Inventory environment
            days: Number of days to simulate
            
        Returns:
            Dictionary containing performance metrics
        """
        
        obs = env.reset()
        
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
           
            action = self.get_action(obs)
            
            obs, reward, done, info = env.step(action)
            
            
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


def optimize_ss_policy(env: InventoryEnv, days: int = 1000) -> Tuple[int, int, int, int]:
    """
    Find optimal (s, S) parameters through grid search
    
    Args:
        env: Inventory environment
        days: Number of days for each simulation
        
    Returns:
        Tuple of optimal parameters (s1, S1, s2, S2)
    """
    print("Optimizing (s, S) policy parameters...")
    
    best_cost = float('inf')
    best_params = None
    
    s_range = range(1, 11)  
    S_range = range(5, 21)  
    
    total_combinations = len(s_range) * len(S_range) * len(s_range) * len(S_range)
    current_combination = 0
    
    for s1 in s_range:
        for S1 in S_range:
            if S1 <= s1:  
                continue
            for s2 in s_range:
                for S2 in S_range:
                    if S2 <= s2:  
                        continue
                        
                    current_combination += 1
                    if current_combination % 100 == 0:
                        print(f"Testing combination {current_combination}/{total_combinations}")
                    
                    
                    policy = SSPolicy(s1, S1, s2, S2)
                    metrics = policy.simulate(env, days)
                    
                    if metrics['total_cost'] < best_cost:
                        best_cost = metrics['total_cost']
                        best_params = (s1, S1, s2, S2)
                        print(f"New best: s1={s1}, S1={S1}, s2={s2}, S2={S2}, "
                              f"cost={best_cost:.2f}")
    
    print(f"Optimization complete! Best parameters: {best_params}")
    print(f"Best total cost: {best_cost:.2f}")
    
    return best_params


def test_ss_policy():
    """Test the (s, S) policy implementation"""
    print("Testing (s, S) Policy Implementation...")
    
    env = InventoryEnv()
    policy = SSPolicy(s1=3, S1=10, s2=4, S2=12)
    print(f"Policy parameters: s1={policy.s1}, S1={policy.s1}, s2={policy.s2}, S2={policy.S2}")
    metrics = policy.simulate(env, days=100)
    print("\nSimulation Results:")
    print(f"Total Cost: {metrics['total_cost']:.2f}")
    print(f"Average Daily Cost: {metrics['average_daily_cost']:.2f}")
    print(f"Total Shortages: {metrics['total_shortages']}")
    print(f"Service Level: {metrics['service_level']:.1f}%")
    print(f"Total Orders: {metrics['total_orders']}")
    
    return metrics


if __name__ == "__main__":
    test_metrics = test_ss_policy()
    
    print("\n" + "="*50)
    print("Running optimization (this may take a while)...")
