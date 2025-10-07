import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any
import random


class InventoryEnv(gym.Env):
    """
    Inventory Management Environment for two products with stochastic demand and lead times.
    Cost Structure:
        - Holding cost (h) = 1 per unit per day
        - Ordering cost (i) = 3 per order + fixed cost (K = 10)
        - Penalty cost (π) = 7 per unit of unmet demand
    """
    
    def __init__(self, max_stock: int = 50, max_order: int = 10):

        super(InventoryEnv, self).__init__()        
        self.max_stock = max_stock
        self.max_order = max_order
        self.holding_cost = 1.0  # h = 1 per unit per day
        self.ordering_cost_per_unit = 3.0  # i = 3 per order
        self.fixed_ordering_cost = 10.0  # K = 10
        self.penalty_cost = 7.0  # π = 7 per unit of unmet demand
        self.product1_demand_probs = [1/6, 1/3, 1/3, 1/6]  # For demands 1,2,3,4
        self.product1_demand_values = [1, 2, 3, 4]
        self.product2_demand_probs = [1/8, 1/4, 1/2, 1/8]  # For demands 2,3,4,5
        self.product2_demand_values = [2, 3, 4, 5]
        self.product1_lead_time_range = (0.5, 1.0)  # Uniform(0.5, 1)
        self.product2_lead_time_range = (0.2, 0.7)  # Uniform(0.2, 0.7)
        self.action_space = spaces.MultiDiscrete([max_order + 1, max_order + 1])
        self.observation_space = spaces.Box(
            low=0, high=max_stock, shape=(2,), dtype=np.int32
        )
        
        # Initialize state
        self.stock = np.array([0, 0], dtype=np.int32)
        self.pending_orders = [] 
        self.current_day = 0
        self.total_cost = 0.0
        
        # variables
        self.daily_costs = []
        self.daily_shortages = []
        self.daily_orders = []
        
        # Per-product 
        self.daily_costs_product1 = []
        self.daily_costs_product2 = []
        self.daily_shortages_product1 = []
        self.daily_shortages_product2 = []
        self.daily_orders_product1 = []
        self.daily_orders_product2 = []
        self.daily_holding_costs_product1 = []
        self.daily_holding_costs_product2 = []
        self.daily_ordering_costs_product1 = []
        self.daily_ordering_costs_product2 = []
        self.daily_penalty_costs_product1 = []
        self.daily_penalty_costs_product2 = []
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        
        self.stock = np.array([0, 0], dtype=np.int32)
        self.pending_orders = []
        self.current_day = 0
        self.total_cost = 0.0
        self.daily_costs = []
        self.daily_shortages = []
        self.daily_orders = []
        
        # Reset per-product tracking variables
        self.daily_costs_product1 = []
        self.daily_costs_product2 = []
        self.daily_shortages_product1 = []
        self.daily_shortages_product2 = []
        self.daily_orders_product1 = []
        self.daily_orders_product2 = []
        self.daily_holding_costs_product1 = []
        self.daily_holding_costs_product2 = []
        self.daily_ordering_costs_product1 = []
        self.daily_ordering_costs_product2 = []
        self.daily_penalty_costs_product1 = []
        self.daily_penalty_costs_product2 = []
        
        return self.stock.copy(), {}
    
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
       
        order1, order2 = action
        
        # Process pending orders (arrivals)
        self._process_pending_orders()
        
        # Generate demand for current day
        demand1 = self._generate_demand(1)
        demand2 = self._generate_demand(2)
        
        # Calculate shortages
        shortage1 = max(0, demand1 - self.stock[0])
        shortage2 = max(0, demand2 - self.stock[1])
        
        # Update stock levels
        self.stock[0] = max(0, self.stock[0] - demand1)
        self.stock[1] = max(0, self.stock[1] - demand2)
        
        # Place new orders if action > 0
        if order1 > 0:
            lead_time = self._generate_lead_time(1)
            arrival_day = self.current_day + int(np.ceil(lead_time))
            self.pending_orders.append((1, order1, arrival_day))
            
        if order2 > 0:
            lead_time = self._generate_lead_time(2)
            arrival_day = self.current_day + int(np.ceil(lead_time))
            self.pending_orders.append((2, order2, arrival_day))
        
        # Calculate per-product costs
        holding_cost1 = self.holding_cost * self.stock[0]
        holding_cost2 = self.holding_cost * self.stock[1]
        ordering_cost1 = self._calculate_ordering_cost(order1, 0)  # Only product 1
        ordering_cost2 = self._calculate_ordering_cost(0, order2)  # Only product 2
        penalty_cost1 = self.penalty_cost * shortage1
        penalty_cost2 = self.penalty_cost * shortage2
        
        # Calculate total costs
        holding_cost = holding_cost1 + holding_cost2
        ordering_cost = ordering_cost1 + ordering_cost2
        penalty_cost = penalty_cost1 + penalty_cost2
        
        daily_cost = holding_cost + ordering_cost + penalty_cost
        self.total_cost += daily_cost
        
        # Track overall metrics
        self.daily_costs.append(daily_cost)
        self.daily_shortages.append(shortage1 + shortage2)
        self.daily_orders.append(order1 + order2)
        
        # Track per-product metrics
        self.daily_costs_product1.append(holding_cost1 + ordering_cost1 + penalty_cost1)
        self.daily_costs_product2.append(holding_cost2 + ordering_cost2 + penalty_cost2)
        self.daily_shortages_product1.append(shortage1)
        self.daily_shortages_product2.append(shortage2)
        self.daily_orders_product1.append(order1)
        self.daily_orders_product2.append(order2)
        self.daily_holding_costs_product1.append(holding_cost1)
        self.daily_holding_costs_product2.append(holding_cost2)
        self.daily_ordering_costs_product1.append(ordering_cost1)
        self.daily_ordering_costs_product2.append(ordering_cost2)
        self.daily_penalty_costs_product1.append(penalty_cost1)
        self.daily_penalty_costs_product2.append(penalty_cost2)
        
        reward = -daily_cost
        self.current_day += 1
        
        # Prepare info dictionary
        info = {
            'daily_cost': daily_cost,
            'holding_cost': holding_cost,
            'ordering_cost': ordering_cost,
            'penalty_cost': penalty_cost,
            'shortage1': shortage1,
            'shortage2': shortage2,
            'demand1': demand1,
            'demand2': demand2,
            'order1': order1,
            'order2': order2,
            'total_cost': self.total_cost,
            'daily_cost_product1': holding_cost1 + ordering_cost1 + penalty_cost1,
            'daily_cost_product2': holding_cost2 + ordering_cost2 + penalty_cost2,
            'holding_cost_product1': holding_cost1,
            'holding_cost_product2': holding_cost2,
            'ordering_cost_product1': ordering_cost1,
            'ordering_cost_product2': ordering_cost2,
            'penalty_cost_product1': penalty_cost1,
            'penalty_cost_product2': penalty_cost2
        }
        
        return self.stock.copy(), reward, False, False, info
    
    def _generate_demand(self, product: int) -> int:
        
        if product == 1:
            return np.random.choice(self.product1_demand_values, p=self.product1_demand_probs)
        else:
            return np.random.choice(self.product2_demand_values, p=self.product2_demand_probs)
    
    def _generate_lead_time(self, product: int) -> float:
        
        if product == 1:
            return np.random.uniform(*self.product1_lead_time_range)
        else:
            return np.random.uniform(*self.product2_lead_time_range)
    
    def _process_pending_orders(self):
        """Process pending orders that arrive today"""
        orders_to_remove = []
        
        for i, (product, quantity, arrival_day) in enumerate(self.pending_orders):
            if arrival_day <= self.current_day:
                # Order arrives
                self.stock[product - 1] += quantity
                orders_to_remove.append(i)
        
        for i in reversed(orders_to_remove):
            del self.pending_orders[i]
    
    def _calculate_ordering_cost(self, order1: int, order2: int) -> float:
        
        cost = 0.0
        
        if order1 > 0:
            cost += self.fixed_ordering_cost + self.ordering_cost_per_unit * order1
            
        if order2 > 0:
            cost += self.fixed_ordering_cost + self.ordering_cost_per_unit * order2
            
        return cost
    
    def render(self, mode: str = 'human'):
        
        if mode == 'human':
            print(f"Day {self.current_day}: Stock=[{self.stock[0]}, {self.stock[1]}], "
                  f"Pending Orders={len(self.pending_orders)}, "
                  f"Total Cost={self.total_cost:.2f}")
    
    def get_metrics(self) -> Dict[str, Any]:
        if not self.daily_costs:
            return {}
        
        # Calculate service levels
        days_with_shortages = sum(1 for s in self.daily_shortages if s > 0)
        days_with_shortages_product1 = sum(1 for s in self.daily_shortages_product1 if s > 0)
        days_with_shortages_product2 = sum(1 for s in self.daily_shortages_product2 if s > 0)
        
        total_days = len(self.daily_costs)
        service_level = (total_days - days_with_shortages) / total_days * 100
        service_level_product1 = (total_days - days_with_shortages_product1) / total_days * 100
        service_level_product2 = (total_days - days_with_shortages_product2) / total_days * 100
            
        return {
            'total_cost': self.total_cost,
            'average_daily_cost': np.mean(self.daily_costs),
            'total_shortages': sum(self.daily_shortages),
            'average_shortages': np.mean(self.daily_shortages),
            'total_orders': sum(self.daily_orders),
            'average_orders': np.mean(self.daily_orders),
            'service_level': service_level,
            'days_simulated': total_days,
            'total_cost_product1': sum(self.daily_costs_product1),
            'total_cost_product2': sum(self.daily_costs_product2),
            'average_daily_cost_product1': np.mean(self.daily_costs_product1),
            'average_daily_cost_product2': np.mean(self.daily_costs_product2),
            'total_shortages_product1': sum(self.daily_shortages_product1),
            'total_shortages_product2': sum(self.daily_shortages_product2),
            'average_shortages_product1': np.mean(self.daily_shortages_product1),
            'average_shortages_product2': np.mean(self.daily_shortages_product2),
            'total_orders_product1': sum(self.daily_orders_product1),
            'total_orders_product2': sum(self.daily_orders_product2),
            'average_orders_product1': np.mean(self.daily_orders_product1),
            'average_orders_product2': np.mean(self.daily_orders_product2),
            'service_level_product1': service_level_product1,
            'service_level_product2': service_level_product2,
            'total_holding_cost_product1': sum(self.daily_holding_costs_product1),
            'total_holding_cost_product2': sum(self.daily_holding_costs_product2),
            'total_ordering_cost_product1': sum(self.daily_ordering_costs_product1),
            'total_ordering_cost_product2': sum(self.daily_ordering_costs_product2),
            'total_penalty_cost_product1': sum(self.daily_penalty_costs_product1),
            'total_penalty_cost_product2': sum(self.daily_penalty_costs_product2)
        }


if __name__ == "__main__":
    env = InventoryEnv()
    
    print("Testing Inventory Environment...")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    obs, _ = env.reset()
    print(f"Initial observation: {obs}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Obs={obs}, Reward={reward:.2f}")
        env.render()
    
    print("\nEnvironment test completed successfully!")
