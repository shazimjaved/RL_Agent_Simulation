import simpy
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class InventorySimCore:
    def __init__(self, 
                 holding_cost: float = 1.0,
                 ordering_cost_per_unit: float = 3.0,
                 fixed_ordering_cost: float = 10.0,
                 penalty_cost: float = 7.0,
                 product1_demand_probs: List[float] = None,
                 product1_demand_values: List[int] = None,
                 product2_demand_probs: List[float] = None,
                 product2_demand_values: List[int] = None,
                 product1_lead_time_range: Tuple[float, float] = (0.5, 1.0),
                 product2_lead_time_range: Tuple[float, float] = (0.2, 0.7),
                 lambda_demand: float = 0.1,
                 max_stock: int = 50):
        # cost parameters
        self.holding_cost = holding_cost
        self.ordering_cost_per_unit = ordering_cost_per_unit
        self.fixed_ordering_cost = fixed_ordering_cost
        self.penalty_cost = penalty_cost
        self.max_stock = max_stock
        
        # parameters
        self.product1_demand_probs = product1_demand_probs or [1/6, 1/3, 1/3, 1/6]
        self.product1_demand_values = product1_demand_values or [1, 2, 3, 4]
        self.product2_demand_probs = product2_demand_probs or [1/8, 1/2, 1/4, 1/8]
        self.product2_demand_values = product2_demand_values or [5, 4, 3, 2]
        self.lambda_demand = lambda_demand  # Exponential inter-arrival rate
        self.product1_lead_time_range = product1_lead_time_range  # Uniform(0.5, 1.0)
        self.product2_lead_time_range = product2_lead_time_range  # Uniform(0.2, 0.7)
        
        # SimPy environment 
        self.env = None
        self.stock = np.array([0, 0], dtype=np.int32)  
        self.pending_orders = []  
        self.demand_history_size = 5 
        self.demand_history = []  
        
        # Metrics tracking
        self.total_cost = 0.0
        self.daily_costs = []  
        self.daily_shortages = []  
        self.daily_orders = []  
        
        # Per-product metrics tracking
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
        
        self.current_decision_time = 0.0 
        self.next_decision_time = 0.0  
        self.decision_interval = 1.0 
        self.episode_length = 1000.0  # Total simulation length
        self.rng = None  
        
        self.demand_process_1 = None  # Product 1 demand process
        self.demand_process_2 = None  # Product 2 demand process
        self.order_events = []  
        
        self.period_demand1 = 0  # Total demand for Product 1 in current period
        self.period_demand2 = 0  # Total demand for Product 2 in current period
        self.period_shortage1 = 0  # Total shortages for Product 1 in current period
        self.period_shortage2 = 0  # Total shortages for Product 2 in current period 
        
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        
        # Creating new SimPy environment
        self.env = simpy.Environment()
        self.stock = np.array([0, 0], dtype=np.int32)  
        self.pending_orders = []  
        self.current_decision_time = 0.0  
        self.next_decision_time = 0.0
        self.total_cost = 0.0  
        self.demand_history = []  
        self.order_events = []  
        self.period_demand1 = 0  
        self.period_demand2 = 0
        self.period_shortage1 = 0
        self.period_shortage2 = 0
        
        # INDEPENDENT demand processes for each product
        self.demand_process_1 = self.env.process(self._demand_arrival_process_product1())
        self.demand_process_2 = self.env.process(self._demand_arrival_process_product2())
        
        # Reset metrics tracking
        self.daily_costs = []
        self.daily_shortages = []
        self.daily_orders = []
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
        
        return self._get_observation()
    
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, Dict[str, Any], bool]:
        order1, order2 = action
        
        # Reset period counters
        self.period_demand1 = 0
        self.period_demand2 = 0
        self.period_shortage1 = 0
        self.period_shortage2 = 0
        
        # Process orders for Product 1
        if order1 > 0:
            lead_time = self._generate_lead_time(1) 
            # SimPy process 
            order_event = self.env.process(self._order_arrival_process(1, order1, lead_time))
            self.order_events.append(order_event)
            arrival_time = self.current_decision_time + lead_time
            arrival_day = int(np.ceil(arrival_time))
            self.pending_orders.append((1, order1, arrival_day))  # Tracking pending order
            
        # Process orders for Product 2 
        if order2 > 0:
            lead_time = self._generate_lead_time(2)
            order_event = self.env.process(self._order_arrival_process(2, order2, lead_time))
            self.order_events.append(order_event)
            arrival_time = self.current_decision_time + lead_time
            arrival_day = int(np.ceil(arrival_time))
            self.pending_orders.append((2, order2, arrival_day))
        next_decision_time = self.current_decision_time + self.decision_interval
        self.env.run(until=next_decision_time)  
    
        period_demand1 = self.period_demand1
        period_demand2 = self.period_demand2
        shortage1 = self.period_shortage1
        shortage2 = self.period_shortage2
        
        # Holding cost
        holding_cost1 = self.holding_cost * self.stock[0]
        holding_cost2 = self.holding_cost * self.stock[1]
        
        # Ordering cost
        ordering_cost1 = self._calculate_ordering_cost(order1, 0)
        ordering_cost2 = self._calculate_ordering_cost(0, order2)
        
        # Penalty cost
        penalty_cost1 = self.penalty_cost * shortage1
        penalty_cost2 = self.penalty_cost * shortage2
        
        # Aggregate costs
        holding_cost = holding_cost1 + holding_cost2
        ordering_cost = ordering_cost1 + ordering_cost2
        penalty_cost_total = penalty_cost1 + penalty_cost2
        
        # Total daily cost 
        daily_cost = holding_cost + ordering_cost + penalty_cost_total
        self.total_cost += daily_cost  # total cost
        
        # metrics
        self.daily_costs.append(daily_cost)
        self.daily_shortages.append(shortage1 + shortage2)
        num_orders = (1 if order1 > 0 else 0) + (1 if order2 > 0 else 0)
        self.daily_orders.append(num_orders)
        
        # Per-product tracking
        self.daily_costs_product1.append(holding_cost1 + ordering_cost1 + penalty_cost1)
        self.daily_costs_product2.append(holding_cost2 + ordering_cost2 + penalty_cost2)
        self.daily_shortages_product1.append(shortage1)
        self.daily_shortages_product2.append(shortage2)
        self.daily_orders_product1.append(1 if order1 > 0 else 0)
        self.daily_orders_product2.append(1 if order2 > 0 else 0)
        self.daily_holding_costs_product1.append(holding_cost1)
        self.daily_holding_costs_product2.append(holding_cost2)
        self.daily_ordering_costs_product1.append(ordering_cost1)
        self.daily_ordering_costs_product2.append(ordering_cost2)
        self.daily_penalty_costs_product1.append(penalty_cost1)
        self.daily_penalty_costs_product2.append(penalty_cost2)
        
        # Reward calculation: Negative of cost
        # This directly matches our objective of minimizing total cost
        reward = -daily_cost
        
        self.current_decision_time = next_decision_time
        terminated = self.current_decision_time >= self.episode_length
       
        info = {
            'daily_cost': daily_cost,
            'holding_cost': holding_cost,
            'ordering_cost': ordering_cost,
            'penalty_cost': penalty_cost_total,
            'shortage1': shortage1,
            'shortage2': shortage2,
            'demand1': period_demand1,
            'demand2': period_demand2,
            'order1': order1,  # Quantity ordered for product 1
            'order2': order2,  # Quantity ordered for product 2
            'num_orders1': 1 if order1 > 0 else 0,  # Number of orders for product 1
            'num_orders2': 1 if order2 > 0 else 0,  # Number of orders for product 2
            'total_cost': self.total_cost,
            'daily_cost_product1': holding_cost1 + ordering_cost1 + penalty_cost1,
            'daily_cost_product2': holding_cost2 + ordering_cost2 + penalty_cost2,
            'holding_cost_product1': holding_cost1,
            'holding_cost_product2': holding_cost2,
            'ordering_cost_product1': ordering_cost1,
            'ordering_cost_product2': ordering_cost2,
            'penalty_cost_product1': penalty_cost1,
            'penalty_cost_product2': penalty_cost2,
        }
        
        return self._get_observation(), reward, info, terminated
    
    def _demand_arrival_process_product1(self):
        while True:
            inter_arrival = self.rng.exponential(1.0 / self.lambda_demand) if self.lambda_demand > 0 else 1.0
            yield self.env.timeout(inter_arrival)  
            
            # Generate and process demand ONLY for Product 1
            demand1 = self._generate_demand(1)
            self._process_demand(1, demand1)
            self.demand_history.append((1, demand1, self.env.now))
            if len(self.demand_history) > self.demand_history_size * 2: 
                self.demand_history.pop(0)  
    
    def _demand_arrival_process_product2(self):
        while True:
            inter_arrival = self.rng.exponential(1.0 / self.lambda_demand) if self.lambda_demand > 0 else 1.0
            yield self.env.timeout(inter_arrival)  
            
            # Generate and process demand ONLY for Product 2
            demand2 = self._generate_demand(2)
            self._process_demand(2, demand2)
            
            # Store in history
            self.demand_history.append((2, demand2, self.env.now))
            if len(self.demand_history) > self.demand_history_size * 2: 
                self.demand_history.pop(0)
    
    def _process_demand(self, product: int, demand: int):
        product_idx = product - 1 
        # Track total demand in current period
        if product == 1:
            self.period_demand1 += demand
        else:
            self.period_demand2 += demand

        # Calculate shortage
        shortage = max(0, demand - self.stock[product_idx])

        # Track shortages for penalty cost calculation
        if product == 1:
            self.period_shortage1 += shortage
        else:
            self.period_shortage2 += shortage
        
        # Update stock: reduce by demand 
        self.stock[product_idx] = max(0, self.stock[product_idx] - demand)
        
    def _order_arrival_process(self, product: int, quantity: int, lead_time: float):
        yield self.env.timeout(lead_time)  # Wait for lead time
        product_idx = product - 1
        self.stock[product_idx] = min(self.stock[product_idx] + quantity, self.max_stock)
        
        # Remove from pending orders list
        for i in range(len(self.pending_orders) - 1, -1, -1):
            p, q, _ = self.pending_orders[i]
            if p == product and q == quantity:
                del self.pending_orders[i]
                break
    
    def _generate_demand(self, product: int) -> int:
       
        if product == 1:
            # Product 1
            return self.rng.choice(self.product1_demand_values, p=self.product1_demand_probs)
        else:
            # Product 2
            return self.rng.choice(self.product2_demand_values, p=self.product2_demand_probs)
    
    def _generate_lead_time(self, product: int) -> float:
        if product == 1:
            # Product 1
            return self.rng.uniform(*self.product1_lead_time_range)
        else:
            # Product 2
            return self.rng.uniform(*self.product2_lead_time_range)
    
    def _calculate_ordering_cost(self, order1: int, order2: int) -> float:
     
        cost = 0.0
        if order1 > 0:
            cost += self.fixed_ordering_cost + self.ordering_cost_per_unit * order1
        if order2 > 0:
            cost += self.fixed_ordering_cost + self.ordering_cost_per_unit * order2
        return cost
    
    def _get_observation(self) -> np.ndarray:
        obs_list = []
        
        # Current stock levels
        obs_list.extend(self.stock.tolist())
        
        # Pending orders quantity
        pending_qty1 = sum(qty for prod, qty, _ in self.pending_orders if prod == 1)
        pending_qty2 = sum(qty for prod, qty, _ in self.pending_orders if prod == 2)
        obs_list.extend([pending_qty1, pending_qty2])
        
        # Demand history
        product1_demands = [demand for prod, demand, _ in self.demand_history if prod == 1]
        product2_demands = [demand for prod, demand, _ in self.demand_history if prod == 2]
        
        recent_product1 = product1_demands[-self.demand_history_size:] if product1_demands else []
        recent_product2 = product2_demands[-self.demand_history_size:] if product2_demands else []
        
        # Interleave demand history 
        for i in range(self.demand_history_size):
            d1 = recent_product1[i] if i < len(recent_product1) else 0
            d2 = recent_product2[i] if i < len(recent_product2) else 0
            obs_list.extend([d1, d2])
        
        # Normalized cumulative cost
        normalized_total_cost = min(self.total_cost / 50000.0, 10.0) 
        obs_list.append(normalized_total_cost)
        
        return np.array(obs_list, dtype=np.float32)
    
    def get_metrics(self) -> Dict[str, Any]:
        if not self.daily_costs:
            return {}
        
        days_with_shortages = sum(1 for s in self.daily_shortages if s > 0)
        days_with_shortages_product1 = sum(1 for s in self.daily_shortages_product1 if s > 0)
        days_with_shortages_product2 = sum(1 for s in self.daily_shortages_product2 if s > 0)
        
        total_days = len(self.daily_costs)
        service_level = (total_days - days_with_shortages) / total_days * 100 if total_days > 0 else 0
        service_level_product1 = (total_days - days_with_shortages_product1) / total_days * 100 if total_days > 0 else 0
        service_level_product2 = (total_days - days_with_shortages_product2) / total_days * 100 if total_days > 0 else 0
        
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

