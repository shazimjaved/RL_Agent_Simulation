import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional
from simpyy import InventorySimCore

class InventoryEnv(gym.Env):
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 max_stock: int = 50, 
                 max_order: int = 10,
                 episode_length: int = 1000,
                 seed: Optional[int] = None):
        
        super(InventoryEnv, self).__init__()
        
        self.max_stock = max_stock
        self.max_order = max_order
        self.episode_length = episode_length
        self.seed_value = seed
        
        # Action space
        self.action_space = spaces.MultiDiscrete([max_order + 1, max_order + 1])
        
        # Observation space
        obs_size = 2 + 2 + (5 * 2) + 1
        self.observation_space = spaces.Box(
            low=0.0, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Initializing SimPy simulation 
        self.sim_core = InventorySimCore(
            holding_cost=1.0,
            ordering_cost_per_unit=3.0,
            fixed_ordering_cost=10.0,
            penalty_cost=7.0,
            product1_demand_probs=[1/6, 1/3, 1/3, 1/6],
            product1_demand_values=[1, 2, 3, 4],
            product2_demand_probs=[1/8, 1/2, 1/4, 1/8],
            product2_demand_values=[5, 4, 3, 2],
            product1_lead_time_range=(0.5, 1.0),
            product2_lead_time_range=(0.2, 0.7),
            lambda_demand=0.1,
            max_stock=max_stock
        )
        
        # episode length
        self.sim_core.episode_length = float(episode_length)
        self.current_step = 0  # Track current step for truncation
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        #Reset environment to initial state 
        reset_seed = seed if seed is not None else self.seed_value
        
        # Reset SimPy core
        obs = self.sim_core.reset(seed=reset_seed)
        self.current_step = 0
        
        return obs, {}
    
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        
        # Clip action to valid range 
        action = np.clip(action, 0, self.max_order)
        
        # Execute step in SimPy core
        obs, reward, info, terminated = self.sim_core.step(tuple(action))
        self.current_step += 1
        
        # Check if episode truncated 
        truncated = self.current_step >= self.episode_length
        info['current_step'] = self.current_step
        info['episode_length'] = self.episode_length
        
        if truncated:
            terminated = True
        
        # it Returns Gymnasium format
        return obs, reward, terminated, False, info
    
    def render(self, mode: str = 'human'):
        if mode == 'human':
            pass
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.sim_core.get_metrics()
    
    def close(self):
        """Clean up resources."""
        pass
if __name__ == "__main__":
    env = InventoryEnv(episode_length=10)
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: action={action}, obs={obs}, reward={reward:.2f}, cost={-reward:.2f}")
        if terminated or truncated:
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Total cost: {-total_reward:.2f}")
    
    metrics = env.get_metrics()
    print(f"\nMetrics: {metrics}")


