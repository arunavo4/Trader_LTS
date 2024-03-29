"""
    PPO Algorithm
"""

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.visionnet_v2 import VisionNetwork
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search

from lib.env.StockTraderEnv import StockTraderEnv

tf = try_import_tf()

if __name__ == "__main__":
    ModelCatalog.register_custom_model("my_model", VisionNetwork)
    tune.run(
        PPOTrainer,
        stop={
            "timesteps_total": 10000000,
        },
        config={
            "env": StockTraderEnv,  # or "corridor" if registered above
            "model": {
                "custom_model": "my_model",
            },
            "vf_share_layers": True,
            "vf_loss_coeff": grid_search([0.5, 1]),
            "vf_clip_param": 10.0,
            "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
            "num_workers": 4,  # parallelism
            "num_gpus": 1,
            "env_config": {
                "initial_balance": 10000,
                "day_step_size": 375,  # IN 375 | US 390
                "look_back_window_size": 375 * 10,  # US 390 * 10 | 375 * 10
                "enable_env_logging": False,
                "observation_window": 84,
                "frame_stack_size": 4,
                "use_leverage": False,
                "hold_reward": False,
                "market": 'in_mkt',  # 'in_mkt' | 'us_mkt'
            },
        },
    )
