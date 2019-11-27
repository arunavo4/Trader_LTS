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

from lib.env.TraderRenkoEnv_v3_lite import StockTradingEnv

tf = try_import_tf()

if __name__ == "__main__":
    ray.init(memory=20000000000,
             object_store_memory=10000000000,
             redis_max_memory=5000000000,
             driver_object_store_memory=2000000000)
    ModelCatalog.register_custom_model("my_model", VisionNetwork)
    tune.run(
        PPOTrainer,
        stop={
            "timesteps_total": 10000000,
        },
        config={
            "env": StockTradingEnv,  # or "corridor" if registered above
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
                "enable_env_logging": False,
                "look_back_window_size": 375 * 10,
                "observation_window": 84,
                "frame_stack_size": 4,
            },
        },
    )
