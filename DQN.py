"""
    Rainbow DQN

    Note: For Policy Gradient Algorithms We need to subclass TFModelV2
    and for QModel Algorithms subclass DistributionalQModel.
"""

import ray
from ray import tune
from ray.rllib.agents.dqn.dqn import DQNTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

from lib.env.USStockEnv import USStockEnv
from lib.model.vision_network import VisionNetwork

tf = try_import_tf()

# register_env("StockTradingEnv", lambda _: StockTradingEnv(10))
ModelCatalog.register_custom_model("NatureCNN", VisionNetwork)

# restore = '/home/skywalker/ray_results/DQN/DQN_StockTradingEnv_2f8c5cc4_2019-11-12_23-08-06i2in8uwu/checkpoint_600',
# resume = True,

tune.run(DQNTrainer,
         max_failures=10,
         checkpoint_freq=10,  # iterations
         checkpoint_at_end=True,
         config={"env": USStockEnv,
                 "num_atoms": 51,
                 "noisy": True,
                 "gamma": 0.99,
                 "lr": .0000625,
                 "hiddens": [512],
                 "learning_starts": 20000,
                 "buffer_size": 1000000,
                 "sample_batch_size": 20,
                 "train_batch_size": 512,
                 "schedule_max_timesteps": 3000000,
                 "exploration_final_eps": 0.01,
                 "exploration_fraction": .1,
                 "target_network_update_freq": 8000,
                 "prioritized_replay": True,
                 "prioritized_replay_alpha": 0.5,
                 "beta_annealing_fraction": 0.2,
                 "final_prioritized_replay_beta": 1.0,
                 "n_step": 3,
                 "num_gpus": 1,
                 "num_workers": 4,
                 "model": {
                     "custom_model": "NatureCNN"
                 },
                 "env_config": {
                     "initial_balance": 10000,
                     "enable_env_logging": False,
                     "look_back_window_size": 390 * 10,  # Indian 375 * 10 | US 390 * 10
                     "observation_window": 84,
                     "frame_stack_size": 4,
                     "use_leverage": False,
                     "market": 'us_mkt',
                 },
                 })  # "eager": True for eager execution
# "num_workers": 4,
