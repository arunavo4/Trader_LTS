"""
    Apex DQN
"""

import ray
from ray import tune
from ray.rllib.agents.dqn import ApexTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

from lib.env.IndianStockEnv import IndianStockEnv
from lib.model.vision_network import VisionNetwork

tf = try_import_tf()

# register_env("StockTradingEnv", lambda _: StockTradingEnv(10))
ModelCatalog.register_custom_model("NatureCNN", VisionNetwork)

tune.run(ApexTrainer,
         max_failures=10,
         checkpoint_freq=10,  # iterations
         checkpoint_at_end=True,
         config={"env": IndianStockEnv,
                 "model": {
                     "custom_model": "NatureCNN"
                 },
                 "double_q": False,
                 "dueling": False,
                 "num_atoms": 1,
                 "noisy": False,
                 "n_step": 3,
                 "lr": .0001,
                 "adam_epsilon": .00015,
                 "hiddens": [512],
                 "buffer_size": 1000000,
                 "schedule_max_timesteps": 2000000,
                 "exploration_final_eps": 0.01,
                 "exploration_fraction": .1,
                 "prioritized_replay_alpha": 0.5,
                 "beta_annealing_fraction": 1.0,
                 "final_prioritized_replay_beta": 1.0,
                 "num_gpus": 1,
                 "num_workers": 32,
                 "num_envs_per_worker": 1,
                 "train_batch_size": 512,
                 "sample_batch_size": 20,
                 "target_network_update_freq": 50000,
                 "timesteps_per_iteration": 25000,
                 "env_config": {
                     "initial_balance": 10000,
                     "enable_env_logging": False,
                     "look_back_window_size": 375 * 10,
                     "observation_window": 84,
                     "frame_stack_size": 4,
                     "use_leverage": False,
                 },
                 })  # "eager": True for eager execution
