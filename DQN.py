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

from lib.env.TraderRenkoEnv_v2_lite import StockTradingEnv
from lib.model.vision_network import VisionNetwork

tf = try_import_tf()

# register_env("StockTradingEnv", lambda _: StockTradingEnv(10))
ModelCatalog.register_custom_model("NatureCNN", VisionNetwork)

# restore = '/home/skywalker/ray_results/DQN/DQN_StockTradingEnv_2f8c5cc4_2019-11-12_23-08-06i2in8uwu/checkpoint_600',
# resume = True,
# max_failures = 10,

# SINGLE WORKER
# ray.init(memory=10000000000,
#          object_store_memory=5000000000,
#          redis_max_memory=2000000000,
#          driver_object_store_memory=1000000000)

# 4 WORKERS
ray.init(memory=20000000000,
         object_store_memory=10000000000,
         redis_max_memory=5000000000,
         driver_object_store_memory=2000000000)

tune.run(DQNTrainer,
         max_failures=10,
         checkpoint_freq=10,  # iterations
         checkpoint_at_end=True,
         config={"env": StockTradingEnv,
                 "num_atoms": 51,
                 "parameter_noise": False,
                 "noisy": True,
                 "gamma": 0.99,
                 "lr": .0001,
                 "hiddens": [512],
                 "learning_starts": 10000,
                 "buffer_size": 50000,
                 "sample_batch_size": 4,
                 "train_batch_size": 32,
                 "schedule_max_timesteps": 2000000,
                 "exploration_final_eps": 0.0,
                 "exploration_fraction": .000001,
                 "target_network_update_freq": 500,
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
                     "enable_env_logging": False,
                     "look_back_window_size": 375 * 10,
                     "observation_window": 84,
                     "frame_stack_size": 4,
                 },
                 })  # "eager": True for eager execution
# "num_workers": 4,
