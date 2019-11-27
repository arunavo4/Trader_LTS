"""
    Apex DQN
"""

import ray
from ray import tune
from ray.rllib.agents.dqn import ApexTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

from lib.env.TraderRenkoEnv_v3_lite import StockTradingEnv
from lib.model.vision_network import VisionNetwork

tf = try_import_tf()

# register_env("StockTradingEnv", lambda _: StockTradingEnv(10))
ModelCatalog.register_custom_model("NatureCNN", VisionNetwork)

ray.init(memory=20000000000,
         object_store_memory=10000000000,
         redis_max_memory=5000000000,
         driver_object_store_memory=2000000000)

tune.run(ApexTrainer,
         max_failures=10,
         checkpoint_freq=10,  # iterations
         checkpoint_at_end=True,
         config={"env": StockTradingEnv,
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
                 "schedule_max_timesteps": 5000000,
                 "exploration_final_eps": 0.01,
                 "exploration_fraction": .1,
                 "prioritized_replay_alpha": 0.5,
                 "beta_annealing_fraction": 1.0,
                 "final_prioritized_replay_beta": 1.0,
                 "num_gpus": 1,
                 "num_workers": 4,
                 "num_envs_per_worker": 2,
                 "sample_batch_size": 20,
                 "train_batch_size": 512,
                 "per_worker_exploration": True,
                 "worker_side_prioritization": True,
                 "target_network_update_freq": 50000,
                 "timesteps_per_iteration": 25000,
                 "env_config": {
                     "enable_env_logging": False,
                     "look_back_window_size": 375 * 10,
                     "observation_window": 84,
                     "frame_stack_size": 4,
                 },
                 })  # "eager": True for eager execution
