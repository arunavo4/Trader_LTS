"""
    Dist DQN
"""

from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

from lib.env.USStockEnv import USStockEnv
from lib.model.vision_network import VisionNetwork

tf = try_import_tf()

ModelCatalog.register_custom_model("NatureCNN", VisionNetwork)

tune.run(DQNTrainer,
         checkpoint_freq=10,  # iterations
         checkpoint_at_end=True,
         config={"env": USStockEnv,
                 "model": {
                     "custom_model": "NatureCNN"
                 },
                 "double_q": False,
                 "dueling": False,
                 "num_atoms": 51,
                 "noisy": False,
                 "n_step": 1,
                 "prioritized_replay": False,
                 "lr": .0000625,
                 "adam_epsilon": .00015,
                 "hiddens": [512],
                 "buffer_size": 1000000,
                 "schedule_max_timesteps": 3000000,
                 "exploration_final_eps": 0.01,
                 "exploration_fraction": .1,
                 "prioritized_replay_alpha": 0.5,
                 "beta_annealing_fraction": 1.0,
                 "final_prioritized_replay_beta": 1.0,
                 "num_gpus": 1,
                 "sample_batch_size": 4,
                 "train_batch_size": 32,
                 "target_network_update_freq": 8000,
                 "timesteps_per_iteration": 10000,
                 "env_config": {
                     "initial_balance": 10000,
                     "enable_env_logging": False,
                     "look_back_window_size": 10,
                     "observation_window": 84,
                     "frame_stack_size": 4,
                     "use_leverage": False,
                     "hold_reward": False,
                 },
                 })  # "eager": True for eager execution
