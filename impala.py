"""
    Impala
"""

import ray
from ray import tune
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

from lib.env.StockTraderEnv import StockTraderEnv
from lib.model.vision_network import VisionNetwork

tf = try_import_tf()

# register_env("StockTradingEnv", lambda _: StockTradingEnv(10))
ModelCatalog.register_custom_model("NatureCNN", VisionNetwork)

tune.run(ImpalaTrainer,
         max_failures=10,
         checkpoint_freq=10,  # iterations
         checkpoint_at_end=True,
         config={"env": StockTraderEnv,
                 "model": {
                     "custom_model": "NatureCNN"
                 },
                 "sample_batch_size": 50,
                 "train_batch_size": 500,
                 "num_gpus": 1,
                 "num_workers": 30,
                 "num_envs_per_worker": 5,
                 "lr_schedule": [
                     [0, 0.0005],
                     [20000000, 0.000000000001],
                 ],
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
                 })  # "eager": True for eager execution
