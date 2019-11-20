"""
    Impala
"""

import ray
from ray import tune
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

from lib.env.TraderRenkoEnv_v2_lite import StockTradingEnv
from lib.model.vision_network import VisionNetwork

tf = try_import_tf()

# register_env("StockTradingEnv", lambda _: StockTradingEnv(10))
ModelCatalog.register_custom_model("NatureCNN", VisionNetwork)

ray.init(memory=20000000000,
         object_store_memory=10000000000,
         redis_max_memory=5000000000,
         driver_object_store_memory=2000000000)

tune.run(ImpalaTrainer,
         max_failures=10,
         checkpoint_freq=10,  # iterations
         checkpoint_at_end=True,
         config={"env": StockTradingEnv,
                 "model": {
                     "custom_model": "NatureCNN"
                 },
                 "sample_batch_size": 50,
                 "train_batch_size": 500,
                 "num_gpus": 1,
                 "num_workers": 4,
                 "num_envs_per_worker": 1,
                 "lr_schedule": [
                     [0, 0.0005],
                     [20000000, 0.000000000001],
                 ],
                 "env_config": {
                     "enable_env_logging": False,
                     "look_back_window_size": 375 * 10,
                     "observation_window": 84,
                     "frame_stack_size": 4,
                 },
                 })  # "eager": True for eager execution
