"""
    Dist DQN
"""

from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.dqn.distributional_q_model import DistributionalQModel
from ray.rllib.models.tf.visionnet_v1 import _get_filter_config
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.utils import try_import_tf

from lib.env.TraderRenkoEnv import StockTradingEnv

tf = try_import_tf()


# ============== Nature CNN Model ==================

class VisionNetwork(DistributionalQModel):
    """Generic vision network implemented in DistributionalQModel API."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **kw):
        super(VisionNetwork, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)

        activation = get_activation_fn(model_config.get("conv_activation"))
        filters = model_config.get("conv_filters")
        if not filters:
            filters = _get_filter_config(obs_space.shape)
        no_final_linear = model_config.get("no_final_linear")
        vf_share_layers = model_config.get("vf_share_layers")

        inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")
        last_layer = inputs

        # Build the action layers
        for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding="same",
                name="conv{}".format(i))(last_layer)
        out_size, kernel, stride = filters[-1]
        if no_final_linear:
            # the last layer is adjusted to be of size num_outputs
            last_layer = tf.keras.layers.Conv2D(
                num_outputs,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding="valid",
                name="conv_out")(last_layer)
            conv_out = last_layer
        else:
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding="valid",
                name="conv{}".format(i + 1))(last_layer)
            conv_out = tf.keras.layers.Conv2D(
                num_outputs, [1, 1],
                activation=None,
                padding="same",
                name="conv_out")(last_layer)

        # Build the value layers
        if vf_share_layers:
            last_layer = tf.keras.layers.Lambda(
                lambda x: tf.squeeze(x, axis=[1, 2]))(last_layer)
            value_out = tf.keras.layers.Dense(
                1,
                name="value_out",
                activation=None,
                kernel_initializer=normc_initializer(0.01))(last_layer)
        else:
            # build a parallel set of hidden layers for the value net
            last_layer = inputs
            for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
                last_layer = tf.keras.layers.Conv2D(
                    out_size,
                    kernel,
                    strides=(stride, stride),
                    activation=activation,
                    padding="same",
                    name="conv_value_{}".format(i))(last_layer)
            out_size, kernel, stride = filters[-1]
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding="valid",
                name="conv_value_{}".format(i + 1))(last_layer)
            last_layer = tf.keras.layers.Conv2D(
                1, [1, 1],
                activation=None,
                padding="same",
                name="conv_value_out")(last_layer)
            value_out = tf.keras.layers.Lambda(
                lambda x: tf.squeeze(x, axis=[1, 2]))(last_layer)

        self.base_model = tf.keras.Model(inputs, [conv_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        model_out, self._value_out = self.base_model(
            tf.cast(input_dict["obs"], tf.float32))
        return tf.squeeze(model_out, axis=[1, 2]), state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


ModelCatalog.register_custom_model("NatureCNN", VisionNetwork)

tune.run(DQNTrainer,
         checkpoint_freq=10,  # iterations
         checkpoint_at_end=True,
         config={"env": StockTradingEnv,
                 "model": {
                     "custom_model": "NatureCNN"
                 },
                 "double_q": False,
                 "dueling": False,
                 "num_atoms": 51,
                 "noisy": False,
                 "n_step": 1,
                 "lr": .0000625,
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
                 "sample_batch_size": 4,
                 "train_batch_size": 32,
                 "target_network_update_freq": 8000,
                 "timesteps_per_iteration": 10000,
                 "env_config": {
                     "enable_env_logging": False,
                     "look_back_window_size": 375 * 10,
                     "observation_window": 84,
                     "frame_stack_size": 4,
                 },
                 })  # "eager": True for eager execution
