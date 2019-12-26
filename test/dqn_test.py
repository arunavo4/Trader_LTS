"""
    Rainbow DQN

    Note: For Policy Gradient Algorithms We need to subclass TFModelV2
    and for QModel Algorithms subclass DistributionalQModel.
"""
import collections
import os
import pickle
from statistics import mean

import gym
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import _flatten_action
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils import try_import_tf
from ray.tune.util import merge_dicts

from lib.env.USStockEnv import USStockEnv
from lib.env.IndianStockEnv import IndianStockEnv
from lib.model.vision_network import VisionNetwork

tf = try_import_tf()

# ================== Register Custom Model ======================
ModelCatalog.register_custom_model("NatureCNN", VisionNetwork)


# ================== Rollout Functions ==========================
def run(args):
    config = {}
    # Load configuration from file
    config_dir = os.path.dirname(args['checkpoint_dir'])
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        if not args['config']:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory.")
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])
    config = merge_dicts(config, args['config'])
    if not args['env']:
        if not config.get("env"):
            print("the following arguments are required: --env")
        args['env'] = config.get("env")

    ray.init()

    cls = get_agent_class(args['run'])
    agent = cls(env=args['env'], config=config)
    agent.restore(args['checkpoint_path'])
    num_steps = int(args['steps'])
    rollout(agent, args['env'], num_steps, args['out'], args['no_render'])


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def rollout(agent, env_name, num_steps, out=None, no_render=True):
    policy_agent_mapping = default_policy_agent_mapping

    if hasattr(agent, "workers"):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: _flatten_action(m.action_space.sample())
            for p, m in policy_map.items()
        }
    else:
        env = gym.make(env_name)
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    if out is not None:
        rollouts = []
    steps = 0
    while steps < (num_steps or steps + 1):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        if out is not None:
            rollout = []
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    a_action = _flatten_action(a_action)  # tuple actions
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, _ = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            if not no_render:
                env.render()
            if out is not None:
                rollout.append([obs, action, next_obs, reward, done])
            steps += 1
            obs = next_obs
        if out is not None:
            rollouts.append(rollout)
        print("\n============ Stats ================")
        print("Episode reward:", reward_total)
        print("Final Amount:", env.net_worth[0])
        print("Avg Daily Profit:", round(mean(env.daily_profit_per), 3))
        print("Days in Episode:", len(env.daily_profit_per))
        print("Wins:", env.wins, "Losses:", env.losses)
        print("====================================\n")

    if out is not None:
        pickle.dump(rollouts, open(out, "wb"))


if __name__ == "__main__":
    args = {
        'checkpoint_dir': '/home/skywalker/ray_results/IMPALA/IMPALA_USStockEnv_0_2019-12-24_04-51-08ce8794fz'
                          '/checkpoint_590',
        'checkpoint_path': '/home/skywalker/ray_results/IMPALA/IMPALA_USStockEnv_0_2019-12-24_04-51-08ce8794fz'
                           '/checkpoint_590/checkpoint-590',
        'config': {"env_config": {
            "initial_balance": 10000,
            "enable_env_logging": True,
            "look_back_window_size": 390 * 10,
            "observation_window": 84,
            "frame_stack_size": 4,
            "use_leverage": False,
            "hold_reward": False,
        }},
        'env': USStockEnv,
        'run': "IMPALA",
        'steps': 10000,
        'out': None,
        'no_render': True
    }

    run(args)

