from collections import deque

import gym
import numpy as np
from gym.spaces import Discrete, Box


class SimpleEnv(gym.Env):

    def __init__(self, config):
        self.k = 4
        self.frames = deque([], maxlen=self.k)
        self.action_space = Discrete(3)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, self.k),
                                     dtype=np.uint8)

    def reset(self):
        ob = np.ones([84, 84], dtype=np.uint8)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def _get_ob(self):
        assert len(self.frames) == self.k
        obs = np.ones([84, 84, 4], dtype=np.uint8)
        for i in range(4):
            obs[:, :, i] = self.frames[i]
        return obs

    def step(self, action):
        assert action in [0, 1, 2], action
        if action == 0:
            reward = 0
        elif action == 1:
            reward = 0.5
        elif action == 2:
            reward = 1
        else:
            reward = -1
        done = True if reward == 2 else False
        return self._get_ob(), reward, done, {}
