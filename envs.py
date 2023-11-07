from collections import OrderedDict, deque
from typing import Any, NamedTuple
import os

import numpy as np

import gym
import pickle

class DummyEnv:
  def observation_spec(self):
    return {'observation': gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)}

  def action_spec(self):
    return {'action': gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)}

  def step(self, action):
    return {'observation': np.zeros((1,)), 'reward': 0.0, 'is_last': False, 'is_terminal': False}

  def reset(self):
    return {'observation': np.zeros((1,)), 'reward': 0.0, 'is_last': False, 'is_terminal': False}


class DMC:
  def __init__(self, env):
    self._env = env
    self._ignored_keys = []

  @property
  def obs_space(self):
    spaces = {
        'observation': self._env.observation_spec(),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }
    return spaces

  @property
  def act_space(self):
    spec = self._env.action_spec()
    action = gym.spaces.Box((spec.minimum)*spec.shape[0], (spec.maximum)*spec.shape[0], shape=spec.shape, dtype=np.float32)
    return {'action': action}

  def step(self, action):
    time_step = self._env.step(action)
    assert time_step.discount in (0, 1)
    obs = {
        'reward': time_step.reward,
        'is_first': False,
        'is_last': time_step.last(),
        'is_terminal': time_step.discount == 0,
        'observation': time_step.observation,
        'action' : action,
        'discount': time_step.discount
    }
    return obs

  def reset(self):
    time_step = self._env.reset()
    obs = {
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
        'observation': time_step.observation,
        'action' : np.zeros_like(self.act_space['action'].sample()),
        'discount': time_step.discount
    }
    return obs

  def __getattr__(self, name):
    if name == 'obs_space':
        return self.obs_space
    if name == 'act_space':
        return self.act_space
    return getattr(self._env, name)






def make(name, obs_type, frame_stack, action_repeat, seed, img_size=84, exorl=False):
    return DMC(DummyEnv())
