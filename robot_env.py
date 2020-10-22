"""
Wraps the sawyer environment so that it can be used from RAD
"""
from typing import Tuple, Dict
import numpy as np
import gym
import cv2
from viceraq.environment import make_sawyer

class RobotEnv(gym.Env):
    metadata = {'render.modes':[]}

    def __init__(self, img_size: int, frame_skip: int):
        super(RobotEnv, self).__init__()
        self._env = make_sawyer(frame_skip)
        shape = (3, img_size, img_size)
        space = gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
        self.observation_space = space
        self.action_space = self._env.action_space
        self._max_episode_steps = self._env._env.env._max_episode_steps
        self._sz = img_size

    def _convert_img(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, (self._sz, self._sz))
        img = img.transpose(2, 0, 1).copy()
        return img

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        reward = 0
        for _ in range(self._frame_skip):
            img, _, done, info = self._env.step(action)
            reward += float(info['angle_success'] == 1.0) #not sure if neccessary
            if done:
                break
        next_state = self._convert_img(img)
        return next_state, reward, done, info

    def reset(self) -> np.ndarray:
        self._env = make_sawyer()
        img = self._env.reset()
        img = self._convert_img(img)
        return img

    def render(self, mode='human'):
        return self._env.render(mode)

    def close(self):
        return self._env.close()
