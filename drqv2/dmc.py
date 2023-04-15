# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
sys.path.append("/home/siri/Desktop/Parking/gym-carla")
import gym_carla

from collections import deque

import numpy as np
import gym

params = {
    'number_of_vehicles': 0,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town04',  # which town to simulate
    'task_mode': 'roundabout',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 500,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 1,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
}

class ExtendedTimeStep:
    def __init__(self, observation, reward, done, info, action, discount) -> None:
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info
        self.action = action
        self.discount = discount

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)

class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, num_repeats):
        super().__init__(env)
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        for i in range(self._num_repeats):
            observation, reward_, done, info = self.env.step(action)
            reward += reward_
            if done:
                break

        return ExtendedTimeStep(observation, reward, done, info, action)

    def observation_spec(self):
        return self.env.observation_space

    def action_spec(self):
        return self.env.action_space

class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_frames):
        super().__init__(env)
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        obs_shape = self.env.observation_space # (32,32,3)
        obs_shape = (obs_shape[2] * num_frames, obs_shape[0], obs_shape[1])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def _transform_observation(self, observation):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return obs

    def _extract_pixels(self, observation):
        return observation.transpose(2, 0, 1).copy()

    def reset(self):
        observation = self.env.reset()
        observation = self._extract_pixels(observation)
        for _ in range(self._num_frames):
            self._frames.append(observation)
        return self._transform_observation(observation)

    def step(self, action):
        observation, reward, done, info, discount = self.env.step(action)
        observation = self._extract_pixels(observation)
        self._frames.append(observation)
        return self._transform_observation(observation), reward, done, info, discount

    def observation_spec(self):
        return self.observation_space

    def action_spec(self):
        return self.env.action_space

class ActionDTypeWrapper(gym.Wrapper):
    def __init__(self, env, dtype):
        super().__init__(env)
        self._dtype = dtype

    def step(self, action):
        action = action.astype(self._dtype)
        return self.env.step(action)

    def observation_spec(self):
        return self.env.observation_space

    def action_spec(self):
        return gym.spaces.Box(low=self.env.action_space.low.astype(self._dtype), 
                              high=self.env.action_space.high.astype(self._dtype),
                              dtype=self._dtype)

    def reset(self):
        return self.env.reset()

class ExtendedTimeStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        observation = self.env.reset()
        action = np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)
        return ExtendedTimeStep(observation, 0, False, {}, action, 1)

    def step(self, action):
        observation, reward, done, info, discount = self.env.step(action)
        return ExtendedTimeStep(observation, reward, done, info, action, discount)
    
    def observation_spec(self):
        return self.env.observation_space

    def action_spec(self):
        return self.env.action_space

def make(frame_stack):

    env = gym.make('carla-v0', params=params)
    if frame_stack > 1:
        env = FrameStackWrapper(env, frame_stack)
    # env = ActionDTypeWrapper(env, np.float32)
    env = ExtendedTimeStepWrapper(env)
    return env