import gym
from gym import spaces
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from enum import IntEnum

import powderworld.dists
from powderworld.sim import PWSim, PWRenderer

class PWRewRule(IntEnum):
    ELEM_COUNT = 1
    
# x = [0, 1, 2]
# y = [0, 1, 2]
#
def PWRewConfig(method, elem, weight, x, y, p1=0):
    return [method, elem, weight, x, y, p1]

class PWGenRule(IntEnum):
    CIRCLE = 1
    CLONER_CIRCLE = 2
    BOXES = 3
    FILL_SLICE = 4
    CLONER_ROOF = 5
    SINE_WAVE = 6
    RAMPS = 7
    ARCHES = 8
    

def PWGenConfig(method, elem, num, p1=0, p2=0, p3=0):
    return [method, elem, num, p1, p2, p3]

def PWTaskConfig():
    return {
        'general': {
            'world_size': 64,
            'obs_type': 'channels'
        },
        'agent': {
            'starting_timesteps': 0,
            'ending_timesteps': 0,
            'num_actions': 50,
            'time_per_action': 1,
            'agent_type': 'disembodied' # [disembodied, embodied]
        }
        'state_gen': {
            'rules': [
                PWGenConfig(PWGenRule.CIRCLE, "sand", num=1)
            ]
        },
        'reward': {
            'rules': [
                PWRewConfig(PWRewRule.ELEM_COUNT, "sand", weight=1, x=2, y=2)
            ],
            'dense_reward_ratio': 0.1,
        }
    }
    

class PWEnvConfig:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


config_default = {
    # GENERAL OPTIONS
    'world_size': 64,
    'obs_type': 'channels', # ['channels', 'rgb']
    # ACTION OPTIONS
    'starting_timesteps': 0,
    'ending_timesteps': 0,
    'num_actions': 50,
    'time_per_action': 1,
    # STATE OPTIONS
    # REWARD OPTIONS
    'dense_reward_ratio': 0.1,
}

class PWEnv(VecEnv):
    def __init__(self, config=None, num_envs=32, device=None, use_jit=True):

        if config is None:
            config = config_default
        self.config = PWEnvConfig(config)
        print(config)

        self.world_size = config['world_size']
        self.num_envs = num_envs
        self.device = device

        self.pw = PWSim(self.device, use_jit)
        self.pwr = PWRenderer(self.device)

        # Default observation_space
        if self.config.obs_type == "channels":
            self.observation_space = spaces.Box(low=-5., high=5., shape=(self.pw.NUM_CHANNEL, self.config.world_size, self.config.world_size), dtype=np.float32)
        elif self.config.obs_type == "rgb":
            self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.config.world_size, self.config.world_size), dtype=np.float32)

        # Action space: (Element(20), X(Size/8), Y(Size/8), X-delta(8), Y-delta(8), WindDir(8))
        self.action_space = spaces.MultiDiscrete(np.array([self.pw.NUM_ELEMENTS, self.config.world_size//8, self.config.world_size//8, 8, 8, 8]))
        super().__init__(self.num_envs, self.observation_space, self.action_space)
        
    def reset(self):
        self.world = torch.zeros((self.num_envs, self.pw.NUM_CHANNEL, 64, 64), dtype=torch.float32, device=self.device)

        self.pw.add_element(self.world[:, :, :, :], "wall")
        self.pw.add_element(self.world[:, :, 1:-1, 1:-1], "empty")

        self.pw.add_element(self.world[:, :, 20:30, 20:30], "water")


        self.num_actions_taken = 0
        return self.world.cpu().numpy()
        
    def lim(self, x):
        return np.clip(1, self.world_size-1, int(x))

    def render(self, mode='rgb_array'):
        assert mode=='rgb_array'
        im = self.pwr.render(self.world[0:1])
        im = Image.fromarray(im)
        im = im.resize((256, 256), Image.NEAREST)
        im = np.array(im)
        return im
                    
    def step_async(self, actions):
        self.actions = actions

    def apply_action(self, world, actions):
        actions = actions.astype(np.int)
        for b in range(len(world)):
            elem, x, y, x_delta, y_delta, wind_dir = actions[b]
            real_x = x*8 + x_delta
            real_y = y*8 + y_delta
            radius = 3
            # TODO: Make winddir actually correct
            winddir = 20 * torch.Tensor([0, 0]).to(self.device)[None,:,None,None]
            self.pw.add_element(world[b:b+1, :, self.lim(real_x-radius):self.lim(real_x+radius), \
                                      self.lim(real_y-radius):self.lim(real_y+radius)], int(elem), winddir)
            
    # Take a step in the RL env.
    def step_wait(self):
        # Add elements to the world
        self.apply_action(self.world, self.actions)
        
        # Simulate world forwards
        for t in self.config.time_per_action:
            self.world = self.pw(self.world)        
        self.num_actions_taken += 1
        
        done = self.num_actions_taken >= self.config.num_actions
        
        # If agent is done taking actions, run env "time_per_action" timesteps for final reward.
        if done: 
            for t in self.config.time_per_action:
                self.world = self.pw(self.world)
        rew = self.get_rew() * (1 if done else self.config.dense_reward_ratio)
        dones = np.array([done] * self.num_envs)
        
        # Get observation
        if done:
            ob = self.reset()
        else:
            ob = self.world.cpu().numpy()
        info = [{}] * self.num_envs
        return ob, rew, dones, info
    
    def get_rew(self):
        return 0
    
    def get_attr(self, attr_name, indices=None):
        return None

    def set_attr(self, attr_name, value, indices=None):
        return None

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return None

    def _get_target_envs(self, indices):
        return None
    
    def seed(self, seed=None):
        return None

    def close(self):
        return None
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        from stable_baselines3.common import env_util
        if indices is None:
            indices = range(self.batch_size)
        return [env_util.is_wrapped(self, wrapper_class) for i in indices]

# class PWDrawEnv(PWGeneralEnv):
#     def __init__(self, test=False, kwargs_pcg=None, total_timesteps=64, batch_size=32, device=None, dense_reward=True, use_jit=True):
#         super().__init__(test=test, kwargs_pcg=kwargs_pcg, total_timesteps=total_timesteps, batch_size=batch_size, device=device, use_jit=use_jit)
#         self.action_space = spaces.MultiDiscrete(np.array([16, 16, 2, 3, 3]))
#         self.test_idx = 0
#         self.reset()

#     def reset(self):
#         self.world = torch.zeros((self.batch_size, self.pw.NUM_CHANNEL, 64, 64), dtype=torch.float32, device=self.device)

#         # Randomly placed elements
#         for b in range(self.batch_size):
#             if self.test:
#                 self.test_idx += 1
#                 powderworld.dists.make_test160(self.pw, self.world[b:b+1], self.test_idx % 160)    
#             else:
#                 args = ['elems', 'num_tasks', 'num_lines', 'num_circles', 'num_squares']
#                 powderworld.dists.make_water_world(self.pw, self.world[b:b+1], **{k: self.kwargs_pcg[k] for k in args})
#                 watery = 10
#                 waterx = 20
#                 # Water
#                 self.pw.add_element(self.world[b:b+1, :, watery-5:watery+5, waterx-15:waterx+15], "water")
#                 self.pw.add_element(self.world[b:b+1, :, watery-4:watery+4, waterx-14:waterx+14], "cloner")
#                 # Border
#                 self.pw.add_element(self.world[b:b+1, :, 40:64, -20:-18], "wall")
#                 self.pw.add_element(self.world[b:b+1, :, :, -18:], "empty")
        
#         self.timestep = 0
#         return self.world.cpu().numpy()
                    
#     def step_wait(self):
#         self.apply_action(self.world, self.actions)
        
#         self.pw.add_element(self.world[:, :, 40:64, -20:-18], "wall")
        
#         for t in range(3):
#             self.world = self.pw(self.world)
#         self.timestep += 1
#         ob = self.world.cpu().numpy()
#         rew = self.get_rew()
#         done = self.timestep >= self.total_timesteps
#         dones = np.array([done] * self.batch_size)
#         if done:
#             ob = self.reset()
#         info = [{}] * self.batch_size
#         return ob, rew, dones, info
    
#     def get_rew(self):
#         rew = torch.sum(self.world[:, 3:4, 40:64, -20:], dim=(1,2,3)).cpu().numpy() / 150
#         return rew

#     def render(self, env_id=0, mode='rgb_array'):
#         assert mode=='rgb_array'
#         im = self.pwr.render(self.world[[env_id]])
#         im = Image.fromarray(im)
#         im = im.resize((256, 256), Image.NEAREST)
#         im = np.array(im)
#         return im

# class PWSandEnv(PWGeneralEnv):
#     def __init__(self, test=False, kwargs_pcg=None, total_timesteps=64, batch_size=32, device=None, use_jit=True):
#         super().__init__(test=test, kwargs_pcg=kwargs_pcg, total_timesteps=total_timesteps, batch_size=batch_size, device=device, use_jit=use_jit)

#     def reset(self):
#         self.world = torch.zeros((self.batch_size, self.pw.NUM_CHANNEL, 64, 64), dtype=torch.float32, device=self.device)

#         # Randomly placed elements
#         for b in range(self.batch_size):
#             if self.test:
#                 powderworld.dists.make_rl_test(self.pw, self.world[b:b+1], b % 8)
#             else:
#                 args = ['elems', 'num_tasks', 'num_lines', 'num_circles', 'num_squares']
#                 powderworld.dists.make_rl_world(self.pw, self.world[b:b+1], **{k: self.kwargs_pcg[k] for k in args})
        
#         self.timestep = 0
#         return self.world.cpu().numpy()
                    
#     def step_wait(self):
#         self.apply_action(self.world, self.actions, force_elem='wind')
        
#         self.world = self.pw(self.world)
#         self.timestep += 1
#         ob = self.world.cpu().numpy()
#         rew = self.get_rew()
#         done = self.timestep >= self.total_timesteps
#         dones = np.array([done] * self.batch_size)
#         if done:
#             ob = self.reset()
#         info = [{}] * self.batch_size
#         return ob, rew, dones, info
    
#     def get_rew(self):
#         # measure the amount of sand in the goal area
#         rew = torch.sum(self.world[:, 2:3, 32-10:32+10, -10:], dim=(1,2,3)).cpu().numpy() / 150
#         return rew
    
#     def render(self, env_id=0, mode='rgb_array'):
#         assert mode=='rgb_array'
#         im = self.pwr.render(self.world[[env_id]])
#         im = Image.fromarray(im)
#         im = im.resize((256, 256), Image.NEAREST)
#         im = np.array(im)
#         return im

# class PWDestroyEnv(PWGeneralEnv):
#     def __init__(self, test=False, kwargs_pcg=None, total_timesteps=5, n_world_settle=64, batch_size=32, device=None, use_jit=True):
#         super().__init__(test=test, kwargs_pcg=kwargs_pcg, total_timesteps=total_timesteps, batch_size=batch_size, device=device, use_jit=use_jit)
#         self.n_world_settle = n_world_settle
#         self.test_idx = 0
    
#     def reset(self):
#         self.world = torch.zeros((self.batch_size, self.pw.NUM_CHANNEL, 64, 64), dtype=torch.float32, device=self.device)

#         # Randomly placed elements
#         for b in range(self.batch_size):
#             if self.test:
#                 self.test_idx += 1
#                 powderworld.dists.make_test160(self.pw, self.world[b:b+1], self.test_idx % 160)
#             else:
#                 args = ['elems', 'num_tasks', 'num_lines', 'num_circles', 'num_squares']
#                 powderworld.dists.make_world(self.pw, self.world[b:b+1], **{k: self.kwargs_pcg[k] for k in args})
        
#         self.timestep = 0
#         return self.world.cpu().numpy()
                    
#     def step_wait(self):
#         self.apply_action(self.world, self.actions)
        
#         self.world = self.pw(self.world)
#         self.timestep += 1
#         ob = self.world.cpu().numpy()
#         rew = self.get_rew()
#         done = self.timestep >= self.total_timesteps
#         dones = np.array([done] * self.batch_size)
#         if done:
#             ob = self.reset()
#         info = [{}] * self.batch_size
#         return ob, rew, dones, info
    
#     def get_rew(self):
#         # measure the amount of sand in the goal area
#         rew = np.zeros(self.world.shape[0])
#         if self.timestep >= self.total_timesteps:
#             for t in range(self.n_world_settle):
#                 self.world = self.pw(self.world)
#             rew = torch.sum(self.world[:, 0:1, :, :], dim=(1,2,3)).cpu().numpy() / 1000
#         return rew
    
#     def render(self, env_id=0, mode='rgb_array'):
#         assert mode=='rgb_array'
#         im = self.pwr.render(self.world[[env_id]])
#         im = Image.fromarray(im)
#         im = im.resize((256, 256), Image.NEAREST)
#         im = np.array(im)
#         return im
