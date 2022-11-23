import gym
from gym import spaces
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn

import powderworld.dists
from powderworld.sim import PWSim, PWRenderer


def lim(x):
    return np.clip(1, 63, int(x))

"""
Different environments:
General environment

Drawing environment
 - start state: zeros
Wind Envrionemnt
 - starting state: pcg
Destroy Environment
 - starting state: pcg

"""

kwargs_pcg_default = dict(hw=(64, 64), elems=['stone'], num_tasks=1000000, num_lines=5, num_circles=0, num_squares=0, has_empty_path=False)

class PWGeneralEnv(VecEnv):
    def __init__(self, test=False, kwargs_pcg=None, total_timesteps=64, batch_size=32, device=None, use_jit=True):
        if kwargs_pcg is None:
            kwargs_pcg = kwargs_pcg_default

        self.test = test
        self.kwargs_pcg = kwargs_pcg
        self.total_timesteps = total_timesteps
        self.batch_size = batch_size
        self.device = device

        self.pw = PWSim(self.device, use_jit)
        self.pwr = PWRenderer(self.device)

        self.hw = self.kwargs_pcg['hw']
        self.action_resolution = 4
        self.hw_action = (a//self.action_resolution for a in self.hw)
        self.directions_all = np.array([(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)])

        # Default observation_space, action_space
        self.observation_space = spaces.Box(low=-5., high=5., shape=(self.pw.NUM_CHANNEL, *self.hw), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(np.array([*self.hw_action, self.pw.NUM_ELEMENTS, 3, 3]))
        super().__init__(self.batch_size, self.observation_space, self.action_space)
        
    def reset(self):
        raise NotImplementedError()

    def render(self, env_id=0, mode='rgb_array'):
        raise NotImplementedError()
                    
    def step_async(self, actions):
        self.actions = actions

    def parse_action(self, action):
        if action.ndim==1:
            action = np.tile(action, (self.batch_size, 1))
        xbin, ybin, elem, dxbin, dybin = [action[..., i] for i in range(5)]
        x = xbin*self.action_resolution+self.action_resolution//2
        y = ybin*self.action_resolution+self.action_resolution//2
        dirx = (dxbin-1)
        diry = (dybin-1)
        # direction = self.directions_all[direction]
        # int cast is required because of np stuff
        return [(x[i], y[i], int(elem[i]), dirx[i], diry[i]) for i in range(self.batch_size)]
    
    def apply_action(self, world, actions, force_elem=None):
        actions = self.parse_action(actions)
        for b in range(len(world)):
            x, y, elem, xdir, ydir = actions[b]
            if force_elem is not None:
                elem = force_elem
            radius = 5
            winddir = 20 * torch.Tensor([xdir, ydir]).to(self.device)[None,:,None,None]
            self.pw.add_element(world[b:b+1, :, lim(x-radius):lim(x+radius), lim(y-radius):lim(y+radius)], elem, winddir)
    
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

class PWDrawEnv(PWGeneralEnv):
    def __init__(self, test=False, kwargs_pcg=None, total_timesteps=64, batch_size=32, device=None, dense_reward=True, use_jit=True):
        super().__init__(test=test, kwargs_pcg=kwargs_pcg, total_timesteps=total_timesteps, batch_size=batch_size, device=device, use_jit=use_jit)
        self.action_space = spaces.MultiDiscrete(np.array([16, 16, 2, 3, 3]))
        self.test_idx = 0
        self.reset()

    def reset(self):
        self.world = torch.zeros((self.batch_size, self.pw.NUM_CHANNEL, 64, 64), dtype=torch.float32, device=self.device)

        # Randomly placed elements
        for b in range(self.batch_size):
            if self.test:
                self.test_idx += 1
                powderworld.dists.make_test160(self.pw, self.world[b:b+1], self.test_idx % 160)    
            else:
                args = ['elems', 'num_tasks', 'num_lines', 'num_circles', 'num_squares']
                powderworld.dists.make_water_world(self.pw, self.world[b:b+1], **{k: self.kwargs_pcg[k] for k in args})
                watery = 10
                waterx = 20
                # Water
                self.pw.add_element(self.world[b:b+1, :, watery-5:watery+5, waterx-15:waterx+15], "water")
                self.pw.add_element(self.world[b:b+1, :, watery-4:watery+4, waterx-14:waterx+14], "cloner")
                # Border
                self.pw.add_element(self.world[b:b+1, :, 40:64, -20:-18], "wall")
                self.pw.add_element(self.world[b:b+1, :, :, -18:], "empty")
        
        self.timestep = 0
        return self.world.cpu().numpy()
                    
    def step_wait(self):
        self.apply_action(self.world, self.actions)
        
        self.pw.add_element(self.world[:, :, 40:64, -20:-18], "wall")
        
        for t in range(3):
            self.world = self.pw(self.world)
        self.timestep += 1
        ob = self.world.cpu().numpy()
        rew = self.get_rew()
        done = self.timestep >= self.total_timesteps
        dones = np.array([done] * self.batch_size)
        if done:
            ob = self.reset()
        info = [{}] * self.batch_size
        return ob, rew, dones, info
    
    def get_rew(self):
        rew = torch.sum(self.world[:, 3:4, 40:64, -20:], dim=(1,2,3)).cpu().numpy() / 150
        return rew

    def render(self, env_id=0, mode='rgb_array'):
        assert mode=='rgb_array'
        im = self.pwr.render(self.world[[env_id]])
        im = Image.fromarray(im)
        im = im.resize((256, 256), Image.NEAREST)
        im = np.array(im)
        return im

class PWSandEnv(PWGeneralEnv):
    def __init__(self, test=False, kwargs_pcg=None, total_timesteps=64, batch_size=32, device=None, use_jit=True):
        super().__init__(test=test, kwargs_pcg=kwargs_pcg, total_timesteps=total_timesteps, batch_size=batch_size, device=device, use_jit=use_jit)

    def reset(self):
        self.world = torch.zeros((self.batch_size, self.pw.NUM_CHANNEL, 64, 64), dtype=torch.float32, device=self.device)

        # Randomly placed elements
        for b in range(self.batch_size):
            if self.test:
                powderworld.dists.make_rl_test(self.pw, self.world[b:b+1], b % 8)
            else:
                args = ['elems', 'num_tasks', 'num_lines', 'num_circles', 'num_squares']
                powderworld.dists.make_rl_world(self.pw, self.world[b:b+1], **{k: self.kwargs_pcg[k] for k in args})
        
        self.timestep = 0
        return self.world.cpu().numpy()
                    
    def step_wait(self):
        self.apply_action(self.world, self.actions, force_elem='wind')
        
        self.world = self.pw(self.world)
        self.timestep += 1
        ob = self.world.cpu().numpy()
        rew = self.get_rew()
        done = self.timestep >= self.total_timesteps
        dones = np.array([done] * self.batch_size)
        if done:
            ob = self.reset()
        info = [{}] * self.batch_size
        return ob, rew, dones, info
    
    def get_rew(self):
        # measure the amount of sand in the goal area
        rew = torch.sum(self.world[:, 2:3, 32-10:32+10, -10:], dim=(1,2,3)).cpu().numpy() / 150
        return rew
    
    def render(self, env_id=0, mode='rgb_array'):
        assert mode=='rgb_array'
        im = self.pwr.render(self.world[[env_id]])
        im = Image.fromarray(im)
        im = im.resize((256, 256), Image.NEAREST)
        im = np.array(im)
        return im

class PWDestroyEnv(PWGeneralEnv):
    def __init__(self, test=False, kwargs_pcg=None, total_timesteps=5, n_world_settle=64, batch_size=32, device=None, use_jit=True):
        super().__init__(test=test, kwargs_pcg=kwargs_pcg, total_timesteps=total_timesteps, batch_size=batch_size, device=device, use_jit=use_jit)
        self.n_world_settle = n_world_settle
        self.test_idx = 0
    
    def reset(self):
        self.world = torch.zeros((self.batch_size, self.pw.NUM_CHANNEL, 64, 64), dtype=torch.float32, device=self.device)

        # Randomly placed elements
        for b in range(self.batch_size):
            if self.test:
                self.test_idx += 1
                powderworld.dists.make_test160(self.pw, self.world[b:b+1], self.test_idx % 160)
            else:
                args = ['elems', 'num_tasks', 'num_lines', 'num_circles', 'num_squares']
                powderworld.dists.make_world(self.pw, self.world[b:b+1], **{k: self.kwargs_pcg[k] for k in args})
        
        self.timestep = 0
        return self.world.cpu().numpy()
                    
    def step_wait(self):
        self.apply_action(self.world, self.actions)
        
        self.world = self.pw(self.world)
        self.timestep += 1
        ob = self.world.cpu().numpy()
        rew = self.get_rew()
        done = self.timestep >= self.total_timesteps
        dones = np.array([done] * self.batch_size)
        if done:
            ob = self.reset()
        info = [{}] * self.batch_size
        return ob, rew, dones, info
    
    def get_rew(self):
        # measure the amount of sand in the goal area
        rew = np.zeros(self.world.shape[0])
        if self.timestep >= self.total_timesteps:
            for t in range(self.n_world_settle):
                self.world = self.pw(self.world)
            rew = torch.sum(self.world[:, 0:1, :, :], dim=(1,2,3)).cpu().numpy() / 1000
        return rew
    
    def render(self, env_id=0, mode='rgb_array'):
        assert mode=='rgb_array'
        im = self.pwr.render(self.world[[env_id]])
        im = Image.fromarray(im)
        im = im.resize((256, 256), Image.NEAREST)
        im = np.array(im)
        return im
