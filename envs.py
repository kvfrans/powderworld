import gym
from gym import spaces
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from powderworld import PowderWorld, PowderWorldRenderer
from PIL import Image
import powder_dists


from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn


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
    def __init__(self, test=False, kwargs_pcg=None, total_timesteps=64, batch_size=32, device=None):
        if kwargs_pcg is None:
            kwargs_pcg = kwargs_pcg_default

        self.test = test
        self.kwargs_pcg = kwargs_pcg
        self.total_timesteps = total_timesteps
        self.batch_size = batch_size
        self.device = device

        self.pw = PowderWorld(self.device)
        self.pwr = PowderWorldRenderer(self.device)

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
    def __init__(self, test=False, kwargs_pcg=None, total_timesteps=20, n_world_settle=0, batch_size=32, device=None, dense_reward=True):
        super().__init__(test=test, kwargs_pcg=kwargs_pcg, total_timesteps=total_timesteps, batch_size=batch_size, device=device)
        self.observation_space = spaces.Box(low=-5., high=5., shape=(2*self.pw.NUM_CHANNEL, *self.hw), dtype=np.float32)

        self.n_world_settle = n_world_settle
        
        self.dense_reward = dense_reward
        
        self.world_start, self.world_end = None, None
        self.world_agent_start, self.world_agent_end = None, None
        self.reset()

    def reset(self):
        self.timestep = 0
        self.world_start = torch.zeros((self.batch_size, self.pw.NUM_CHANNEL, 64, 64), dtype=torch.float32, device=self.device)
        self.world_agent_start = torch.zeros((self.batch_size, self.pw.NUM_CHANNEL, 64, 64), dtype=torch.float32, device=self.device)
        powder_dists.make_empty_world(self.pw, self.world_start)
        powder_dists.make_empty_world(self.pw, self.world_agent_start)

        # Randomly placed elements
        for b in range(self.batch_size):
            if self.test:
                powder_dists.make_test(self.pw, self.world_start[b:b+1], b % 8)
            else:
                args = ['elems', 'num_tasks', 'num_lines', 'num_circles', 'num_squares']
                powder_dists.make_world(self.pw, self.world_start[b:b+1], **{k: self.kwargs_pcg[k] for k in args})
        
        self.world_end = self.rollout_world(self.world_start)
            
        ob = torch.cat([self.world_agent_start, self.world_end], dim=1).cpu().numpy()
        return ob
    
    def rollout_world(self, world):
        world_end = world.clone()
        for _ in range(self.n_world_settle):
            world_end = self.pw(world_end)
        return world_end
                    
    def step_wait(self):
        # temporary force draw sand
        self.apply_action(self.world_agent_start, self.actions, force_elem='sand')
        
        rew, done = self.get_rew(), (self.timestep == self.total_timesteps-1)
        
        ob = torch.cat([self.world_agent_start, self.world_end], dim=1).cpu().numpy()
        dones = np.array([done] * self.batch_size)
        if done:
            ob = self.reset()
        info = [{}] * self.batch_size
        self.timestep += 1
        return ob, rew, dones, info
    
    def get_rew(self):
        if not self.dense_reward and self.timestep < self.total_timesteps-1: # sparse reward and not the end
            return 0
        
        # rollout env
        self.world_agent_end = self.rollout_world(self.world_agent_start)
        
        world_blur = F.max_pool2d(self.world_agent_end, kernel_size=3, stride=1)
        world_truth_blur = F.max_pool2d(self.world_end, kernel_size=3, stride=1)
        rew = -F.mse_loss(world_blur[:,:14], world_truth_blur[:,:14], reduction='none')
        rew = torch.sum(rew, dim=(1,2,3)).cpu().numpy() / 500
        return rew

    def render(self, env_id=0, mode='rgb_array'):
        assert mode=='rgb_array'
        a = self.pwr.render(self.world_agent_start[[env_id]])
        b = self.pwr.render(self.world_end[[env_id]])
        return np.concatenate([a, b], axis=1)

class PWSandEnv(PWGeneralEnv):
    def __init__(self, test=False, kwargs_pcg=None, total_timesteps=64, batch_size=32, device=None):
        super().__init__(test=test, kwargs_pcg=kwargs_pcg, total_timesteps=total_timesteps, batch_size=batch_size, device=device)

    def reset(self):
        self.world = torch.zeros((self.batch_size, self.pw.NUM_CHANNEL, 64, 64), dtype=torch.float32, device=self.device)

        # Randomly placed elements
        for b in range(self.batch_size):
            if self.test:
                powder_dists.make_rl_test(self.pw, self.world[b:b+1], b % 8)
            else:
                args = ['elems', 'num_tasks', 'num_lines', 'num_circles', 'num_squares', 'has_empty_path']
                powder_dists.make_rl_world(self.pw, self.world[b:b+1], **{k: self.kwargs_pcg[k] for k in args})
        
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
        return self.pwr.render(self.world[[env_id]])

class PWDestroyEnv(PWDrawEnv):
    def __init__(self, test=False, kwargs_pcg=None, total_timesteps=5, n_world_settle=0, batch_size=32, device=None):
        super().__init__(test=test, kwargs_pcg=kwargs_pcg, total_timesteps=total_timesteps, batch_size=batch_size, device=device)
    
    def reset(self):
        self.timestep = 0
        self.world_start = torch.zeros((self.batch_size, self.pw.NUM_CHANNEL, 64, 64), dtype=torch.float32, device=self.device)
        self.world_agent_start = torch.zeros((self.batch_size, self.pw.NUM_CHANNEL, 64, 64), dtype=torch.float32, device=self.device)
        powder_dists.make_empty_world(self.pw, self.world_start)
        powder_dists.make_empty_world(self.pw, self.world_agent_start)
        
        # Randomly placed elements
        for b in range(self.batch_size):
            if self.test:
                powder_dists.make_test(self.pw, self.world_agent_start[b:b+1], b % 8)
            else:
                args = ['elems', 'num_tasks', 'num_lines', 'num_circles', 'num_squares']
                powder_dists.make_world(self.pw, self.world_agent_start[b:b+1], **{k: self.kwargs_pcg[k] for k in args})
        
        self.truth_elems = np.random.randint(1,6, size=(self.batch_size,))
        for b in range(self.batch_size):
            radius = 4
            elem = int(self.truth_elems[b])
            x1 = np.random.randint(1,8) * 8 + 4
            y1 = np.random.randint(1,8) * 8 + 4
            self.pw.add_element(self.world_agent_start[b:b+1, :, lim(x1-radius):lim(x1+radius), lim(y1-radius):lim(y1+radius)], elem)
            
        self.world_end = self.world_start.clone()
        for _ in range(self.n_world_settle):
            self.world_end = self.pw(self.world_end)
            
        ob = torch.cat([self.world_agent_start, self.world_end], dim=1).cpu().numpy()
        return ob




