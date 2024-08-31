import numpy as np
import torch
from PIL import Image, ImageDraw
from enum import IntEnum
from collections import namedtuple
import gymnasium
from gymnasium import spaces

from sim import PWSim, PWRenderer, interp

PWGenConfigTuple = namedtuple('PWGenConfig', ['method', 'elem', 'num', 'y', 'x'])
def PWGenConfig(method, elem, num, y=0, x=0):
    return PWGenConfigTuple(method, elem, num, y, x)


class PWEnv(gymnasium.Env):
    def __init__(self, device=None, use_jit=False, obs_type='elems', flatten_actions=True):
        
        self.device = device
        self.pw = PWSim(self.device, use_jit)
        self.pwr = PWRenderer(self.device)
        self.render_mode = 'rgb_array_list'
        self.render_frames = []
        self.obs_type = obs_type
        self.elems = ['sand', 'water', 'wood', 'fire', 'plant', 'stone']
        self.world_size = 32
        self.lim = lambda x : np.clip(int(x), 1, self.world_size-2)

        # Default observation_space
        if self.obs_type == 'elems':
            self.observation_space = spaces.Box(low=0., high=21, shape=(5, self.world_size, self.world_size), dtype=np.uint8)
        elif self.obs_type == "rgb":
            self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.world_size, self.world_size), dtype=np.float32)

        if flatten_actions:
            # Flat Action Space: (Element * X * Y)
            self.action_space = spaces.Discrete(len(self.elems) * self.world_size//8 * self.world_size//8)
        else:
            # Action space: (Element(20), X(Size/8), Y(Size/8), X-delta(8), Y-delta(8), WindDir(8))
            # self.action_space = spaces.MultiDiscrete(np.array([len(self.elems), self.world_size//8, self.world_size//8, 8, 8]))
            raise NotImplementedError()
        
    def reset(self):
        np_world = np.zeros((1, self.world_size, self.world_size), dtype=np.uint8)
        np_world[:, 0, :] = 1
        np_world[:, -1, :] = 1
        np_world[:, :, 0] = 1
        np_world[:, :, -1] = 1
        self.world = self.pw.np_to_pw(np_world).clone()
        self.cpu_world = self.world.cpu().numpy()
        self.t = 0
        return self.get_obs(), None
    
    def render(self):
        im = self.pwr.render(self.world)
        im = Image.fromarray(im)
        im = im.resize((256, 256), Image.NEAREST)
        im = np.array(im)
        return im
                    
    def step_async(self, actions):
        self.actions = actions
        
    def get_obs(self):
        return self.cpu_world[0,0] # Integer of elements.
    
    # Take a step in the RL env.
    def step(self, action):
        # Step world.
        self.world = self.pw(self.world)

        # Apply actions.
        np_action_world = np.zeros((1, self.world_size, self.world_size), dtype=np.uint8)
        y, x, element_id = np.unravel_index(action, (8, 8, len(self.elems)))
        elem = [2, 3, 5, 7, 8, 9][element_id] # Convert from action_id to pw elem id.
        brush_size = self.world_size//8
        real_x = x*brush_size
        real_y = y*brush_size
        np_action_world[:, real_y:real_y+brush_size, real_x:real_x+brush_size] = elem
        world_delta = self.pw.np_to_pw(np_action_world)
        self.world = interp(~self.pw.get_bool(world_delta, 'empty') & ~self.pw.get_bool(self.world, 'wall'), self.world, world_delta)

        self.t += 1
        self.cpu_world = self.world.cpu().numpy()
        obs = self.get_obs()
        return obs, 0, False, self.t >= 1000, None