import gym
from gym import spaces
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from enum import IntEnum
from collections import namedtuple
import copy

from powderworld.base_vec_env import VecEnv
import powderworld.gen
import powderworld.rews
from powderworld.sim import PWSim, PWRenderer, interp


class PWRewRule(IntEnum):
    ELEM_COUNT = 1
    ELEM_DESTROY = 2
    DELTA = 3
    
# x = [0, 1, 2, 3]
# y = [0, 1, 2, 3]
PWRewConfigTuple = namedtuple('PWRewConfig', ['method', 'elem', 'weight', 'x', 'y', 'p1'])
def PWRewConfig(method, elem, weight, x, y, p1=0):
    return PWRewConfigTuple(method, elem, weight, x, y, p1)

class PWGenRule(IntEnum):
    CIRCLE = 1
    CLONER_CIRCLE = 2
    BOXES = 3
    FILL_SLICE = 4
    CLONER_ROOF = 5
    SINE_WAVE = 6
    RAMPS = 7
    ARCHES = 8
    CONTAINER = 9
    
PWGenConfigTuple = namedtuple('PWGenConfig', ['method', 'elem', 'num', 'p1', 'p2', 'p3'])
def PWGenConfig(method, elem, num, p1=0, p2=0, p3=0):
    return PWGenConfigTuple(method, elem, num, p1, p2, p3)

def PWTaskConfig():
    return {
        # General configs should be shared across all PWTasks in a PWTaskDist.
        'general': {
            'world_size': 64,
            'obs_type': 'elems',
            'num_task_variations': 10000000
        },
        'agent': {
            'starting_timesteps': 0,
            'ending_timesteps': 0,
            'num_actions': 50,
            'time_per_action': 1,
            'agent_type': 'disembodied', # [disembodied, embodied]
            'disabled_elements': []
        },
        'state_gen': {
            'rules': [
                # PWGenConfig(PWGenRule.CIRCLE, "sand", num=1)
            ],
            'seed': 0,
        },
        'reward': {
            'rules': [
                # PWRewConfig(PWRewRule.ELEM_COUNT, "sand", weight=1, x=3, y=3)
            ],
            'dense_reward_ratio': 0.1,
        }
    }


class PWTask():
    def __init__(self, pw, config):
        self.config = config
        self.pw = pw
        
    def reset(self):
        self.num_actions_taken = 0
        rand = np.random.RandomState(self.config['state_gen']['seed'])
        new_world = powderworld.gen.init_world(self.config['general']['world_size'], self.config['general']['world_size'])
        for rule in self.config['state_gen']['rules']:
            for _ in range(rule.num):
                if rule.method == PWGenRule.CIRCLE:
                    powderworld.gen.do_circle(new_world, rand, rule.elem)
                elif rule.method == PWGenRule.CLONER_CIRCLE:
                    powderworld.gen.do_cloner_circle(new_world, rand, rule.elem)
                elif rule.method == PWGenRule.BOXES:
                    powderworld.gen.do_boxes(new_world, rand, rule.elem, rule.p1)
                elif rule.method == PWGenRule.FILL_SLICE:
                    powderworld.gen.do_fill_slice(new_world, rand, rule.elem, rule.p1)
                elif rule.method == PWGenRule.CLONER_ROOF:
                    powderworld.gen.do_cloner_roof(new_world, rand, rule.elem)
                elif rule.method == PWGenRule.SINE_WAVE:
                    powderworld.gen.do_sine_wave(new_world, rand, rule.elem, [rule.p1, rule.p2])
                elif rule.method == PWGenRule.RAMPS:
                    powderworld.gen.do_ramps(new_world, rand, rule.elem)
                elif rule.method == PWGenRule.ARCHES:
                    powderworld.gen.do_arches(new_world, rand, rule.elem)
                elif rule.method == PWGenRule.CONTAINER:
                    powderworld.gen.do_container(new_world, rand, rule.elem, rule.p1, rule.p2)
        powderworld.gen.do_edges(new_world, rand)
        return new_world
        
    def get_rew(self, world, world_before):
        rew = 0
        for rule in self.config['reward']['rules']:
            if rule.method == PWRewRule.ELEM_COUNT:
                rew += powderworld.rews.rew_elem_exists(world, rule.elem, rule.y, rule.x) * rule.weight
            elif rule.method == PWRewRule.ELEM_DESTROY:
                rew += powderworld.rews.rew_elem_destroy(world, rule.elem, rule.y, rule.x) * rule.weight
            elif rule.method == PWRewRule.DELTA:
                rew += powderworld.rews.rew_delta(world, world_before, rule.y, rule.x) * rule.weight
        return rew


class PWEnv(VecEnv):
    def __init__(self, num_envs=32, device=None, use_jit=True, flatten_actions=False):
        
        self.num_envs = num_envs
        self.device = device
        self.pw = PWSim(self.device, use_jit)
        self.pwr = PWRenderer(self.device)
        self.flatten_actions = flatten_actions
        self.render_mode = 'rgb_array_list'
        self.set_attr('is_recording', False)
        self.render_frames = []
                
        self.tasks = []
        for n in range(self.num_envs):
            self.tasks.append(PWTask(self.pw, self.generate_task_config()))
            
        self.world_size = self.tasks[0].config['general']['world_size']
        self.lim = lambda x : np.clip(int(x), 1, self.world_size-2)

        # Default observation_space
        if self.tasks[0].config["general"]["obs_type"] == "elems":
            self.observation_space = spaces.Box(low=0., high=21, shape=(self.world_size, self.world_size), dtype=np.uint8)
            
        elif self.tasks[0].config["general"]["obs_type"] == "channels":
            self.observation_space = spaces.Box(low=-5., high=5., shape=(self.pw.NUM_CHANNEL, \
                                                                         self.world_size, self.world_size), dtype=np.float32)
        elif self.tasks[0].config["general"]["obs_type"] == "rgb":
            self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.world_size, self.world_size), dtype=np.float32)

        if flatten_actions:
            # Flat Action Space: (Element * X * Y)
            self.action_space = spaces.Discrete(self.pw.NUM_ELEMENTS * 8 * 8)
        else:
            # Action space: (Element(20), X(Size/8), Y(Size/8), X-delta(8), Y-delta(8), WindDir(8))
            self.action_space = spaces.MultiDiscrete(np.array([self.pw.NUM_ELEMENTS, self.world_size//8, self.world_size//8, 8, 8, 8]))
        super().__init__(self.num_envs, self.observation_space, self.action_space)
        
    def generate_task_config(self):
        raise NotImplementedError("PWEnv requires a task generator.")
        
    def reset(self):
        np_world = np.zeros((self.num_envs, self.world_size, self.world_size), dtype=np.uint8)
                
        self.tasks = []
        for n in range(self.num_envs):
            self.tasks.append(PWTask(self.pw, self.generate_task_config()))
        for task_iter in range(len(self.tasks)):
            np_world[task_iter:task_iter+1] = self.tasks[task_iter].reset()
            
        self.world = self.pw.np_to_pw(np_world).clone()
        self.cpu_world = self.world.cpu().numpy()
        self.cpu_world_before = np.copy(self.cpu_world)
        
        return self.cpu_world[:,0]
        
    def make_img(self, world_slice):
        im = self.pwr.render(world_slice)
        im = Image.fromarray(im)
        im = im.resize((256, 256), Image.NEAREST)
        im = np.array(im)
        return im
    
    def render(self, mode='rgb_array_list', task_id=0):
        return self.render_frames
    
#     def render(self, mode='rgb_array_list', task_id=0):
#         im = self.pwr.render(self.world[task_id:task_id+1])
#         im = Image.fromarray(im)
#         im = im.resize((256, 256), Image.NEAREST)
#         im = np.array(im)
#         return im
    
    
                    
    def step_async(self, actions):
        self.actions = actions
        
    def apply_action_batch(self):
        actions = self.actions
        np_world = np.zeros((self.num_envs, self.world_size, self.world_size), dtype=np.uint8)
        np_bool = np.zeros((self.num_envs, 1, self.world_size, self.world_size), dtype=np.bool)
        np_wind = np.zeros((self.num_envs, 2, self.world_size, self.world_size), dtype=np.float16)
        
        if len(actions.shape) == 1:            
            y, x, element_id = np.unravel_index(actions, (8, 8, 21))
            elem = element_id
            real_x = x*8
            real_y = y*8
        else:
            # Full MultiDiscrete
            elem = actions[:, 0]
            real_x = actions[:, 1]*8 + actions[:, 3]
            real_y = actions[:, 2]*8 + actions[:, 4]
            
        for i, t in enumerate(self.tasks):
            if elem[i] in t.config["agent"]["disabled_elements"]:
                elem[i] = 0
                        
        # Create a meshgrid of offsets
        radius = 3
        y_offset, x_offset = np.mgrid[-radius+1:radius, -radius+1:radius]

        # Add the offsets to real_x and real_y to get the coordinates
        x_coords = real_x[:, None, None] + x_offset
        y_coords = real_y[:, None, None] + y_offset

        # Clip the coordinates to the world size
        x_coords = np.clip(x_coords, 1, self.world_size - 2)
        y_coords = np.clip(y_coords, 1, self.world_size - 2)

        # Use advanced indexing to set the values in np_world
        np_world[np.arange(self.num_envs)[:, None, None], y_coords, x_coords] = elem[:, None, None]
        np_bool[np.arange(self.num_envs)[:, None, None], :, y_coords, x_coords] = True
        
        world_bool = torch.from_numpy(np_bool).to(self.device)
        world_delta = self.pw.np_to_pw(np_world)
        self.world = interp(world_bool, self.world, world_delta)
        

    # Take a step in the RL env.
    def step_wait(self):
        self.apply_action_batch()
        
        is_recording = self.get_attr('is_recording')
        self.render_frames = []

        self.cpu_world_before = np.copy(self.cpu_world)
        action_times = np.array([task.config["agent"]["time_per_action"] for task in self.tasks])
        for t in range(np.max(action_times)):
            does_tick = action_times > 0
            action_times[does_tick] -= 1
            if does_tick.all():
                self.world = self.pw(self.world)
            else:
                self.world[does_tick] = self.pw(self.world[does_tick])
            if is_recording and does_tick[0]:
                self.render_frames.append(self.make_img(self.world[0:1]))

        for task in self.tasks:
            task.num_actions_taken += 1
        dones = np.array([task.num_actions_taken >= task.config["agent"]["num_actions"] for task in self.tasks], dtype=np.bool)
        
        # For any tasks that are finished, run forwards ending_timesteps times.
        if dones.any():
            done_world = self.world[dones]
            ending_timesteps = np.array([task.config["agent"]["ending_timesteps"] for task in self.tasks])[dones]
            for t in range(np.max(ending_timesteps)):
                does_tick = ending_timesteps > 0 
                ending_timesteps[does_tick] -= 1
                if does_tick.all():
                    done_world = self.pw(done_world)
                else:
                    done_world[does_tick] = self.pw(done_world[does_tick])
                if is_recording and dones[0]:
                    self.render_frames.append(self.make_img(done_world[0:1]))
            self.world = done_world
        
        self.cpu_world = self.world.cpu().numpy()
        
        rews = np.array([self.tasks[ta].get_rew(self.cpu_world[ta:ta+1], self.cpu_world_before[ta:ta+1]) for ta in range(len(self.tasks))])
        ratios = np.array([task.config["reward"]["dense_reward_ratio"] if ~dones[ta] else 1 for ta, task in enumerate(self.tasks)])
        rews = rews * ratios
        
        if dones.any():
            new_np_world = np.zeros((dones.sum(), self.world_size, self.world_size), dtype=np.uint8)
            for ta in range(len(self.tasks)):
                if dones[ta]:
                    self.tasks[ta] = PWTask(self.pw, self.generate_task_config())
                    new_np_world[ta:ta+1] = self.tasks[ta].reset()
            self.world[dones] = self.pw.np_to_pw(new_np_world)
                
        obs = self.cpu_world[:,0]
        info = [{}] * self.num_envs
        return obs, rews, dones, info
    
    def get_attr(self, attr_name, indices=None):
        return getattr(self, attr_name)

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

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

class PWEnvSingle(PWEnv):
    def __init__(self, config, num_envs=32, device=None, use_jit=True, flatten_actions=False):
        self.config = config
        super().__init__(num_envs, device, use_jit, flatten_actions)
        
    def generate_task_config(self):
        config = copy.deepcopy(self.config)
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        return config