import gym
from gym import spaces
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from powderworld import PowderWorld
from PIL import Image
import powder_dists


from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn


def lim(x):
    return np.clip(1, 63, int(x))

class PowderWorldSandEnv(VecEnv):
    def __init__(self, test=False, elems=['stone'], num_tasks=1000000, num_lines=5, num_circles=0, num_squares=0, has_empty_path=False):
        self.device = 'cuda'
        self.render_mode = 'rgb_array'
        self.pw = PowderWorld(self.device)
        self.BATCH_SIZE = 32
        self.test = test
        self.elems = elems
        self.num_tasks = num_tasks
        self.num_lines = num_lines
        self.num_circles = num_circles
        self.num_squares = num_squares
        self.has_empty_path = has_empty_path

        # Observation Space = [64x64 world state matrix.]
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(self.pw.NUM_CHANNEL, 64, 64), dtype=np.float32)
        # Action Space = A stroke of [ElementType, X1, Y1, X2, Y2]
        # self.action_space = spaces.MultiDiscrete(np.array([14, 64, 64, 64, 64]))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))
        super().__init__(self.BATCH_SIZE, self.observation_space, self.action_space)
        
        self.reset()
        
    def reset(self):
        self.world = torch.zeros((self.BATCH_SIZE, self.pw.NUM_CHANNEL, 64, 64), dtype=torch.float32, device=self.device)

        # Randomly placed elements
        for b in range(self.BATCH_SIZE):
            if self.test:
                powder_dists.make_rl_test(self.pw, self.world[b:b+1], b % 8)
            else:
                powder_dists.make_rl_world(self.pw, self.world[b:b+1], self.elems, self.num_tasks, self.num_lines, self.num_circles, self.num_squares, self.has_empty_path)
        
        self.timestep = 0
        self.dummy_reward = np.zeros((self.BATCH_SIZE,))
        return self.world.cpu().numpy()
                    
    def step_async(self, actions):
        self.actions = actions
        
    def step_wait(self):
        for b in range(self.BATCH_SIZE):
            x, y, xdir, ydir = self.actions[b][0], self.actions[b][1], self.actions[b][2], self.actions[b][3]
            x = 32+int(x*32)
            y = 32+int(y*32)
            radius = 5
            wind = 5 * torch.Tensor([xdir, ydir]).to(self.device)[None,:,None,None]
            self.pw.add_element(self.world[b:b+1, :, lim(x-radius):lim(x+radius), lim(y-radius):lim(y+radius)], 'wind', wind)
            self.dummy_reward[b] += xdir
        self.world = self.pw(self.world)
        self.timestep += 1
        ob = self.world.cpu().numpy()
        rew = self.get_rew()
        done = self.timestep >= 64
        dones = np.array([done] * self.BATCH_SIZE)
        if done:
            ob = self.reset()
        info = [{}] * self.BATCH_SIZE
        # print(ob.shape, rew.shape, dones, info)
        return ob, rew, dones, info
    
    def get_rew(self):
        rew = torch.sum(self.world[:, 2:3, 32-10:32+10, -10:], dim=(1,2,3)).cpu().numpy() / 150
        return rew
    
    def get_images(self, num_img=None):
        if num_img is None:
            num_img = self.BATCH_SIZE
        imgs = []
        for b in range(self.BATCH_SIZE):
            imgs.append(self.pw.render(self.world[b:b+1]).astype("uint8"))
        return imgs

    def render(self, mode="rgb_array"):
        im = self.get_images(1)[0]
        im = Image.fromarray(im)
        im = im.resize((256, 256), Image.NEAREST)
        im = np.array(im)
        return im
    
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
            indices = range(self.BATCH_SIZE)
        return [env_util.is_wrapped(self, wrapper_class) for i in indices]
    
    
class PowderWorldPlaceEnv(VecEnv):
    def __init__(self, test=False, elems=['stone'], num_tasks=1000000, num_lines=5, num_circles=0, num_squares=0):
        self.device = 'cuda'
        self.render_mode = 'rgb_array'
        self.pw = PowderWorld(self.device)
        self.BATCH_SIZE = 32
        self.test = test
        self.elems = elems
        self.num_tasks = num_tasks
        self.num_lines = num_lines
        self.num_circles = num_circles
        self.num_squares = num_squares

        # Observation Space = [64x64 world state matrix.]
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(self.pw.NUM_CHANNEL*2, 64, 64), dtype=np.float32)
        # Action Space = A stroke of [ElementType, X1, Y1, X2, Y2]
#         self.action_space = spaces.MultiDiscrete(np.array([5, 36]))
        self.action_space = spaces.Discrete(36)
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))
        super().__init__(self.BATCH_SIZE, self.observation_space, self.action_space)
        
        self.log_start = torch.zeros((self.BATCH_SIZE, self.pw.NUM_CHANNEL, 64, 64), device=self.device)
        self.log_truth_start = torch.zeros((self.BATCH_SIZE, self.pw.NUM_CHANNEL, 64, 64), device=self.device)
        self.log_truth_after = torch.zeros((self.BATCH_SIZE, self.pw.NUM_CHANNEL, 64, 64), device=self.device)
        self.log_agent_start = torch.zeros((self.BATCH_SIZE, self.pw.NUM_CHANNEL, 64, 64), device=self.device)
        self.log_agent_after = torch.zeros((self.BATCH_SIZE, self.pw.NUM_CHANNEL, 64, 64), device=self.device)
        
        self.reset()
        
    def reset(self):
        self.world = torch.zeros((self.BATCH_SIZE, self.pw.NUM_CHANNEL, 64, 64), dtype=torch.float32, device=self.device)

        # Randomly placed elements
        for b in range(self.BATCH_SIZE):
            if self.test:
                powder_dists.make_test(self.pw, self.world[b:b+1], b % 8)
            else:
                powder_dists.make_world(self.pw, self.world[b:b+1], self.elems, self.num_tasks, self.num_lines, self.num_circles, self.num_squares)
        
        self.log_start_old = self.world.clone()
        self.world_truth = self.world.clone()
#         print("Making truth elems")
        self.truth_elems = np.random.randint(1,6, size=(self.BATCH_SIZE,))
        for b in range(self.BATCH_SIZE):
            radius = 4
            elem = int(self.truth_elems[b])
            x1 = np.random.randint(1,8) * 8 + 4
            y1 = np.random.randint(1,8) * 8 + 4
            self.pw.add_element(self.world_truth[b:b+1, :, lim(x1-radius):lim(x1+radius), lim(y1-radius):lim(y1+radius)], elem)
        self.log_truth_start_old = self.world_truth.clone()
        for t in range(32):
            self.world_truth = self.pw(self.world_truth)
        self.log_truth_after_old = self.world_truth.clone()
        
        self.timestep = 0
        return torch.cat([self.world, self.world_truth], dim=1).cpu().numpy()
                    
    def step_async(self, actions):
        self.actions = actions
        
    def step_wait(self):
        self.log_start = self.world.clone()
        self.log_truth = self.world_truth.clone()
        
#         print(self.actions[0])
#         print("Using truth elems", self.truth_elems)
        for b in range(self.BATCH_SIZE):
#             elem, xy = self.actions[b][0], self.actions[b][1]
            xy = self.actions[b]
            elem = int(self.truth_elems[b])

            x = (int(xy // 6)+1) * 8 + 4
            y = (int(xy % 6)+1) * 8 + 4
            elem = int(elem)
            radius = 4
            self.pw.add_element(self.world[b:b+1, :, lim(x-radius):lim(x+radius), lim(y-radius):lim(y+radius)], elem)
        self.log_agent_start = self.world.clone()
        for t in range(32):
            self.world = self.pw(self.world)
        self.log_agent_after = self.world.clone()
        self.log_start = self.log_start_old
        self.log_truth_start = self.log_truth_start_old
        self.log_truth_after = self.log_truth_after_old


        self.timestep += 1
        ob = torch.cat([self.world, self.world_truth], dim=1).cpu().numpy()
        rew = self.get_rew()
        done = True
        dones = np.array([done] * self.BATCH_SIZE)
        if done:
            ob = self.reset()
        info = [{}] * self.BATCH_SIZE
        # print(ob.shape, rew.shape, dones, info)
        return ob, rew, dones, info
    
    def get_rew(self):
        world_blur = F.max_pool2d(self.world, kernel_size=3, stride=1)
        world_truth_blur = F.max_pool2d(self.world_truth, kernel_size=3, stride=1)
        rew = -F.mse_loss(world_blur[:,:14], world_truth_blur[:,:14], reduction='none')
        rew = torch.sum(rew, dim=(1,2,3)).cpu().numpy() / 500
        return rew
    
    def get_images(self, num_img=None):
        if num_img is None:
            num_img = self.BATCH_SIZE
        imgs = []
        for b in range(num_img):
            imgs.append(np.concatenate([
                self.pw.render(self.log_start[b:b+1]).astype("uint8"),
                self.pw.render(self.log_truth_start[b:b+1]).astype("uint8"),
                self.pw.render(self.log_truth_after[b:b+1]).astype("uint8"),
                self.pw.render(self.log_agent_start[b:b+1]).astype("uint8"),
                self.pw.render(self.log_agent_after[b:b+1]).astype("uint8"),
            ], axis=0))
        return imgs

    def render(self, mode="rgb_array"):
        im = self.get_images(4)
        im_big = np.concatenate(im, axis=1)
        print(im_big.shape)
        im = Image.fromarray(im_big)
        im = im.resize((256*4, 256*5), Image.NEAREST)
        im = np.array(im)
        print(im.shape)
        return im
    
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
            indices = range(self.BATCH_SIZE)
        return [env_util.is_wrapped(self, wrapper_class) for i in indices]