from distutils.util import strtobool

from powderworld import PowderWorld, PowderWorldRenderer
import powder_dists
import torch
import numpy as np
from IPython import display as ipythondisplay
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from gym import spaces

import powder_env
import envs

from stable_baselines3 import PPO, DQN
import torch.nn as nn

from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import wandb
from wandb.integration.sb3 import WandbCallback

import argparse

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, features_dim=256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            obs = torch.as_tensor(observation_space.sample()).float()
            n_flatten = self.cnn(obs[None]).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations):
        return self.linear(self.cnn(observations))

def train_agent(args):
    if args.track:
        run = wandb.init(
            name=args.exp_name,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True)
    
    all_elems = ['empty', 'sand', 'water', 'wall', 'plant', 'stone', 'lava']
    kwargs_pcg = dict(hw=(64,64), elems=all_elems[:args.n_elems], num_tasks=100000, 
                      num_lines=args.n_lines, num_circles=args.n_circles, num_squares=args.n_squares, has_empty_path=False)
    
    if args.env=='sand':
        env = envs.PWSandEnv(False, kwargs_pcg, device=args.device)
    elif args.env=='draw':
        env = envs.PWDrawEnv(False, kwargs_pcg, device=args.device)
    elif args.env=='destroy':
        env = envs.PWDestroyEnv(False, kwargs_pcg, device=args.device)
        
    env = VecMonitor(env)
    if args.track:
        env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=500)
    
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=20),
    )
    
    model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs, n_steps=64, batch_size=64, verbose=1, tensorboard_log=f"runs/{run.id}")
    callback = WandbCallback(gradient_save_freq=10000, model_save_path=f"models/{run.id}", verbose=2)
    model.learn(total_timesteps=args.n_steps, callback=callback)
    
    if args.track:
        run.finish()
    
parser = argparse.ArgumentParser()
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
parser.add_argument("--exp-name", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default='cuda:0')

parser.add_argument("--env", type=str, default='sand')
parser.add_argument("--n_elems", type=int, default=4)
parser.add_argument("--n_lines", type=int, default=0)
parser.add_argument("--n_circles", type=int, default=0)
parser.add_argument("--n_squares", type=int, default=0)

parser.add_argument("--n_steps", type=int, default=500_000)

def main():
    args = parser.parse_args()
    print(args)
    train_agent(args)

if __name__=='__main__':
    main()


