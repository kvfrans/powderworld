import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import stable_baselines3
from tqdm import tqdm
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, DQN
import torch.nn as nn
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from powderworld import PWSim
from powderworld.envs import PWSandEnv, PWDestroyEnv, PWDrawEnv

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, features_dim=256, obs_feature_dim=20):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        sequential_layers = [
            nn.Conv2d(obs_feature_dim, 32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        ]
        
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(*sequential_layers)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        
    def forward(self, observations):
        return self.linear(self.cnn(observations))


def train(run_name, env_name, kwargs_pcg, args):
    print("env name is", env_name)
    env_fn = None
    if env_name == 'sand':
        env_fn = PWSandEnv
    elif env_name == 'draw':
        env_fn = PWDrawEnv
    elif env_name == 'destroy':
        env_fn = PWDestroyEnv

    env = env_fn(test=False, kwargs_pcg=kwargs_pcg, device='cuda')
    env = VecMonitor(env)
    env = VecVideoRecorder(env, "videos", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
    
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256, obs_feature_dim=env.observation_space.shape[0]),
    )

    model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs, n_steps=args.ppo_steps, batch_size=args.ppo_batchsize, verbose=1)
    model.learn(total_timesteps=1_000_000)


if __name__ == "__main__":
    print("Loading args")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir')
    parser.add_argument('--num_elems', type=int, default=6)
    parser.add_argument('--num_lines', type=int, default=5)
    parser.add_argument('--num_circles', type=int, default=5)
    parser.add_argument('--num_squares', type=int, default=5)
    parser.add_argument('--num_tasks', type=int, default=100000)
    parser.add_argument('--env_name')
    
    parser.add_argument('--ppo_steps', type=int, default=64)
    parser.add_argument('--ppo_batchsize', type=int, default=512)
    
    args = parser.parse_args()
    print(args)
    
    if args.env_name is None:
        raise Exception("Provide an environment to train on !")
    
    all_elems = ['empty', 'sand', 'water', 'wall', 'plant', 'fire', 'wood', 'ice', 'lava', 'dust', 'cloner', 'gas', 'acid', 'stone']
    elems = all_elems[:args.num_elems]
    kwargs_pcg = dict(hw=(64,64), elems=elems, num_tasks=args.num_tasks, num_lines=args.num_lines,
                      num_circles=args.num_circles, num_squares=args.num_squares)
    train(args.savedir, args.env_name, kwargs_pcg, args)
