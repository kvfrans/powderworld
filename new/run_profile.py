import sys
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
import io
import base64
import time
import imageio
from torch.profiler import profile, record_function, ProfilerActivity


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
from powderworld import *
from powderworld.env import PWEnv, PWEnvSingle, PWTaskConfig, PWGenConfig, PWGenRule
# BENCHMARK


def run_benchmark(PWClass, bs):
    with torch.no_grad():
        torch.cuda.empty_cache()

        pw = PWClass(device, use_jit=False)
        world = torch.zeros((bs, pw.NUM_CHANNEL, 64, 64), dtype=torch.float16).to(device)
        pw.add_element(world[:, :, :, :], "empty")
        for i in range(10):
            world = pw(world, do_skips=False)
        
        torch.cuda.synchronize()
        start = time.time()
        
        for i in range(100):
            world = pw(world, do_skips=False)

        torch.cuda.synchronize()
        total_time = time.time() - start
        
#         for rule in pw.update_rules_jit:
#             rand_movement = torch.rand((world.shape[0], 1, world.shape[2], world.shape[3]), dtype=pw_type, device=pw.device) # For gravity, flowing.
#             rand_interact = torch.rand((world.shape[0], 1, world.shape[2], world.shape[3]), dtype=pw_type, device=pw.device) # For element-wise int.
#             rand_element = torch.rand((world.shape[0], 1, world.shape[2], world.shape[3]), dtype=pw_type, device=pw.device)
#             info = (rand_movement, rand_interact, rand_element)
#             torch.cuda.synchronize()
#             start = time.time()
        
#             for i in range(100):
#                 world = rule(world, info)

#             torch.cuda.synchronize()
#             total_time = time.time() - start
#             print("{}, Time: {}".format(rule.__class__.__name__, total_time))
        
    print("Time:", total_time)
    
    
def run_benchmark_agent(PWClass, bs):
    with torch.no_grad():
        torch.cuda.empty_cache()

        config = PWTaskConfig()
        new_rules = []
        new_rules.append(PWGenConfig(PWGenRule.SINE_WAVE, "stone", 1, 40, 50))
        config['state_gen']['rules'] = new_rules


        envs = PWEnvSingle(config, use_jit=False, device="cuda", num_envs=bs)
        envs.is_vector_env = True
        envs.reset()
        
        for i in tqdm(range(10)):
            action = np.array([envs.action_space.sample() for _ in range(envs.num_envs)])
            observation, reward, done, info = envs.step(action)
        
        torch.cuda.synchronize()
        start = time.time()

        for i in tqdm(range(100)):
            action = np.array([envs.action_space.sample() for _ in range(envs.num_envs)])
            observation, reward, done, info = envs.step(action)
        
        torch.cuda.synchronize()
        total_time = time.time() - start
        envs.close()

        
    print("Time:", total_time)

# print("Normal benchmark")
# run_benchmark(PWSim, 32)

# print("Normal benchmark")
# run_benchmark(PWSim, 128)

# print("Normal benchmark")
# run_benchmark(PWSim, 256)

# print("Agent benchmark")
# run_benchmark_agent(PWSim, 32)

print("Agent benchmark")
run_benchmark_agent(PWSim, 128)

# print("Agent benchmark")
# run_benchmark_agent(PWSim, 256)

# run_benchmark(PWSim, 256)


