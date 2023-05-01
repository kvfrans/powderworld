import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import time
import types
import torch.jit as jit
import threading
from collections import namedtuple
from typing import Dict,Tuple,Optional,List

device = "cuda:2"
pw_type = torch.float16

from torch import Tensor
torch._C._jit_set_nvfuser_horizontal_mode(True)
torch._C._jit_set_nvfuser_single_node_mode(True)

def get_below(x):
    return torch.roll(x, shifts=-1, dims=2)
def get_above(x):
    return torch.roll(x, shifts=1, dims=2)
def get_left(x):
    return torch.roll(x, shifts=1, dims=3)
def get_right(x):
    return torch.roll(x, shifts=-1, dims=3)

def get_center_view(x):
    return x[:, :, 1:63, 1:63]
def get_below_view(x):
    return x[:, :, 2:64, 1:63]
def get_above_view(x):
    return x[:, :, 0:62, 1:63]
def get_left_view(x):
    return x[:, :, 1:63, 2:64]
def get_right_view(x):
    return x[:, :, 1:63, 0:62]

@torch.jit.script # JIT decorator
def interp(switch, if_false, if_true):
    return (~switch)*if_false + (switch)*if_true

@torch.jit.script # JIT decorator
def interp2(switch_a, switch_b, if_false, if_a, if_b):
    return ((~switch_a)&(~switch_b))*if_false + (switch_a)*if_a + (switch_b)*if_b

def new_world():
    world = torch.zeros((32, 8, 64, 64), dtype=pw_type).to(device)
    return world

def benchmark(func, inputs, iters=100):
    if type(inputs) is tuple:
        with torch.no_grad():
            for i in range(10):
                _ = func(*inputs)
            torch.cuda.synchronize()
            start = time.time()
            for i in range(iters):
                _ = func(*inputs)
            torch.cuda.synchronize()
            total_time = time.time() - start
            print("Time: {}".format(total_time))
    else:
        with torch.no_grad():
            for i in range(10):
                _ = func(inputs)
            torch.cuda.synchronize()
            start = time.time()
            for i in range(iters):
                _ = func(inputs)
            torch.cuda.synchronize()
            total_time = time.time() - start
            print("Time: {}".format(total_time))

# ===========================================
# Lesson: Don't have [:] in fused operators!!
# ===========================================
        
# Adding to world (0.23)
@torch.jit.script
def add(world):
    x = world + 1
    y = x + 2
    return y * 3
# benchmark(add, new_world(), 10000)

# Adding to world (0.65)
@torch.jit.script
def add2(world):
    x = world[:] + 1
    y = x[:] + 2
    return y[:] * 3
# benchmark(add2, new_world(), 10000)

# Adding to world (0.65)
@torch.jit.script
def add3(world):
    x = world[:5] + 1
    y = x[:] + 2
    return y[:] * 3
# benchmark(add3, new_world(), 10000)

# ===========================================
# Question: Which get_above is better? For gravity
# JIT Trace and Script is the same. Trace=(0.015), Script=(0.016), None=(0.026)
# ===========================================

# 0.057
@torch.jit.script
def above1(world):
    above = get_above(world)
    return above + 1
# benchmark(above1, new_world(), 1000)

# 0.031
@torch.jit.script
def above2(world):
    above = world[:, :, 1:-1]
    return above + 1
# benchmark(above2, new_world(), 1000)

# ===========================================
# Question: Which order is better? For gravity
# Yes, it is better to do all the shifts first, then kernel the pointwise.
# JIT Trace and Script is the same. Trace=(0.015), Script=(0.016), None=(0.026)
# OK! The kernel idea is slower. Because in the end we have to read from the conv result anyways.
# ===========================================


# (0.022, 0.016)
@torch.jit.script
def grav1(world):
    currDensity = 0
    density = world[:, 1:2]
    density_delta = get_above(density) - density # Delta between ABOVE and current
    is_density_above_greater = (density_delta > 0)
    is_density_below_less = get_below(is_density_above_greater)
    is_density_current = (density == currDensity)
    is_density_above_current = get_above(is_density_current)
    is_gravity = (world[:, 2:3] == 1)
    is_center_and_below_gravity = get_below(is_gravity) & is_gravity
    is_center_and_above_gravity = get_above(is_gravity) & is_gravity

    # These should never be both true for the same block
    does_become_below = is_density_current & is_density_below_less & is_center_and_below_gravity
    does_become_above = is_density_above_greater & is_density_above_current & is_center_and_above_gravity
    return does_become_below | does_become_above
benchmark(grav1, new_world(), 100)

# (0.013, 0.011)
@torch.jit.script
def grav3(world):
    currDensity = 0
    above = get_above(world)
    below = get_below(world)
    
    density = world[:, 1:2]
    is_density_above_greater = (above[:, 1:2] - world[:, 1:2] > 0)
    is_density_below_less = (below[:, 1:2] - world[:, 1:2] <= 0)
    is_density_current = (world[:, 1:2] == currDensity)
    is_density_above_current = (above == currDensity)
    is_gravity = (world[:, 2:3] == 1)
    is_gravity_above = above[:, 2:3] == 1
    is_gravity_below = below[:, 2:3] == 1
    
#     with torch.jit.strict_fusion():
    is_center_and_below_gravity = is_gravity_below & is_gravity
    is_center_and_above_gravity = is_gravity_above & is_gravity
    does_become_below = is_density_current & is_density_below_less & is_center_and_below_gravity
    does_become_above = is_density_above_greater & is_density_above_current & is_center_and_above_gravity

    return does_become_below | does_become_above
benchmark(grav3, new_world(), 100)


# # (0.013, 0.011)
# grav_kernel = torch.zeros((6,8*3,3,3), device=device, dtype=pw_type)
# # is_density_above_greater
# grav_kernel[0, 1, 1, 1] = -1
# grav_kernel[0, 1, 2, 1] = 1
# # is_density_above_greater
# grav_kernel[1, 1, 1, 1] = -1
# grav_kernel[1, 1, 0, 1] = 1
# # density_current
# grav_kernel[2, 1, 1, 1] = 1
# # density_current_above
# grav_kernel[3, 1, 0, 1] = 1
# # is_gravity_above
# grav_kernel[4, 2, 0, 1] = 1
# # is_gravity_below
# grav_kernel[5, 2, 2, 1] = 1

# # (0.022)
# @torch.jit.script
# def grav2(world : Tensor, kernel: Tensor):
#     currDensity = 0
#     above = get_above(world)
#     below = get_below(world)
#     expanded_world = torch.cat([above, world, below], dim=1) # [Batch, Channel1+Channel2+Channel3]
    
#     grav_values = F.conv2d(expanded_world, kernel, padding=1) > 0
#     is_density_above_greater = grav_values[0]
#     is_density_below_less = grav_values[1]
#     is_density_current = grav_values[2]
#     is_gravity = grav_values[3]
#     is_gravity_above = grav_values[4]
#     is_gravity_below = grav_values[5]
    
#     is_center_and_below_gravity = is_gravity_below & is_gravity
#     is_center_and_above_gravity = is_gravity_above & is_gravity
#     does_become_below = is_density_current & is_density_below_less & is_center_and_below_gravity
#     does_become_above = is_density_above_greater & is_density_above_greater & is_center_and_above_gravity
    
#     return does_become_below | does_become_above
# benchmark(grav2, (new_world(), grav_kernel), 100)

# @torch.jit.script
# def help_density_greater(above, world):
#     return (above[:, 1:2] - world[:, 1:2] > 0)
# # (0.013, 0.011)
# @torch.jit.script
# def grav4(world):
#     currDensity = 0
#     above = get_above(world)
#     below = get_below(world)
    
#     is_density_above_greater_fut = torch.jit.fork(help_density_greater, above, below)
#     is_density_below_less_fut = torch.jit.fork(help_density_greater, above, below)
#     is_density_current_fut = torch.jit.fork(help_density_greater, above, below)
#     is_density_above_current_fut = torch.jit.fork(help_density_greater, above, below)
#     is_gravity_fut = torch.jit.fork(help_density_greater, above, below)
#     is_gravity_above_fut = torch.jit.fork(help_density_greater, above, below)
#     is_gravity_below_fut = torch.jit.fork(help_density_greater, above, below)

#     is_density_above_greater = torch.jit.wait(is_density_above_greater_fut)
#     is_density_below_less = torch.jit.wait(is_density_above_greater_fut)
#     is_density_current = torch.jit.wait(is_density_above_greater_fut)
#     is_density_above_current = torch.jit.wait(is_density_above_greater_fut)
#     is_gravity = torch.jit.wait(is_density_above_greater_fut)
#     is_gravity_above = torch.jit.wait(is_density_above_greater_fut)
#     is_gravity_below =torch.jit.wait(is_density_above_greater_fut)
    
#     is_center_and_below_gravity = is_gravity_below & is_gravity
#     is_center_and_above_gravity = is_gravity_above & is_gravity
#     does_become_below = is_density_current & is_density_below_less & is_center_and_below_gravity
#     does_become_above = is_density_above_greater & is_density_above_current & is_center_and_above_gravity
    
#     return does_become_below | does_become_above
# benchmark(grav4, new_world(), 100)

# ===========================================
# Question: Which get_above is better?
# Answer: It doesnt matter, they're like the same...
# ===========================================

# (0.016)
@torch.jit.script
def grav1(world):
    currDensity = 0
    density = world[:, 1:2]
    density_delta = get_above(density) - density # Delta between ABOVE and current
    is_density_above_greater = (density_delta > 0)
    is_density_below_less = get_below(is_density_above_greater)
    is_density_current = (density == currDensity)
    is_density_above_current = get_above(is_density_current)
    is_gravity = (world[:, 2:3] == 1)
    is_center_and_below_gravity = get_below(is_gravity) & is_gravity
    is_center_and_above_gravity = get_above(is_gravity) & is_gravity

    # These should never be both true for the same block
    does_become_below = is_density_current & is_density_below_less & is_center_and_below_gravity
    does_become_above = is_density_above_greater & is_density_above_current & is_center_and_above_gravity
    return does_become_below | does_become_above
# benchmark(grav1, new_world(), 100)

# (0.016)
@torch.jit.script
def grav2(world):
    currDensity = 0
    density = world[:, 1:2]
    is_density_above_greater = (get_above_view(density) - get_center_view(density) > 0)
    is_density_below_less = (get_below_view(density) - get_center_view(density) <= 0)
    is_density_current = (density == currDensity)
    is_density_above_current = get_above_view(is_density_current)
    is_gravity = (world[:, 2:3] == 1)
    is_center_and_below_gravity = get_below_view(is_gravity) & get_center_view(is_gravity)
    is_center_and_above_gravity = get_above_view(is_gravity) & get_center_view(is_gravity)

    # These should never be both true for the same block
    does_become_below = is_density_current & is_density_below_less & is_center_and_below_gravity
    does_become_above = is_density_above_greater & is_density_above_current & is_center_and_above_gravity
    return does_become_below | does_become_above
# benchmark(grav1, new_world(), 100)