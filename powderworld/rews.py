import torch
import numpy as np
from powderworld.sim import pw_elements

def get_slice(world, y, x):
    if x == -1 or y == -1:
        return world
    slice_size_y = int(world.shape[2] / 4)
    slice_size_x = int(world.shape[3] / 4)
    return world[:, :, slice_size_y*y:slice_size_y*(y+1), slice_size_x*x:slice_size_x*(x+1)]

def rew_elem_exists(world, elem, y, x):
    elem_id = pw_elements[elem][0]
    return (get_slice(world, y, x)[:, 0] == elem_id).sum()

def rew_elem_destroy(world, elem, y, x):
    elem_id = pw_elements[elem][0]
    return -(get_slice(world, y, x)[:, 0] == elem_id).sum()

def rew_delta(world, world_before, y, x):
    return (get_slice(world, y, x)[:, :len(pw_elements)] - get_slice(world_before, y, x)[:, :len(pw_elements)]).sum()