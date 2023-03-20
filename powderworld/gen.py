import torch
import numpy as np
import skimage.draw
import os
from powderworld.sim import pw_elements, pw_type

def lim(world_size, x):
    return np.clip(int(x), 0, world_size-1)

def lim_gen(world_size):
    return lambda x : lim(world_size, x)

def trapez(y,y0,w):
    return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)

def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
    if r0 == r1 and c0 == c1:
        r1 += 1
    # print(r0, c0, r1, c1, w)
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

    # The following is now always < 1 in abs
    slope = (r1-r0) / (c1-c0)

    # Adjust weight by the slope
    w *= np.sqrt(1+np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * slope + (c1*r0-c0*r1) / (c1-c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w/2)
    yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1,1), w).flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0, xx >= rmin, xx < rmax))
    # mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])

def add_elem(world_slice, elem):
    elem_num = pw_elements[elem][0]
    world_slice[...] = elem_num
    
def add_elem_mask(world, mask, elem):
    elem_num = pw_elements[elem][0]
    world[mask] = elem_num
    
def add_elem_rc(world_slice, rr, cc, elem):
    elem_num = pw_elements[elem][0]
    world_slice[:,rr,cc] = elem_num

def init_world(xsize=64, ysize=64, batchsize=1):
    world = np.zeros((batchsize, xsize, ysize), dtype=np.uint8)
    return world

# Behavior to create Wall edges around a world.
def do_edges(world, rand):
    add_elem(world[:, :1, :], "wall")
    add_elem(world[:, -1:, :], "wall")
    add_elem(world[:, :, :1], "wall")
    add_elem(world[:, :, -1:], "wall")
    
# Behavior to make a circle. Don't overlap any elements.
def do_circle(world, rand, elem):
    for t in range(10):
        radius = rand.randint(5, 20)
        x1 = rand.randint(world.shape[1])
        y1 = rand.randint(world.shape[2])
        rr, cc = skimage.draw.disk((x1, y1), radius, shape=(world.shape[1], world.shape[2]))
        # If enough of the space is empty
        if world[:,rr,cc].sum() < 10:
            add_elem_rc(world, rr, cc, elem)
            if rand.rand() < 0.5:
                rr, cc = skimage.draw.disk((x1, y1), max(1, radius-5), shape=(world.shape[1], world.shape[2]))
                add_elem_rc(world, rr, cc, 'empty')
            break
            
# Behavior to make a circle with cloner. Don't overlap any elements.
def do_cloner_circle(world, rand, elem):
    for t in range(10):
        radius = rand.randint(5, 10)
        x1 = rand.randint(world.shape[1])
        y1 = rand.randint(world.shape[2])
        rr, cc = skimage.draw.disk((x1, y1), radius, shape=(world.shape[1], world.shape[2]))
        # If enough of the space is empty
        if world[:,rr,cc].sum() < 10:
            add_elem_rc(world, rr, cc, elem)
            rr, cc = skimage.draw.disk((x1, y1), max(1, radius-2), shape=(world.shape[1], world.shape[2]))
            add_elem_rc(world, rr, cc, 'cloner')
            break
            
# Behavior to make square boxes. Don't overlap any elements.
def do_boxes(world, rand, elem, empty_roof):
    lim_y = lim_gen(world.shape[1])
    lim_x = lim_gen(world.shape[2])
    for t in range(10):
        radius = rand.randint(8, 20)
        radius_small = radius-rand.randint(3,7)
        y1 = rand.randint(radius, world.shape[1]-radius)
        x1 = rand.randint(radius, world.shape[2]-radius)
        mask = np.zeros(world.shape, bool)
        mask[:, lim_y(y1-radius):lim_y(y1+radius), lim_x(x1-radius):lim_x(x1+radius)] = True
        mask[:, lim_y(y1-radius_small):lim_y(y1+radius_small), lim_x(x1-radius_small):lim_x(x1+radius_small)] = False
        if empty_roof:
            mask[:, lim_y(y1-radius):lim_y(y1+radius_small), lim_x(x1-radius_small):lim_x(x1+radius_small)] = False
        if world[mask].sum() <= 10:
            add_elem_mask(world, mask, elem)
            break
            
# Behavior to fill in a slice of the map.
def do_fill_slice(world, rand, elem, direc):
    if direc == 0:
        amount = rand.randint(1, int(world.shape[1]/3))
        add_elem(world[:, :amount, :], elem)
    elif direc == 1:
        amount = rand.randint(1, int(world.shape[1]/3))
        add_elem(world[:, -amount:, :], elem)
    elif direc == 2:
        amount = rand.randint(1, int(world.shape[2]/3))
        add_elem(world[:, :, :amount], elem)
    elif direc == 3:
        amount = rand.randint(1, int(world.shape[2]/3))
        add_elem(world[:, :, -amount:], elem)
        
# Cloner Roof
def do_cloner_roof(world, rand, elem):
    add_elem(world[:,:3,:], "cloner")
    add_elem(world[:,3:6,:], elem)
        
# Behavior to make a sine wave. May overlap.
def do_sine_wave(world, rand, elem, y_range=[10, 20]):
    elem_id = pw_elements[elem][0]
    center_x = rand.randint(world.shape[2])
    spread_x = rand.randint(world.shape[2]) + 10
    start_x = lim(world.shape[2], center_x - spread_x)
    end_x = lim(world.shape[2], center_x + spread_x)
    radius = rand.randint(2, 4)
    amplitude = rand.randint(5, 10)
    frequency = np.pi/rand.randint(8, 15)
    origin = rand.randint(y_range[0], y_range[1])
    do_double_wave = rand.rand() < 0.3
    for x in range(start_x, end_x, 1):
        y = int(origin + amplitude*np.sin(x*frequency))
        big_slice = world[:, y-radius-1:y+radius+1, x-radius-1:x+radius+1]
        if np.sum((big_slice != elem_id) & (big_slice != 0)) >= 5:
            break
        add_elem(world[:, y-radius:y+radius, x-radius:x+radius], elem)
        if do_double_wave:
            y = int(origin - amplitude*np.sin(x*frequency))
            big_slice = world[:, y-radius-1:y+radius+1, x-radius-1:x+radius+1]
            if np.sum((big_slice != elem_id) & (big_slice != 0)) >= 5:
                break
            add_elem(world[:, y-radius:y+radius, x-radius:x+radius], elem)
            

# Behavior to make 45-degree ramps
def do_ramps(world, rand, elem):
    elem_id = pw_elements[elem][0]
    radius = rand.randint(1, 3)
    count = 0
    
    y = 10+rand.randint(10)
    x = rand.randint(world.shape[2])
    center_x = rand.randint(world.shape[2]/2) + world.shape[2]/4
    
    while y < world.shape[1]-10:
        count += 1
        if count > 10:
            break
        delta_y = 1
        delta_x = 1 if x < center_x else -1

        full_t = rand.randint(10)+10
        for t in range(full_t):
            ny = y+delta_y*t
            nx = x+delta_x*t
            add_elem(world[:, ny-radius:ny+radius, nx-radius:nx+radius], elem)
            
        rand_t = rand.randint(full_t)
        y = y + delta_y*rand_t
        x = x + delta_x*rand_t
        
# Behavior to make arches. Won't overlap, but will touch.
def do_arches(world, rand, elem):
    lim_y = lim_gen(world.shape[1])
    lim_x = lim_gen(world.shape[2])
    elem_id = pw_elements[elem][0]
    radius = rand.randint(1, 3)
    count = 0
    
    x_length = rand.randint(3, 12)
    y = rand.randint(10, world.shape[1])
    x = rand.randint(x_length, world.shape[2]-x_length)
    failed_left = False
    failed_right = False
    
    mask = np.zeros(world.shape, bool)
    
    for t in range(x_length):
        ny = y
        nx = x + t
        world_slice = world[:, lim_y(ny-radius):lim_y(ny+radius), lim_x(nx-radius):lim_x(nx+radius)]
        if world_slice.sum() != 0:
            failed_right = True
            break
        mask[:, lim_y(ny-radius):lim_y(ny+radius), lim_x(nx-radius):lim_x(nx+radius)] = True
    for t in range(x_length):
        ny = y
        nx = x - t
        world_slice = world[:, lim_y(ny-radius):lim_y(ny+radius), lim_x(nx-radius):lim_x(nx+radius)]
        if world_slice.sum() != 0:
            failed_left = True
            break
        mask[:, lim_y(ny-radius):lim_y(ny+radius), lim_x(nx-radius):lim_x(nx+radius)] = True
    if not failed_right:
        for t in range(world.shape[1]):
            ny = y + t
            nx = x + x_length
            world_slice = world[:, lim_y(ny-radius):lim_y(ny+radius), lim_x(nx-radius):lim_x(nx+radius)]
            if world_slice.sum() != 0:
                break
            mask[:, lim_y(ny-radius):lim_y(ny+radius), lim_x(nx-radius):lim_x(nx+radius)] = True
    if not failed_left:
        for t in range(world.shape[1]):
            ny = y + t
            nx = x - x_length
            world_slice = world[:, lim_y(ny-radius):lim_y(ny+radius), lim_x(nx-radius):lim_x(nx+radius)]
            if world_slice.sum() != 0:
                break
            mask[:, lim_y(ny-radius):lim_y(ny+radius), lim_x(nx-radius):lim_x(nx+radius)] = True
    
    add_elem_mask(world, mask, elem)

    
        

# def make_world(world, elems=['empty','sand', 'water', 'wall'], num_tasks=1000000, num_lines=5, num_circles=0, num_squares=0):
#     seed = np.random.randint(num_tasks)
#     rand = np.random.RandomState(seed)
    
#     pw.add_element(world[:, :, :, :], "empty")

#     for s in range(rand.randint(0, num_lines+1)):
#         radius = rand.randint(1,10)
#         elem = rand.choice(elems)
#         x1 = rand.randint(64)
#         x2 = rand.randint(64)
#         y1 = rand.randint(64)
#         y2 = rand.randint(64)
#         rr, cc, _ = weighted_line(x1, y1, x2, y2, radius, 0, 64)
#         pw.add_element_rc(world, rr, cc, elem)
#     for s in range(rand.randint(0, num_circles+1)):
#         radius = rand.randint(5,20)
#         elem = rand.choice(elems)
#         x1 = rand.randint(64)
#         y1 = rand.randint(64)
#         rr, cc = skimage.draw.disk((x1, y1), radius, shape=(64,64))
#         pw.add_element_rc(world, rr, cc, elem)
#     for s in range(rand.randint(0, num_squares+1)):
#         radius = rand.randint(5,20)
#         elem = rand.choice(elems)
#         x1 = rand.randint(64)
#         y1 = rand.randint(64)
#         pw.add_element(world[:, :, lim(x1-radius):lim(x1+radius), lim(y1-radius):lim(y1+radius)], elem)
        
#     pw.add_element(world[:, :, 0:1, :], "wall")
#     pw.add_element(world[:, :, 63:64, :], "wall")
#     pw.add_element(world[:, :, :, 0:1], "wall")
#     pw.add_element(world[:, :, :, 63:64], "wall")
    
# def make_test160(world, test_num):
#     world[:] = saved_worlds[test_num]
    
# def make_test(world, test_num):
#     pw.add_element(world[:, :, :, :], "empty")
#     if test_num == 0:
#         # Sand on Water.
#         pw.add_element(world[:, :, 32:, :], 'water')
#         rr, cc = skimage.draw.disk((32, 32), 10, shape=(64,64))
#         pw.add_element_rc(world, rr, cc, 'sand')
#     elif test_num == 1:
#         # Stone Arches.
#         rr, cc, _ = weighted_line(63, 5, 32, 5, 3, 0, 64)
#         pw.add_element_rc(world, rr, cc, 'stone')
#         rr, cc, _ = weighted_line(63, 15, 32, 15, 3, 0, 64)
#         pw.add_element_rc(world, rr, cc, 'stone')
#         rr, cc, _ = weighted_line(32, 5, 32, 15, 3, 0, 64)
#         pw.add_element_rc(world, rr, cc, 'stone')
#         rr, cc, _ = weighted_line(63, 40, 32, 40, 3, 0, 64)
#         pw.add_element_rc(world, rr, cc, 'stone')
#         rr, cc, _ = weighted_line(63, 60, 32, 60, 3, 0, 64)
#         pw.add_element_rc(world, rr, cc, 'stone')
#         rr, cc, _ = weighted_line(32, 45, 32, 55, 3, 0, 64)
#         pw.add_element_rc(world, rr, cc, 'stone')
#     elif test_num == 2:
#         # Ice and Plants in Water.
#         pw.add_element(world[:, :, :, :], 'water')
#         rr, cc = skimage.draw.disk((16, 32), 10, shape=(64,64))
#         pw.add_element_rc(world, rr, cc, 'ice')
#         rr, cc = skimage.draw.disk((48, 32), 10, shape=(64,64))
#         pw.add_element_rc(world, rr, cc, 'plant')
#     elif test_num == 3:
#         # Burning Vines.
#         rr, cc, _ = weighted_line(0, 0, 63, 63, 3, 0, 64)
#         pw.add_element_rc(world, rr, cc, 'plant')
#         rr, cc, _ = weighted_line(32, 32, 0, 63, 3, 0, 64)
#         pw.add_element_rc(world, rr, cc, 'wood')
#         rr, cc, _ = weighted_line(48, 16, 16, 16, 3, 0, 64)
#         pw.add_element_rc(world, rr, cc, 'wood')
#         pw.add_element(world[:,:,48:,48:], 'fire')
#     elif test_num == 4:
#         # Acid on Wood.
#         pw.add_element(world[:, :, 32:, :], 'wood')
#         rr, cc = skimage.draw.disk((16, 32), 10, shape=(64,64))
#         pw.add_element_rc(world, rr, cc, 'acid')
#     elif test_num == 5:
#         # Water Flowing.
#         rr, cc, _ = weighted_line(0, 0, 32, 32, 3, 0, 64)
#         pw.add_element_rc(world, rr, cc, 'wall')
#         rr, cc, _ = weighted_line(63, 16, 32, 48, 3, 0, 64)
#         pw.add_element_rc(world, rr, cc, 'wall')
#         pw.add_element(world[:,:,:16,16:32], 'water')
#     elif test_num == 6:
#         # Dust on Lava.
#         pw.add_element(world[:, :, 32:, :], 'lava')
#         pw.add_element(world[:, :, :, 16-2:16+2], 'stone')
#         pw.add_element(world[:, :, :, 48-2:48+2], 'stone')
#         rr, cc = skimage.draw.disk((16, 32), 10, shape=(64,64))
#         pw.add_element_rc(world, rr, cc, 'dust')
#     elif test_num == 7:
#         # Gas Flowing.
#         rr, cc, _ = weighted_line(0, 16, 32, 48, 3, 0, 64)
#         pw.add_element_rc(world, rr, cc, 'wall')
#         rr, cc, _ = weighted_line(63, 0, 32, 32, 3, 0, 64)
#         pw.add_element_rc(world, rr, cc, 'wall')
#         pw.add_element(world[:,:,48:,16:32], 'gas')
#     pw.add_element(world[:, :, 0:1, :], "wall")
#     pw.add_element(world[:, :, 63:64, :], "wall")
#     pw.add_element(world[:, :, :, 0:1], "wall")
#     pw.add_element(world[:, :, :, 63:64], "wall")
    
    
# def make_rl_world(world, elems=['empty','sand', 'water', 'wall'], num_tasks=1000000, num_lines=5, num_circles=0, num_squares=0, has_empty_path=False):
#     seed = np.random.randint(num_tasks)
#     rand = np.random.RandomState(seed)
#     new_elems = elems.copy()
#     if 'sand' in new_elems:
#         new_elems.remove('sand')
    
#     pw.add_element(world[:, :, :, :], "empty")
#     sandy = rand.randint(8,54)
#     sandx = rand.randint(8,44)

#     for s in range(rand.randint(0, num_lines+1)):
#         radius = rand.randint(1,10)
#         elem = rand.choice(new_elems)
#         x1 = rand.randint(64)
#         x2 = rand.randint(64)
#         y1 = rand.randint(64)
#         y2 = rand.randint(64)
#         rr, cc, _ = weighted_line(x1, y1, x2, y2, radius, 0, 64)
#         pw.add_element_rc(world, rr, cc, elem)
#     for s in range(rand.randint(0, num_circles+1)):
#         radius = rand.randint(5,20)
#         elem = rand.choice(new_elems)
#         x1 = rand.randint(64)
#         y1 = rand.randint(64)
#         rr, cc = skimage.draw.disk((x1, y1), radius, shape=(64,64))
#         pw.add_element_rc(world, rr, cc, elem)
#     for s in range(rand.randint(0, num_squares+1)):
#         radius = rand.randint(5,20)
#         elem = rand.choice(new_elems)
#         x1 = rand.randint(64)
#         y1 = rand.randint(64)
#         pw.add_element(world[:, :, lim(x1-radius):lim(x1+radius), lim(y1-radius):lim(y1+radius)], elem)
    
#     # Wall
#     pw.add_element(world[:, :, 0:1, :], "wall")
#     pw.add_element(world[:, :, 63:64, :], "wall")
#     pw.add_element(world[:, :, :, 0:1], "wall")
#     pw.add_element(world[:, :, :, 63:64], "wall")
#     # Sand
#     pw.add_element(world[:, :, lim(sandy-5):lim(sandy+5), lim(sandx-5):lim(sandx+5)], "sand")
#     # Border
#     pw.add_element(world[:, :, :, -10:-1], "wall")
#     pw.add_element(world[:, :, 32-10:32+10, -10:-1], "empty")
    
# def make_rl_test(world, test_num):
#     pw.add_element(world[:, :, :, :], "empty")
#     if test_num == 0:
#         sandy = 32
#         sandx = 8
#     elif test_num == 1:
#         pw.add_element(world[:, :, 16:48, 32:36], "wall")
#         sandy = 32
#         sandx = 8
#     elif test_num == 2:
#         pw.add_element(world[:, :, :, :], "wall")
#         sandy = 8
#         sandx = 8
#         rr, cc, _ = weighted_line(0, 0, 32, 63, 15, 0, 64)
#         pw.add_element_rc(world, rr, cc, 'empty')
#     elif test_num == 3:
#         pw.add_element(world[:, :, :, :], "wall")
#         sandy = 32
#         sandx = 8
#         rr, cc, _ = weighted_line(32, 8, 0, 8, 25, 0, 64)
#         pw.add_element_rc(world, rr, cc, 'empty')
#         rr, cc, _ = weighted_line(8, 8, 8, 50, 25, 0, 64)
#         pw.add_element_rc(world, rr, cc, 'empty')
#         rr, cc, _ = weighted_line(0, 50, 32, 50, 25, 0, 64)
#         pw.add_element_rc(world, rr, cc, 'empty')
#     elif test_num == 4:
#         sandy = 54
#         sandx = 8
#         pw.add_element(world[:, :, 40:64, :], "water")
#     elif test_num == 5:
#         sandy = 32
#         sandx = 8
#         pw.add_element(world[:, :, 40:64, 0:16], "wall")
#         pw.add_element(world[:, :, :, 30:40], "plant")
#         pw.add_element(world[:, :, 55:64, 40:45], "wall")
#         pw.add_element(world[:, :, 55:64, 45:], "lava")
#     elif test_num == 6:
#         sandy = 32
#         sandx = 8
#         pw.add_element(world[:, :, 40:64, 0:16], "wall")
#         pw.add_element(world[:, :, :, 30:45], "stone")
#     elif test_num == 7:
#         sandy = 32
#         sandx = 8
#         pw.add_element(world[:, :, 42:44, :], "lava")
#         pw.add_element(world[:, :, 44:46, :], "stone")
#         pw.add_element(world[:, :, 46:, :], "dust")
        
#         pw.add_element(world[:, :, :20, :], "lava")
#         pw.add_element(world[:, :, 20:22, :], "stone")
#         pw.add_element(world[:, :, 22:24:, :], "dust")
#         pw.add_element(world[:, :, 24:26, :], "plant")
#         pw.add_element(world[:, :, :26, 40:], "wall")
#     else:
#         sandy = 32
#         sandx = 8
    
#     # Wall
#     pw.add_element(world[:, :, 0:1, :], "wall")
#     pw.add_element(world[:, :, 63:64, :], "wall")
#     pw.add_element(world[:, :, :, 0:1], "wall")
#     pw.add_element(world[:, :, :, 63:64], "wall")
#     # Sand       
#     pw.add_element(world[:, :, lim(sandy-5):lim(sandy+5), lim(sandx-5):lim(sandx+5)], "sand")
#     # Border
#     pw.add_element(world[:, :, :, -10:-1], "wall")
#     pw.add_element(world[:, :, 32-10:32+10, -10:-1], "empty")
    