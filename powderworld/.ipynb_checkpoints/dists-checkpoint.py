import torch
import numpy as np
import skimage.draw
import os

saved_worlds = torch.from_numpy(np.load(os.path.join(os.path.dirname(__file__), '160worlds.npz'))['arr_0'].astype('float'))

def lim(x):
    return np.clip(1, 63, int(x))

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

def init_world(pw, xsize=64, ysize=64, batchsize=1):
    return torch.zeros((batchsize, pw.NUM_CHANNEL, xsize, ysize), dtype=torch.float32, device=pw.device)

def make_empty_world(pw, world):
    pw.add_element(world[:, :, :, :], "empty")
    pw.add_element(world[:, :, 0:1, :], "wall")
    pw.add_element(world[:, :, 63:64, :], "wall")
    pw.add_element(world[:, :, :, 0:1], "wall")
    pw.add_element(world[:, :, :, 63:64], "wall")

def make_world(pw, world, elems=['empty','sand', 'water', 'wall'], num_tasks=1000000, num_lines=5, num_circles=0, num_squares=0):
    seed = np.random.randint(num_tasks)
    rand = np.random.RandomState(seed)
    
    pw.add_element(world[:, :, :, :], "empty")

    for s in range(rand.randint(0, num_lines+1)):
        radius = rand.randint(1,10)
        elem = rand.choice(elems)
        x1 = rand.randint(64)
        x2 = rand.randint(64)
        y1 = rand.randint(64)
        y2 = rand.randint(64)
        rr, cc, _ = weighted_line(x1, y1, x2, y2, radius, 0, 64)
        pw.add_element_rc(world, rr, cc, elem)
    for s in range(rand.randint(0, num_circles+1)):
        radius = rand.randint(5,20)
        elem = rand.choice(elems)
        x1 = rand.randint(64)
        y1 = rand.randint(64)
        rr, cc = skimage.draw.disk((x1, y1), radius, shape=(64,64))
        pw.add_element_rc(world, rr, cc, elem)
    for s in range(rand.randint(0, num_squares+1)):
        radius = rand.randint(5,20)
        elem = rand.choice(elems)
        x1 = rand.randint(64)
        y1 = rand.randint(64)
        pw.add_element(world[:, :, lim(x1-radius):lim(x1+radius), lim(y1-radius):lim(y1+radius)], elem)
        
    pw.add_element(world[:, :, 0:1, :], "wall")
    pw.add_element(world[:, :, 63:64, :], "wall")
    pw.add_element(world[:, :, :, 0:1], "wall")
    pw.add_element(world[:, :, :, 63:64], "wall")
    
def make_test160(pw, world, test_num):
    world[:] = saved_worlds[test_num]
    
def make_test(pw, world, test_num):
    pw.add_element(world[:, :, :, :], "empty")
    if test_num == 0:
        # Sand on Water.
        pw.add_element(world[:, :, 32:, :], 'water')
        rr, cc = skimage.draw.disk((32, 32), 10, shape=(64,64))
        pw.add_element_rc(world, rr, cc, 'sand')
    elif test_num == 1:
        # Stone Arches.
        rr, cc, _ = weighted_line(63, 5, 32, 5, 3, 0, 64)
        pw.add_element_rc(world, rr, cc, 'stone')
        rr, cc, _ = weighted_line(63, 15, 32, 15, 3, 0, 64)
        pw.add_element_rc(world, rr, cc, 'stone')
        rr, cc, _ = weighted_line(32, 5, 32, 15, 3, 0, 64)
        pw.add_element_rc(world, rr, cc, 'stone')
        rr, cc, _ = weighted_line(63, 40, 32, 40, 3, 0, 64)
        pw.add_element_rc(world, rr, cc, 'stone')
        rr, cc, _ = weighted_line(63, 60, 32, 60, 3, 0, 64)
        pw.add_element_rc(world, rr, cc, 'stone')
        rr, cc, _ = weighted_line(32, 45, 32, 55, 3, 0, 64)
        pw.add_element_rc(world, rr, cc, 'stone')
    elif test_num == 2:
        # Ice and Plants in Water.
        pw.add_element(world[:, :, :, :], 'water')
        rr, cc = skimage.draw.disk((16, 32), 10, shape=(64,64))
        pw.add_element_rc(world, rr, cc, 'ice')
        rr, cc = skimage.draw.disk((48, 32), 10, shape=(64,64))
        pw.add_element_rc(world, rr, cc, 'plant')
    elif test_num == 3:
        # Burning Vines.
        rr, cc, _ = weighted_line(0, 0, 63, 63, 3, 0, 64)
        pw.add_element_rc(world, rr, cc, 'plant')
        rr, cc, _ = weighted_line(32, 32, 0, 63, 3, 0, 64)
        pw.add_element_rc(world, rr, cc, 'wood')
        rr, cc, _ = weighted_line(48, 16, 16, 16, 3, 0, 64)
        pw.add_element_rc(world, rr, cc, 'wood')
        pw.add_element(world[:,:,48:,48:], 'fire')
    elif test_num == 4:
        # Acid on Wood.
        pw.add_element(world[:, :, 32:, :], 'wood')
        rr, cc = skimage.draw.disk((16, 32), 10, shape=(64,64))
        pw.add_element_rc(world, rr, cc, 'acid')
    elif test_num == 5:
        # Water Flowing.
        rr, cc, _ = weighted_line(0, 0, 32, 32, 3, 0, 64)
        pw.add_element_rc(world, rr, cc, 'wall')
        rr, cc, _ = weighted_line(63, 16, 32, 48, 3, 0, 64)
        pw.add_element_rc(world, rr, cc, 'wall')
        pw.add_element(world[:,:,:16,16:32], 'water')
    elif test_num == 6:
        # Dust on Lava.
        pw.add_element(world[:, :, 32:, :], 'lava')
        pw.add_element(world[:, :, :, 16-2:16+2], 'stone')
        pw.add_element(world[:, :, :, 48-2:48+2], 'stone')
        rr, cc = skimage.draw.disk((16, 32), 10, shape=(64,64))
        pw.add_element_rc(world, rr, cc, 'dust')
    elif test_num == 7:
        # Gas Flowing.
        rr, cc, _ = weighted_line(0, 16, 32, 48, 3, 0, 64)
        pw.add_element_rc(world, rr, cc, 'wall')
        rr, cc, _ = weighted_line(63, 0, 32, 32, 3, 0, 64)
        pw.add_element_rc(world, rr, cc, 'wall')
        pw.add_element(world[:,:,48:,16:32], 'gas')
    pw.add_element(world[:, :, 0:1, :], "wall")
    pw.add_element(world[:, :, 63:64, :], "wall")
    pw.add_element(world[:, :, :, 0:1], "wall")
    pw.add_element(world[:, :, :, 63:64], "wall")
    
    
def make_rl_world(pw, world, elems=['empty','sand', 'water', 'wall'], num_tasks=1000000, num_lines=5, num_circles=0, num_squares=0, has_empty_path=False):
    seed = np.random.randint(num_tasks)
    rand = np.random.RandomState(seed)
    new_elems = elems.copy()
    if 'sand' in new_elems:
        new_elems.remove('sand')
    
    pw.add_element(world[:, :, :, :], "empty")
    sandy = rand.randint(8,54)
    sandx = rand.randint(8,44)

    for s in range(rand.randint(0, num_lines+1)):
        radius = rand.randint(1,10)
        elem = rand.choice(new_elems)
        x1 = rand.randint(64)
        x2 = rand.randint(64)
        y1 = rand.randint(64)
        y2 = rand.randint(64)
        rr, cc, _ = weighted_line(x1, y1, x2, y2, radius, 0, 64)
        pw.add_element_rc(world, rr, cc, elem)
    for s in range(rand.randint(0, num_circles+1)):
        radius = rand.randint(5,20)
        elem = rand.choice(new_elems)
        x1 = rand.randint(64)
        y1 = rand.randint(64)
        rr, cc = skimage.draw.disk((x1, y1), radius, shape=(64,64))
        pw.add_element_rc(world, rr, cc, elem)
    for s in range(rand.randint(0, num_squares+1)):
        radius = rand.randint(5,20)
        elem = rand.choice(new_elems)
        x1 = rand.randint(64)
        y1 = rand.randint(64)
        pw.add_element(world[:, :, lim(x1-radius):lim(x1+radius), lim(y1-radius):lim(y1+radius)], elem)
    
    # Wall
    pw.add_element(world[:, :, 0:1, :], "wall")
    pw.add_element(world[:, :, 63:64, :], "wall")
    pw.add_element(world[:, :, :, 0:1], "wall")
    pw.add_element(world[:, :, :, 63:64], "wall")
    # Sand
    pw.add_element(world[:, :, lim(sandy-5):lim(sandy+5), lim(sandx-5):lim(sandx+5)], "sand")
    # Border
    pw.add_element(world[:, :, :, -10:-1], "wall")
    pw.add_element(world[:, :, 32-10:32+10, -10:-1], "empty")
    
def make_rl_test(pw, world, test_num):
    pw.add_element(world[:, :, :, :], "empty")
    if test_num == 0:
        sandy = 32
        sandx = 8
    elif test_num == 1:
        pw.add_element(world[:, :, 16:48, 32:36], "wall")
        sandy = 32
        sandx = 8
    elif test_num == 2:
        pw.add_element(world[:, :, :, :], "wall")
        sandy = 8
        sandx = 8
        rr, cc, _ = weighted_line(0, 0, 32, 63, 15, 0, 64)
        pw.add_element_rc(world, rr, cc, 'empty')
    elif test_num == 3:
        pw.add_element(world[:, :, :, :], "wall")
        sandy = 32
        sandx = 8
        rr, cc, _ = weighted_line(32, 8, 0, 8, 25, 0, 64)
        pw.add_element_rc(world, rr, cc, 'empty')
        rr, cc, _ = weighted_line(8, 8, 8, 50, 25, 0, 64)
        pw.add_element_rc(world, rr, cc, 'empty')
        rr, cc, _ = weighted_line(0, 50, 32, 50, 25, 0, 64)
        pw.add_element_rc(world, rr, cc, 'empty')
    elif test_num == 4:
        sandy = 54
        sandx = 8
        pw.add_element(world[:, :, 40:64, :], "water")
    elif test_num == 5:
        sandy = 32
        sandx = 8
        pw.add_element(world[:, :, 40:64, 0:16], "wall")
        pw.add_element(world[:, :, :, 30:40], "plant")
        pw.add_element(world[:, :, 55:64, 40:45], "wall")
        pw.add_element(world[:, :, 55:64, 45:], "lava")
    elif test_num == 6:
        sandy = 32
        sandx = 8
        pw.add_element(world[:, :, 40:64, 0:16], "wall")
        pw.add_element(world[:, :, :, 30:45], "stone")
    elif test_num == 7:
        sandy = 32
        sandx = 8
        pw.add_element(world[:, :, 42:44, :], "lava")
        pw.add_element(world[:, :, 44:46, :], "stone")
        pw.add_element(world[:, :, 46:, :], "dust")
        
        pw.add_element(world[:, :, :20, :], "lava")
        pw.add_element(world[:, :, 20:22, :], "stone")
        pw.add_element(world[:, :, 22:24:, :], "dust")
        pw.add_element(world[:, :, 24:26, :], "plant")
        pw.add_element(world[:, :, :26, 40:], "wall")
    else:
        sandy = 32
        sandx = 8
    
    # Wall
    pw.add_element(world[:, :, 0:1, :], "wall")
    pw.add_element(world[:, :, 63:64, :], "wall")
    pw.add_element(world[:, :, :, 0:1], "wall")
    pw.add_element(world[:, :, :, 63:64], "wall")
    # Sand       
    pw.add_element(world[:, :, lim(sandy-5):lim(sandy+5), lim(sandx-5):lim(sandx+5)], "sand")
    # Border
    pw.add_element(world[:, :, :, -10:-1], "wall")
    pw.add_element(world[:, :, 32-10:32+10, -10:-1], "empty")
    