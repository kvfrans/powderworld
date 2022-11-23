import gym
import torch
import numpy as np
from IPython import display as ipythondisplay
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from powderworld import PWSim, PWRenderer
import powderworld.dists

# ============
# WORLD MODEL NETWORK
# ============

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
    
class Encoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs
    
class Decoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def get_features(self, x, encoder_features):
        ftrs = []
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
            ftrs.append(x)
        return ftrs
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        enc_chs=(14, 32, 64, 128)
        dec_chs=(128, 64, 32)
        num_class=14
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        return out

# ============
# DATA GENERATION
# ============


TOTAL_STEP = 128
BATCH_SIZE = 64

def lim(x):
    return np.clip(1, 63, int(x))

@torch.no_grad()
def make_data_batch(pw, elems, dist, num_tasks=1000000, num_lines=5, num_circles=0, num_squares=0, step_size=8):
    STEP_SIZE = step_size
    PER_ITER = BATCH_SIZE*int(TOTAL_STEP/STEP_SIZE) # 1024
    ci = 0
    data_batch = torch.zeros((PER_ITER, 2, 64, 64), dtype=torch.int64, device=pw.device)
    world = torch.zeros((BATCH_SIZE, pw.NUM_CHANNEL, 64, 64), dtype=torch.float32, device=pw.device)
    if dist == "train":
        for b in range(BATCH_SIZE):
            powderworld.dists.make_world(pw, world[b:b+1], elems, num_tasks, num_lines, num_circles, num_squares)
    elif dist == "test":
        for b in range(BATCH_SIZE):
            powderworld.dists.make_test(pw, world[b:b+1], b % 8)
    elif dist == "test160":
        for b in range(BATCH_SIZE):
            powderworld.dists.make_test160(pw, world[b:b+1], b)

    last_world = torch.argmax(world[:, :14], dim=1).clone().int()
    for t in range(TOTAL_STEP):
        if (t+1) % STEP_SIZE == 0:
            new_world = torch.argmax(world[:, :14], dim=1).clone().int()
            data_batch[ci:ci+BATCH_SIZE, 0] = last_world
            data_batch[ci:ci+BATCH_SIZE, 1] = new_world
            last_world = new_world
            ci += BATCH_SIZE
        world = pw(world)
    return data_batch

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = torch.zeros((capacity, 2, 64, 64), dtype=torch.int64, device=device)
        self.capacity = capacity
        self.size = 0
        self.idx = 0

    def append(self, data_batch):
        dlen = data_batch.shape[0]
        self.buffer[self.idx:self.idx+dlen] = data_batch
        self.idx = (self.idx+dlen) % self.capacity
        self.size = min(self.capacity, self.size + dlen)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return self.buffer[indices]

def train(savedir, device, num_elems, num_tasks, num_lines, num_circles, num_squares, step_size):
    config = {
        'num_elems': num_elems,
        'num_tasks': num_tasks,
        'num_lines': num_lines,
        'num_circles': num_circles,
        'num_squares': num_squares
    }
    print("Training with {}".format(config))
    model = WorldModel().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=.005)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    pw_device = device
    pw = PWSim(pw_device)
    pwr = PWRenderer(pw_device)
        
    def expand(x):
        x = F.one_hot(x, 14)
        x = torch.permute(x, (0,3,1,2)).float()
        return x
    
    all_elems = ['empty', 'sand', 'water', 'wall', 'plant', 'fire', 'wood', 'ice', 'lava', 'dust', 'cloner', 'gas', 'acid', 'stone']
    elems = all_elems[:num_elems]
    
    # Test on Transfer set.
    buffer_test = ReplayBuffer(1024, pw_device)
    buffer_test.append(make_data_batch(pw, elems, 'test',
        num_tasks=None, num_lines=None, num_circles=None, num_squares=None, step_size=step_size))
    idx = torch.randperm(buffer_test.buffer.shape[0])
    buffer_test.buffer = buffer_test.buffer[idx].view(buffer_test.buffer.size())
    
    # Test on Transfer 160 set.
    buffer_test160 = ReplayBuffer(1024, pw_device)
    buffer_test160.append(make_data_batch(pw, elems, 'test160',
        num_tasks=None, num_lines=None, num_circles=None, num_squares=None, step_size=step_size))
    idx = torch.randperm(buffer_test160.buffer.shape[0])
    buffer_test160.buffer = buffer_test160.buffer[idx].view(buffer_test160.buffer.size())
    
    bsize = 32
    buffer = ReplayBuffer(1024 * bsize, pw_device)
    for i in tqdm(range(bsize)):
        buffer.append(make_data_batch(pw, elems, 'train',
            num_tasks=num_tasks, num_lines=num_lines, num_circles=num_circles, num_squares=num_squares, step_size=step_size))

    BATCH_SIZE = 64
    for it in tqdm(range(5000)):
        # Make new data
        buffer.append(make_data_batch(pw, elems, 'train',
            num_tasks=num_tasks, num_lines=num_lines, num_circles=num_circles, num_squares=num_squares, step_size=step_size))
        
        # Sample training Data
        batch = buffer.sample(BATCH_SIZE)
        start = expand(batch[:, 0])
        end = batch[:, 1]
        
        # Train
        optim.zero_grad()
        out = model(start)
        loss = loss_fn(out, end)
        loss.backward()
        optim.step()
        
        log = {}
        log['loss'] = loss.item()
        
        # Debug Log
        if it % 100 == 0:
            with torch.no_grad():
                t_start = expand(buffer_test.buffer[:, 0])
                t_end = buffer_test.buffer[:, 1]
                t_out = model(t_start)
                test_loss = loss_fn(t_out, t_end).item()
                log['loss_test'] = test_loss
                
                t_start = expand(buffer_test160.buffer[:, 0])
                t_end = buffer_test160.buffer[:, 1]
                t_out = model(t_start)
                test_loss = loss_fn(t_out, t_end).item()
                log['loss_test160'] = test_loss
        
                print(log)
                
        if (it+1) % 1000 == 0:
            torch.save(model.state_dict(), '/models/wm_{}.pt'.format(savedir))


# 

import sys
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir')
    parser.add_argument('--num_elems', type=int, default=14)
    parser.add_argument('--num_tasks', type=int, default=5)
    parser.add_argument('--num_lines', type=int, default=5)
    parser.add_argument('--num_circles', type=int, default=5)
    parser.add_argument('--num_squares', type=int, default=5)
    parser.add_argument('--step_size', type=int, default=8)

    args = parser.parse_args()
    print(args)

    train(args.savedir, 'cuda', args.num_elems, args.num_tasks, args.num_lines, args.num_circles, args.num_squares, args.step_size)




