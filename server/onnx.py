import torch.onnx
import torch
import numpy as np
import powderworld.dists
from powderworld import PWSim, PWRenderer

device = 'cuda'
world = torch.zeros((1, 64, 128, 20), dtype=torch.float32, device=device)

class PwTranspose(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.pw = PWSim(device, use_jit=False)
    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.pw(x)
        return x.permute(0,2,3,1)
    
class PwrTranspose(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.pwr = PWRenderer(device)
    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.pwr(x)
        x = x.permute(1,2,0) # 64 x 64 x 3
        x = torch.cat([x, torch.ones((64,128,1)).to(device)], dim=2)
        x = (x*255).int()
        return x

print("Running PW Compile Check")
pw = PwTranspose(device)
pwr = PwrTranspose(device)


world = torch.zeros((4, pw.pw.NUM_CHANNEL, 64, 128), dtype=torch.float32, device=device)
for b in range(4):
    powderworld.dists.make_world(pw.pw, world[b:b+1], num_lines=5, num_circles=0, num_squares=0)
world = world.permute(0,2,3,1)
world2_torch = pw(torch.clone(world))
render_torch = pwr(world2_torch)

print("Compiling PW")
torch.onnx.export(pw,               # model being run
                  world,                         # model input (or a tuple for multiple inputs)
                  "pw.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

print("Compiling PW Renderer")
torch.onnx.export(pwr,               # model being run
                  world,                         # model input (or a tuple for multiple inputs)
                  "pwr.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

