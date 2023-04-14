import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import types
import torch.jit as jit
import threading
from collections import namedtuple
from typing import Dict,Tuple,Optional,List
 
Info = namedtuple('Info', ['rand_movement', 'rand_interact', 'rand_element'])

# pw_type = torch.float16
# pw_type = torch.float32

# ================ REGISTER ELEMENTS. =================
# Name:    ID, Density, GravityInter
pw_elements = {
    "empty": (0, 1,    1),
    "wall":  (1, 4,    0),
    "sand":  (2, 3,    1),
    "water": (3, 2,    1),
    "gas":   (4, 0,    1),
    "wood":  (5, 4,    0),
    "ice":   (6, 4,    0),
    "fire":  (7, 0,    1),
    "plant": (8, 4,    0),
    "stone": (9, 3,    1),
    "lava":  (10,3,    1),
    "acid":  (11,2,    1),
    "dust":  (12,2,    1),
    "cloner":(13,4,    0),
    "agentFish":        (14, 2, 1),
    "agentBird":        (15, 4, 0),
    "agentKangaroo":    (16, 3, 1),
    "agentMole":        (17, 3, 1),
    "agentLemming":     (18, 3, 1),
    "agentSnake":       (19, 4, 0),
    "agentRobot":       (20, 3, 1),
}

pw_element_names = [
    "empty",
    "wall",
    "sand",
    "water",
    "gas",
    "wood",
    "ice",
    "fire",
    "plant",
    "stone",
    "lava",
    "acid",
    "dust",
    "cloner",
    "agentFish",
    "agentBird",
    "agentKangaroo",
    "agentMole",
    "agentLemming",
    "agentSnake",
    "agentRobot"
]

# ================================================
# ============= HELPERS ==================
# ================================================

def get_below(x):
    return torch.roll(x, shifts=-1, dims=2)
def get_above(x):
    return torch.roll(x, shifts=1, dims=2)
def get_left(x):
    return torch.roll(x, shifts=1, dims=3)
def get_right(x):
    return torch.roll(x, shifts=-1, dims=3)

def get_in_cardinal_direction(x, directions):
    y = get_right(x) * (directions==0)
    y = y + get_below(x) * (directions==2)
    y = y + get_left(x) * (directions==4)
    y = y + get_above(x) * (directions==6)
    return y

@torch.jit.script # JIT decorator
def interp(switch, if_false, if_true):
    return (~switch)*if_false + (switch)*if_true

@torch.jit.script # JIT decorator
def interp_int(switch, if_false, if_true: int):
    return (~switch)*if_false + (switch)*if_true

@torch.jit.script # JIT decorator
def interp2(switch_a, switch_b, if_false, if_a, if_b):
    return ((~switch_a)&(~switch_b))*if_false + (switch_a)*if_a + (switch_b)*if_b

@torch.jit.script # JIT decorator
def interp_swaps8(swaps, world, w0, w1, w2, w3, w4, w5, w6, w7):
    new_world = world*(swaps == -1)
    new_world += w0*(swaps == 0)
    new_world += w1*(swaps == 1)
    new_world += w2*(swaps == 2)
    new_world += w3*(swaps == 3)
    new_world += w4*(swaps == 4)
    new_world += w5*(swaps == 5)
    new_world += w6*(swaps == 6)
    new_world += w7*(swaps == 7)
    return new_world

@torch.jit.script # JIT decorator
def interp_swaps4(swaps, world, w0, w1, w2, w3):
    new_world = world*(swaps == -1)
    new_world += w0*(swaps == 0)
    new_world += w1*(swaps == 1)
    new_world += w2*(swaps == 2)
    new_world += w3*(swaps == 3)
    return new_world

# ================================================
# ============= GENERAL CLASS =================
# ================================================
        

class PWSim(torch.nn.Module):
    def __init__(self, device, use_jit=True):
        with torch.no_grad():
            super().__init__()
            if isinstance(device, str):
                device = torch.device(device)
            self.device = device
            self.use_jit = use_jit

            self.elements = pw_elements
            self.element_names = pw_element_names
            
            self.NUM_ELEMENTS = len(self.elements)
            # [ElementID(0), Density(1), GravityInter(2), VelocityField(3, 4), Color(5), Custom1(6), Custom2(7), Custom3(8)]
            # Custom 1              Custom 2          Custom 3
            # BirdVelX              BirdVelY
            #                                         DidGravity
            # FluidMomentum
            # FluidMomentum         KangarooJump
            # MoleDirection
            # SnakeDirection.       SnakeEnergy
            self.NUM_CHANNEL = 1 + 1 + 1 + 2 + 1 + 3
            self.pw_type = torch.float16 if 'cuda' in device.type else torch.float32

            # ================ TORCH KERNELS =================
            self.elem_vecs = {}
            self.elem_vecs_array = nn.Embedding(self.NUM_ELEMENTS, self.NUM_CHANNEL, device=device, dtype=self.pw_type)
            for elem_name, elem in self.elements.items():
                elem_vec = torch.zeros(self.NUM_CHANNEL, device=device)
                elem_vec[0] = elem[0]
                elem_vec[1] = elem[1]
                elem_vec[2] = elem[2]
                self.elem_vecs[elem_name] = elem_vec[None, :, None, None]
                self.elem_vecs_array.weight[elem[0]] = elem_vec

            self.neighbor_kernel = torch.ones((1, 1, 3, 3), device=device, dtype=self.pw_type)
            self.zero = torch.zeros((1,1), device=device, dtype=self.pw_type)
            self.one = torch.ones((1,1), device=device, dtype=self.pw_type)

            self.up = torch.Tensor([-1,0]).to(device)[None,:,None,None]
            self.down = torch.Tensor([1,0]).to(device)[None,:,None,None]
            self.left = torch.Tensor([0,-1]).to(device)[None,:,None,None]
            self.right = torch.Tensor([0,1]).to(device)[None,:,None,None]
            
            self.register_update_rules()
    
    # ==================================================
    # ============ REGISTER UPDATE RULES ===============
    # ==================================================
    def register_update_rules(self):
        """
        Overwrite this function with your own set of update rules to change behavior.
        """
        self.update_rules = [
            BehaviorStone(self),
            BehaviorMole(self),
            BehaviorGravity(self),
            BehaviorSand(self),
            BehaviorLemming(self),
            BehaviorFluidFlow(self),
            BehaviorIce(self),
            BehaviorWater(self),
            BehaviorFire(self),
            BehaviorPlant(self),
            BehaviorLava(self),
            BehaviorAcid(self),
            BehaviorCloner(self),
            BehaviorFish(self),
            BehaviorBird(self) ,
            BehaviorKangaroo(self),
            BehaviorSnake(self),
            BehaviorVelocity(self),
        ]
        self.update_rules_jit = None
    

    # =========== WORLD EDITING HELPERS ====================
    def add_element(self, world_slice, element_name, wind=None):
        if isinstance(element_name, int):
            element_name = self.element_names[element_name]
        
        if element_name == "wind":
            world_slice[:,3:5] = wind
        else:
            world_slice[...] = self.elem_vecs[element_name]
            if element_name == "agentSnake":
                world_slice[:,7] = 1
            
    def add_element_rc(self, world_slice, rr, cc, element_name):
        if isinstance(element_name, int):
            element_name = self.element_names[element_name]
        world_slice[:,:,rr,cc] = self.elem_vecs[element_name]
        
    def id_to_pw(self, world_ids):
        with torch.no_grad():
            world = self.elem_vecs_array(world_ids)
            world = torch.permute(world, (0,3,1,2))
            return world
        
    def np_to_pw(self, np_world):
        with torch.no_grad():
            np_world_ids = torch.from_numpy(np_world).int().to(self.device)
            return self.id_to_pw(np_world_ids)

    
    # =========== UPDATE HELPERS ====================
    def get_elem(self, world, elemname):
        elem_id = self.elements[elemname][0]
        return (world[:, 0:1] == elem_id).to(self.pw_type)
    
    def get_bool(self, world, elemname):
        elem_id = self.elements[elemname][0]
        return (world[:, 0:1] == elem_id)
    
    def direction_func(self, d, x):
        if d == 0:
            return get_right(x)
        elif d == 1:
            return get_right(get_below(x))
        elif d == 2:
            return get_below(x)
        elif d == 3:
            return get_left(get_below(x))
        elif d == 4:
            return get_left(x)
        elif d == 5:
            return get_left(get_above(x))
        elif d == 6:
            return get_above(x)
        elif d == 7:
            return get_right(get_above(x))
    
    def forward(self, world, do_skips=False):
        with torch.no_grad():
            # Helper Functions
            rand_movement = torch.rand((world.shape[0], 1, world.shape[2], world.shape[3]), dtype=self.pw_type, device=self.device) # For gravity
            rand_interact = torch.rand((world.shape[0], 1, world.shape[2], world.shape[3]), dtype=self.pw_type, device=self.device) # For element-wise
            rand_element = torch.rand((world.shape[0], 1, world.shape[2], world.shape[3]), dtype=self.pw_type, device=self.device) # For self-element.

            info = (rand_movement, rand_interact, rand_element)
            
            if self.update_rules_jit is None:    
                if not self.use_jit:
                    self.update_rules_jit = self.update_rules
                else:
                    print("Slow run to compile JIT.")
                    self.update_rules_jit = []
                    for update_rule in self.update_rules:
                        self.update_rules_jit.append(torch.jit.trace(update_rule, (world, info)))
                    print("Done compiling JIT.")
            
            if do_skips:
                skips = []
                for update_rule in self.update_rules:
                    skips.append(update_rule.check_filter(world))
                for i, update_rule_jit in enumerate(self.update_rules_jit):
                    if skips[i]:
                        world = update_rule_jit(world, info)
            else:
                for update_rule in self.update_rules_jit:
                    world = update_rule(world, info)
                
        return world

# ================================================
# ============== RENDERER ========================
# ================================================
class PWRenderer(torch.nn.Module):
    def __init__(self, device):
        with torch.no_grad():
            super().__init__()
            if isinstance(device, str):
                device = torch.device(device)
            pw_type = torch.float16 if 'cuda' in device.type else torch.float32
            self.elem_vecs_array = nn.Embedding(len(pw_elements), 3, device=device)
            self.elem_vecs_array.weight.data = (torch.Tensor([
                [236, 240, 241], #EMPTY #ECF0F1
                [108, 122, 137], #WALL #6C7A89
                [243, 194, 58], #SAND #F3C23A
                [75, 119, 190], #WATER #4B77BE
                [179, 157, 219], #GAS #875F9A
                [202, 105, 36], #WOOD #CA6924
                [137, 196, 244], #ICE #89C4F4
                [249, 104, 14], #FIRE #F9680E
                [38, 194, 129], #PLANT #26C281
                [38, 67, 72], #STONE #264348
                [157, 41, 51], #LAVA #9D2933
                [176, 207, 120], #ACID #B0CF78
                [255, 179, 167], #DUST #FFB3A7
                [191, 85, 236], #CLONER #BF55EC
                [0, 229, 255], #AGENT FISH
                [61, 90, 254], #AGENT BIRD #3D5AFE
                [121, 85, 72], #AGENT KANGAROO #795548
                [56, 142, 60], #AGENT MOLE #388E3C
                [158, 157, 36], #AGENT LEMMING
                [198, 40, 40], #AGENT SNAKE 
                [224, 64, 251], #AGENT ROBOT 
            ]).to(device).to(pw_type) / 255.0)
            self.vector_color_kernel = torch.Tensor([200, 100, 100]).to(device)
            self.vector_color_kernel /= 255.0
            self.vector_color_kernel = self.vector_color_kernel[:, None, None]
    
    # ================ RENDERING ====================
    def forward(self, world):
        with torch.no_grad():
            img = self.elem_vecs_array(world[0:1, 0].int())[0].permute(2,0,1)
            velocity_field = world[0, 3:5]
            velocity_field_magnitudes = torch.norm(velocity_field, dim=0)[None]

            velocity_field_angles_raw = (1/(2*torch.pi)) * torch.acos(velocity_field[1] / (velocity_field_magnitudes+0.001))
            is_y_lessthan_zero = (velocity_field[0] < 0)
            velocity_field_angles_raw = interp(switch=is_y_lessthan_zero, if_false=velocity_field_angles_raw, if_true=(1 - velocity_field_angles_raw))
            velocity_field_angles = velocity_field_angles_raw
            
            velocity_field_colors = self.vector_color_kernel

            velocity_field_display = torch.clamp(velocity_field_magnitudes/5, 0, 0.5)
            img = (1-velocity_field_display)*img + velocity_field_display*velocity_field_colors
            img = torch.clamp(img, 0, 1)
            return img
    def render(self, world):
        img = self(world)
        img = img.detach().cpu()
        img = img.permute(1, 2, 0).numpy()
        img = (img*255).astype(np.uint8)
        return img
        

# =========================================================================
# ====================== CORE UPDATE BEHAVIORS ============================
# =========================================================================

class BehaviorGravity(torch.nn.Module):
    """
    Run gravity procedure.
    Loop through each possible density (1-5).
    In kernel form, compute for each block:
        IF density == currDensity && density BELOW is less && both gravity-affected -> Become below.
        (Else)If density ABOVE is greater && density ABOVE == currDensity && both gravity-affected -> Become above.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def check_filter(self, world):
        return True
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        
        world[:, 8:9] = interp(switch=(world[:, 2:3] == 1), if_false=world[:, 8:9], if_true=self.pw.zero)
            
        above = get_above(world)
        below = get_below(world)

        is_density_below_less = (below[:, 1:2] - world[:, 1:2] < 0)
        is_gravity = (world[:, 2:3] == 1)
        is_gravity_below = below[:, 2:3] == 1

        is_center_and_below_gravity = is_gravity_below & is_gravity
        does_become_below = is_density_below_less & is_center_and_below_gravity
        does_become_above = get_above(does_become_below)
        
        has_overlap = does_become_below & does_become_above
        does_become_below_real = does_become_below & ~has_overlap
        does_become_above_real = get_above(does_become_below_real)
        
        world[:] = interp2(switch_a=does_become_below_real, switch_b=does_become_above_real, if_false=world, if_a=below, if_b=above)
        world[:, 8:9] = interp(switch=does_become_above_real, if_false=world[:, 8:9], if_true=self.pw.one)
            
        return world
    
class BehaviorSand(torch.nn.Module):
    """
    Run sand-piling procedure. 
    Loop over each piling block type. In kernel form, for each block:
        If dir=left and BELOW_LEFT density is less && both gravity-affected -> Become below-left.
        If ABOVE_RIGHT dir=left and ABOVE_RIGHT density is less && both gravity-affected -> Become above-right.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def check_filter(self, world):
        return True
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        fall_dir = (rand_movement > 0.5)
        not_did_gravity = ~(world[:, 8:9] > 0)
        for fallLeft in [True, False]:
            get_in_dir = get_left if fallLeft else get_right
            get_in_not_dir = get_right if fallLeft else get_left
            world_below_left = get_in_dir(get_below(world))
            world_above_right = get_in_not_dir(get_above(world))

            fall_dir = (rand_movement > 0.5) if fallLeft else (rand_movement <= 0.5)
            rand_above_right = get_in_not_dir(get_above(fall_dir))

            is_below_left_density_lower = (world[:, 1:2] - world_below_left[:, 1:2]) > 0
            is_above_right_density_higher = (world_above_right[:, 1:2] - world[:, 1:2]) > 0
            is_gravity = (world[:, 2:3] == 1)
            is_below_left_gravity = world_below_left[:, 2:3] == 1
            is_above_right_gravity = world_above_right[:, 2:3] == 1
            is_element = self.pw.get_bool(world, 'sand') | self.pw.get_bool(world, 'dust')
            is_above_right_element = self.pw.get_bool(world_above_right, 'sand') | self.pw.get_bool(world_above_right, 'dust')

            is_matching_fall = fall_dir
            is_above_right_matching_fall = rand_above_right
            not_did_gravity = ~(world[:, 8:9] > 0)
            not_did_gravity_below_left = ~(world_below_left[:, 8:9] > 0)
            not_did_gravity_above_right = ~(world_above_right[:, 8:9] > 0)

            does_become_below_left = is_element & not_did_gravity_below_left & is_matching_fall & is_below_left_density_lower \
                                    & is_below_left_gravity & not_did_gravity
            does_become_above_right = is_above_right_element & not_did_gravity_above_right & is_above_right_matching_fall \
                                    & is_above_right_density_higher & is_above_right_gravity & not_did_gravity

            world[:] = interp2(switch_a=does_become_below_left, switch_b=does_become_above_right,
                              if_false=world, if_a=world_below_left, if_b=world_above_right)
        return world
    
class BehaviorStone(torch.nn.Module):
    """Run stone-stability procedure. If a stone is next to two stones, turn gravity off. Otherwise, turn it on."""
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
        self.stone_kernel = torch.zeros((1,1,3,3), device=self.pw.device, dtype=self.pw.pw_type)
        self.stone_kernel[0, 0, 0, 0] = 1
        self.stone_kernel[0, 0, 0, 2] = 1
    def check_filter(self, world):
        return True
    def forward(self, world, info):
        stone = self.pw.get_elem(world, "stone")
        has_stone_supports = F.conv2d(stone, self.stone_kernel, padding=1)
        world[:, 2:3] = (1-stone)*world[:, 2:3] + stone*(has_stone_supports < 2)
        return world
    
class BehaviorFluidFlow(torch.nn.Module):
    """
    Run fluid-flowing procedure. Same as sand-piling, but move LEFT/RIGHT instead of BELOW-LEFT/BELOW-RIGHT.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def check_filter(self, world):
        return True
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        new_fluid_momentum = torch.zeros((world.shape[0], 1, world.shape[2], world.shape[3]), dtype=self.pw.pw_type, device=self.pw.device)
        for fallLeft in [True, False]:
            get_in_dir = get_left if fallLeft else get_right
            get_in_not_dir = get_right if fallLeft else get_left
            fall_dir = ((rand_movement + world[:, 6:7] + new_fluid_momentum) > 0.5)
            is_matching_fall = fall_dir if fallLeft else (~fall_dir)
            world_left = get_in_dir(world)
            world_right = get_in_not_dir(world)

            is_air_move = self.pw.get_bool(world, "agentKangaroo") | self.pw.get_bool(world, "agentLemming")
            is_element = self.pw.get_bool(world, "empty") | self.pw.get_bool(world, "water") | self.pw.get_bool(world, "gas") | \
                self.pw.get_bool(world, "lava")  | self.pw.get_bool(world, "acid") | is_air_move
            is_left_density_lower = ((world[:, 1:2] - world_left[:, 1:2]) > 0)
            is_gravity = world[:, 2:3] == 1
            is_left_gravity = world_left[:, 2:3] == 1
            not_did_gravity_left = ~(world[:, 8:9] > 0) | is_air_move

            does_become_left = is_matching_fall & is_element & not_did_gravity_left & is_left_density_lower & is_left_gravity & is_gravity
            does_become_right = get_in_not_dir(does_become_left)
            
            has_overlap = does_become_left & does_become_right
            does_become_left_real = does_become_left & ~has_overlap
            does_become_right_real = get_in_not_dir(does_become_left_real)
            
            new_fluid_momentum += does_become_right_real * (2 if fallLeft else -2)

            world[:] = interp2(switch_a=does_become_left_real, switch_b=does_become_right_real,
                          if_false=world, if_a=world_left, if_b=world_right)

        # todo: fix this and find out why
        # OK, we know why, it's because new_fluid_momentum has excess info everywhere. we want to just update this for all fluid elements.
        world[:, 6:7, :, :] = \
            interp(switch=(self.pw.get_bool(world, "empty") | self.pw.get_bool(world, "water") | self.pw.get_bool(world, "gas") | \
                           self.pw.get_bool(world, "lava")  | self.pw.get_bool(world, "acid") | self.pw.get_bool(world, "agentKangaroo") | self.pw.get_bool(world, "agentLemming")), 
                   if_false=world[:, 6:7, :, :], 
                   if_true=new_fluid_momentum)

        return world
        
class BehaviorIce(torch.nn.Module):
    """
    Ice melting. Ice touching water or air turns to water.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def check_filter(self, world):
        return True
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        ice_chance = rand_interact
        ice_melting_neighbors = self.pw.get_elem(world, "empty") + self.pw.get_elem(world, "fire") \
            + self.pw.get_elem(world, "lava") + self.pw.get_elem(world, "water")
        ice_can_melt = (F.conv2d(ice_melting_neighbors, self.pw.neighbor_kernel, padding=1) > 1)
        does_turn_water = self.pw.get_bool(world, "ice") & ice_can_melt & (ice_chance < 0.02)
        world[:] = interp(switch=does_turn_water, if_false=world, if_true=self.pw.elem_vecs['water'])
        return world
    
class BehaviorWater(torch.nn.Module):
    """
    Water freezing. Water touching 3+ ices can turn to ice.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def check_filter(self, world):
        return True
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        ice_chance = rand_element
        water_can_freeze = (F.conv2d(self.pw.get_elem(world, "ice"), self.pw.neighbor_kernel, padding=1) >= 3)
        does_turn_ice = self.pw.get_bool(world, "water") & water_can_freeze & (ice_chance < 0.05)
        world[:] = interp(switch=does_turn_ice, if_false=world, if_true=self.pw.elem_vecs['ice'])
        return world
        
class BehaviorFire(torch.nn.Module):
    """
    Fire burning.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def check_filter(self, world):
        return True
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        burn_chance = rand_interact
        fire_and_lava = self.pw.get_elem(world, "fire") + self.pw.get_elem(world, "lava")
        has_fire_neighbor = F.conv2d(fire_and_lava, self.pw.neighbor_kernel, padding=1) > 0
        does_burn_wood = self.pw.get_bool(world, "wood") & (burn_chance < 0.05)
        does_burn_bird = self.pw.get_bool(world, "agentBird") & (burn_chance < 0.05)
        does_burn_plant = self.pw.get_bool(world, "plant") & (burn_chance < 0.2)
        does_burn_agent = (self.pw.get_bool(world, "agentFish") | self.pw.get_bool(world, "agentLemming") | self.pw.get_bool(world, "agentKangaroo") | self.pw.get_bool(world, "agentMole")) & (burn_chance < 0.2)
        does_burn_gas = self.pw.get_bool(world, "gas") & (burn_chance < 0.2)
        does_burn_dust = self.pw.get_bool(world, "dust")
        does_burn_ice = self.pw.get_bool(world, "ice") & (burn_chance < 0.2) & has_fire_neighbor
        does_burn = (does_burn_wood | does_burn_plant | does_burn_gas | does_burn_dust | does_burn_bird | does_burn_agent) & has_fire_neighbor
        
        # Velocity for fire
        world[:,3:5] -= 8*get_left(does_burn & has_fire_neighbor)*self.pw.left
        world[:,3:5] -= 8*get_above(does_burn & has_fire_neighbor)*self.pw.up
        world[:,3:5] -= 8*get_below(does_burn & has_fire_neighbor)*self.pw.down
        world[:,3:5] -= 8*get_right(does_burn & has_fire_neighbor)*self.pw.right
        
        world[:,3:5] -= 30*get_left(does_burn_dust & has_fire_neighbor)*self.pw.left
        world[:,3:5] -= 30*get_above(does_burn_dust & has_fire_neighbor)*self.pw.up
        world[:,3:5] -= 30*get_below(does_burn_dust & has_fire_neighbor)*self.pw.down
        world[:,3:5] -= 30*get_right(does_burn_dust & has_fire_neighbor)*self.pw.right
        
        world[:] = interp(switch=does_burn, if_false=world, if_true=self.pw.elem_vecs['fire'])
        world[:] = interp(switch=does_burn_ice, if_false=world, if_true=self.pw.elem_vecs['water'])

        #Fire spread. (Fire+burnable, or Lava)=> creates a probability to spread to air.
        burnables = self.pw.get_elem(world, "wood") + self.pw.get_elem(world, "plant") + self.pw.get_elem(world, "gas") + \
            self.pw.get_elem(world, "dust") + self.pw.get_bool(world, "agentFish") + self.pw.get_bool(world, "agentBird") + \
            self.pw.get_bool(world, "agentKangaroo") + self.pw.get_bool(world, "agentMole") + self.pw.get_bool(world, "agentLemming")
        fire_with_burnable_neighbor = F.conv2d(burnables, self.pw.neighbor_kernel, padding=1) * fire_and_lava
        in_fire_range = F.conv2d(fire_with_burnable_neighbor + self.pw.get_elem(world, "lava"), self.pw.neighbor_kernel, padding=1)
        does_burn_empty = self.pw.get_bool(world, "empty") & (in_fire_range > 0) & (burn_chance < 0.3)
        world[:] = interp(switch=does_burn_empty, if_false=world, if_true=self.pw.elem_vecs['fire'])

        # Fire fading. Fire just has a chance to fade, if not next to a burnable neighbor.
        fire_chance = rand_element
        has_burnable_neighbor = F.conv2d(burnables, self.pw.neighbor_kernel, padding=1)
        does_fire_turn_empty = self.pw.get_bool(world, "fire") & (fire_chance < 0.4) & (has_burnable_neighbor == 0)
        world[:] = interp(switch=does_fire_turn_empty, if_false=world, if_true=self.pw.elem_vecs['empty'])
        
        return world

        
class BehaviorPlant(torch.nn.Module):
    """
    Plants-growing. If there is water next to plant, and < 4 neighbors, chance to grow there.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def check_filter(self, world):
        return True
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        plant_chance = rand_interact
        plant_counts = F.conv2d(self.pw.get_elem(world, "plant"), self.pw.neighbor_kernel, padding=1)
        does_plantgrow = self.pw.get_bool(world, "water") & (plant_chance < 0.05)
        does_plantgrow_plant = does_plantgrow & (plant_counts <= 3) & (plant_counts >= 1)
        does_plantgrow_empty = does_plantgrow & (plant_counts > 3)
        
        wood_ice_counts = F.conv2d(self.pw.get_elem(world, "ice") + self.pw.get_elem(world, "wood"), self.pw.neighbor_kernel, padding=1)
        does_plantgrow_plant = does_plantgrow_plant | ((wood_ice_counts > 0) & (plant_chance < 0.2) & self.pw.get_bool(world, "empty") & (plant_counts > 0))
        
        world[:] = interp2(switch_a=does_plantgrow_plant, switch_b=does_plantgrow_empty,
                              if_false=world, if_a=self.pw.elem_vecs['plant'], if_b=self.pw.elem_vecs['empty'])
        return world
        
class BehaviorLava(torch.nn.Module):
    """
    Lava-water interaction. Lava that is touching water turns to stone.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def check_filter(self, world):
        return True
    def forward(self, world, info):
        water_counts = F.conv2d(self.pw.get_elem(world, "water"), self.pw.neighbor_kernel, padding=1)
        does_turn_stone = (water_counts > 0) & self.pw.get_bool(world, "lava")
        world[:] = interp(switch=does_turn_stone, if_false=world, if_true=self.pw.elem_vecs['stone'])
        
        lava_counts = F.conv2d(self.pw.get_elem(world, "lava"), self.pw.neighbor_kernel, padding=1)
        does_turn_stone = (lava_counts > 0) & self.pw.get_bool(world, "sand")
        world[:] = interp(switch=does_turn_stone, if_false=world, if_true=self.pw.elem_vecs['stone'])
        return world
        
class BehaviorAcid(torch.nn.Module):
    """
    Acid destroys everything except wall and cloner.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def check_filter(self, world):
        return True
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        acid_rand = (rand_interact < 0.2)
        is_block = ~(self.pw.get_bool(world, "empty") | self.pw.get_bool(world, "wall") | self.pw.get_bool(world, "acid") | self.pw.get_bool(world, "cloner") | self.pw.get_bool(world, "agentSnake") | self.pw.get_bool(world, "gas"))
        is_acid = self.pw.get_bool(world, "acid")
        does_acid_dissapear = (is_acid & acid_rand & get_below(is_block)) | (is_acid & acid_rand & get_above(is_block))
        does_block_dissapear = (is_block & get_above(acid_rand) & get_above(is_acid)) \
            | (is_block & get_below(acid_rand) & get_below(is_acid))
        does_dissapear = does_acid_dissapear | does_block_dissapear
        world[:] = interp(switch=does_dissapear, if_false=world, if_true=self.pw.elem_vecs['gas'])
        return world
        
class BehaviorCloner(torch.nn.Module):
    """
    Cloner keeps track of the first element it touches, and then replaces neighboring empty blocks with that element.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def check_filter(self, world):
        return self.pw.get_bool(world, "cloner").any().item()
    def forward(self, world, info):
        cloner_assigns = world[:, 6:7, :, :]
        is_not_cloner = ~self.pw.get_bool(world, "cloner")
        labels = world[:, 0:1]
        for get_dir in [get_below, get_above, get_left, get_right]:
            is_cloner_empty = self.pw.get_bool(world, "cloner") & ((cloner_assigns == 0) | (cloner_assigns == 13))
            dir_labels = get_dir(labels)
            # TODO: replace this with a switch() function that covers edge cases. When closer_assigns is twice.
            world[:, 6:7, :, :] = interp2(switch_a=is_not_cloner, switch_b=is_cloner_empty,
                                                                            if_a=cloner_assigns,
                                                                            if_b=dir_labels,
                                                                            if_false=cloner_assigns)

        # Cloner produce
        cloner_assigns_ids = torch.clamp(world[:,6], min=0, max=self.pw.NUM_ELEMENTS-1).int()
        cloner_assigns_vec = self.pw.elem_vecs_array(cloner_assigns_ids)
        cloner_assigns_vec = torch.permute(cloner_assigns_vec, (0,3,1,2))
        for get_dir in [get_below, get_above, get_left, get_right]:
            cloner_assigns_vec_dir = get_dir(cloner_assigns_vec)
            is_dir_cloner_not_empty = get_dir(self.pw.get_bool(world, "cloner") & ((cloner_assigns != 0) & (cloner_assigns != 13))) \
                & self.pw.get_bool(world, "empty")
            world[:] = interp(switch=is_dir_cloner_not_empty, if_false=world, if_true=cloner_assigns_vec_dir)
        return world

class BehaviorVelocity(torch.nn.Module):
    """
    Velocity field movement
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def check_filter(self, world):
        return (world[:, 3:5].abs() > 0.9).any().item()
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        velocity_field = world[:, 3:5]

        for n in range(2):
            velocity_field_magnitudes = torch.norm(velocity_field, dim=1)[:, None]
            velocity_field_angles_raw = (1/(2*torch.pi)) * torch.acos(velocity_field[:,1:2] / (velocity_field_magnitudes+0.001))
            is_y_lessthan_zero = (velocity_field[:,0:1] < 0)
            velocity_field_angle = interp(switch=is_y_lessthan_zero, if_false=velocity_field_angles_raw, if_true=(1 - velocity_field_angles_raw))
            velocity_field_delta = velocity_field.clone()
            velocity_angle_int = torch.remainder(torch.floor(velocity_field_angle * 8 + 0.5), 8)
            is_velocity_enough = (velocity_field_magnitudes > (1.0 if n == 0 else 2.0)) & (~self.pw.get_bool(world, "wall"))
                
            dw = []
            for angle in [0,1,2,3,4,5,6,7]:
                dw.append(self.pw.direction_func(angle, world))
                
            swaps = -torch.ones((world.shape[0], 1, world.shape[2], world.shape[3]), dtype=self.pw.pw_type, device=self.pw.device)
            for angle in [0,1,2,3,4,5,6,7]:
                direction_empty = self.pw.get_bool(dw[angle], "empty")
                direction_swap = self.pw.direction_func(angle, swaps)
                match = (velocity_angle_int == angle) & is_velocity_enough & (swaps == -1) & (direction_swap == -1) & direction_empty
                opposite_match = self.pw.direction_func((angle+4) % 8, match)
                swaps = interp_int(match, swaps, angle)
                swaps = interp_int(opposite_match, swaps, (angle+4) % 8)                
            
            velocity_field_old = velocity_field.clone()
            world[:] = interp_swaps8(swaps, world, dw[0], dw[1], dw[2], dw[3], dw[4], dw[5], dw[6], dw[7])
            world[:, 3:5] = world[:, 3:5]*0.5 + velocity_field_old * 0.5


        # Velocity field reduction
        velocity_field *= 0.95
        for i in range(1):
            velocity_field[:, 0:1] = F.conv2d(velocity_field[:, 0:1], self.pw.neighbor_kernel/18, padding=1) + velocity_field[:, 0:1]*0.5
            velocity_field[:, 1:2] = F.conv2d(velocity_field[:, 1:2], self.pw.neighbor_kernel/18, padding=1) + velocity_field[:, 1:2]*0.5
        world[:, 3:5] = velocity_field
        return world
    
    
class BehaviorFish(torch.nn.Module):
    """
    Fish move randomly.
    IF (Fish & direction) -> Become opposite direction.
    IF in opposite direction is (Fish & direction) -> become Fish.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def check_filter(self, world):
        return self.pw.get_bool(world, "agentFish").any().item()
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        
        # Small issue here: the order matters because sometimes fish move twice if they roll the correct angle.
        # We could fix it by keeping track of new fish and not allowing them to move.
        
        for angle in [0,1,2,3]:
            is_gravity = world[:, 2:3] == 1
            is_angle_match = torch.floor(rand_movement * 4) == angle
            density = world[:, 1:2]
            is_empty_in_dir = self.pw.direction_func(angle*2, is_gravity & (density <= 2))
            is_fish = self.pw.get_bool(world, "agentFish")
            opposite_world = self.pw.direction_func(angle*2, world)
            does_become_opposite = is_angle_match & is_empty_in_dir & is_fish & self.pw.direction_func(angle*2, ~is_fish) & (rand_interact < 0.2)
            does_become_fish = self.pw.direction_func(((angle*2) + 4) % 8, does_become_opposite)

            world[:] = interp2(switch_a=does_become_fish, switch_b=does_become_opposite,
                          if_false=world, if_a=self.pw.elem_vecs['agentFish'], if_b=opposite_world)

        return world
    
class BehaviorBird(torch.nn.Module):
    """
    Birds have a random velocity, and create velocity in that direction.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
        self.obstacle_kernel = torch.cat([(torch.arange(7)-3)[None, None, :, None].expand(1, 1, 7,7),
                                          (torch.arange(7)-3)[None, None, None, :].expand(1, 1, 7,7)], dim=0).to(self.pw.device).to(self.pw.pw_type)
        self.flocking_kernel = torch.ones((2,2,13,13), device=self.pw.device, dtype=self.pw.pw_type)
        self.flocking_kernel[1,0] = 0
        self.flocking_kernel[0,1] = 0
        self.flocking_kernel[:,:,3,3] = 0
    def check_filter(self, world):
        return self.pw.get_bool(world, "agentBird").any().item()
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        
        is_empty_bird_vel = (world[:, 6:7] == 0) & (world[:, 7:8] == 0)
        is_empty_bird = is_empty_bird_vel & self.pw.get_bool(world, "agentBird")
        random_dirs = torch.cat([torch.cos(rand_movement*torch.pi*2), torch.sin(rand_movement*torch.pi*2)], dim=1)
        bird_vel = world[:, 6:8].clone()
        bird_vel = interp(switch=is_empty_bird, if_false=bird_vel, if_true=random_dirs)

        
        not_empty = (~self.pw.get_bool(world, "empty")).to(self.pw.pw_type)
        vel_delta_obstacle = -F.conv2d(not_empty, self.obstacle_kernel, padding=3)
        vel_delta_flocking = 1 * F.conv2d(bird_vel * self.pw.get_elem(world, "agentBird"), self.flocking_kernel, padding=6)
        
        bird_vel += self.pw.get_elem(world, "agentBird")*(vel_delta_obstacle + vel_delta_flocking)
        bird_vel = F.normalize(bird_vel + 0.01, dim=1)
        
        original_68 = world[:, 6:8].clone()
        world[:, 3:5] += self.pw.get_elem(world, "agentBird")*bird_vel
        world[:, 6:8] = interp(self.pw.get_bool(world, "agentBird"), original_68, bird_vel)
        
        return world

class BehaviorKangaroo(torch.nn.Module):
    """
    Kangaroos move left/right and also randomly jump and pick up blocks.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def check_filter(self, world):
        return True
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        
        density = world[:, 1:2]
        fluid_momentum = world[:, 6:7]
        kangaroo_jump_state = world[:, 7:8]
        is_kangaroo = self.pw.get_bool(world, "agentKangaroo")
        
        is_kangaroo_jump = is_kangaroo & (rand_element < 0.05) & get_below(density >= 3)
        kangaroo_jump_state = interp(switch=is_kangaroo_jump, if_false=kangaroo_jump_state, if_true=self.pw.one)
        kangaroo_jump_state = interp(switch=is_kangaroo, if_false=kangaroo_jump_state, if_true=kangaroo_jump_state-0.1)
        world[:, 7:8] = kangaroo_jump_state
                
        world[:, 3:5] += (is_kangaroo & (kangaroo_jump_state > 0))*self.pw.up*4

        return world

class BehaviorMole(torch.nn.Module):
    """
    Moles burrow through solids.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def check_filter(self, world):
        return self.pw.get_bool(world, "agentMole").any().item()
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        
        beetle_dir = world[:, 6:7]
        
        is_beetle = self.pw.get_bool(world, "agentMole")
        does_beetle_dir_change = is_beetle & (rand_element < 0.1)
        new_beetle_dir = torch.floor(rand_movement * 4)
        beetle_dir = interp(switch=does_beetle_dir_change, if_false=beetle_dir, if_true=new_beetle_dir)
        world[:, 6:7] = beetle_dir
        
        density = world[:, 1:2]
        is_beetle_num = self.pw.get_elem(world, "agentMole")
        has_supports = F.conv2d((density >= 3).to(self.pw.pw_type), self.pw.neighbor_kernel, padding=1)
        world[:, 2:3] = (1-is_beetle_num)*world[:, 2:3] + is_beetle_num*(has_supports < 2)
        
                        
        dw = []
        for angle in [0,1,2,3]:
            dw.append(self.pw.direction_func(angle*2, world))
        
        for angle in [0,1,2,3]:
            is_angle_match = (beetle_dir == angle)
            is_solid_in_dir = dw[angle][:, 1:2] >= 3
            is_wall_in_dir = self.pw.get_bool(dw[angle], "wall")
            is_empty_in_dir = self.pw.get_bool(dw[angle], "empty")
            is_beetle_in_dor = self.pw.get_bool(dw[angle], "agentMole")
            does_move_in_dir = ((rand_element < 0.5) & is_solid_in_dir & ~is_wall_in_dir) | ((rand_element < 0.1) & ~is_wall_in_dir)
            does_become_opposite = is_angle_match & does_move_in_dir & is_beetle & ~is_beetle_in_dor
            does_become_beetle = self.pw.direction_func(((angle*2) + 4) % 8, does_become_opposite)
            
            resulting_world = interp(switch=is_solid_in_dir, if_false=dw[angle], if_true=self.pw.elem_vecs['empty'])

            world[:] = interp2(switch_a=does_become_beetle, switch_b=does_become_opposite,
                          if_false=world, if_a=dw[(angle + 2) % 4], if_b=resulting_world)

        return world
        
class BehaviorLemming(torch.nn.Module):
    """
    If lemmings run into a block, they move up.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def check_filter(self, world):
        return True
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        
        fluid_momentum = world[:, 6:7, :, :]
        for fallLeft in [True, False]:
            fall_dir = (fluid_momentum > 0.5)
            is_matching_fall = fall_dir if fallLeft else (~fall_dir)
            get_in_dir = get_left if fallLeft else get_right
            get_in_not_dir = get_right if fallLeft else get_left
            is_element = self.pw.get_bool(world, "agentLemming")

            density = world[:, 1:2]
            density_forward = get_in_dir(density)
            density_above = get_above(density)
            density_forward_above = get_in_dir(get_above(density))

            is_forward_density_higher = (density_forward - density) >= 0
            is_above_density_lower = (density_above - density) < 0
            is_density_forward_above_lower = (density_forward_above - density) < 0
            
            does_become_above = is_element & is_forward_density_higher & is_above_density_lower & is_density_forward_above_lower
            does_become_below = get_below(does_become_above)

            world_above = get_above(world)
            world_below = get_below(world)
            world[:] = interp2(switch_a=does_become_above, switch_b=does_become_below,
                          if_false=world, if_a=world_above, if_b=world_below)
        
        return world
    
    
class BehaviorSnake(torch.nn.Module):
    """
    [s1][s2][e]
    [s=does_become_opposite -> become snake_trail]
    [e=does_become_snake -> become swap]
    Snake is working, but we need to prevent turns into itself / turns into the wall.
    How to do so? How about don't turn if there is a wall present or there is more snake present?
    
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def check_filter(self, world):
        return self.pw.get_bool(world, "agentSnake").any().item()
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element = info
        
        snake_dir = world[:, 6:7]
        was_snake = self.pw.get_bool(world, "agentSnake").clone()
        old_snake_dir = snake_dir.clone()
        old_snake_energy = world[:, 7:8].clone()
        
        does_become_trail = torch.zeros((world.shape[0], 1, world.shape[2], world.shape[3]), device=self.pw.device, dtype=torch.bool)
        does_become_snake = torch.zeros((world.shape[0], 1, world.shape[2], world.shape[3]), device=self.pw.device, dtype=torch.bool)
        dir_snake_came_from = torch.zeros((world.shape[0], 1, world.shape[2], world.shape[3]), device=self.pw.device, dtype=self.pw.pw_type)
        ones = torch.ones((world.shape[0], 1, world.shape[2], world.shape[3]), device=self.pw.device, dtype=self.pw.pw_type)
        
        does_turn = (rand_movement < 0.1)
        
        # For each angle, if it's a snake, become trail.
        # If coming from behind is this angle -> does_become_this_angle = (It becomes a snake. There is a snake before entering the tile.)
        # If random, or infront is a wall, does_turn.
        # Dir_snake_come_from = which angle entering from?
        for angle in [0,1,2,3]:
            is_angle_match = (snake_dir == angle)
            is_snake_angle = self.pw.get_bool(world, "agentSnake") & is_angle_match        
            does_become_trail = does_become_trail | is_snake_angle
            does_become_this_angle = self.pw.direction_func(((angle*2) + 4) % 8, is_snake_angle) & ~self.pw.get_bool(world, "wall")
            does_turn = does_turn | (self.pw.direction_func(angle*2, self.pw.get_bool(world, "wall") | self.pw.get_bool(world, "agentSnake")) & does_become_this_angle)
            world[:] = interp((does_become_snake&does_become_this_angle), world, self.pw.elem_vecs['empty'])
            does_become_snake = does_become_snake | does_become_this_angle
            dir_snake_came_from = interp(switch=does_become_this_angle, if_false=dir_snake_came_from, if_true=ones*angle)    
        
        # Perform Swaps
        acid_or_not = interp(rand_element < 0.05, self.pw.elem_vecs['empty'], self.pw.elem_vecs['acid'])
        snake_trail = interp(old_snake_energy > 0, acid_or_not, self.pw.elem_vecs['agentSnake'])
        world[:] = interp(does_become_trail, if_false=world, if_true=snake_trail)
        world[:] = interp(does_become_snake, if_false=world, if_true=self.pw.elem_vecs['agentSnake'])
        
        # This is where I am about to turn in.
        turned_dir_came_from = (dir_snake_came_from + 1 - 2*(rand_element < 0.5).int()) % 4
        in_dir = get_in_cardinal_direction(world, turned_dir_came_from*2)
        # BUT, don't turn if there is a snake or wall in that direction!
        does_turn = does_turn & ~self.pw.get_bool(in_dir, "agentSnake") & ~self.pw.get_bool(in_dir, "wall")
        
        is_snake = self.pw.get_bool(world, "agentSnake")
        dir_snake_came_from = interp(switch=does_turn, if_false=dir_snake_came_from, if_true=turned_dir_came_from)
        new_snake_dir = interp(switch=(is_snake & ~was_snake), if_false=old_snake_dir, if_true=dir_snake_came_from)
        
        world[:, 6:7] = interp(is_snake, world[:, 6:7], new_snake_dir)
        world[:, 7:8] = interp(is_snake, world[:, 7:8], old_snake_energy-0.1)

        return world