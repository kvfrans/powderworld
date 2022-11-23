import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import types
import torch.jit as jit
from collections import namedtuple
 
Info = namedtuple('Info', ['rand_movement', 'rand_interact', 'rand_element', 'velocity_field', 'did_gravity'])

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

@torch.jit.script # JIT decorator
def interp(switch, if_false, if_true):
    return (~switch)*if_false + (switch)*if_true

@torch.jit.script # JIT decorator
def interp2(switch_a, switch_b, if_false, if_a, if_b):
    return ((~switch_a)&(~switch_b))*if_false + (switch_a)*if_a + (switch_b)*if_b

# ================================================
# ============= GENERAL CLASS =================
# ================================================

class PWSim(torch.nn.Module):
    def __init__(self, device, use_jit=True):
        with torch.no_grad():
            super().__init__()
            self.device = device
            self.use_jit = use_jit

            # ================ REGISTER ELEMENTS. =================
            # Name:    ID, Density, GravityInter
            self.elements = {
                "empty": (0, 1,    1),
                "wall":  (1, 5,    0),
                "sand":  (2, 4,    1),
                "water": (3, 3,    1),
                "gas":   (4, 0,    1),
                "wood":  (5, 5,    0),
                "ice":   (6, 5,    0),
                "fire":  (7, 0,    1),
                "plant": (8, 5,    0),
                "stone": (9, 4,    1),
                "lava":  (10,4,    1),
                "acid":  (11,3,    1),
                "dust":  (12,2,    1),
                "cloner":(13,5,    0),
            }
            self.element_names = [
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
                "cloner",]
            self.NUM_ELEMENTS = len(self.elements)
            # [ElementID(14), Density(1), GravityInter(1), FluidMomentum(1), ClonerID(1), VelocityField(2)] = 20
            self.NUM_CHANNEL = self.NUM_ELEMENTS + 1 + 1 + 1 + 1 + 2

            # ================ TORCH KERNELS =================
            self.elem_vecs = {}
            self.elem_vecs_array = nn.Embedding(self.NUM_ELEMENTS, self.NUM_CHANNEL, device=device)
            for elem_name, elem in self.elements.items():
                elem_vec = torch.zeros(self.NUM_CHANNEL, device=device)
                elem_vec[elem[0]] = 1
                elem_vec[self.NUM_ELEMENTS] = elem[1]
                elem_vec[self.NUM_ELEMENTS+1] = elem[2]
                self.elem_vecs[elem_name] = elem_vec[None, :, None, None]
                self.elem_vecs_array.weight[elem[0]] = elem_vec

            self.neighbor_kernel = torch.ones((1, 1, 3, 3), device=device)

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
            BehaviorGravity(self),
            BehaviorSand(self),
            BehaviorFluidFlow(self),
            BehaviorIce(self),
            BehaviorWater(self),
            BehaviorFire(self),
            BehaviorPlant(self),
            BehaviorLava(self),
            BehaviorAcid(self),
            BehaviorCloner(self),
            BehaviorVelocity(self),
        ]
        self.update_rules_jit = None
    

    # =========== WORLD EDITING HELPERS ====================
    def add_element(self, world_slice, element_name, wind=None):
        if isinstance(element_name, int):
            element_name = self.element_names[element_name]
        
        if element_name == "wind":
            world_slice[:,self.NUM_ELEMENTS+4:self.NUM_ELEMENTS+6] = wind
        else:
            elem_id, elem_dens, elem_grav = self.elements[element_name]
            elemnt_vec = torch.zeros(self.NUM_CHANNEL, device=world_slice.device)
            elemnt_vec[elem_id] = 1
            elemnt_vec[self.NUM_ELEMENTS] = elem_dens
            elemnt_vec[self.NUM_ELEMENTS+1] = elem_grav
            world_slice[...] = elemnt_vec[None, :, None, None]
            
    def add_element_rc(self, world_slice, rr, cc, element_name):
        if isinstance(element_name, int):
            element_name = self.element_names[element_name]
            
        elem_id, elem_dens, elem_grav = self.elements[element_name]
        elemnt_vec = torch.zeros(self.NUM_CHANNEL, device=world_slice.device)
        elemnt_vec[elem_id] = 1
        elemnt_vec[self.NUM_ELEMENTS] = elem_dens
        elemnt_vec[self.NUM_ELEMENTS+1] = elem_grav
        world_slice[:,:,rr,cc] = elemnt_vec[None, :, None]
    
    
    # =========== UPDATE HELPERS ====================
    def get_elem(self, world, elemname):
        elemid = self.elements[elemname][0]
        return(world[:, elemid:elemid+1])
    
    def get_bool(self, world, elemname):
        elemid = self.elements[elemname][0]
        return(world[:, elemid:elemid+1] == 1)
    
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
    
    def forward(self, world):
        with torch.no_grad():
            # Helper Functions
            rand_movement = torch.rand((world.shape[0], 1, world.shape[2], world.shape[3]), device=self.device) # For gravity, flowing.
            rand_interact = torch.rand((world.shape[0], 1, world.shape[2], world.shape[3]), device=self.device) # For element-wise interactions.
            rand_element = torch.rand((world.shape[0], 1, world.shape[2], world.shape[3]), device=self.device) # For self-element behavior.
            velocity_field = torch.clone(world[:, self.NUM_ELEMENTS+4:self.NUM_ELEMENTS+6])
            did_gravity = torch.zeros((world.shape[0], 1, world.shape[2], world.shape[3]), device=self.device)
            
            info = Info(rand_movement, rand_interact, rand_element, velocity_field, did_gravity)
            
            if self.update_rules_jit is None:            
                if not self.use_jit:
                    self.update_rules_jit = self.update_rules
                else:
                    print("Slow run to compile JIT.")
                    self.update_rules_jit = []
                    for update_rule in self.update_rules:
                        self.update_rules_jit.append(torch.jit.trace(update_rule, (world, info)))
            
            for update_rule in self.update_rules_jit:
                world, info = update_rule(world, info)
        return world
        

# ================================================
# ============== RENDERER ========================
# ================================================
class PWRenderer(torch.nn.Module):
    def __init__(self, device):
        with torch.no_grad():
            super().__init__()
            self.color_kernel = torch.Tensor([
                [236, 240, 241], #EMPTY #ECF0F1
                [108, 122, 137], #WALL #6C7A89
                [243, 194, 58], #SAND #F3C23A
                [75, 119, 190], #WATER #4B77BE
                [135, 95, 154], #GAS #875F9A
                [202, 105, 36], #WOOD #CA6924
                [137, 196, 244], #ICE #89C4F4
                [249, 104, 14], #FIRE #F9680E
                [38, 194, 129], #PLANT #26C281
                [38, 67, 72], #STONE #264348
                [157, 41, 51], #LAVA #9D2933
                [176, 207, 120], #ACID #B0CF78
                [255, 179, 167], #DUST #FFB3A7
                [191, 85, 236], #CLONER #BF55EC
            ]).to(device)
            self.color_kernel /= 255.0
            self.color_kernel = self.color_kernel.T[:, :, None, None]
#             self.vector_color_kernel = torch.Tensor([
#                 [68, 1, 84],
#                 [64, 67, 135],
#                 [41, 120, 142],
#                 [34, 167, 132],
#                 [121, 209, 81],
#                 [253, 231, 36],
#                 [68, 1, 84],
#             ]).to(device)
            self.vector_color_kernel = torch.Tensor([200, 100, 100]).to(device)
            self.vector_color_kernel /= 255.0
            self.vector_color_kernel = self.vector_color_kernel[:, None, None]
            self.NUM_ELEMENTS = self.color_kernel.shape[1]
    
    # ================ RENDERING ====================
    def forward(self, world):
        with torch.no_grad():
            img = F.conv2d(world[:, :self.NUM_ELEMENTS], self.color_kernel)[0]
            velocity_field = world[0, self.NUM_ELEMENTS+4:self.NUM_ELEMENTS+6]
            velocity_field_magnitudes = torch.norm(velocity_field, dim=0)[None]

            velocity_field_angles_raw = (1/(2*torch.pi)) * torch.acos(velocity_field[1] / (velocity_field_magnitudes+0.001))
            is_y_lessthan_zero = (velocity_field[0] < 0)
            velocity_field_angles_raw = interp(switch=is_y_lessthan_zero, if_false=velocity_field_angles_raw, if_true=(1 - velocity_field_angles_raw))
            velocity_field_angles = velocity_field_angles_raw
            
            velocity_field_colors = self.vector_color_kernel
#             velocity_field_colors = torch.zeros_like(img)
#             for c in range(7):
#                 velocity_field_colors += self.vector_color_kernel[c] * torch.clamp(1 - 7*torch.abs((velocity_field_angles - c/6)), 0, 1)

            velocity_field_display = torch.clamp(velocity_field_magnitudes/2, 0, 0.5)
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
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element, velocity_field, did_gravity = info
        for currDensity in [0,1,2,3,4]:
            density = world[:, self.pw.NUM_ELEMENTS:self.pw.NUM_ELEMENTS+1]
            density_delta = get_above(density) - density # Delta between ABOVE and current
            is_density_above_greater = (density_delta > 0)
            is_density_below_less = get_below(is_density_above_greater) # If BELOW has density_above_greater, then density_below_less
            is_density_current = (density == currDensity)
            is_density_above_current = get_above(is_density_current)
            is_gravity = (world[:, self.pw.NUM_ELEMENTS+1:self.pw.NUM_ELEMENTS+2] == 1)
            is_center_and_below_gravity = get_below(is_gravity) & is_gravity
            is_center_and_above_gravity = get_above(is_gravity) & is_gravity

            # These should never be both true for the same block
            does_become_below = is_density_current & is_density_below_less & is_center_and_below_gravity
            does_become_above = is_density_above_greater & is_density_above_current & is_center_and_above_gravity

            did_gravity += does_become_above

            world_above = get_above(world)
            world_below = get_below(world)
            world[:] = interp2(switch_a=does_become_below, switch_b=does_become_above,
                              if_false=world, if_a=world_below, if_b=world_above)
        info = Info(rand_movement, rand_interact, rand_element, velocity_field, did_gravity)
        return world, info
    
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
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element, velocity_field, did_gravity = info
        for elem_name in ['sand', 'dust']:
            elem = self.pw.element_names.index(elem_name)
            fall_dir = (rand_movement > 0.5)
            not_did_gravity = (did_gravity <= 0)
            for fallLeft in [True, False]:
                get_in_dir = get_left if fallLeft else get_right
                get_in_not_dir = get_right if fallLeft else get_left
                is_element = (world[:, elem:elem+1] == 1)
                is_above_right_element = get_in_not_dir(get_above(is_element))
                density = world[:, self.pw.NUM_ELEMENTS:self.pw.NUM_ELEMENTS+1]
                is_matching_fall = fall_dir if fallLeft else (~fall_dir)
                is_above_right_matching_fall = get_in_not_dir(get_above(is_matching_fall))
                is_below_left_density_lower = ((density - get_in_dir(get_below(density))) > 0)
                is_above_right_density_higher = ((get_in_not_dir(get_above(density)) - density) > 0)
                is_gravity = (world[:, self.pw.NUM_ELEMENTS+1:self.pw.NUM_ELEMENTS+2] == 1)
                is_below_left_gravity = get_in_dir(get_below(is_gravity)) & is_gravity
                is_above_right_gravity = get_in_not_dir(get_above(is_gravity)) & is_gravity
                not_did_gravity_below_left = get_in_dir(get_below(not_did_gravity)) & not_did_gravity
                not_did_gravity_above_right = get_in_not_dir(get_above(not_did_gravity)) & not_did_gravity

                does_become_below_left = is_element & not_did_gravity_below_left & is_matching_fall & is_below_left_density_lower & is_below_left_gravity
                does_become_above_right = is_above_right_element & not_did_gravity_above_right & is_above_right_matching_fall \
                                        & is_above_right_density_higher & is_above_right_gravity

                world_below_left = get_in_dir(get_below(world))
                world_above_right = get_in_not_dir(get_above(world))

                world[:] = interp2(switch_a=does_become_below_left, switch_b=does_become_above_right,
                              if_false=world, if_a=world_below_left, if_b=world_above_right)
        info = Info(rand_movement, rand_interact, rand_element, velocity_field, did_gravity)
        return world, info
    
class BehaviorStone(torch.nn.Module):
    """Run stone-stability procedure. If a stone is next to two stones, turn gravity off. Otherwise, turn it on."""
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
        self.stone_kernel = torch.zeros((1,1,3,3), device=self.pw.device)
        self.stone_kernel[0, 0, 0, 0] = 1
        self.stone_kernel[0, 0, 0, 2] = 1
    def forward(self, world, info):
        stone = self.pw.get_elem(world, "stone")
        has_stone_supports = F.conv2d(stone, self.stone_kernel, padding=1)
        world[:, self.pw.NUM_ELEMENTS+1:self.pw.NUM_ELEMENTS+2] = \
            (1-stone)*world[:, self.pw.NUM_ELEMENTS+1:self.pw.NUM_ELEMENTS+2] + stone*(has_stone_supports < 2)
        return world, info
    
class BehaviorFluidFlow(torch.nn.Module):
    """
    Run fluid-flowing procedure. Same as sand-piling, but move LEFT/RIGHT instead of BELOW-LEFT/BELOW-RIGHT.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element, velocity_field, did_gravity = info
        fluid_momentum = world[:, self.pw.NUM_ELEMENTS+2:self.pw.NUM_ELEMENTS+3, :, :]
        new_fluid_momentum = torch.zeros((world.shape[0], 1, world.shape[2], world.shape[3]), device=self.pw.device)
        fall_rand = rand_movement
        for elem_name in ['empty', 'water', 'gas', 'lava', 'acid']:
            elem = self.pw.element_names.index(elem_name)
            for fallLeft in [True, False]:
                fall_dir = ((fall_rand + fluid_momentum + new_fluid_momentum) > 0.5)
                not_did_gravity = (did_gravity <= 0)
                get_in_dir = get_left if fallLeft else get_right
                get_in_not_dir = get_right if fallLeft else get_left
                is_element = (world[:, elem:elem+1] == 1)
                is_right_element = get_in_not_dir(is_element)
                density = world[:, self.pw.NUM_ELEMENTS:self.pw.NUM_ELEMENTS+1]
                is_matching_fall = fall_dir if fallLeft else (~fall_dir)
                is_right_matching_fall = get_in_not_dir(is_matching_fall)
                is_left_density_lower = ((density - get_in_dir(density)) > 0)
                is_right_density_higher = ((get_in_not_dir(density) - density) > 0)
                is_gravity = world[:, self.pw.NUM_ELEMENTS+1:self.pw.NUM_ELEMENTS+2] == 1
                is_left_gravity = get_in_dir(is_gravity) & is_gravity
                is_right_gravity = get_in_not_dir(is_gravity) & is_gravity
                not_did_gravity_left = not_did_gravity
                not_did_gravity_right = get_in_not_dir(not_did_gravity)

                does_become_left = is_matching_fall & is_element & not_did_gravity_left & is_left_density_lower & is_left_gravity
                does_become_right = is_right_matching_fall & is_right_element & not_did_gravity_right & is_right_density_higher & is_right_gravity

                new_fluid_momentum += does_become_right * (2 if fallLeft else -2)

                world_left = get_in_dir(world)
                world_right = get_in_not_dir(world)
                world[:] = interp2(switch_a=does_become_left, switch_b=does_become_right,
                              if_false=world, if_a=world_left, if_b=world_right)

        world[:, self.pw.NUM_ELEMENTS+2:self.pw.NUM_ELEMENTS+3, :, :] = new_fluid_momentum
        info = Info(rand_movement, rand_interact, rand_element, velocity_field, did_gravity)
        return world, info
        
class BehaviorIce(torch.nn.Module):
    """
    Ice melting. Ice touching water or air turns to water.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element, velocity_field, did_gravity = info
        ice_chance = rand_interact
        ice_melting_neighbors = self.pw.get_elem(world, "empty") + self.pw.get_elem(world, "fire") \
            + self.pw.get_elem(world, "lava") + self.pw.get_elem(world, "water")
        ice_can_melt = (F.conv2d(ice_melting_neighbors, self.pw.neighbor_kernel, padding=1) > 1)
        does_turn_water = self.pw.get_bool(world, "ice") & ice_can_melt & (ice_chance < 0.02)
        world[:] = interp(switch=does_turn_water, if_false=world, if_true=self.pw.elem_vecs['water'])
        info = Info(rand_movement, rand_interact, rand_element, velocity_field, did_gravity)
        return world, info
    
class BehaviorWater(torch.nn.Module):
    """
    Water freezing. Water touching 3+ ices can turn to ice.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element, velocity_field, did_gravity = info
        ice_chance = rand_element
        water_can_freeze = (F.conv2d(self.pw.get_elem(world, "ice"), self.pw.neighbor_kernel, padding=1) >= 3)
        does_turn_ice = self.pw.get_bool(world, "water") & water_can_freeze & (ice_chance < 0.05)
        world[:] = interp(switch=does_turn_ice, if_false=world, if_true=self.pw.elem_vecs['ice'])
        info = Info(rand_movement, rand_interact, rand_element, velocity_field, did_gravity)
        return world, info
        
class BehaviorFire(torch.nn.Module):
    """
    Fire burning.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element, velocity_field, did_gravity = info
        burn_chance = rand_interact
        fire_and_lava = self.pw.get_elem(world, "fire") + self.pw.get_elem(world, "lava")
        has_fire_neighbor = F.conv2d(fire_and_lava, self.pw.neighbor_kernel, padding=1) > 0
        does_burn_wood = self.pw.get_bool(world, "wood") & (burn_chance < 0.05)
        does_burn_plant = self.pw.get_bool(world, "plant") & (burn_chance < 0.2)
        does_burn_gas = self.pw.get_bool(world, "gas") & (burn_chance < 0.2)
        does_burn_dust = self.pw.get_bool(world, "dust")
        does_burn_ice = self.pw.get_bool(world, "ice") & (burn_chance < 0.2) & has_fire_neighbor
        does_burn = (does_burn_wood | does_burn_plant | does_burn_gas | does_burn_dust) & has_fire_neighbor
        
        # Velocity for fire
        velocity_field -= 2*get_left(does_burn & has_fire_neighbor)*self.pw.left
        velocity_field -= 2*get_above(does_burn & has_fire_neighbor)*self.pw.up
        velocity_field -= 2*get_below(does_burn & has_fire_neighbor)*self.pw.down
        velocity_field -= 2*get_right(does_burn & has_fire_neighbor)*self.pw.right
        
        
        velocity_field -= 20*get_left(does_burn_dust & has_fire_neighbor)*self.pw.left
        velocity_field -= 20*get_above(does_burn_dust & has_fire_neighbor)*self.pw.up
        velocity_field -= 20*get_below(does_burn_dust & has_fire_neighbor)*self.pw.down
        velocity_field -= 20*get_right(does_burn_dust & has_fire_neighbor)*self.pw.right
        
        world[:] = interp(switch=does_burn, if_false=world, if_true=self.pw.elem_vecs['fire'])
        world[:] = interp(switch=does_burn_ice, if_false=world, if_true=self.pw.elem_vecs['water'])

        #Fire spread. (Fire+burnable, or Lava)=> creates a probability to spread to air.
        burnables = self.pw.get_elem(world, "wood") + self.pw.get_elem(world, "plant") + self.pw.get_elem(world, "gas") + self.pw.get_elem(world, "dust")
        fire_with_burnable_neighbor = F.conv2d(burnables, self.pw.neighbor_kernel, padding=1) * fire_and_lava
        in_fire_range = F.conv2d(fire_with_burnable_neighbor + self.pw.get_elem(world, "lava"), self.pw.neighbor_kernel, padding=1)
        does_burn_empty = self.pw.get_bool(world, "empty") & (in_fire_range > 0) & (burn_chance < 0.3)
        world[:] = interp(switch=does_burn_empty, if_false=world, if_true=self.pw.elem_vecs['fire'])

        # Fire fading. Fire just has a chance to fade, if not next to a burnable neighbor.
        fire_chance = rand_element
        has_burnable_neighbor = F.conv2d(burnables, self.pw.neighbor_kernel, padding=1)
        does_fire_turn_empty = self.pw.get_bool(world, "fire") & (fire_chance < 0.4) & (has_burnable_neighbor == 0)
        world[:] = interp(switch=does_fire_turn_empty, if_false=world, if_true=self.pw.elem_vecs['empty'])
        
        info = Info(rand_movement, rand_interact, rand_element, velocity_field, did_gravity)
        return world, info

        
class BehaviorPlant(torch.nn.Module):
    """
    Plants-growing. If there is water next to plant, and < 4 neighbors, chance to grow there.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element, velocity_field, did_gravity = info
        plant_chance = rand_interact
        plant_counts = F.conv2d(self.pw.get_elem(world, "plant"), self.pw.neighbor_kernel, padding=1)
        does_plantgrow = self.pw.get_bool(world, "water") & (plant_chance < 0.05)
        does_plantgrow_plant = does_plantgrow & (plant_counts <= 3) & (plant_counts >= 1)
        does_plantgrow_empty = does_plantgrow & (plant_counts > 3)
        world[:] = interp2(switch_a=does_plantgrow_plant, switch_b=does_plantgrow_empty,
                              if_false=world, if_a=self.pw.elem_vecs['plant'], if_b=self.pw.elem_vecs['empty'])
        info = Info(rand_movement, rand_interact, rand_element, velocity_field, did_gravity)
        return world, info
        
class BehaviorLava(torch.nn.Module):
    """
    Lava-water interaction. Lava that is touching water turns to stone.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def forward(self, world, info):
        water_counts = F.conv2d(self.pw.get_elem(world, "water"), self.pw.neighbor_kernel, padding=1)
        does_turn_stone = (water_counts > 0) & self.pw.get_bool(world, "lava")
        world[:] = interp(switch=does_turn_stone, if_false=world, if_true=self.pw.elem_vecs['stone'])
        return world, info
        
class BehaviorAcid(torch.nn.Module):
    """
    Acid destroys everything except wall and cloner.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element, velocity_field, did_gravity = info
        acid_rand = (rand_interact < 0.2)
        is_block = ~(self.pw.get_bool(world, "empty") | self.pw.get_bool(world, "wall") | self.pw.get_bool(world, "acid") | self.pw.get_bool(world, "cloner"))
        is_acid = self.pw.get_bool(world, "acid")
        does_acid_dissapear = (is_acid & acid_rand & get_below(is_block)) | (is_acid & acid_rand & get_above(is_block))
        does_block_dissapear = (is_block & get_above(acid_rand) & get_above(is_acid)) \
            | (is_block & get_below(acid_rand) & get_below(is_acid))
        does_dissapear = does_acid_dissapear | does_block_dissapear
        world[:] = interp(switch=does_dissapear, if_false=world, if_true=self.pw.elem_vecs['empty'])
        info = Info(rand_movement, rand_interact, rand_element, velocity_field, did_gravity)
        return world, info
        
class BehaviorCloner(torch.nn.Module):
    """
    Cloner keeps track of the first element it touches, and then replaces neighboring empty blocks with that element.
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def forward(self, world, info):
        cloner_assigns = world[:,self.pw.NUM_ELEMENTS+3:self.pw.NUM_ELEMENTS+4]
        is_not_cloner = ~self.pw.get_bool(world, "cloner")
        labels = torch.argmax(world[:, :self.pw.NUM_ELEMENTS], dim=1)[:, None]
        for get_dir in [get_below, get_above, get_left, get_right]:
            is_cloner_empty = self.pw.get_bool(world, "cloner") & ((cloner_assigns == 0) | (cloner_assigns == 13))
            dir_labels = get_dir(labels)
            world[:,self.pw.NUM_ELEMENTS+3:self.pw.NUM_ELEMENTS+4] = (~(is_cloner_empty|is_not_cloner))*cloner_assigns + is_cloner_empty*dir_labels

        # Cloner produce
        cloner_assigns_vec = self.pw.elem_vecs_array(world[:,self.pw.NUM_ELEMENTS+3].int())
        cloner_assigns_vec = torch.permute(cloner_assigns_vec, (0,3,1,2))
        for get_dir in [get_below, get_above, get_left, get_right]:
            cloner_assigns_vec_dir = get_dir(cloner_assigns_vec)
            is_dir_cloner_not_empty = get_dir(self.pw.get_bool(world, "cloner") & ((cloner_assigns != 0) & (cloner_assigns != 13))) \
                & self.pw.get_bool(world, "empty")
            world[:] = interp(switch=is_dir_cloner_not_empty, if_false=world, if_true=cloner_assigns_vec_dir)
        return world, info


@torch.jit.script # JIT decorator
def interp_vel(switch_a, switch_b, world, if_a, if_b):
    return world - switch_a*world - switch_b*world + switch_a*if_a + switch_b*if_b

class BehaviorVelocity(torch.nn.Module):
    """
    Velocity field movement
    """
    def __init__(self, pw):
        super().__init__()
        self.pw = pw
    def forward(self, world, info):
        rand_movement, rand_interact, rand_element, velocity_field, did_gravity = info
        for n in range(2):
            velocity_field_magnitudes = torch.norm(velocity_field, dim=1)[:, None]
            velocity_field_angles_raw = (1/(2*torch.pi)) * torch.acos(velocity_field[:,1:2] / (velocity_field_magnitudes+0.001))
            is_y_lessthan_zero = (velocity_field[:,0:1] < 0)
            velocity_field_angle = interp(switch=is_y_lessthan_zero, if_false=velocity_field_angles_raw, if_true=(1 - velocity_field_angles_raw))
            velocity_field_delta = velocity_field.clone()
            
            for angle in [0,1,2,3,4,5,6,7]:
                is_angle_match = (torch.remainder(torch.floor(velocity_field_angle * 8 + 0.5), 8) == angle)
                is_velocity_enough = (velocity_field_magnitudes > 0.1)
                is_empty_in_dir = self.pw.direction_func(angle, self.pw.get_bool(world, "empty")) & (~self.pw.get_bool(world, "wall"))

                does_become_empty = is_angle_match & is_velocity_enough & is_empty_in_dir
                does_become_opposite = self.pw.direction_func((angle+4) % 8, does_become_empty)
                opposite_world = self.pw.direction_func((angle+4) % 8, world)

                world[:] = interp_vel(does_become_empty, does_become_opposite, world, self.pw.elem_vecs['empty'], opposite_world)

                angle_velocity_field = self.pw.direction_func(angle, velocity_field)
                opposite_velocity_field = self.pw.direction_func((angle+4) % 8, velocity_field)
                
                velocity_field_delta[:] -= does_become_empty*velocity_field*0.5
                velocity_field_delta[:] += does_become_opposite*opposite_velocity_field*0.5
            velocity_field = velocity_field_delta


        # Velocity field reduction
        velocity_field *= 0.95
        for i in range(1):
            velocity_field[:, 0:1] = F.conv2d(velocity_field[:, 0:1], self.pw.neighbor_kernel/18, padding=1) + velocity_field[:, 0:1]*0.5
            velocity_field[:, 1:2] = F.conv2d(velocity_field[:, 1:2], self.pw.neighbor_kernel/18, padding=1) + velocity_field[:, 1:2]*0.5
        world[:, self.pw.NUM_ELEMENTS+4:self.pw.NUM_ELEMENTS+6] = velocity_field
        info = Info(rand_movement, rand_interact, rand_element, velocity_field, did_gravity)
        return world, info