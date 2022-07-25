import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import types
import torch.jit as jit

# ============= HELPERS ==================
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

class PowderWorld(torch.nn.Module):
    def __init__(self, device):
        with torch.no_grad():
            super().__init__()
            self.device = device

            # ================ BASIC VARIABLE STUFF =================
            # Name:    ID, Density, GravityInter
            self.elements = {
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
                "dust":  (12,3,    1),
                "cloner":(13,4,    0),
            }
            self.NUM_ELEMENTS = len(self.elements)
            # [ElementID(N), Density(1), GravityInter(1), FluidMomentum(1), ClonerID(1)]
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
            self.stone_kernel = torch.zeros((1,1,3,3), device=device)
            self.stone_kernel[0, 0, 0, 0] = 1
            self.stone_kernel[0, 0, 0, 2] = 1

            self.up = torch.Tensor([0,-1]).to(device)[None,:,None,None]
            self.left = torch.Tensor([-1,0]).to(device)[None,:,None,None]
            self.right = torch.Tensor([1,0]).to(device)[None,:,None,None]

            # ================ FOR RENDERING =================
            self.color_kernel = torch.Tensor([
                [236, 240, 241], #EMPTY
                [108, 122, 137], #WALL
                [243, 194, 58], #SAND
                [75, 119, 190], #WATER
                [135, 95, 154], #GAS
                [202, 105, 36], #WOOD
                [137, 196, 244], #ICE
                [249, 104, 14], #FIRE
                [38, 194, 129], #PLANT
                [38, 67, 72], #STONE
                [157, 41, 51], #LAVA
                [176, 207, 120], #ACID
                [255, 179, 167], #DUST
                [191, 85, 236], #CLONER
            ]).to(device)
            self.color_kernel /= 255.0
            self.color_kernel = self.color_kernel.T[:, :, None, None]
            self.vector_color_kernel = torch.Tensor([
                [68, 1, 84],
                [64, 67, 135],
                [41, 120, 142],
                [34, 167, 132],
                [121, 209, 81],
                [253, 231, 36],
                [68, 1, 84],
            ]).to(device)
            self.vector_color_kernel /= 255.0
            self.vector_color_kernel = self.vector_color_kernel[:, :, None, None]


    # ================ RENDERING ====================
    def render(self, world):
        img = F.conv2d(world[:, :self.NUM_ELEMENTS], self.color_kernel)[0]
        velocity_field = world[0, self.NUM_ELEMENTS+4:self.NUM_ELEMENTS+6]
        velocity_field_magnitudes = torch.linalg.vector_norm(velocity_field, dim=0)[None]

        velocity_field_angles_raw = (1/(2*torch.pi)) * torch.acos(velocity_field[1] / (velocity_field_magnitudes+0.001))
        is_y_lessthan_zero = (velocity_field[0] < 0)
        velocity_field_angles_raw = interp(switch=is_y_lessthan_zero, if_false=velocity_field_angles_raw, if_true=(1 - velocity_field_angles_raw))
        velocity_field_angles = velocity_field_angles_raw
        velocity_field_colors = torch.zeros_like(img)
        for c in range(7):
            velocity_field_colors += self.vector_color_kernel[c] * torch.clamp(1 - 7*torch.abs((velocity_field_angles - c/6)), 0, 1)

        velocity_field_display = torch.clamp(velocity_field_magnitudes/2, 0, 0.5)
        img = (1-velocity_field_display)*img + velocity_field_display*velocity_field_colors
        img = torch.clamp(img, 0, 1)
        img = img.detach().cpu()
        img = img.permute(1, 2, 0).numpy()
        img = (img*255).astype(np.uint8)
        return img

    # =========== WORLD EDITING HELPERS ====================
    def add_element(self, world_slice, element_name, wind=None):
        if element_name == "wind":
            world_slice[:,self.NUM_ELEMENTS+4:self.NUM_ELEMENTS+6] = wind
        else:
            elem_id, elem_dens, elem_grav = self.elements[element_name]
            elemnt_vec = torch.zeros(self.NUM_CHANNEL, device=self.device)
            elemnt_vec[elem_id] = 1
            elemnt_vec[self.NUM_ELEMENTS] = elem_dens
            elemnt_vec[self.NUM_ELEMENTS+1] = elem_grav
            world_slice[...] = elemnt_vec[None, :, None, None]
    
    
    # =========== POWDERWORLD STEP FUNCTION ====================
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
            
            # Run stone-stability procedure. If a stone is next to two stones, turn gravity off. Otherwise, turn it on.
            stone = self.get_elem(world, "stone")
            has_stone_supports = F.conv2d(stone, self.stone_kernel, padding=1)
            world[:, self.NUM_ELEMENTS+1:self.NUM_ELEMENTS+2] = \
                (1-stone)*world[:, self.NUM_ELEMENTS+1:self.NUM_ELEMENTS+2] + stone*(has_stone_supports < 2)

            # Run gravity procedure.
            # Loop through each possible density (1-5).
            # In kernel form, compute for each block:
                # IF density == currDensity && density BELOW is less && both gravity-affected -> Become below.
                # (Else)If density ABOVE is greater && density ABOVE == currDensity && both gravity-affected -> Become above.
            did_gravity = torch.zeros((world.shape[0], 1, world.shape[2], world.shape[3]), device=self.device)
            for currDensity in [0,1,2,3]:
                density = world[:, self.NUM_ELEMENTS:self.NUM_ELEMENTS+1]
                density_delta = get_above(density) - density # Delta between ABOVE and current
                is_density_above_greater = (density_delta > 0)
                is_density_below_less = get_below(is_density_above_greater) # If BELOW has density_above_greater, then density_below_less
                is_density_current = (density == currDensity)
                is_density_above_current = get_above(is_density_current)
                is_gravity = (world[:, self.NUM_ELEMENTS+1:self.NUM_ELEMENTS+2] == 1)
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

            # Run sand-piling procedure. Loop over each piling block type. In kernel form, for each block:
                # If dir=left and BELOW_LEFT density is less && both gravity-affected -> Become below-left.
                # If ABOVE_RIGHT dir=left and ABOVE_RIGHT density is less && both gravity-affected -> Become above-right.
            for elem in [2, 12]: # Later: only pick sand-piling elements.
                fall_dir = (rand_movement > 0.5)
                not_did_gravity = (did_gravity <= 0)
                for fallLeft in [True, False]:
                    get_in_dir = get_left if fallLeft else get_right
                    get_in_not_dir = get_right if fallLeft else get_left
                    is_element = (world[:, elem:elem+1] == 1)
                    is_above_right_element = get_in_not_dir(get_above(is_element))
                    density = world[:, self.NUM_ELEMENTS:self.NUM_ELEMENTS+1]
                    is_matching_fall = fall_dir if fallLeft else (~fall_dir)
                    is_above_right_matching_fall = get_in_not_dir(get_above(is_matching_fall))
                    is_below_left_density_lower = ((density - get_in_dir(get_below(density))) > 0)
                    is_above_right_density_higher = ((get_in_not_dir(get_above(density)) - density) > 0)
                    is_gravity = (world[:, self.NUM_ELEMENTS+1:self.NUM_ELEMENTS+2] == 1)
                    is_below_left_gravity = get_in_dir(get_below(is_gravity)) & is_gravity
                    is_above_right_gravity = get_in_not_dir(get_above(is_gravity)) & is_gravity
                    not_did_gravity_below_left = get_in_dir(get_below(not_did_gravity)) & not_did_gravity
                    not_did_gravity_above_right = get_in_not_dir(get_above(not_did_gravity)) & not_did_gravity

                    does_become_below_left = is_element & not_did_gravity_below_left & is_matching_fall & is_below_left_density_lower & is_below_left_gravity
                    does_become_above_right = is_above_right_element & not_did_gravity_above_right & is_above_right_matching_fall \
                                            & is_above_right_density_higher & is_above_right_gravity

                    world_below_left = get_in_dir(get_below(world))
                    world_above_right = get_in_not_dir(get_above(world))

                    debug_two_assigns = does_become_below_left & does_become_above_right
                    # if torch.any(debug_two_assigns > 0):
                    #     print("Assign Error")
                    
                    world[:] = interp2(switch_a=does_become_below_left, switch_b=does_become_above_right,
                                  if_false=world, if_a=world_below_left, if_b=world_above_right)

            # Run water-flowing procedure. Same as sand-piling, but move LEFT/RIGHT instead of BELOW-LEFT/BELOW-RIGHT.
            fluid_momentum = world[:, self.NUM_ELEMENTS+2:self.NUM_ELEMENTS+3, :, :]
            new_fluid_momentum = torch.zeros((world.shape[0], 1, world.shape[2], world.shape[3]), device=self.device)
            fall_rand = rand_movement
            for elem in [0, 3, 4, 10, 11]: # Later: only pick water-flowing elements.
                for fallLeft in [True, False]:
                    fall_dir = ((fall_rand + fluid_momentum + new_fluid_momentum) > 0.5)
                    not_did_gravity = (did_gravity <= 0)
                    get_in_dir = get_left if fallLeft else get_right
                    get_in_not_dir = get_right if fallLeft else get_left
                    is_element = (world[:, elem:elem+1] == 1)
                    is_right_element = get_in_not_dir(is_element)
                    density = world[:, self.NUM_ELEMENTS:self.NUM_ELEMENTS+1]
                    is_matching_fall = fall_dir if fallLeft else (~fall_dir)
                    is_right_matching_fall = get_in_not_dir(is_matching_fall)
                    is_left_density_lower = ((density - get_in_dir(density)) > 0)
                    is_right_density_higher = ((get_in_not_dir(density) - density) > 0)
                    is_gravity = world[:, self.NUM_ELEMENTS+1:self.NUM_ELEMENTS+2] == 1
                    is_left_gravity = get_in_dir(is_gravity) & is_gravity
                    is_right_gravity = get_in_not_dir(is_gravity) & is_gravity
                    not_did_gravity_left = not_did_gravity
                    not_did_gravity_right = get_in_not_dir(not_did_gravity)

                    does_become_left = is_matching_fall & is_element & not_did_gravity_left & is_left_density_lower & is_left_gravity
                    does_become_right = is_right_matching_fall & is_right_element & not_did_gravity_right & is_right_density_higher & is_right_gravity

                    new_fluid_momentum += does_become_right * (2 if fallLeft else -2)

                    world_left = get_in_dir(world)
                    world_right = get_in_not_dir(world)

                    debug_two_assigns = does_become_left & does_become_right
                    # if torch.any(debug_two_assigns > 0):
                    #     print("Assign Error")
                    
                    world[:] = interp2(switch_a=does_become_left, switch_b=does_become_right,
                                  if_false=world, if_a=world_left, if_b=world_right)

            world[:, self.NUM_ELEMENTS+2:self.NUM_ELEMENTS+3, :, :] = new_fluid_momentum

            # Ice melting. Ice touching water or air turns to water.
            ice_chance = rand_interact
            ice_melting_neighbors = self.get_elem(world, "empty") + self.get_elem(world, "fire") \
                + self.get_elem(world, "lava") + self.get_elem(world, "water")
            ice_can_melt = (F.conv2d(ice_melting_neighbors, self.neighbor_kernel, padding=1) > 1)
            does_turn_water = self.get_bool(world, "ice") & ice_can_melt & (ice_chance < 0.02)
            world[:] = interp(switch=does_turn_water, if_false=world, if_true=self.elem_vecs['water'])
            
            # Water freezing. Water touching 3+ ices can turn to ice.
            ice_chance = rand_element
            water_can_freeze = (F.conv2d(self.get_elem(world, "ice"), self.neighbor_kernel, padding=1) >= 3)
            does_turn_ice = self.get_bool(world, "water") & water_can_freeze & (ice_chance < 0.05)
            world[:] = interp(switch=does_turn_ice, if_false=world, if_true=self.elem_vecs['ice'])

            # Fire burning. If fire next to a burnable, chance to burn it to fire.
            burn_chance = rand_interact
            fire_and_lava = self.get_elem(world, "fire") + self.get_elem(world, "lava")
            has_fire_neighbor = F.conv2d(fire_and_lava, self.neighbor_kernel, padding=1) > 0
            does_burn_wood = self.get_bool(world, "wood") & (burn_chance < 0.05)
            does_burn_plant = self.get_bool(world, "plant") & (burn_chance < 0.2)
            does_burn_gas = self.get_bool(world, "gas") & (burn_chance < 0.2)
            does_burn_dust = self.get_bool(world, "dust")
            does_burn_ice = self.get_bool(world, "ice") & (burn_chance < 0.2) & has_fire_neighbor
            does_burn = (does_burn_wood | does_burn_plant | does_burn_gas | does_burn_dust) & has_fire_neighbor
            # Velocity for fire
            velocity_field += 10*get_left(does_burn & has_fire_neighbor)*self.up
            velocity_field += 10*(does_burn & has_fire_neighbor)*self.left
            velocity_field += 10*get_right(does_burn & has_fire_neighbor)*self.right
            velocity_field += 20*get_left(does_burn_dust & has_fire_neighbor)*self.up
            velocity_field += 20*(does_burn_dust & has_fire_neighbor)*self.left
            velocity_field += 20*get_right(does_burn_dust & has_fire_neighbor)*self.up
            
            world[:] = interp(switch=does_burn, if_false=world, if_true=self.elem_vecs['fire'])
            world[:] = interp(switch=does_burn_ice, if_false=world, if_true=self.elem_vecs['water'])

            #Fire spread. (Fire+burnable, or Lava)=> creates a probability to spread to air.
            burnables = self.get_elem(world, "wood") + self.get_elem(world, "plant") + self.get_elem(world, "gas") + self.get_elem(world, "dust")
            fire_with_burnable_neighbor = F.conv2d(burnables, self.neighbor_kernel, padding=1) * fire_and_lava
            in_fire_range = F.conv2d(fire_with_burnable_neighbor + self.get_elem(world, "lava"), self.neighbor_kernel, padding=1)
            does_burn_empty = self.get_bool(world, "empty") & (in_fire_range > 0) & (burn_chance < 0.3)
            world[:] = interp(switch=does_burn_empty, if_false=world, if_true=self.elem_vecs['fire'])

            # Fire fading. Fire just has a chance to fade, if not next to a burnable neighbor.
            fire_chance = rand_element
            has_burnable_neighbor = F.conv2d(burnables, self.neighbor_kernel, padding=1)
            does_fire_turn_empty = self.get_bool(world, "fire") & (fire_chance < 0.4) & (has_burnable_neighbor == 0)
            world[:] = interp(switch=does_fire_turn_empty, if_false=world, if_true=self.elem_vecs['empty'])

            # Plants-growing. If there is water next to plant, and < 4 neighbors, chance to grow there.
            plant_chance = rand_interact
            plant_counts = F.conv2d(self.get_elem(world, "plant"), self.neighbor_kernel, padding=1)
            does_plantgrow = self.get_bool(world, "water") & (plant_chance < 0.05)
            does_plantgrow_plant = does_plantgrow & (plant_counts <= 3) & (plant_counts >= 1)
            does_plantgrow_empty = does_plantgrow & (plant_counts > 3)
            world[:] = interp2(switch_a=does_plantgrow_plant, switch_b=does_plantgrow_empty,
                                  if_false=world, if_a=self.elem_vecs['plant'], if_b=self.elem_vecs['empty'])

            # Lava-water interaction. Lava that is touching water turns to stone.
            water_counts = F.conv2d(self.get_elem(world, "water"), self.neighbor_kernel, padding=1)
            does_turn_stone = (water_counts > 0) & self.get_bool(world, "lava")
            world[:] = interp(switch=does_turn_stone, if_false=world, if_true=self.elem_vecs['stone'])

            # Acid destroys everything except wall.
            acid_rand = (rand_interact < 0.2)
            is_block = ~(self.get_bool(world, "empty") | self.get_bool(world, "wall") | self.get_bool(world, "acid"))
            is_acid = self.get_bool(world, "acid")
            does_acid_dissapear = (is_acid & acid_rand & get_below(is_block)) + (is_acid & acid_rand & get_above(is_block))
            does_block_dissapear = (is_block & get_above(acid_rand) & get_above(is_acid)) \
                | (is_block & get_below(acid_rand) & get_below(is_acid))
            does_dissapear = does_acid_dissapear | does_block_dissapear
            world[:] = interp(switch=does_dissapear, if_false=world, if_true=self.elem_vecs['empty'])

            # Cloner assign (right), also clear assignments from non-cloner
            cloner_assigns = world[:,self.NUM_ELEMENTS+3:self.NUM_ELEMENTS+4]
            is_not_cloner = ~self.get_bool(world, "cloner")
            labels = torch.argmax(world[:, :self.NUM_ELEMENTS], dim=1)[:, None]
            for get_dir in [get_below, get_above, get_left, get_right]:
                is_cloner_empty = self.get_bool(world, "cloner") & ((cloner_assigns == 0) | (cloner_assigns == 13))
                dir_labels = get_dir(labels)
                world[:,self.NUM_ELEMENTS+3:self.NUM_ELEMENTS+4] = (~(is_cloner_empty|is_not_cloner))*cloner_assigns + is_cloner_empty*dir_labels

            # Cloner produce
            cloner_assigns_vec = self.elem_vecs_array(world[:,self.NUM_ELEMENTS+3].int())
            cloner_assigns_vec = torch.permute(cloner_assigns_vec, (0,3,1,2))
            for get_dir in [get_below, get_above, get_left, get_right]:
                cloner_assigns_vec_dir = get_dir(cloner_assigns_vec)
                is_dir_cloner_not_empty = get_dir(self.get_bool(world, "cloner") & ((cloner_assigns != 0) & (cloner_assigns != 13))) \
                    & self.get_bool(world, "empty")
                world[:] = interp(switch=is_dir_cloner_not_empty, if_false=world, if_true=cloner_assigns_vec_dir)

#             # Velocity field movement
            for n in range(2):
                velocity_field_magnitudes = torch.linalg.vector_norm(velocity_field, dim=1)[:, None]
                velocity_field_angles_raw = (1/(2*torch.pi)) * torch.acos(velocity_field[:,1:2] / (velocity_field_magnitudes+0.001))
                is_y_lessthan_zero = (velocity_field[:,0:1] < 0)
                velocity_field_angle = interp(switch=is_y_lessthan_zero, if_false=velocity_field_angles_raw, if_true=(1 - velocity_field_angles_raw))
                for angle in [0,1,2,3,4,5,6,7]:
                    is_angle_match = (torch.remainder(torch.floor(velocity_field_angle * 8 + 0.5), 8) == angle)
                    is_velocity_enough = (velocity_field_magnitudes > 0.1)
                    is_empty_in_dir = self.direction_func(angle, self.get_bool(world, "empty")) * (~self.get_bool(world, "wall"))

                    does_become_empty = is_angle_match & is_velocity_enough & is_empty_in_dir
                    does_become_opposite = self.direction_func((angle+4) % 8, does_become_empty)
                    opposite_world = self.direction_func((angle+4) % 8, world)

                    does_become_empty = does_become_empty.float()
                    does_become_opposite = does_become_opposite.float()

                    world[:] = (1-does_become_empty-does_become_opposite)*world + (does_become_empty)*self.elem_vecs['empty'] \
                        + (does_become_opposite)*opposite_world

                    angle_velocity_field = self.direction_func(angle, velocity_field)
                    opposite_velocity_field = self.direction_func((angle+4) % 8, velocity_field)
                    velocity_field[:] = (1-does_become_empty-does_become_opposite)*velocity_field + (does_become_empty)*angle_velocity_field \
                        + (does_become_opposite)*opposite_velocity_field


            # Velocity field reduction
            velocity_field *= 0.95
            # velocity_field -= torch.sign(velocity_field)*0.02
            for i in range(1):
                velocity_field[:, 0:1] = F.conv2d(velocity_field[:, 0:1], self.neighbor_kernel/18, padding=1) + velocity_field[:, 0:1]*0.5
                velocity_field[:, 1:2] = F.conv2d(velocity_field[:, 1:2], self.neighbor_kernel/18, padding=1) + velocity_field[:, 1:2]*0.5

            world[:, self.NUM_ELEMENTS+4:self.NUM_ELEMENTS+6] = velocity_field


            # sanity_check_world = torch.sum(world[:,:self.NUM_ELEMENTS,:,:], dim=1)
            # sanity_check_world_2 = world[:,:self.NUM_ELEMENTS,:,:]
            # if torch.any(sanity_check_world != 1) or torch.any(sanity_check_world_2 < 0):
            #     print("Sanity Check Failed! A kernel operator resulted in multiple blocks assigned.")
            return world