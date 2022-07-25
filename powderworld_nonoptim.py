import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import types

# ================ BASIC VARIABLE STUFF =================
# Name:    ID, Density, GravityInter
elements = {
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
NUM_ELEMENTS = len(elements)
# [ElementID(N), Density(1), GravityInter(1), FluidMomentum(1), ClonerID(1)]
NUM_CHANNEL = NUM_ELEMENTS + 1 + 1 + 1 + 1 + 2


# ================ RENDERING =================
color_kernel = None
vector_color_kernel = None
def render(world, device):
    with torch.no_grad():
        global color_kernel
        global vector_color_kernel
        if color_kernel == None:
            color_kernel = torch.Tensor([
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
            color_kernel /= 255.0
            color_kernel = color_kernel.T[:, :, None, None]
            vector_color_kernel = torch.Tensor([
                [68, 1, 84],
                [64, 67, 135],
                [41, 120, 142],
                [34, 167, 132],
                [121, 209, 81],
                [253, 231, 36],
                [68, 1, 84],
            ]).to(device)
            vector_color_kernel /= 255.0
            vector_color_kernel = vector_color_kernel[:, :, None, None]


        img = F.conv2d(world[:, :NUM_ELEMENTS], color_kernel)[0]
        velocity_field = world[0, NUM_ELEMENTS+4:NUM_ELEMENTS+6]
        velocity_field_magnitudes = torch.linalg.vector_norm(velocity_field, dim=0)[None]
        
        velocity_field_angles_raw = (1/(2*torch.pi)) * torch.acos(velocity_field[1] / (velocity_field_magnitudes+0.001))
        is_y_lessthan_zero = (velocity_field[0] < 0).float()
        velocity_field_angles_raw = (1-is_y_lessthan_zero)*velocity_field_angles_raw + (is_y_lessthan_zero)*(1 - velocity_field_angles_raw)
        velocity_field_angles = velocity_field_angles_raw
        velocity_field_colors = torch.zeros_like(img)
        for c in range(7):
            velocity_field_colors += vector_color_kernel[c] * torch.clamp(1 - 7*torch.abs((velocity_field_angles - c/6)), 0, 1)

        velocity_field_display = torch.clamp(velocity_field_magnitudes/2, 0, 0.5)
        img = (1-velocity_field_display)*img + velocity_field_display*velocity_field_colors
        img = torch.clamp(img, 0, 1)
        img = img.detach().cpu()
        img = img.permute(1, 2, 0).numpy()
        img = (img*255).astype(np.uint8)
        return img

# =========== WORLD EDITING HELPERS ====================
def add_element(world_slice, element_name, device, wind=None):
    if element_name == "wind":
        world_slice[:,NUM_ELEMENTS+4:NUM_ELEMENTS+6] = wind
    else:
        elem_id, elem_dens, elem_grav = elements[element_name]
        elemnt_vec = torch.zeros(NUM_CHANNEL).to(device)
        elemnt_vec[elem_id] = 1
        elemnt_vec[NUM_ELEMENTS] = elem_dens
        elemnt_vec[NUM_ELEMENTS+1] = elem_grav
        world_slice[...] = elemnt_vec[None, :, None, None]
    
    
# =========== POWDERWORLD STEP FUNCTION ====================
def get_below(x):
    return torch.roll(x, shifts=-1, dims=2)
def get_above(x):
    return torch.roll(x, shifts=1, dims=2)
def get_left(x):
    return torch.roll(x, shifts=1, dims=3)
def get_right(x):
    return torch.roll(x, shifts=-1, dims=3)
def get_elem(world, elemname):
    elemid = elements[elemname][0]
    return(world[:, elemid:elemid+1])

# Helper Kernels
def make_kernels(device):
    kernels = types.SimpleNamespace()
    with torch.no_grad():
        kernels.elem_vecs = {}
        kernels.elem_vecs_array = nn.Embedding(NUM_ELEMENTS, NUM_CHANNEL).to(device)
        for elem_name, elem in elements.items():
            elem_vec = torch.zeros(NUM_CHANNEL).to(device)
            elem_vec[elem[0]] = 1
            elem_vec[NUM_ELEMENTS] = elem[1]
            elem_vec[NUM_ELEMENTS+1] = elem[2]
            kernels.elem_vecs[elem_name] = elem_vec[None, :, None, None]
            kernels.elem_vecs_array.weight[elem[0]] = elem_vec

        kernels.neighbor_kernel = torch.ones((1, 1, 3, 3)).to(device)
        kernels.stone_kernel = torch.zeros((1,1,3,3)).to(device)
        kernels.stone_kernel[0, 0, 0, 0] = 1
        kernels.stone_kernel[0, 0, 0, 2] = 1
        
        kernels.up = torch.Tensor([0,-1]).to(device)[None,:,None,None]
        kernels.left = torch.Tensor([-1,0]).to(device)[None,:,None,None]
        kernels.right = torch.Tensor([1,0]).to(device)[None,:,None,None]
    return kernels

def step(world, device, kernels):
    with torch.no_grad():
        # Helper Functions
        rand_movement = torch.rand((world.shape[0], 1, world.shape[2], world.shape[3])).to(device) # For gravity, flowing.
        rand_interact = torch.rand((world.shape[0], 1, world.shape[2], world.shape[3])).to(device) # For element-wise interactions.
        rand_element = torch.rand((world.shape[0], 1, world.shape[2], world.shape[3])).to(device) # For self-element behavior.
        velocity_field = torch.clone(world[:, NUM_ELEMENTS+4:NUM_ELEMENTS+6])

        # Run stone-stability procedure. If a stone is next to two stones, turn gravity off. Otherwise, turn it on.
        stone = get_elem(world, "stone")
        has_stone_supports = F.conv2d(stone, kernels.stone_kernel, padding=1)
        world[:, NUM_ELEMENTS+1:NUM_ELEMENTS+2] = (1-stone)*world[:, NUM_ELEMENTS+1:NUM_ELEMENTS+2] + stone*(has_stone_supports < 2).float()
        
        # Run gravity procedure.
        # Loop through each possible density (1-5).
        # In kernel form, compute for each block:
            # IF density == currDensity && density BELOW is less && both gravity-affected -> Become below.
            # (Else)If density ABOVE is greater && density ABOVE == currDensity && both gravity-affected -> Become above.
        did_gravity = torch.zeros((world.shape[0], 1, world.shape[2], world.shape[3])).to(device)
        for currDensity in [0,1,2,3]:
            density = world[:, NUM_ELEMENTS:NUM_ELEMENTS+1]
            density_delta = get_above(density) - density # Delta between ABOVE and current
            is_density_above_greater = (density_delta > 0).float() 
            is_density_below_less = get_below(is_density_above_greater) # If BELOW has density_above_greater, then density_below_less
            is_density_current = (density == currDensity).float()
            is_density_above_current = get_above(is_density_current)
            is_gravity = world[:, NUM_ELEMENTS+1:NUM_ELEMENTS+2]
            is_center_and_below_gravity = get_below(is_gravity) * is_gravity
            is_center_and_above_gravity = get_above(is_gravity) * is_gravity

            # These should never be both true for the same block
            does_become_below = is_density_current * is_density_below_less * is_center_and_below_gravity
            does_become_above = is_density_above_greater * is_density_above_current * is_center_and_above_gravity

            did_gravity += does_become_above

            world_above = get_above(world)
            world_below = get_below(world)
            world[:] = world*(1-does_become_below-does_become_above) + does_become_below*world_below + does_become_above*world_above

        # Run sand-piling procedure. Loop over each piling block type. In kernel form, for each block:
            # If dir=left and BELOW_LEFT density is less && both gravity-affected -> Become below-left.
            # If ABOVE_RIGHT dir=left and ABOVE_RIGHT density is less && both gravity-affected -> Become above-right.
        for elem in [2, 12]: # Later: only pick sand-piling elements.
            fall_dir = (rand_movement > 0.5).float()
            not_did_gravity = (did_gravity <= 0).float()
            for fallLeft in [True, False]:
                get_in_dir = get_left if fallLeft else get_right
                get_in_not_dir = get_right if fallLeft else get_left
                is_element = world[:, elem:elem+1]
                is_above_right_element = get_in_not_dir(get_above(is_element))
                density = world[:, NUM_ELEMENTS:NUM_ELEMENTS+1]
                is_matching_fall = fall_dir if fallLeft else (1-fall_dir)
                is_above_right_matching_fall = get_in_not_dir(get_above(is_matching_fall))
                is_below_left_density_lower = ((density - get_in_dir(get_below(density))) > 0).float()
                is_above_right_density_higher = ((get_in_not_dir(get_above(density)) - density) > 0).float()
                is_gravity = world[:, NUM_ELEMENTS+1:NUM_ELEMENTS+2]
                is_below_left_gravity = get_in_dir(get_below(is_gravity)) * is_gravity
                is_above_right_gravity = get_in_not_dir(get_above(is_gravity)) * is_gravity
                not_did_gravity_below_left = get_in_dir(get_below(not_did_gravity)) * not_did_gravity
                not_did_gravity_above_right = get_in_not_dir(get_above(not_did_gravity)) * not_did_gravity
                
                does_become_below_left = is_element * not_did_gravity_below_left * is_matching_fall * is_below_left_density_lower * is_below_left_gravity
                does_become_above_right = is_above_right_element * not_did_gravity_above_right * is_above_right_matching_fall \
                                        * is_above_right_density_higher * is_above_right_gravity

                world_below_left = get_in_dir(get_below(world))
                world_above_right = get_in_not_dir(get_above(world))

                debug_two_assigns = does_become_below_left * does_become_above_right
                if torch.any(debug_two_assigns > 0):
                    print("Assign Error")

                world[:] = world*(1-does_become_below_left-does_become_above_right) \
                    + does_become_below_left*world_below_left + does_become_above_right*world_above_right

        # Run water-flowing procedure. Same as sand-piling, but move LEFT/RIGHT instead of BELOW-LEFT/BELOW-RIGHT.
        fluid_momentum = world[:, NUM_ELEMENTS+2:NUM_ELEMENTS+3, :, :]
        new_fluid_momentum = torch.zeros((world.shape[0], 1, world.shape[2], world.shape[3])).to(device)
        fall_rand = rand_movement
        for elem in [0, 3, 4, 10, 11]: # Later: only pick water-flowing elements.
            for fallLeft in [True, False]:
                fall_dir = ((fall_rand + fluid_momentum + new_fluid_momentum) > 0.5).float()
                not_did_gravity = (did_gravity <= 0).float()
                get_in_dir = get_left if fallLeft else get_right
                get_in_not_dir = get_right if fallLeft else get_left
                is_element = world[:, elem:elem+1]
                is_right_element = get_in_not_dir(is_element)
                density = world[:, NUM_ELEMENTS:NUM_ELEMENTS+1]
                is_matching_fall = fall_dir if fallLeft else (1-fall_dir)
                is_right_matching_fall = get_in_not_dir(is_matching_fall)
                is_left_density_lower = ((density - get_in_dir(density)) > 0).float()
                is_right_density_higher = ((get_in_not_dir(density) - density) > 0).float()
                is_gravity = world[:, NUM_ELEMENTS+1:NUM_ELEMENTS+2]
                is_left_gravity = get_in_dir(is_gravity) * is_gravity
                is_right_gravity = get_in_not_dir(is_gravity) * is_gravity
                # not_did_gravity_left = get_in_dir(not_did_gravity) * not_did_gravity
                # not_did_gravity_right = get_in_not_dir(not_did_gravity) * not_did_gravity
                not_did_gravity_left = not_did_gravity
                not_did_gravity_right = get_in_not_dir(not_did_gravity)

                does_become_left = is_matching_fall * is_element * not_did_gravity_left * is_left_density_lower * is_left_gravity
                does_become_right = is_right_matching_fall * is_right_element * not_did_gravity_right * is_right_density_higher * is_right_gravity

                new_fluid_momentum += does_become_right * (2 if fallLeft else -2)

                world_left = get_in_dir(world)
                world_right = get_in_not_dir(world)

                debug_two_assigns = does_become_left * does_become_right
                if torch.any(debug_two_assigns > 0):
                    print("Assign Error")

                world[:] = world*(1-does_become_left-does_become_right) \
                    + does_become_left*world_left + does_become_right*world_right
        world[:, NUM_ELEMENTS+2:NUM_ELEMENTS+3, :, :] = new_fluid_momentum
        
        # # Ice freezing. Run a conv kernel to check for adjacent ices. Sum this to get probability that water turns to ice.
        # ice_chance = rand_interact
        # ice_counts = F.conv2d(get_elem(world, "ice"), kernels.neighbor_kernel, padding=1)
        # does_turn_ice = get_elem(world, "water") * (((ice_counts/9)) > 3*ice_chance).float()
        # world[:] = (1-does_turn_ice)*world + does_turn_ice*kernels.elem_vecs['ice']
        
        # Ice melting. Ice touching water or air turns to water.
        ice_chance = rand_interact
        ice_melting_neighbors = get_elem(world, "empty") + get_elem(world, "fire") + get_elem(world, "lava") + get_elem(world, "water")
        ice_can_melt = (F.conv2d(ice_melting_neighbors, kernels.neighbor_kernel, padding=1) > 1).float()
        does_turn_ice = get_elem(world, "ice") * ice_can_melt * (ice_chance < 0.02).float()
        world[:] = (1-does_turn_ice)*world + does_turn_ice*kernels.elem_vecs['water']
        
        # Fire burning. If fire next to a burnable, chance to burn it to fire.
        burn_chance = rand_interact
        fire_and_lava = get_elem(world, "fire") + get_elem(world, "lava")
        has_fire_neighbor = F.conv2d(fire_and_lava, kernels.neighbor_kernel, padding=1)
        does_burn_wood = get_elem(world, "wood") * (burn_chance < 0.05).float()
        does_burn_plant = get_elem(world, "plant") * (burn_chance < 0.2).float()
        does_burn_gas = get_elem(world, "gas") * (burn_chance < 0.2).float()
        does_burn_dust = get_elem(world, "dust")
        does_burn_ice = get_elem(world, "ice") * (burn_chance < 0.2).float() * (has_fire_neighbor > 0).float()
        does_burn = (does_burn_wood + does_burn_plant + does_burn_gas + does_burn_dust)*(has_fire_neighbor > 0).float()
        
        velocity_field += 10*get_left(does_burn * (has_fire_neighbor > 0).float())*kernels.up
        velocity_field += 10*(does_burn * (has_fire_neighbor > 0).float())*kernels.left
        velocity_field += 10*get_right(does_burn * (has_fire_neighbor > 0).float())*kernels.right
        velocity_field += 20*get_left(does_burn_dust * (has_fire_neighbor > 0).float())*kernels.up
        velocity_field += 20*(does_burn_dust * (has_fire_neighbor > 0).float())*kernels.left
        velocity_field += 20*get_right(does_burn_dust * (has_fire_neighbor > 0).float())*kernels.up
        world[:] = (1-does_burn)*world + (does_burn)*kernels.elem_vecs['fire']
        world[:] = (1-does_burn_ice)*world + (does_burn_ice)*kernels.elem_vecs['water']
        

        
        #Fire spread. (Fire+burnable, or Lava)=> creates a probability to spread to air.
        burnables = get_elem(world, "wood") + get_elem(world, "plant") + get_elem(world, "gas") + get_elem(world, "dust")
        fire_with_burnable_neighbor = F.conv2d(burnables, kernels.neighbor_kernel, padding=1) * fire_and_lava
        in_fire_range = F.conv2d(fire_with_burnable_neighbor + get_elem(world, "lava"), kernels.neighbor_kernel, padding=1)
        does_burn_empty = get_elem(world, "empty") * (in_fire_range > 0).float() * (burn_chance < 0.3).float()
        world[:] = (1-does_burn_empty)*world + (does_burn_empty)*kernels.elem_vecs['fire']

        # Fire fading. Fire just has a chance to fade, if not next to a burnable neighbor.
        fire_chance = rand_element
        has_burnable_neighbor = F.conv2d(burnables, kernels.neighbor_kernel, padding=1)
        does_fire_turn_empty = get_elem(world, "fire") * (fire_chance < 0.4).float() * (has_burnable_neighbor == 0).float()
        world[:] = (1-does_fire_turn_empty)*world + does_fire_turn_empty*kernels.elem_vecs['empty']
        
        # Plants-growing. If there is water next to plant, and < 4 neighbors, chance to grow there.
        plant_chance = rand_interact
        plant_counts = F.conv2d(get_elem(world, "plant"), kernels.neighbor_kernel, padding=1)
        does_plantgrow = get_elem(world, "water") * (plant_chance < 0.05).float()
        does_plantgrow_plant = does_plantgrow * (plant_counts <= 3).float() * (plant_counts >= 1).float()
        does_plantgrow_empty = does_plantgrow * (plant_counts > 3).float()
        world[:] = (1-does_plantgrow_plant-does_plantgrow_empty)*world + does_plantgrow_plant*kernels.elem_vecs['plant'] \
            + does_plantgrow_empty*kernels.elem_vecs['empty']
        
        # Lava-water interaction. Lava that is touching water turns to stone.
        water_counts = F.conv2d(get_elem(world, "water"), kernels.neighbor_kernel, padding=1)
        does_turn_stone = (water_counts > 0).float() * get_elem(world, "lava")
        world[:] = (1-does_turn_stone)*world + does_turn_stone*kernels.elem_vecs['stone']
        
        # Acid destroys everything except wall.
        acid_rand = (rand_interact < 0.2).float()
        is_block = 1 - get_elem(world, "empty") - get_elem(world, "wall") - get_elem(world, "acid")
        is_acid = get_elem(world, "acid")
        does_acid_dissapear = (is_acid * acid_rand * get_below(is_block)) + (is_acid * acid_rand * get_above(is_block))
        does_block_dissapear = (is_block * get_above(acid_rand) * get_above(is_acid)) + (is_block * get_below(acid_rand) * get_below(is_acid))
        does_dissapear = torch.clamp(does_acid_dissapear + does_block_dissapear, 0, 1)
        world[:] = (1-does_dissapear)*world + does_dissapear*kernels.elem_vecs['empty']

        # Cloner assign (right), also clear assignments from non-cloner
        cloner_assigns = world[:,NUM_ELEMENTS+3:NUM_ELEMENTS+4]
        is_not_cloner = 1 - get_elem(world, "cloner")
        labels = torch.argmax(world[:, :NUM_ELEMENTS], dim=1)[:, None]
        for get_dir in [get_below, get_above, get_left, get_right]:
            is_cloner_empty = get_elem(world, "cloner") * ((cloner_assigns == 0).float() + (cloner_assigns == 13).float())
            dir_labels = get_dir(labels)
            world[:,NUM_ELEMENTS+3:NUM_ELEMENTS+4] = (1-is_cloner_empty-is_not_cloner)*cloner_assigns + is_cloner_empty*dir_labels
        
        # Cloner produce
        cloner_assigns_vec = kernels.elem_vecs_array(world[:,NUM_ELEMENTS+3].int())
        cloner_assigns_vec = torch.permute(cloner_assigns_vec, (0,3,1,2))
        for get_dir in [get_below, get_above, get_left, get_right]:
            cloner_assigns_vec_dir = get_dir(cloner_assigns_vec)
            is_dir_cloner_not_empty = get_dir(get_elem(world, "cloner") * ((cloner_assigns != 0).float() * (cloner_assigns != 13).float())) \
                * get_elem(world, "empty")
            world[:] = (1-is_dir_cloner_not_empty)*world + is_dir_cloner_not_empty*cloner_assigns_vec_dir        
        
        # Velocity field movement
        for n in range(2):
            velocity_field_magnitudes = torch.linalg.vector_norm(velocity_field, dim=1)[:, None]
            velocity_field_angles_raw = (1/(2*torch.pi)) * torch.acos(velocity_field[:,1:2] / (velocity_field_magnitudes+0.001))
            is_y_lessthan_zero = (velocity_field[:,0:1] < 0)
            velocity_field_angle = (1-is_y_lessthan_zero)*velocity_field_angles_raw + (is_y_lessthan_zero)*(1 - velocity_field_angles_raw)
            direction_func = [
                lambda x : get_right(x),
                lambda x : get_right(get_below(x)),
                lambda x : get_below(x),
                lambda x : get_left(get_below(x)),
                lambda x : get_left(x),
                lambda x : get_left(get_above(x)),
                lambda x : get_above(x),
                lambda x : get_right(get_above(x)),
            ]
            for angle in [0,1,2,3,4,5,6,7]:
                is_angle_match = (torch.remainder(torch.floor(velocity_field_angle * 8 + 0.5), 8) == angle)
                is_velocity_enough = (velocity_field_magnitudes > 0.1).float()
                is_empty_in_dir = direction_func[angle](get_elem(world, "empty")) * (1-get_elem(world, "wall"))

                does_become_empty = is_angle_match * is_velocity_enough * is_empty_in_dir
                does_become_opposite = direction_func[(angle+4) % 8](does_become_empty)
                opposite_world = direction_func[(angle+4) % 8](world)

                world[:] = (1-does_become_empty-does_become_opposite)*world + (does_become_empty)*kernels.elem_vecs['empty'] \
                    + (does_become_opposite)*opposite_world
                
                angle_velocity_field = direction_func[angle](velocity_field)
                opposite_velocity_field = direction_func[(angle+4) % 8](velocity_field)
                xsum = torch.sum(velocity_field)
                debugsum = does_become_empty + does_become_opposite
                does_become_empty *= 1
                does_become_opposite *= 1
                velocity_field[:] = (1-does_become_empty-does_become_opposite)*velocity_field + (does_become_empty)*angle_velocity_field \
                    + (does_become_opposite)*opposite_velocity_field
            
        # Velocity field reduction
        velocity_field *= 0.95
        # velocity_field -= torch.sign(velocity_field)*0.02
        for i in range(1):
            velocity_field[:, 0:1] = F.conv2d(velocity_field[:, 0:1], kernels.neighbor_kernel/18, padding=1) + velocity_field[:, 0:1]*0.5
            velocity_field[:, 1:2] = F.conv2d(velocity_field[:, 1:2], kernels.neighbor_kernel/18, padding=1) + velocity_field[:, 1:2]*0.5
        
        world[:, NUM_ELEMENTS+4:NUM_ELEMENTS+6] = velocity_field
        

        sanity_check_world = torch.sum(world[:,:NUM_ELEMENTS,:,:], dim=1)
        sanity_check_world_2 = world[:,:NUM_ELEMENTS,:,:]
        if torch.any(sanity_check_world != 1) or torch.any(sanity_check_world_2 < 0):
            print("Sanity Check Failed! A kernel operator resulted in multiple blocks assigned.")
    return world