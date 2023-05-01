from powderworld.env import *
from powderworld.sim import pw_elements, pw_element_names

# =====================================
# How Powderworld environments work:
# - A TaskConfig defines a single task in Powderworld.
# - Powderworld Gym environments are *distributions* of tasks.
# - The base class is PWEnv. This defines an OpenAI Gym Environment. It needs a 'generate_task_config' function.
# - To make a new task distribution, extend PWEnv with a generator function.

# =====================================

def elem_id(name):
    return pw_elements[name][0]

# This runs at the bottom of the file.
env_list_train = [
    'PWTaskGenBuildBirdTunnel',
    'PWTaskGenBuildKangarooTunnel',
    'PWTaskGenBuildLemmingBridge',
    'PWTaskGenBuildLemmingTunnelWood',
    'PWTaskGenCollapseStoneTower',
    'PWTaskGenCollapseWoodTower',
    'PWTaskGenCreateStone',
    'PWTaskGenDestroyAll',
    'PWTaskGenExplodeDust',
    'PWTaskGenFillWood',
    
    'PWTaskGenFillWoodLavaTrapLid',
    'PWTaskGenFloodGas',
    'PWTaskGenFloodWater',
    'PWTaskGenGasMove',
    'PWTaskGenHerdBirds',
    'PWTaskGenLavaMove',
    'PWTaskGenLavaTowerPenalty',
    'PWTaskGenMakePlantSpread',
    'PWTaskGenPlantAndWater',
    'PWTaskGenPlantBurn',
    
    'PWTaskGenRemoveBirds',
    'PWTaskGenSandDustPlace',
    'PWTaskGenSandMove',
    'PWTaskGenSandPlace',
    'PWTaskGenSandTowerPenalty',
    'PWTaskGenStoneTower',
    'PWTaskGenWoodPlace',
    'PWTaskGenWaterBlock',
    'PWTaskGenSandBlock',
    'PWTaskGenFillIce',
]

env_list_test = [
    'PWTaskGenBuildLemmingTunnelStone',
    'PWTaskGenBuildKangarooBridge',
    'PWTaskGenFillWoodLavaTrap',
    'PWTaskGenGasAndWater',
    'PWTaskGenPlantAndLava',
    'PWTaskGenStoneTowerPenalty',
    'PWTaskGenWaterMove',
    'PWTaskGenPreserveIce',
    'PWTaskGenMakePlantSpreadWood',
    'PWTaskGenPlantGasBurn',
]

env_list = env_list_train + env_list_test

    
# ===================
# Hand-Designed Tasks
# ===================

class PWTaskGenWoodPlace(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Place wood in the bottom-right corner."
        
        config['reward']['matrix'][48:, 48:] = [elem_id('wood'), 1]
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CONTAINER, "wall", num=1, y=48, x=48))
        return config

class PWTaskGenSandPlace(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Place sand in the bottom-right corner."
        
        config['reward']['matrix'][48:, 48:] = [elem_id('sand'), 1]
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CONTAINER, "wall", num=1, y=48, x=48))
        return config
    
class PWTaskGenSandDustPlace(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Place sand in the bottom-right corner. Place dust in the bottom-left corner."
        
        config['reward']['matrix'][48:, 48:] = [elem_id('sand'), 1]
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CONTAINER, "wall", num=1, y=48, x=48))
        config['reward']['matrix'][48:, 0:16] = [elem_id('dust'), 1]
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CONTAINER, "wall", num=1, y=48, x=0))
        return config

class PWTaskGenPlantBurn(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Place fire to burn a generated circle of plants."

        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, "plant", num=1))
        config['reward']['matrix'][:, :] = [elem_id('plant'), -1]
        return config
    
class PWTaskGenPlantGasBurn(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Place fire to burn a generated circle of plants and gas."

        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, "plant", num=1))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, "gas", num=1))
        config['reward']['matrix'][:, :] = [elem_id('empty'), 1]
        return config
    
class PWTaskGenGasMove(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Move a generated circle of gas to the top-right corner. Cannot place gas."
        
        config['agent']['disabled_elements'] = [pw_elements["gas"][0]]
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, "gas", num=1))
        config['reward']['matrix'][0:16, 48:] = [elem_id('gas'), 1]
        return config
    
class PWTaskGenStoneTower(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Build a tower of stone that reaches top of world."

        config['reward']['matrix'][0:16, :] = [elem_id('stone'), 1]
        return config
    
class PWTaskGenStoneTowerPenalty(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Build a tower of stone that reaches top of world. Minimize elements present."

        config['reward']['matrix'][0:16, :] = [elem_id('stone'), 1]
        config['reward']['matrix'][16:, :] = [elem_id('empty'), 0.1]
        return config
    
class PWTaskGenSandTowerPenalty(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Build a tower of sand that reaches top of world. Minimize elements present."

        config['reward']['matrix'][0:16, :] = [elem_id('sand'), 1]
        config['reward']['matrix'][16:, :] = [elem_id('empty'), 0.1]
        return config
    
class PWTaskGenLavaTowerPenalty(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Build a tower of lava that reaches top of world. Minimize elements present."

        config['reward']['matrix'][0:16, :] = [elem_id('lava'), 1]
        config['reward']['matrix'][16:, :] = [elem_id('empty'), 0.1]
        return config
    
class PWTaskGenWaterMove(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Direct water from a generated source into a container in bottom-right."
        config['agent']['disabled_elements'] = [elem_id('water'), elem_id('ice'), elem_id('cloner')]

        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CLONER_CIRCLE, "water", num=2))
        x = np.random.randint(0, 48) 
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CONTAINER, "wall", num=1, y=48, x=x))
        config['reward']['matrix'][48:, x:x+16] = [elem_id('water'), 1]
        return config
    
class PWTaskGenWaterBlock(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Keep water from hitting the floor."

        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "water", num=1, y=48, x=0))
        config['reward']['matrix'][48:, :] = [elem_id('water'), -1]
        config['reward']['matrix'][:48, :] = [elem_id('water'), 0.2]
        return config
    
class PWTaskGenSandBlock(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Keep sand from hitting the floor."

        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "sand", num=1, y=48, x=0))
        config['reward']['matrix'][48:, :] = [elem_id('sand'), -1]
        config['reward']['matrix'][:48, :] = [elem_id('sand'), 0.2]
        return config
    
class PWTaskGenSandMove(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Direct sand from a generated source into a container in bottom-right."
        config['agent']['disabled_elements'] = [elem_id('sand'), elem_id('cloner')]

        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CLONER_CIRCLE, "sand", num=2))
        x = np.random.randint(0, 48) 
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CONTAINER, "wall", num=1, y=48, x=x))
        config['reward']['matrix'][48:, x:x+16] = [elem_id('sand'), 1]
        return config
    
class PWTaskGenLavaMove(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Direct lava from a generated source into a container in bottom-right."
        config['agent']['disabled_elements'] = [elem_id('lava'), elem_id('cloner')]

        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CLONER_CIRCLE, "lava", num=2))
        x = np.random.randint(0, 48) 
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CONTAINER, "wall", num=1, y=48, x=x))
        config['reward']['matrix'][48:, x:x+16] = [elem_id('lava'), 1]
        return config
    
class PWTaskGenDestroyAll(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Destroy everything."

        for i in range(5):
            elem = np.random.choice(pw_element_names)
            config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, elem, num=1))
        config['reward']['matrix'][:, :] = [elem_id('empty'), 1]
        return config
    
class PWTaskGenCollapseStoneTower(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Knock down a generated stone tower."
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.ARCHES, 'stone', num=5))
        config['reward']['matrix'][:32, :] = [elem_id('stone'), -1]
        return config
    
class PWTaskGenCollapseWoodTower(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Knock down a generated wood tower. No fire or lava."
        config['agent']['disabled_elements'] = [elem_id('lava'), elem_id('fire')]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.ARCHES, 'wood', num=5))
        config['reward']['matrix'][:32, :] = [elem_id('wood'), -1]
        return config
    
class PWTaskGenFillWood(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Fill the world with wood."
        config['reward']['matrix'][:, :] = [elem_id('wood'), 1]
        return config
    
class PWTaskGenFillIce(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Fill the world with ice."
        config['reward']['matrix'][:, :] = [elem_id('ice'), 1]
        return config
    
    
class PWTaskGenFillWoodLavaTrap(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Fill the world with wood. There is a lava trap."
        config['reward']['matrix'][:, :] = [elem_id('wood'), 1]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "lava", num=1, y=48, x=0))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "lava", num=1, y=48, x=16))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "lava", num=1, y=48, x=32))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "lava", num=1, y=48, x=48))
        return config
    
class PWTaskGenFillWoodLavaTrapLid(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Fill the world with wood. There is a closed lava trap."
        config['reward']['matrix'][:, :] = [elem_id('wood'), 1]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "stone", num=1, y=46, x=0))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "stone", num=1, y=46, x=16))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "stone", num=1, y=46, x=32))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "stone", num=1, y=46, x=48))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "lava", num=1, y=48, x=0))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "lava", num=1, y=48, x=16))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "lava", num=1, y=48, x=32))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "lava", num=1, y=48, x=48))
        return config
    
class PWTaskGenExplodeDust(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Set fire to a cache of dust."
        config['reward']['matrix'][:, :] = [elem_id('dust'), -1]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "dust", num=1, y=48, x=0))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "dust", num=1, y=48, x=16))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "dust", num=1, y=48, x=32))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "dust", num=1, y=48, x=48))
        return config
    
class PWTaskGenBuildLemmingBridge(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Build a bridge so lemmings can cross a chasm."
        config['agent']['disabled_elements'] = [elem_id('agentLemming'), elem_id('cloner')]

        lemming_height = np.random.randint(0,2)
        goal_height = np.random.randint(1,4)
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.SMALL_CIRCLE, "agentLemming", num=1, y=lemming_height*16, x=0))
        for y in range(lemming_height+1, 4):
            config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wall", num=1, y=y*16, x=0))

        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CONTAINER, "wall", num=1, y=goal_height*16, x=48))
        for y in range(goal_height+1, 4):
            config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wall", num=1, y=y*16, x=48))
        
        config['reward']['matrix'][goal_height*16:goal_height*16+16, 48:] = [elem_id('agentLemming'), 1]
        return config
    
class PWTaskGenBuildLemmingTunnelStone(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Clear a stone tunnel so lemmings can cross a wall."
        config['agent']['disabled_elements'] = [elem_id('agentLemming'), elem_id('cloner')]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.SMALL_CIRCLE, "agentLemming", num=1, y=32, x=0))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wall", num=1, y=48, x=0))
        
        for wall_y in range(3):
            config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "stone", num=1, y=wall_y*16, x=16))
            config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "stone", num=1, y=wall_y*16, x=32))
            
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wall", num=1, y=48, x=16))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wall", num=1, y=48, x=32))

        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CONTAINER, "wall", num=1, y=48, x=48))
        
        config['reward']['matrix'][32:48, 48:] = [elem_id('agentLemming'), 1]
        return config
    
class PWTaskGenBuildLemmingTunnelWood(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Clear a wood tunnel so lemmings can cross a wall."
        config['agent']['disabled_elements'] = [elem_id('agentLemming'), elem_id('cloner')]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.SMALL_CIRCLE, "agentLemming", num=1, y=32, x=0))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wall", num=1, y=48, x=0))
        
        for wall_y in range(3):
            config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wood", num=1, y=wall_y*16, x=16))
            config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wood", num=1, y=wall_y*16, x=32))
            
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wall", num=1, y=48, x=16))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wall", num=1, y=48, x=32))

        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CONTAINER, "wall", num=1, y=48, x=48))
        
        config['reward']['matrix'][32:48, 48:] = [elem_id('agentLemming'), 1]
        return config
    
class PWTaskGenBuildKangarooTunnel(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Clear a plant tunnel so kangaroos can cross a wall."
        config['agent']['disabled_elements'] = [elem_id('agentKangaroo'), elem_id('cloner')]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.SMALL_CIRCLE, "agentKangaroo", num=1, y=32, x=0))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wall", num=1, y=48, x=0))
        
        for wall_y in range(3):
            config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "stone", num=1, y=wall_y*16, x=16))
            config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "stone", num=1, y=wall_y*16, x=32))
            
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wall", num=1, y=48, x=16))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wall", num=1, y=48, x=32))

        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CONTAINER, "wall", num=1, y=48, x=48))
        
        config['reward']['matrix'][32:48, 48:] = [elem_id('agentKangaroo'), 1]
        return config
    
class PWTaskGenBuildBirdTunnel(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Clear an ice tunnel so birds can cross a wall."
        config['agent']['disabled_elements'] = [elem_id('agentBird'), elem_id('cloner')]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.SMALL_CIRCLE, "agentBird", num=1, y=32, x=0))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wall", num=1, y=48, x=0))
        
        for wall_y in range(3):
            config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "ice", num=1, y=wall_y*16, x=16))
            config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "ice", num=1, y=wall_y*16, x=32))
            
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wall", num=1, y=48, x=16))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wall", num=1, y=48, x=32))

        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CONTAINER, "wall", num=1, y=48, x=48))
        
        config['reward']['matrix'][:, 48:] = [elem_id('agentBird'), 1]
        return config
    
class PWTaskGenBuildKangarooBridge(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Build a bridge so kangaroos can cross a chasm."
        config['agent']['disabled_elements'] = [elem_id('agentKangaroo'), elem_id('cloner')]

        lemming_height = np.random.randint(0,2)
        goal_height = np.random.randint(1,4)
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.SMALL_CIRCLE, "agentKangaroo", num=1, y=lemming_height*16, x=0))
        for y in range(lemming_height+1, 4):
            config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wall", num=1, y=y*16, x=0))

        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CONTAINER, "wall", num=1, y=goal_height*16, x=48))
        for y in range(goal_height+1, 4):
            config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wall", num=1, y=y*16, x=48))
        
        config['reward']['matrix'][goal_height*16:goal_height*16+16, 48:] = [elem_id('agentKangaroo'), 1]
        return config
    
class PWTaskGenRemoveBirds(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Remove a flock of birds from the world."
        config['agent']['disabled_elements'] = [elem_id('cloner')]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, "agentBird", num=2))
        config['reward']['matrix'][:, :] = [elem_id('agentBird'), -1]

        return config
    
class PWTaskGenHerdBirds(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Herd a flock of generated birds into a container."
        config['agent']['disabled_elements'] = [elem_id('agentBird'), elem_id('cloner')]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, "agentBird", num=1))
        x = np.random.randint(0, 48) 
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CONTAINER, "wall", num=1, y=48, x=x))
        config['reward']['matrix'][48:, x:x+16] = [elem_id('agentBird'), 1]

        return config
    
class PWTaskGenMakePlantSpread(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Spread a plant as much as possible."
        config['agent']['disabled_elements'] = [elem_id('plant')]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, "plant", num=1))        
        config['reward']['matrix'][:, :] = [elem_id('plant'), 1]
        return config
    
class PWTaskGenMakePlantSpreadWood(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Spread a plant as much as possible. No water or cloners."
        config['agent']['disabled_elements'] = [elem_id('plant'), elem_id('water'), elem_id('ice'), elem_id('cloner')]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, "plant", num=1))        
        config['reward']['matrix'][:, :] = [elem_id('plant'), 1]
        return config


class PWTaskGenCreateStone(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Given a generated sea of water, create stone."
        config['agent']['disabled_elements'] = [elem_id('stone')]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "water", num=1, y=48, x=0))    
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "water", num=1, y=48, x=16))        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "water", num=1, y=48, x=32))        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "water", num=1, y=48, x=48))        

        config['reward']['matrix'][:, :] = [elem_id('stone'), 1]

        return config
    
class PWTaskGenPreserveIce(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Given generated circles of ice, maintain the ice."
        config['agent']['disabled_elements'] = [elem_id('ice'), elem_id('cloner')]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, "ice", num=2))          

        config['reward']['matrix'][:, :] = [elem_id('ice'), 1]

        return config
    
class PWTaskGenFloodWater(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Flood a map with water."
        
        config['reward']['matrix'][:, :] = [elem_id('water'), 1]
        return config
    
class PWTaskGenFloodGas(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Flood a map with gas. Cannot place gas."
        config['agent']['disabled_elements'] = [pw_elements["gas"][0]]
        
        config['reward']['matrix'][:, :] = [elem_id('gas'), 1]
        return config
    
class PWTaskGenPlantAndLava(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Grow plants on the left side, and place lava on the right side."
        
        config['reward']['matrix'][:, 0:32] = [elem_id('plant'), 1]
        config['reward']['matrix'][:, 32:] = [elem_id('lava'), 1]

        return config
    
class PWTaskGenGasAndWater(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Create gas on the left side, and water on the right side."
        
        config['reward']['matrix'][:, 0:32] = [elem_id('gas'), 1]
        config['reward']['matrix'][:, 32:] = [elem_id('water'), 1]

        return config
    
class PWTaskGenPlantAndWater(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Grow plants on the left side, and place water on the right side."
        
        config['reward']['matrix'][:, 0:32] = [elem_id('plant'), 1]
        config['reward']['matrix'][:, 32:] = [elem_id('water'), 1]
        return config
    



# ===================
# Drawing Tasks
# ===================
    
def make_random_goal(reward_matrix, num_generation, weight):
    dummy_config = PWTaskConfig()
    for n in range(num_generation):
        elem = np.random.choice(pw_element_names)
        gen_rule = np.random.choice([
            PWGenRule.CIRCLE,
            PWGenRule.CLONER_CIRCLE,
            PWGenRule.BOXES,
            PWGenRule.FILL_SLICE,
            PWGenRule.CLONER_ROOF,
            PWGenRule.SINE_WAVE,
            PWGenRule.ARCHES,
            PWGenRule.RAMPS,
        ])
        dummy_config['state_gen']['rules'].append(PWGenConfig(gen_rule, elem, num=1))
    dummy_task = PWTask(None, dummy_config)
    np_world = dummy_task.reset()[0]
    
    mask = (reward_matrix[:, :, 1] == 0) & ~(np_world[:, :] == 0)
    reward_matrix[mask, 0] = np_world[mask]
    reward_matrix[mask, 1] = weight
    
    
class PWTaskGenDraw(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Draw a given image on a blank world."
        # config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        # config['agent']['num_actions'] = 100
        # config['agent']['time_per_action'] = 2
        
        make_random_goal(config['reward']['matrix'], 3, 1)
        return config
    
saved_worlds = None
class PWTaskGenDraw160(PWTaskGen):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Draw a given image on a blank world."
        # config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        # config['agent']['num_actions'] = 100
        # config['agent']['time_per_action'] = 2
        
        global saved_worlds
        if saved_worlds is None:
            import os
            saved_worlds = np.load(os.path.join(os.path.dirname(__file__), '160worlds.npz'))['arr_0'].astype('float')
            saved_worlds = np.argmax(saved_worlds[:, :14], axis=1)
            print(saved_worlds.shape)

        config['reward']['matrix'][:, :, 0] = saved_worlds[0]
        config['reward']['matrix'][:, :, 1] = 1
        return config