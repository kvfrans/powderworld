from powderworld.env import *
from powderworld.sim import pw_elements, pw_element_names

# =====================================
# How Powderworld environments work:
# - A TaskConfig defines a single task in Powderworld.
# - Powderworld Gym environments are *distributions* of tasks.
# - The base class is PWEnv. This defines an OpenAI Gym Environment. It needs a 'generate_task_config' function.
# - To make a new task distribution, extend PWEnv with a generator function.
# 
# 
#
# 
#

# =====================================

env_list = [
    'PWEnvSandPlace',
    'PWEnvPlantBurn',
    'PWEnvPlantBurnOne',
    'PWEnvSandMove',
    'PWEnvStoneTower',
    'PWEnvStoneTowerPenalty',
    'PWEnvWaterMove',
    'PWEnvDestroyAll',
    'PWEnvCollapseStoneTower',
    'PWEnvBuildLemmingBridge',
    'PWEnvHerdBirds',
    'PWEnvMakePlantSpread',
    'PWEnvCreateStone',
    'PWEnvPreserveIce',
    'PWEnvFloodWater',
    'PWEnvFloodGas',
    'PWEnvPlantAndLava',
]

def elem_id(name):
    return pw_elements[name][0]
    
    

class PWEnvSandPlace(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Place sand in the bottom-right corner."
        config['agent']['num_actions'] = 20
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        
        config['reward']['matrix'][48:, 48:] = [elem_id('sand'), 1]
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CONTAINER, "wood", num=1, y=48, x=48))
        return config
    
class PWEnvPlantBurnOne(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Place fire to burn a generated circle of plants"
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        
        config['agent']['num_actions'] = 1
        config['agent']['time_per_action'] = 10
        config['agent']['ending_timesteps'] = 50
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, "plant", num=1))
        
        config['reward']['matrix'][:, :] = [elem_id('plant'), -1]
        return config
    
class PWEnvPlantBurn(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Place fire to burn a generated circle of plants"
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        
        config['agent']['num_actions'] = 4
        config['agent']['time_per_action'] = 10
        config['agent']['ending_timesteps'] = 50
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, "plant", num=1))
        
        config['reward']['matrix'][:, :] = [elem_id('plant'), -1]
        return config
    
class PWEnvSandMove(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Move a generated circle of sand to the top-right corner. Cannot place sand."
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['agent']['num_actions'] = 50
        config['agent']['time_per_action'] = 4
        config['agent']['disabled_elements'] = [pw_elements["sand"][0]]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, "sand", num=1))
        config['reward']['matrix'][0:16, 48:] = [elem_id('sand'), 1]
        return config
    
class PWEnvStoneTower(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Build a tower of stone that reaches top of world."
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['agent']['num_actions'] = 50
        config['agent']['time_per_action'] = 4
        
        config['reward']['matrix'][0:16, :] = [elem_id('stone'), 1]
        return config
    
class PWEnvStoneTowerPenalty(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['desc'] = "Build a tower of stone that reaches top of world. Minimize elements present."
        config['agent']['num_actions'] = 50
        config['agent']['time_per_action'] = 4
        
        config['reward']['matrix'][0:16, :] = [elem_id('stone'), 1]
        config['reward']['matrix'][16:, :] = [elem_id('empty'), 0.1]
        return config
    
    
class PWEnvWaterMove(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['desc'] = "Direct water from a generated source into a container in bottom-right."
        config['agent']['num_actions'] = 5
        config['agent']['time_per_action'] = 4
        config['agent']['disabled_elements'] = [elem_id('water'), elem_id('ice')]
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CLONER_CIRCLE, "water", num=1))
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        
        x = np.random.randint(0, 48) 
        config['reward']['matrix'][48:, x:x+16] = [elem_id('water'), 1]
        return config
    
class PWEnvDestroyAll(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['desc'] = "Destroy everything."
        config['agent']['num_actions'] = 10
        config['agent']['time_per_action'] = 4
        config['agent']['ending_timesteps'] = 50
        
        for i in range(5):
            elem = np.random.choice(pw_element_names)
            config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, elem, num=1))
        
        config['reward']['matrix'][:, :] = [elem_id('empty'), 1]
        return config
    
class PWEnvCollapseStoneTower(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['desc'] = "Knock down a generated stone tower."
        config['agent']['num_actions'] = 10
        config['agent']['time_per_action'] = 4
        config['agent']['ending_timesteps'] = 50
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.ARCHES, 'stone', num=5))
        
        config['reward']['matrix'][:32, :] = [elem_id('stone'), -1]

            
        return config
    
class PWEnvBuildLemmingBridge(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['desc'] = "Build a bridge so lemmings can cross a chasm."
        config['agent']['num_actions'] = 50
        config['agent']['time_per_action'] = 1
        config['agent']['ending_timesteps'] = 50
        config['agent']['disabled_elements'] = [elem_id('agentLemming')]

        
        lemming_height = np.random.randint(0,4) * 16
        goal_height = np.random.randint(1,4) * 16
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.SMALL_CIRCLE, "agentLemming", num=1, y=lemming_height, x=0))
        for y in range(lemming_height+1, 4):
            config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wall", num=1, y=y, x=0))

        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CONTAINER, "wall", num=1, y=goal_height, x=3))
        for y in range(goal_height+1, 4):
            config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "wall", num=1, y=y, x=3))
        
        config['reward']['matrix'][goal_height:goal_height+16, 48:] = [elem_id('agentLemming'), 1]
        return config
    
class PWEnvHerdBirds(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Herd a flock of generated birds into a container."
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['agent']['num_actions'] = 100
        config['agent']['time_per_action'] = 2
        config['agent']['disabled_elements'] = [elem_id('agentBird'), elem_id('cloner')]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, "agentBird", num=1))
        x = np.random.randint(0, 48) 
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CONTAINER, "wall", num=1, y=48, x=x))
        config['reward']['matrix'][48:, x:x+16] = [elem_id('agentBird'), 1]

        return config
    
class PWEnvMakePlantSpread(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Spread a plant as much as possible."
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['agent']['num_actions'] = 10
        config['agent']['time_per_action'] = 4
        config['agent']['ending_timesteps'] = 50
        config['agent']['disabled_elements'] = [elem_id('plant')]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, "plant", num=1))        
        config['reward']['matrix'][:, :] = [elem_id('plant'), 1]
        return config


class PWEnvCreateStone(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Given a generated sea of water, create stone."
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['agent']['num_actions'] = 20
        config['agent']['time_per_action'] = 10
        config['agent']['disabled_elements'] = [elem_id('stone')]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "water", num=1, y=3, x=0))    
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "water", num=1, y=3, x=1))        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "water", num=1, y=3, x=2))        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.FILLED_BOX, "water", num=1, y=3, x=3))        

        config['reward']['matrix'][:, :] = [elem_id('stone'), 1]

        return config
    
class PWEnvPreserveIce(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Given generated circles of ice, maintain the ice."
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['agent']['num_actions'] = 100
        config['agent']['time_per_action'] = 1
        config['agent']['ending_timesteps'] = 50
        config['agent']['disabled_elements'] = [pw_elements["ice"][0]]
        
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, "ice", num=2))          

        config['reward']['matrix'][:, :] = [elem_id('ice'), 1]

        return config
    
class PWEnvFloodWater(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Flood a map with water."
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['agent']['num_actions'] = 20
        config['agent']['time_per_action'] = 10
        config['agent']['ending_timesteps'] = 50
        
        config['reward']['matrix'][:, :] = [elem_id('water'), 1]
        return config
    
class PWEnvFloodGas(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Flood a map with gas. Cannot place gas."
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['agent']['num_actions'] = 20
        config['agent']['time_per_action'] = 10
        config['agent']['ending_timesteps'] = 50
        config['agent']['disabled_elements'] = [pw_elements["gas"][0]]
        
        config['reward']['matrix'][:, :] = [elem_id('gas'), 1]
        return config
    
class PWEnvPlantAndLava(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Grow plants on the left side, and place lava on the right side."
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['agent']['num_actions'] = 100
        config['agent']['time_per_action'] = 2
        
        config['reward']['matrix'][:, 0:32] = [elem_id('plant'), 1]
        config['reward']['matrix'][:, 32:] = [elem_id('lava'), 1]

        return config
