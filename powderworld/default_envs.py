from powderworld.env import *
from powderworld.sim import pw_elements, pw_type, pw_element_names

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

class PWEnvSandPlace(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Place sand in the bottom-right corner."
        config['agent']['num_actions'] = 20
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['reward']['rules'].append(PWRewConfig(PWRewRule.ELEM_COUNT, "sand", weight=1, y=3, x=3))
        return config
    
class PWEnvPlantBurn(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Place fire to burn a circle of plants"
        config['agent']['num_actions'] = 4
        config['agent']['time_per_action'] = 10
        config['agent']['ending_timesteps'] = 50
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, "plant", num=1))
        config['reward']['rules'].append(PWRewConfig(PWRewRule.ELEM_DESTROY, "plant", weight=1, y=-1, x=-1))
        return config
    
class PWEnvSandMove(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Move a circle of sand to the top-right corner. Cannot place sand."
        config['agent']['num_actions'] = 50
        config['agent']['time_per_action'] = 4
        config['agent']['disabled_elements'] = [pw_elements["sand"][0]]
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, "sand", num=1))
        config['reward']['rules'].append(PWRewConfig(PWRewRule.ELEM_DESTROY, "sand", weight=1, y=0, x=3))
        return config
    
# class PWEnvStoneTower(PWEnv):
#     def generate_task_config(self):
#         config = PWTaskConfig()
#         config['desc'] = "Build a tower of stone that reaches top of world."
#         config['agent']['num_actions'] = 50
#         config['agent']['time_per_action'] = 4
#         config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
#         config['reward']['rules'].append(PWRewConfig(PWRewRule.ELEM_COUNT, "stone", weight=1, y=0, x=0))
#         config['reward']['rules'].append(PWRewConfig(PWRewRule.ELEM_COUNT, "stone", weight=1, y=0, x=1))
#         config['reward']['rules'].append(PWRewConfig(PWRewRule.ELEM_COUNT, "stone", weight=1, y=0, x=2))
#         config['reward']['rules'].append(PWRewConfig(PWRewRule.ELEM_COUNT, "stone", weight=1, y=0, x=3))
#         return config
    
class PWEnvStoneTower(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Build a tower of stone that reaches top of world. Minimize elements present."
        config['agent']['num_actions'] = 50
        config['agent']['time_per_action'] = 4
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['reward']['rules'].append(PWRewConfig(PWRewRule.ELEM_COUNT, "stone", weight=1, y=0, x=0))
        config['reward']['rules'].append(PWRewConfig(PWRewRule.ELEM_COUNT, "stone", weight=1, y=0, x=1))
        config['reward']['rules'].append(PWRewConfig(PWRewRule.ELEM_COUNT, "stone", weight=1, y=0, x=2))
        config['reward']['rules'].append(PWRewConfig(PWRewRule.ELEM_COUNT, "stone", weight=1, y=0, x=3))
        config['reward']['rules'].append(PWRewConfig(PWRewRule.ELEM_COUNT, "empty", weight=0.2, x=-1, y=-1))
        return config
    
    
class PWEnvWaterMove(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Direct water from a source into a container in bottom-right."
        config['agent']['num_actions'] = 5
        config['agent']['time_per_action'] = 4
        config['agent']['disabled_elements'] = [pw_elements["water"][0]]
        config['agent']['disabled_elements'] = [pw_elements["ice"][0]]
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CLONER_CIRCLE, "water", num=1))
        config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CONTAINER, "wall", num=1, p1=3, p2=1))
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['reward']['rules'].append(PWRewConfig(PWRewRule.ELEM_COUNT, "water", weight=1, y=3, x=1))
        return config
    
class PWEnvDestroyAll(PWEnv):
    def generate_task_config(self):
        config = PWTaskConfig()
        config['desc'] = "Destroy everything."
        config['agent']['num_actions'] = 10
        config['agent']['time_per_action'] = 4
        config['agent']['ending_timesteps'] = 50
        
        for i in range(5):
            elem = np.random.choice(pw_element_names)
            config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, elem, num=1))
        
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['reward']['rules'].append(PWRewConfig(PWRewRule.ELEM_COUNT, "empty", weight=1, y=-1, x=-1))
        return config