from powderworld.env import *

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
        config['agent']['num_actions'] = 20
        config['state_gen']['seed'] = np.random.randint(config['general']['num_task_variations'])
        config['reward']['rules'].append(PWRewConfig(PWRewRule.ELEM_COUNT, "sand", weight=1, x=3, y=3))
        return config