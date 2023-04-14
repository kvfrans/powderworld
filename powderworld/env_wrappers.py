from powderworld.env import *
from powderworld.sim import pw_elements, pw_element_names

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