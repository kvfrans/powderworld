from powderworld.env import *
from powderworld.sim import pw_elements, pw_element_names
from powderworld.default_envs import elem_id, env_list, make_random_goal
from powderworld.default_envs import PWTaskGenDraw

class PWWrapper():
    def __init__(self, task_gen):
        self.task_gen = task_gen

class PWWrapperCircles(PWWrapper):
    def __init__(self, task_gen):
        self.task_gen = task_gen
    def generate_task_config(self):
        config = self.task_gen.generate_task_config()
        config['desc'] += " There are three generated circles."
        
        for i in range(3):
            elem = np.random.choice(pw_element_names)
            config['state_gen']['rules'].append(PWGenConfig(PWGenRule.CIRCLE, elem, num=1))
        
        return config
    
class PWWrapperRandomGeneration(PWWrapper):
    def __init__(self, task_gen, num_generation=1):
        self.task_gen = task_gen
        self.num_generation = num_generation
        
    def generate_task_config(self):
        config = self.task_gen.generate_task_config()
        config['desc'] += " There are {} generated structures.".format(self.num_generation)
        
        for n in range(self.num_generation):
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
            config['state_gen']['rules'].append(PWGenConfig(gen_rule, elem, num=1))
        return config
    
class PWWrapperRandomRewards(PWWrapper):
    def __init__(self, task_gen, num_generation=1):
        self.task_gen = task_gen
        self.num_generation = num_generation
        
    def generate_task_config(self):
        config = self.task_gen.generate_task_config()
        config['desc'] += " {} small bonus rewards are included.".format(self.num_generation)
        
        for n in range(self.num_generation):
            make_random_goal(config['reward']['matrix'], 1, np.random.choice([0.2, -0.2]))

        return config
    
class PWWrapperMultiTaskGen(PWWrapper):
    def __init__(self, task_gens):
        self.task_gens = task_gens
        
    def generate_task_config(self):
        task_gen = np.random.choice(self.task_gens)
        return task_gen.generate_task_config()
    
class PWWrapperMultiTaskGenAuto(PWWrapper):
    def __init__(self, num_task_gens=30):
        task_gen_list = np.random.choice(powderworld.default_envs.env_list_train, num_task_gens, replace=False)
        self.task_gens = [getattr(powderworld.default_envs, task_gen)() for task_gen in task_gen_list]
        
    def generate_task_config(self):
        task_gen = np.random.choice(self.task_gens)
        return task_gen.generate_task_config()
    
class PWWrapperTestTasks(PWWrapper):
    def __init__(self):
        task_gen_list = powderworld.default_envs.env_list_test
        self.task_gens = [getattr(powderworld.default_envs, task_gen)() for task_gen in task_gen_list]
        
    def generate_task_config(self):
        task_gen = np.random.choice(self.task_gens)
        return task_gen.generate_task_config()
    
class PWTaskGenAll(PWWrapper):
    def __init__(self):
        task_gen_list = powderworld.default_envs.env_list
        self.task_gens = [getattr(powderworld.env_list, task_gen)() for task_gen in task_gen_list]
        self.draw_task = PWTaskGenDraw()
        
    def generate_task_config(self):
        if np.random.uniform() < 0.5:
            task_gen = np.random.choice(self.task_gens)
        else:
            task_gen = self.draw_task
        return task_gen.generate_task_config()
    