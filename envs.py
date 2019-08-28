import gym


class BaseEnv:
    def __init__(self, env_string):
        self.env_string = env_string
        self._instance = None

    def get_env(self):
        if self._instance is None:
            self._instance = gym.make(self.env_string)
        return self._instance

    def get_state_space_dim(self):
        return self.get_env().observation_space.shape[0]

    def get_action_space_dim(self):
        return 0
        #return self.get_env().action_space.shape[0]



class EnvGenerator:
    def __init__(self, env_string):
        self.env_string = env_string

    def get_env_wrapper(self):
        return BaseEnv(self.env_string)

    def get_env_definitions(self):
        env = self.get_env_wrapper()
        return env.get_state_space_dim(), env.get_action_space_dim()
