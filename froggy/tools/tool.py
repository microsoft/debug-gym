from abc import ABC, abstractmethod


class EnvironmentTool(ABC):
    name: str = None
    action: str = None
    instructions: str = None

    def __init__(self):
        self.environment = None

    def register(self, environment):
        from froggy.envs.env import RepoEnv

        if not isinstance(environment, RepoEnv):
            raise ValueError("The environment must be a RepoEnv instance.")

        self.environment = environment

    @abstractmethod
    def use(self, action, environment):
        raise NotImplementedError("use method must be implemented in the subclass.")
