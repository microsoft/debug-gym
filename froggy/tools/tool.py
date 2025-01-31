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

    def is_triggered(self, action):
        # e.g. ```pdb b src/main.py:42```
        return action.startswith(self.action)

    @abstractmethod
    def use(self, action, environment):
        raise NotImplementedError("use method must be implemented in the subclass.")

    @abstractmethod
    def reset(self):
        raise NotImplementedError("reset method must be implemented in the subclass.")
