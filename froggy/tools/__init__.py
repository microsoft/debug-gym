from dataclasses import dataclass
from .toolbox import Toolbox

class EnvironmentTool:
    name: str = None
    description: str = None
    instructions: str = None

    def __init__(self):
        self.environment = None

    def register(self, environment):
        self.environment = environment

    def is_triggered(self, text):
        raise NotImplementedError("is_triggered method must be implemented.")

    def use(self, action, environment):
        raise NotImplementedError("use method must be implemented.")
