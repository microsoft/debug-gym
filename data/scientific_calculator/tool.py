class EnvironmentTool:
    name: str = None
    action: str = None
    instructions: str = None

    def __init__(self):
        self.environment = None

    def use(self, action):
        raise NotImplementedError("use method must be implemented in the subclass.")
