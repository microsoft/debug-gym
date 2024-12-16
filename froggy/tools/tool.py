class EnvironmentTool:
    name: str = None
    description: str = None
    instructions: str = None

    def __init__(self):
        self.environment = None
        self._states = {}

    def register(self, environment):
        self.environment = environment

    def is_triggered(self, text):
        raise NotImplementedError("is_triggered method must be implemented.")

    def use(self, action, environment):
        raise NotImplementedError("use method must be implemented.")

    @property
    def states(self):
        # this is a placeholder, should be overridden by subclasses
        return self._states

    def load_states(self, states):
        pass
