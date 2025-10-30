class ConfigManager:
    """A configuration manager that allows dot notation access to configuration values."""
    _cache = {}
    
    def __init__(self, config_dict=None):
        self._config = config_dict or {}
        self._defaults = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'timeout': 30
            },
            'api': {
                'base_url': 'http://api.example.com',
                'version': 'v1'
            }
        }

    def __getattr__(self, name):
        if name in self._cache:
            return self._cache[name]

        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                nested = ConfigManager(value)
                self._cache[name] = nested
                return nested
            return value
        elif name in self._defaults:
            value = self._defaults[name]
            if isinstance(value, dict):
                return ConfigManager(value)
            return value
        raise AttributeError(f"Configuration '{name}' not found")

    def set(self, name, value):
        self._config[name] = value
