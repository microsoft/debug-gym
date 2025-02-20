import unittest

from config_code import ConfigManager


class TestConfigManager(unittest.TestCase):
    def setUp(self):
        # Clear the cache between tests
        ConfigManager._cached_configs = {}

    def test_config_updates(self):
        # Create initial config
        config = ConfigManager({
            'database': {
                'host': 'initial-host',
                'port': 5432
            }
        })

        # First access creates cached value
        initial_db = config.database
        self.assertEqual(initial_db.host, 'initial-host')

        # Update the configuration
        config.set('database', {
            'host': 'new-host',
            'port': 5432
        })

        # This should return the new value but will return cached value
        self.assertEqual(config.database.host, 'new-host')

    def test_multiple_instances(self):
        config1 = ConfigManager({
            'database': {'host': 'host1'}
        })
        config2 = ConfigManager({
            'database': {'host': 'host2'}
        })

        # Access both configs
        _ = config1.database
        _ = config2.database

        # Update first config
        config1.set('database', {'host': 'new-host'})

        # Access both configs again
        # Due to shared cache, this might affect both instances
        self.assertEqual(config1.database.host, 'new-host')
        self.assertEqual(config2.database.host, 'host2')


if __name__ == '__main__':
    unittest.main()