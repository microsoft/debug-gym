"""Mock filelock module for testing purposes."""

class FileLock:
    """Mock FileLock class."""
    
    def __init__(self, lock_file):
        self.lock_file = lock_file
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass