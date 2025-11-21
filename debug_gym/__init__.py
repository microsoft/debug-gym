from debug_gym.version import __version__

__all__ = ["__version__"]


def __getattr__(name):
    """Lazy load FrogyToolParser only when explicitly requested"""
    if name == "FrogyToolParser":
        from debug_gym.frogboss import FrogyToolParser

        return FrogyToolParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
