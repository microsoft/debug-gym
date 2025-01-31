### config:
dynamic attribute access and management

1. `_cache = {}` needs to be instance-level, not shared
2. Clear cache when updating values in set()
```
    if name in self._cache:
        del self._cache[name]
```

