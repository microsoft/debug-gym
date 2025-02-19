from typing import Any, Callable, Dict, Optional, Type


class Toolbox:
    _tool_registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str = None, config_cls: Optional[Any] = None) -> Callable:
        def decorator(subclass: Type) -> Type:
            name_ = name or subclass.__name__.lower().replace("tool", "")
            if name_ in cls._tool_registry:
                if subclass != cls._tool_registry[name_][0]:
                    raise ValueError(f"Cannot register '{name_}' multiple times.")
                return subclass

            cls._tool_registry[name_] = (subclass, config_cls)
            subclass.registered_name = name_
            return subclass

        return decorator

    @classmethod
    def get_tool(cls, name: str, **kwargs) -> Any:
        base_name = name.split(":")[0]
        if base_name not in cls._tool_registry:
            raise ValueError(f"Unknown tool {base_name}")

        tool_cls, _ = cls._tool_registry[base_name]

        return tool_cls(**kwargs)
