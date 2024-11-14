import importlib 
from pathlib import Path
from typing import Type, Dict, Optional, Any, Callable

class Toolbox:
    _tool_registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str=None, config_cls: Optional[Any] = None) -> Callable:
        def decorator(subclass: Type) -> Type:
            nonlocal name
            if name is None:
                name = subclass.__name__.lower().replace('tool', '')
            if name in cls._tool_registry:
                if subclass != cls._tool_registry[name][0]:
                    raise ValueError(
                        f"Cannot register '{name}' multiple times."
                    )
                return subclass

            cls._tool_registry[name] = (subclass, config_cls)
            subclass.registered_name = name
            return subclass

        return decorator

    @classmethod
    def load_tools(cls):
        tools_dir = Path(__file__).parent
        for tool_file in tools_dir.glob("*.py"):
            if tool_file.stem not in ["__init__", "registry"]:
                importlib.import_module(f"froggy.tools.{tool_file.stem}")

    @classmethod
    def get_tool(cls, name: str, **kwargs) -> Any:
        base_name = name.split(':')[0]
        if base_name not in cls._tool_registry:
            raise ValueError(f"Unknown tool {base_name}")
            
        tool_cls, _ = cls._tool_registry[base_name]
        
        if ':' in name:
            subtype = name.split(':')[1]
            return tool_cls.get(subtype)
        return tool_cls(**kwargs)