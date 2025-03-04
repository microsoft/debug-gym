import pytest

from debug_gym.gym.tools.toolbox import Toolbox


def test_register_tool():
    @Toolbox.register()
    class MyTool:
        def __init__(self, value=None):
            self.value = value

    assert "my" in Toolbox._tool_registry
    assert Toolbox._tool_registry["my"][0] is MyTool


def test_register_tool_with_name():
    @Toolbox.register(name="custom")
    class AnotherTool:
        pass

    assert "custom" in Toolbox._tool_registry
    assert Toolbox._tool_registry["custom"][0] is AnotherTool


def test_get_tool():
    @Toolbox.register()
    class TestTool:
        def __init__(self, data=None):
            self.data = data

    instance = Toolbox.get_tool("test", data=123)
    assert isinstance(instance, TestTool)
    assert instance.data == 123


def test_get_tool_unknown_name():
    with pytest.raises(ValueError) as exc:
        Toolbox.get_tool("not_registered")
    assert "Unknown tool not_registered" in str(exc.value)


def test_register_existing_tool_raises():
    @Toolbox.register()
    class DuplicateTool:
        pass

    with pytest.raises(ValueError) as exc:

        @Toolbox.register()
        class DuplicateTool:
            pass

    assert "Cannot register 'duplicate' multiple times." in str(exc.value)
