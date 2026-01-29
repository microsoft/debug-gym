"""Tests for MCP Proxy Tool functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.mcp_proxy import (
    MCPClientSSE,
    MCPToolFactory,
    MCPToolRegistry,
    _async_register_mcp_server_sse,
)
from debug_gym.gym.tools.toolbox import Toolbox

# Mock MCP server tools for testing
MOCK_TOOLS = [
    {
        "name": "echo",
        "description": "Echo the input back",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message to echo"}
            },
            "required": ["message"],
        },
    },
    {
        "name": "add",
        "description": "Add two numbers",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    },
]


@pytest.fixture
def clean_toolbox():
    """Fixture to save and restore Toolbox registry."""
    original = Toolbox._tool_registry.copy()
    yield
    Toolbox._tool_registry = original


def create_mock_mcp_client():
    """Create a mock MCP client that simulates server responses."""
    mock_client = AsyncMock(spec=MCPClientSSE)
    mock_client.url = "http://localhost:19876/sse"
    mock_client.headers = {}
    mock_client._session = MagicMock()
    mock_client._context_stack = MagicMock()

    mock_client.start = AsyncMock()
    mock_client.stop = AsyncMock()
    mock_client.initialize = AsyncMock(
        return_value={
            "serverInfo": {"name": "test-server", "version": "1.0.0"},
            "capabilities": {},
        }
    )
    mock_client.list_tools = AsyncMock(return_value=MOCK_TOOLS)

    def mock_call_tool(tool_name, arguments):
        if tool_name == "echo":
            return {
                "content": [{"type": "text", "text": arguments.get("message", "")}],
                "isError": False,
            }
        elif tool_name == "add":
            result = arguments.get("a", 0) + arguments.get("b", 0)
            return {
                "content": [{"type": "text", "text": str(result)}],
                "isError": False,
            }
        return {"content": [{"type": "text", "text": "Unknown tool"}], "isError": True}

    mock_client.call_tool = AsyncMock(side_effect=mock_call_tool)

    return mock_client


class TestMCPClientSSE:
    """Tests for MCPClientSSE."""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test MCPClientSSE can be initialized with url and headers."""
        client = MCPClientSSE(
            "http://localhost:8000/sse", {"Authorization": "Bearer token"}
        )
        assert client.url == "http://localhost:8000/sse"
        assert client.headers == {"Authorization": "Bearer token"}
        assert client._session is None
        assert client._context_stack is None


class TestMCPToolFactory:
    """Tests for MCPToolFactory."""

    def test_convert_simple_schema(self):
        """Test JSON Schema conversion to EnvironmentTool format."""
        schema = MOCK_TOOLS[1]["inputSchema"]  # "add" tool schema

        arguments = MCPToolFactory._convert_schema(schema)

        assert "a" in arguments
        assert arguments["a"]["type"] == ["number"]
        assert "First number" in arguments["a"]["description"]

        assert "b" in arguments
        assert arguments["b"]["type"] == ["number"]

    def test_convert_schema_with_enum(self):
        """Test schema conversion with enum values."""
        schema = {
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "Operation mode",
                    "enum": ["read", "write", "append"],
                }
            }
        }

        arguments = MCPToolFactory._convert_schema(schema)
        assert arguments["mode"]["enum"] == ["read", "write", "append"]

    @pytest.mark.asyncio
    async def test_create_tool_class(self):
        """Test creating an EnvironmentTool class from MCP definition."""
        mock_client = create_mock_mcp_client()

        tool_class = MCPToolFactory.create_tool_class(
            mock_client, MOCK_TOOLS[0], tool_prefix="test_"  # "echo" tool
        )

        # Verify tool class attributes
        assert tool_class.name == "test_echo"
        assert tool_class.description == "Echo the input back"
        assert "message" in tool_class.arguments

        # Verify tool can be instantiated
        tool_instance = tool_class()
        assert hasattr(tool_instance, "use")

    @pytest.mark.asyncio
    async def test_dynamic_tool_execution(self):
        """Test that dynamically created tools can execute."""
        mock_client = create_mock_mcp_client()

        tool_class = MCPToolFactory.create_tool_class(
            mock_client, MOCK_TOOLS[0]
        )  # "echo"
        tool = tool_class()

        # Mock environment
        mock_env = MagicMock()

        # Execute tool
        result = tool.use(mock_env, message="Hello, World!")

        # Verify execution
        assert isinstance(result, Observation)
        assert result.source == tool.name
        assert "Hello, World!" in result.observation
        mock_client.call_tool.assert_called_once_with(
            "echo", {"message": "Hello, World!"}
        )

    @pytest.mark.asyncio
    async def test_add_tool_execution(self):
        """Test the add tool executes correctly."""
        mock_client = create_mock_mcp_client()

        tool_class = MCPToolFactory.create_tool_class(
            mock_client, MOCK_TOOLS[1]
        )  # "add"
        tool = tool_class()

        mock_env = MagicMock()
        result = tool.use(mock_env, a=5, b=3)

        assert isinstance(result, Observation)
        assert "8" in result.observation
        mock_client.call_tool.assert_called_once_with("add", {"a": 5, "b": 3})


class TestMCPToolRegistry:
    """Tests for MCPToolRegistry."""

    @pytest.mark.asyncio
    async def test_registry_initialization(self):
        """Test registry can be initialized."""
        registry = MCPToolRegistry()
        assert registry.clients == {}
        assert registry.registered_tools == []

    @pytest.mark.asyncio
    async def test_add_server_sse_discovery(self):
        """Test adding an SSE server discovers and registers tools."""
        registry = MCPToolRegistry()

        with patch("debug_gym.gym.tools.mcp_proxy.MCPClientSSE") as mock_client_class:
            mock_client_class.return_value = create_mock_mcp_client()

            tools = await registry.add_server_sse(
                server_id="test-server",
                url="http://localhost:19876/sse",
                tool_prefix="test_",
                auto_register=False,
            )

            # Verify tools were created from mock server
            assert len(tools) == 2
            assert tools[0].name == "test_echo"
            assert tools[1].name == "test_add"

            # Verify client was added
            assert "test-server" in registry.clients

    @pytest.mark.asyncio
    async def test_remove_server(self):
        """Test removing a server closes its connection."""
        registry = MCPToolRegistry()

        mock_client = create_mock_mcp_client()
        registry.clients["test"] = mock_client

        await registry.remove_server("test")

        assert "test" not in registry.clients
        mock_client.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_all_servers(self):
        """Test cleanup stops all servers."""
        registry = MCPToolRegistry()

        for i in range(3):
            registry.clients[f"server{i}"] = create_mock_mcp_client()

        await registry.cleanup()

        assert len(registry.clients) == 0


class TestIntegration:
    """Integration tests for MCP tool system."""

    @pytest.mark.asyncio
    async def test_tool_registration_flow(self, clean_toolbox):
        """Test complete flow from server registration to tool availability."""
        with patch("debug_gym.gym.tools.mcp_proxy.MCPClientSSE") as mock_client_class:
            mock_client_class.return_value = create_mock_mcp_client()

            await _async_register_mcp_server_sse(
                server_id="integration-test",
                url="http://localhost:19876/sse",
                tool_prefix="int_",
            )

            # Verify tools are in Toolbox
            assert "int_echo" in Toolbox._tool_registry
            assert "int_add" in Toolbox._tool_registry

            # Get and use tool from Toolbox
            echo_tool = Toolbox.get_tool("int_echo")
            assert echo_tool is not None
            assert echo_tool.name == "int_echo"

            add_tool = Toolbox.get_tool("int_add")
            assert add_tool is not None
            assert add_tool.name == "int_add"

    @pytest.mark.asyncio
    async def test_tool_execution_end_to_end(self, clean_toolbox):
        """Test tools can be executed end-to-end with mock server."""
        with patch("debug_gym.gym.tools.mcp_proxy.MCPClientSSE") as mock_client_class:
            mock_client_class.return_value = create_mock_mcp_client()

            await _async_register_mcp_server_sse(
                server_id="e2e-test", url="http://localhost:19876/sse", tool_prefix=""
            )

            mock_env = MagicMock()

            # Execute echo tool - get_tool returns instance
            echo_tool = Toolbox.get_tool("echo")
            result = echo_tool.use(mock_env, message="test message")
            assert "test message" in result.observation

            # Execute add tool
            add_tool = Toolbox.get_tool("add")
            result = add_tool.use(mock_env, a=10, b=20)
            assert "30" in result.observation


class TestRegisterMcpServers:
    """Tests for register_mcp_servers function in experiment.py."""

    def test_register_mcp_servers_from_config(self, clean_toolbox):
        """Test registering MCP servers from experiment config."""
        from debug_gym.experiment import register_mcp_servers

        config = {
            "mcp_servers": {
                "test-server": {
                    "url": "http://localhost:19876/sse",
                    "tool_prefix": "test_",
                }
            }
        }

        mock_env = MagicMock()
        mock_logger = MagicMock()

        with patch(
            "debug_gym.gym.tools.mcp_proxy.register_mcp_server_sse"
        ) as mock_register:
            mock_register.return_value = []

            register_mcp_servers(mock_env, config, mock_logger)

            mock_register.assert_called_once_with(
                server_id="test-server",
                url="http://localhost:19876/sse",
                headers=None,
                tool_prefix="test_",
            )

    def test_register_mcp_servers_with_headers(self, clean_toolbox):
        """Test registering MCP servers with custom headers."""
        from debug_gym.experiment import register_mcp_servers

        config = {
            "mcp_servers": {
                "auth-server": {
                    "url": "http://localhost:19876/sse",
                    "headers": {"Authorization": "Bearer token"},
                    "tool_prefix": "auth_",
                }
            }
        }

        mock_env = MagicMock()
        mock_logger = MagicMock()

        with patch(
            "debug_gym.gym.tools.mcp_proxy.register_mcp_server_sse"
        ) as mock_register:
            mock_register.return_value = []

            register_mcp_servers(mock_env, config, mock_logger)

            mock_register.assert_called_once_with(
                server_id="auth-server",
                url="http://localhost:19876/sse",
                headers={"Authorization": "Bearer token"},
                tool_prefix="auth_",
            )

    def test_register_mcp_servers_multiple(self, clean_toolbox):
        """Test registering multiple MCP servers."""
        from debug_gym.experiment import register_mcp_servers

        config = {
            "mcp_servers": {
                "server1": {
                    "url": "http://localhost:8001/sse",
                    "tool_prefix": "s1_",
                },
                "server2": {
                    "url": "http://localhost:8002/sse",
                    "tool_prefix": "s2_",
                },
            }
        }

        mock_env = MagicMock()
        mock_logger = MagicMock()

        with patch(
            "debug_gym.gym.tools.mcp_proxy.register_mcp_server_sse"
        ) as mock_register:
            mock_register.return_value = []

            register_mcp_servers(mock_env, config, mock_logger)

            assert mock_register.call_count == 2

    def test_register_mcp_servers_empty_config(self, clean_toolbox):
        """Test with no MCP servers in config."""
        from debug_gym.experiment import register_mcp_servers

        config = {}
        mock_env = MagicMock()
        mock_logger = MagicMock()

        with patch(
            "debug_gym.gym.tools.mcp_proxy.register_mcp_server_sse"
        ) as mock_register:
            register_mcp_servers(mock_env, config, mock_logger)

            mock_register.assert_not_called()

    def test_register_mcp_servers_invalid_config(self, clean_toolbox):
        """Test with invalid MCP server config (missing url)."""
        from debug_gym.experiment import register_mcp_servers

        config = {
            "mcp_servers": {
                "no-url": {},  # Missing url
                "valid": {"url": "http://localhost:8000/sse"},  # Valid
            }
        }

        mock_env = MagicMock()
        mock_logger = MagicMock()

        with patch(
            "debug_gym.gym.tools.mcp_proxy.register_mcp_server_sse"
        ) as mock_register:
            mock_register.return_value = []

            register_mcp_servers(mock_env, config, mock_logger)

            # Only valid config should be registered
            mock_register.assert_called_once()
            # Should log warning for invalid config
            assert mock_logger.warning.call_count == 1

    def test_register_mcp_servers_handles_errors(self, clean_toolbox):
        """Test that registration errors are logged but don't stop other registrations."""
        from debug_gym.experiment import register_mcp_servers

        config = {
            "mcp_servers": {
                "failing": {"url": "http://localhost:8001/sse"},
                "working": {"url": "http://localhost:8002/sse"},
            }
        }

        mock_env = MagicMock()
        mock_logger = MagicMock()

        with patch(
            "debug_gym.gym.tools.mcp_proxy.register_mcp_server_sse"
        ) as mock_register:
            # First call fails, second succeeds
            mock_register.side_effect = [Exception("Connection failed"), []]

            register_mcp_servers(mock_env, config, mock_logger)

            assert mock_register.call_count == 2
            mock_logger.error.assert_called_once()

    def test_register_mcp_servers_adds_tools_to_env(self, clean_toolbox):
        """Test that tools are added to the environment."""
        from debug_gym.experiment import register_mcp_servers

        config = {
            "mcp_servers": {
                "test-server": {
                    "url": "http://localhost:19876/sse",
                }
            }
        }

        mock_env = MagicMock()
        mock_logger = MagicMock()

        # Create mock tool classes that return instances
        mock_tool_instance = MagicMock()
        mock_tool_instance.name = "test_tool"
        mock_tool_class = MagicMock(return_value=mock_tool_instance)

        with patch(
            "debug_gym.gym.tools.mcp_proxy.register_mcp_server_sse"
        ) as mock_register:
            mock_register.return_value = [mock_tool_class]

            register_mcp_servers(mock_env, config, mock_logger)

            # Verify tool was instantiated and added to env
            mock_tool_class.assert_called_once()
            mock_env.add_tool.assert_called_once_with(mock_tool_instance)
