"""Tests for MCP Proxy Tool functionality."""

import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.mcp_proxy import MCPTool, discover_mcp_tools


def create_mock_mcp_session(tool_responses=None):
    """Create a mock MCP session that simulates server responses.

    Args:
        tool_responses: Dict mapping tool_name -> response function
    """
    tool_responses = tool_responses or {
        "echo": lambda args: args.get("message", ""),
        "add": lambda args: str(args.get("a", 0) + args.get("b", 0)),
    }

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()

    # Mock list_tools response
    mock_list_result = MagicMock()
    echo_tool = MagicMock()
    echo_tool.name = "echo"  # Set name explicitly to avoid MagicMock auto-attributes
    echo_tool.description = "Echo the input back"
    echo_tool.inputSchema = {
        "type": "object",
        "properties": {"message": {"type": "string", "description": "Message to echo"}},
        "required": ["message"],
    }
    add_tool = MagicMock()
    add_tool.name = "add"  # Set name explicitly to avoid MagicMock auto-attributes
    add_tool.description = "Add two numbers"
    add_tool.inputSchema = {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
        "required": ["a", "b"],
    }
    mock_list_result.tools = [echo_tool, add_tool]
    mock_session.list_tools = AsyncMock(return_value=mock_list_result)

    # Mock call_tool response
    async def mock_call_tool(tool_name, arguments):
        result = MagicMock()
        if tool_name in tool_responses:
            text = tool_responses[tool_name](arguments)
            result.content = [MagicMock(text=text)]
        else:
            result.content = [MagicMock(text=f"Unknown tool: {tool_name}")]
        return result

    mock_session.call_tool = mock_call_tool
    return mock_session


@pytest.fixture
def mock_mcp_server():
    """Fixture that mocks the MCP server connection at the SDK level."""
    session = create_mock_mcp_session()

    @asynccontextmanager
    async def mock_sse_client(url, headers=None):
        yield (AsyncMock(), AsyncMock())

    @asynccontextmanager
    async def mock_streamable_http_client(
        url, http_client=None, terminate_on_close=True
    ):
        yield (AsyncMock(), AsyncMock(), AsyncMock())

    @asynccontextmanager
    async def mock_client_session(read_stream, write_stream):
        yield session

    # Create mock modules
    mock_mcp = MagicMock()
    mock_mcp.ClientSession = mock_client_session
    mock_mcp_client_sse = MagicMock()
    mock_mcp_client_sse.sse_client = mock_sse_client
    mock_mcp_client_streamable_http = MagicMock()
    mock_mcp_client_streamable_http.streamable_http_client = mock_streamable_http_client

    with patch.dict(
        sys.modules,
        {
            "mcp": mock_mcp,
            "mcp.client": MagicMock(),
            "mcp.client.sse": mock_mcp_client_sse,
            "mcp.client.streamable_http": mock_mcp_client_streamable_http,
        },
    ):
        yield session


class TestMCPTool:
    """Tests for MCPTool."""

    def test_tool_initialization(self):
        """Test MCPTool can be initialized with required parameters."""
        tool = MCPTool(
            url="http://localhost:8000/sse",
            mcp_tool_name="echo",
            tool_name="test_echo",
            description="Test echo tool",
            input_schema={
                "properties": {"message": {"type": "string", "description": "Msg"}},
                "required": ["message"],
            },
        )
        assert tool.name == "test_echo"
        assert tool.description == "Test echo tool"
        assert "message" in tool.arguments
        assert tool._url == "http://localhost:8000/sse"
        assert tool._mcp_tool_name == "echo"

    def test_tool_defaults_name_from_mcp_tool_name(self):
        """Test tool name defaults to mcp_tool_name if not provided."""
        tool = MCPTool(url="http://localhost:8000/sse", mcp_tool_name="my_tool")
        assert tool.name == "my_tool"
        assert tool.description == "MCP tool: my_tool"

    def test_tool_default_transport_is_sse(self):
        """Test that default transport is SSE."""
        tool = MCPTool(url="http://localhost:8000/sse", mcp_tool_name="test")
        assert tool._transport == "sse"

    def test_tool_transport_sse(self):
        """Test MCPTool with SSE transport."""
        tool = MCPTool(
            url="http://localhost:8000/sse",
            mcp_tool_name="test",
            transport="sse",
        )
        assert tool._transport == "sse"

    def test_tool_transport_streamable_http(self):
        """Test MCPTool with streamable_http transport."""
        tool = MCPTool(
            url="https://api.example.com/mcp",
            mcp_tool_name="test",
            transport="streamable_http",
        )
        assert tool._transport == "streamable_http"

    def test_convert_schema_simple(self):
        """Test JSON Schema conversion to EnvironmentTool format."""
        tool = MCPTool(
            url="http://localhost:8000/sse",
            mcp_tool_name="test",
            input_schema={
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        )
        assert "a" in tool.arguments
        assert tool.arguments["a"]["type"] == ["number"]
        assert "First number" in tool.arguments["a"]["description"]
        assert "b" in tool.arguments
        assert tool.arguments["b"]["type"] == ["number"]

    def test_convert_schema_with_optional(self):
        """Test schema conversion with optional fields."""
        tool = MCPTool(
            url="http://localhost:8000/sse",
            mcp_tool_name="test",
            input_schema={
                "properties": {
                    "required_field": {"type": "string", "description": "Required"},
                    "optional_field": {"type": "string", "description": "Optional"},
                },
                "required": ["required_field"],
            },
        )
        assert tool.arguments["required_field"]["type"] == ["string"]
        assert tool.arguments["optional_field"]["type"] == ["string", "null"]

    def test_convert_schema_with_enum(self):
        """Test schema conversion with enum values."""
        tool = MCPTool(
            url="http://localhost:8000/sse",
            mcp_tool_name="test",
            input_schema={
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "Mode",
                        "enum": ["read", "write"],
                    }
                }
            },
        )
        assert tool.arguments["mode"]["enum"] == ["read", "write"]

    def test_tool_execution(self, mock_mcp_server):
        """Test tool executes and returns result from server."""
        tool = MCPTool(
            url="http://localhost:8000/sse",
            mcp_tool_name="echo",
        )
        mock_env = MagicMock()

        result = tool.use(mock_env, message="Hello, World!")

        assert isinstance(result, Observation)
        assert result.source == "echo"
        assert "Hello, World!" in result.observation

    def test_add_tool_execution(self, mock_mcp_server):
        """Test add tool executes correctly."""
        tool = MCPTool(
            url="http://localhost:8000/sse",
            mcp_tool_name="add",
        )
        mock_env = MagicMock()

        result = tool.use(mock_env, a=5, b=3)

        assert isinstance(result, Observation)
        assert "8" in result.observation


class TestMCPToolSerialization:
    """Tests for MCPTool serialization (pickle, deepcopy)."""

    def test_pickle_roundtrip(self):
        """Test MCPTool can be pickled and unpickled."""
        import pickle

        tool = MCPTool(
            url="http://localhost:8000/sse",
            mcp_tool_name="echo",
            tool_name="test_echo",
            description="Test echo tool",
            input_schema={
                "properties": {"message": {"type": "string", "description": "Msg"}},
                "required": ["message"],
            },
            headers={"Authorization": "Bearer token"},
        )

        # Pickle and unpickle
        pickled = pickle.dumps(tool)
        restored = pickle.loads(pickled)

        # Verify attributes are preserved
        assert restored.name == "test_echo"
        assert restored.description == "Test echo tool"
        assert restored._url == "http://localhost:8000/sse"
        assert restored._mcp_tool_name == "echo"
        assert restored._headers == {"Authorization": "Bearer token"}
        assert "message" in restored.arguments

    def test_pickle_with_used_tool(self, mock_mcp_server):
        """Test MCPTool can be pickled after use."""
        import pickle

        tool = MCPTool(
            url="http://localhost:8000/sse",
            mcp_tool_name="echo",
        )
        mock_env = MagicMock()

        # Execute tool
        tool.use(mock_env, message="test")

        # Pickle should succeed
        pickled = pickle.dumps(tool)
        restored = pickle.loads(pickled)

        # Verify attributes are preserved
        assert restored._url == tool._url
        assert restored._mcp_tool_name == tool._mcp_tool_name

    def test_deepcopy(self):
        """Test MCPTool can be deep copied."""
        import copy

        tool = MCPTool(
            url="http://localhost:8000/sse",
            mcp_tool_name="add",
            tool_name="test_add",
            description="Add numbers",
            input_schema={
                "properties": {
                    "a": {"type": "number", "description": "First"},
                    "b": {"type": "number", "description": "Second"},
                },
                "required": ["a", "b"],
            },
            headers={"X-Custom": "header"},
        )

        # Deep copy
        copied = copy.deepcopy(tool)

        # Verify it's a different object
        assert copied is not tool

        # Verify attributes are preserved
        assert copied.name == "test_add"
        assert copied.description == "Add numbers"
        assert copied._url == "http://localhost:8000/sse"
        assert copied._mcp_tool_name == "add"
        assert copied._headers == {"X-Custom": "header"}
        assert copied.arguments == tool.arguments

    def test_deepcopy_with_used_tool(self, mock_mcp_server):
        """Test MCPTool can be deep copied after use."""
        import copy

        tool = MCPTool(
            url="http://localhost:8000/sse",
            mcp_tool_name="echo",
        )
        mock_env = MagicMock()

        # Execute tool
        tool.use(mock_env, message="test")

        # Deep copy should succeed
        copied = copy.deepcopy(tool)

        # Verify copy has same attributes
        assert copied._url == tool._url
        assert copied._mcp_tool_name == tool._mcp_tool_name

    def test_deepcopy_preserves_history(self):
        """Test that deepcopy preserves history list."""
        import copy

        tool = MCPTool(
            url="http://localhost:8000/sse",
            mcp_tool_name="echo",
        )
        # Add some mock history
        tool.history = [{"action": "test1"}, {"action": "test2"}]

        copied = copy.deepcopy(tool)

        # History should be copied
        assert copied.history == [{"action": "test1"}, {"action": "test2"}]
        # But should be independent
        assert copied.history is not tool.history


class TestDiscoverMcpTools:
    """Tests for discover_mcp_tools function."""

    def test_discover_returns_tools(self, mock_mcp_server):
        """Test discover_mcp_tools returns MCPTool instances."""
        tools = discover_mcp_tools(
            url="http://localhost:8000/sse",
            tool_prefix="test_",
        )

        assert len(tools) == 2
        assert all(isinstance(t, MCPTool) for t in tools)
        assert tools[0].name == "test_echo"
        assert tools[0].description == "Echo the input back"
        assert tools[1].name == "test_add"
        assert tools[1].description == "Add two numbers"

    def test_discover_with_tool_filter(self, mock_mcp_server):
        """Test discover_mcp_tools respects tool_filter."""
        tools = discover_mcp_tools(
            url="http://localhost:8000/sse",
            tool_filter=["echo"],
        )

        assert len(tools) == 1
        assert tools[0].name == "echo"

    def test_discover_with_headers(self, mock_mcp_server):
        """Test discover_mcp_tools passes headers."""
        tools = discover_mcp_tools(
            url="http://localhost:8000/sse",
            headers={"Authorization": "Bearer token"},
        )

        assert len(tools) == 2
        # Headers should be stored in each tool
        assert tools[0]._headers == {"Authorization": "Bearer token"}

    def test_discover_with_transport_sse(self, mock_mcp_server):
        """Test discover_mcp_tools with SSE transport."""
        tools = discover_mcp_tools(
            url="http://localhost:8000/sse",
            transport="sse",
        )

        assert len(tools) == 2
        assert tools[0]._transport == "sse"
        assert tools[1]._transport == "sse"

    def test_discover_with_transport_streamable_http(self, mock_mcp_server):
        """Test discover_mcp_tools with streamable_http transport."""
        tools = discover_mcp_tools(
            url="http://localhost:8000/mcp",
            transport="streamable_http",
        )

        assert len(tools) == 2
        assert tools[0]._transport == "streamable_http"
        assert tools[1]._transport == "streamable_http"

    def test_discover_default_transport_is_sse(self, mock_mcp_server):
        """Test that default transport is SSE."""
        tools = discover_mcp_tools(url="http://localhost:8000/sse")

        assert len(tools) == 2
        assert tools[0]._transport == "sse"


class TestDiscoveredToolExecution:
    """Test that discovered tools can execute against mock server."""

    def test_discovered_echo_tool(self, mock_mcp_server):
        """Test discovered echo tool executes correctly."""
        tools = discover_mcp_tools(url="http://localhost:8000/sse")
        echo_tool = next(t for t in tools if t.name == "echo")

        mock_env = MagicMock()
        result = echo_tool.use(mock_env, message="test message")

        assert isinstance(result, Observation)
        assert "test message" in result.observation

    def test_discovered_add_tool(self, mock_mcp_server):
        """Test discovered add tool executes correctly."""
        tools = discover_mcp_tools(url="http://localhost:8000/sse")
        add_tool = next(t for t in tools if t.name == "add")

        mock_env = MagicMock()
        result = add_tool.use(mock_env, a=10, b=20)

        assert isinstance(result, Observation)
        assert "30" in result.observation


class TestRegisterMcpServers:
    """Tests for register_mcp_servers function in experiment.py."""

    def test_register_mcp_servers_from_config(self, mock_mcp_server):
        """Test registering MCP servers from experiment config."""
        from debug_gym.experiment import register_mcp_servers

        config = {
            "mcp_servers": {
                "test-server": {
                    "url": "http://localhost:8000/sse",
                    "tool_prefix": "test_",
                }
            }
        }

        mock_env = MagicMock()
        mock_logger = MagicMock()

        register_mcp_servers(mock_env, config, mock_logger)

        # Should have added 2 tools (echo and add)
        assert mock_env.add_tool.call_count == 2

    def test_register_mcp_servers_with_transport(self, mock_mcp_server):
        """Test registering MCP servers with explicit transport."""
        from debug_gym.experiment import register_mcp_servers

        config = {
            "mcp_servers": {
                "sse-server": {
                    "url": "http://localhost:8000/sse",
                    "transport": "sse",
                },
            }
        }

        mock_env = MagicMock()
        mock_logger = MagicMock()

        register_mcp_servers(mock_env, config, mock_logger)

        # Should have added 2 tools (echo and add)
        assert mock_env.add_tool.call_count == 2
        added_tool = mock_env.add_tool.call_args_list[0][0][0]
        assert added_tool._transport == "sse"

    def test_register_mcp_servers_with_streamable_http_transport(self, mock_mcp_server):
        """Test registering MCP servers with streamable_http transport."""
        from debug_gym.experiment import register_mcp_servers

        config = {
            "mcp_servers": {
                "http-server": {
                    "url": "https://api.example.com/mcp",
                    "transport": "streamable_http",
                },
            }
        }

        mock_env = MagicMock()
        mock_logger = MagicMock()

        register_mcp_servers(mock_env, config, mock_logger)

        # Should have added 2 tools (echo and add)
        assert mock_env.add_tool.call_count == 2
        added_tool = mock_env.add_tool.call_args_list[0][0][0]
        assert added_tool._transport == "streamable_http"

    def test_register_mcp_servers_default_transport_is_sse(self, mock_mcp_server):
        """Test that default transport is SSE when not specified in config."""
        from debug_gym.experiment import register_mcp_servers

        config = {
            "mcp_servers": {
                "default-server": {
                    "url": "http://localhost:8000/sse",
                    # no transport specified
                },
            }
        }

        mock_env = MagicMock()
        mock_logger = MagicMock()

        register_mcp_servers(mock_env, config, mock_logger)

        assert mock_env.add_tool.call_count == 2
        added_tool = mock_env.add_tool.call_args_list[0][0][0]
        assert added_tool._transport == "sse"

    def test_register_mcp_servers_with_tool_filter(self, mock_mcp_server):
        """Test registering MCP servers with tool filter."""
        from debug_gym.experiment import register_mcp_servers

        config = {
            "mcp_servers": {
                "filtered-server": {
                    "url": "http://localhost:8000/sse",
                    "tools": ["echo"],
                }
            }
        }

        mock_env = MagicMock()
        mock_logger = MagicMock()

        register_mcp_servers(mock_env, config, mock_logger)

        # Should have added only 1 tool (echo)
        assert mock_env.add_tool.call_count == 1
        added_tool = mock_env.add_tool.call_args[0][0]
        assert added_tool.name == "echo"

    def test_register_mcp_servers_empty_config(self):
        """Test with no MCP servers in config."""
        from debug_gym.experiment import register_mcp_servers

        config = {}
        mock_env = MagicMock()
        mock_logger = MagicMock()

        register_mcp_servers(mock_env, config, mock_logger)

        mock_env.add_tool.assert_not_called()

    def test_register_mcp_servers_invalid_config(self, mock_mcp_server):
        """Test with invalid MCP server config (missing url)."""
        from debug_gym.experiment import register_mcp_servers

        config = {
            "mcp_servers": {
                "no-url": {},  # Missing url
                "valid": {"url": "http://localhost:8000/sse"},
            }
        }

        mock_env = MagicMock()
        mock_logger = MagicMock()

        register_mcp_servers(mock_env, config, mock_logger)

        # Only valid config should add tools
        assert mock_env.add_tool.call_count == 2  # echo and add from valid server
        assert mock_logger.warning.call_count == 1

    def test_register_mcp_servers_handles_errors(self):
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

        with patch("debug_gym.gym.tools.mcp_proxy.discover_mcp_tools") as mock_discover:
            # First call fails, second succeeds
            mock_discover.side_effect = [Exception("Connection failed"), []]

            register_mcp_servers(mock_env, config, mock_logger)

            assert mock_discover.call_count == 2
            mock_logger.error.assert_called_once()
