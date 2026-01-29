"""MCP Proxy Tool - Dynamically register MCP server tools as EnvironmentTools.

This module provides functionality to connect to an MCP (Model Context Protocol) server,
discover its tools, and dynamically register them as native EnvironmentTools in the gym.

Uses the official MCP Python SDK for HTTP+SSE transport.
"""

import asyncio
import threading
from typing import Any, Dict, List, Optional, Type

from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox

# Global event loop for MCP clients (needed for HTTP/SSE transport)
_background_loop: Optional[asyncio.AbstractEventLoop] = None
_background_thread: Optional[threading.Thread] = None
_loop_lock = threading.Lock()


def _ensure_background_loop():
    """Ensure a background event loop is running for MCP clients."""
    global _background_loop, _background_thread

    with _loop_lock:
        if _background_loop is None or not _background_loop.is_running():

            def run_loop():
                global _background_loop
                _background_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_background_loop)
                _background_loop.run_forever()

            _background_thread = threading.Thread(target=run_loop, daemon=True)
            _background_thread.start()

            # Wait for loop to be ready
            import time

            while _background_loop is None:
                time.sleep(0.01)

    return _background_loop


def _run_coroutine_in_background_loop(coro):
    """Run a coroutine in the background event loop and wait for result."""
    loop = _ensure_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=60)  # 60 second timeout
    except Exception as e:
        print(f"Error running coroutine in background loop: {e}")
        import traceback

        traceback.print_exc()
        raise


class MCPClientSSE:
    """Client for communicating with an MCP server via HTTP+SSE transport using official MCP SDK."""

    def __init__(self, url: str, headers: Dict[str, str] = None):
        """Initialize HTTP+SSE MCP client.

        Args:
            url: The SSE endpoint URL (e.g., 'http://127.0.0.1:8000/sse')
            headers: Optional HTTP headers to include in requests
        """
        self.url = url
        self.headers = headers or {}
        self._session = None
        self._context_stack = None

    async def start(self):
        """Start the HTTP+SSE connection using MCP SDK."""
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError:
            raise RuntimeError(
                "mcp package is required for SSE transport. Install with: pip install mcp"
            )

        from contextlib import AsyncExitStack

        self._context_stack = AsyncExitStack()

        # Connect using the official MCP SDK sse_client
        read_stream, write_stream = await self._context_stack.enter_async_context(
            sse_client(self.url, headers=self.headers if self.headers else None)
        )

        # Create and initialize the session
        self._session = await self._context_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

    async def stop(self):
        """Stop the HTTP+SSE connection and cleanup."""
        if self._context_stack:
            await self._context_stack.aclose()
            self._context_stack = None
            self._session = None

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the MCP session.

        Returns:
            Server capabilities and information
        """
        if not self._session:
            raise RuntimeError("Client not started. Call start() first.")

        result = await self._session.initialize()
        return {
            "serverInfo": {
                "name": result.serverInfo.name if result.serverInfo else "unknown",
                "version": (
                    result.serverInfo.version if result.serverInfo else "unknown"
                ),
            },
            "capabilities": (
                result.capabilities.model_dump() if result.capabilities else {}
            ),
        }

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools from the MCP server.

        Returns:
            List of tool definitions
        """
        if not self._session:
            raise RuntimeError("Client not started. Call start() first.")

        result = await self._session.list_tools()
        tools = []
        for tool in result.tools:
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema if tool.inputSchema else {},
                }
            )
        return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if not self._session:
            raise RuntimeError("Client not started. Call start() first.")

        result = await self._session.call_tool(tool_name, arguments)

        # Convert result to dict format expected by callers
        content = []
        for item in result.content:
            if hasattr(item, "text"):
                content.append({"type": "text", "text": item.text})
            elif hasattr(item, "data"):
                content.append(
                    {"type": "image", "data": item.data, "mimeType": item.mimeType}
                )
            else:
                content.append({"type": "unknown", "data": str(item)})

        return {
            "content": content,
            "isError": result.isError if hasattr(result, "isError") else False,
        }


class MCPToolFactory:
    """Factory for creating EnvironmentTool classes from MCP tool definitions."""

    @staticmethod
    def create_tool_class(
        mcp_client: MCPClientSSE, tool_def: Dict[str, Any], tool_prefix: str = ""
    ) -> Type[EnvironmentTool]:
        """Create an EnvironmentTool class from an MCP tool definition.

        Args:
            mcp_client: The MCP client instance to use for tool calls
            tool_def: MCP tool definition containing name, description, and inputSchema
            tool_prefix: Optional prefix to add to tool names (e.g., "mcp_")

        Returns:
            A new EnvironmentTool class
        """
        tool_name = tool_def.get("name", "unknown")
        prefixed_name = f"{tool_prefix}{tool_name}" if tool_prefix else tool_name
        tool_description = tool_def.get("description", f"MCP tool: {tool_name}")
        input_schema = tool_def.get("inputSchema", {})

        # Convert MCP JSON Schema to EnvironmentTool arguments format
        tool_arguments = MCPToolFactory._convert_schema(input_schema)

        class DynamicMCPTool(EnvironmentTool):
            """Dynamically created MCP tool."""

            name: str = prefixed_name
            description: str = tool_description
            arguments: Dict[str, Any] = tool_arguments
            _mcp_client = mcp_client
            _mcp_tool_name = tool_name

            def use(self, environment, **kwargs) -> Observation:
                """Execute the MCP tool via async client."""
                try:
                    # Run async call in background loop
                    result = _run_coroutine_in_background_loop(
                        self._mcp_client.call_tool(self._mcp_tool_name, kwargs)
                    )

                    # Format result
                    if isinstance(result, dict):
                        # Handle MCP response format
                        content = result.get("content", [])
                        if content and isinstance(content, list):
                            # Extract text from content items
                            texts = []
                            for item in content:
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "text"
                                ):
                                    texts.append(item.get("text", ""))
                            output = "\n".join(texts) if texts else str(result)
                        else:
                            output = str(result)
                    else:
                        output = str(result)

                    return Observation(self.name, output)

                except Exception as e:
                    return Observation(
                        self.name,
                        f"Error calling MCP tool '{self._mcp_tool_name}': {str(e)}",
                    )

        # Set a meaningful class name for debugging
        DynamicMCPTool.__name__ = f"MCP_{tool_name.replace('-', '_').title()}Tool"
        DynamicMCPTool.__qualname__ = DynamicMCPTool.__name__

        return DynamicMCPTool

    @staticmethod
    def _convert_schema(json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MCP JSON Schema to EnvironmentTool arguments format.

        Args:
            json_schema: JSON Schema from MCP tool definition

        Returns:
            Arguments dict in EnvironmentTool format
        """
        arguments = {}
        properties = json_schema.get("properties", {})
        required_fields = set(json_schema.get("required", []))

        for prop_name, prop_def in properties.items():
            prop_type = prop_def.get("type", "string")
            prop_desc = prop_def.get("description", f"Parameter {prop_name}")

            # Convert JSON Schema type to EnvironmentTool type format
            type_list = [prop_type]
            if prop_name not in required_fields:
                type_list.append("null")

            arguments[prop_name] = {
                "type": type_list,
                "description": prop_desc,
            }

            # Add enum if present
            if "enum" in prop_def:
                arguments[prop_name]["enum"] = prop_def["enum"]

        return arguments


class MCPToolRegistry:
    """Registry for managing MCP server connections and dynamically registered tools."""

    def __init__(self):
        self.clients: Dict[str, MCPClientSSE] = {}
        self.registered_tools: List[EnvironmentTool] = []

    async def add_server_sse(
        self,
        server_id: str,
        url: str,
        headers: Dict[str, str] = None,
        tool_prefix: str = "",
        auto_register: bool = True,
    ) -> List[Type[EnvironmentTool]]:
        """Add an SSE-based MCP server and optionally register its tools.

        Args:
            server_id: Unique identifier for this server
            url: The SSE endpoint URL (e.g., 'http://localhost:8000/sse')
            headers: Optional HTTP headers to include in requests
            tool_prefix: Prefix to add to tool names (e.g., "mcp_" or "graph_")
            auto_register: Whether to automatically register tools with Toolbox

        Returns:
            List of created EnvironmentTool classes
        """
        # Create and start SSE client
        client = MCPClientSSE(url, headers)
        await client.start()
        await client.initialize()

        self.clients[server_id] = client

        # Discover tools
        tool_defs = await client.list_tools()

        # Create and register tool classes
        tool_classes = []
        for tool_def in tool_defs:
            tool_class = MCPToolFactory.create_tool_class(client, tool_def, tool_prefix)

            if auto_register:
                # Register with Toolbox using the tool's name attribute
                Toolbox.register(name=tool_class.name)(tool_class)

            tool_classes.append(tool_class)

        return tool_classes

    async def remove_server(self, server_id: str):
        """Remove an MCP server and close its connection.

        Args:
            server_id: The server identifier to remove
        """
        if server_id in self.clients:
            await self.clients[server_id].stop()
            del self.clients[server_id]

    async def cleanup(self):
        """Stop all MCP server connections."""
        for client in self.clients.values():
            await client.stop()
        self.clients.clear()


# Global registry instance
_global_registry = MCPToolRegistry()


def get_mcp_registry() -> MCPToolRegistry:
    """Get the global MCP tool registry instance."""
    return _global_registry


async def _async_register_mcp_server_sse(
    server_id: str,
    url: str,
    headers: Dict[str, str] = None,
    tool_prefix: str = "",
) -> List[Type[EnvironmentTool]]:
    """Internal async function to register an SSE-based MCP server."""
    registry = get_mcp_registry()
    return await registry.add_server_sse(
        server_id=server_id,
        url=url,
        headers=headers,
        tool_prefix=tool_prefix,
        auto_register=True,
    )


def register_mcp_server_sse(
    server_id: str,
    url: str,
    headers: Dict[str, str] = None,
    tool_prefix: str = "",
) -> List[Type[EnvironmentTool]]:
    """Register an SSE-based MCP server and its tools.

    This function is synchronous and can be called from regular Python code.
    The MCP client connection will be maintained in a background event loop.

    Args:
        server_id: Unique identifier for this server
        url: The SSE endpoint URL (e.g., 'http://localhost:8000/sse')
        headers: Optional HTTP headers to include in requests
        tool_prefix: Prefix to add to tool names

    Returns:
        List of registered EnvironmentTool classes

    Example:
        >>> tools = register_mcp_server_sse(
        ...     "my-server",
        ...     "http://localhost:8000/sse",
        ...     tool_prefix="mcp_"
        ... )
        >>> # Now tools are available: mcp_tool1, mcp_tool2, etc.
    """
    return _run_coroutine_in_background_loop(
        _async_register_mcp_server_sse(
            server_id=server_id, url=url, headers=headers, tool_prefix=tool_prefix
        )
    )
