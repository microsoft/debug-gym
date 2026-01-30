"""MCP Proxy Tool - Each tool manages its own MCP session.

This module provides MCP tools where each tool instance holds its own session
to the MCP server, simplifying lifecycle management.
"""

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool

# Background event loop for async MCP operations
_background_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_lock = threading.Lock()
_background_loop_initializing = False
_background_loop_ready = threading.Event()


def _get_background_loop() -> asyncio.AbstractEventLoop:
    """Get or create a background event loop for MCP operations."""
    global _background_loop, _background_loop_initializing

    def run_loop():
        """Background thread target that creates and runs the event loop."""
        global _background_loop, _background_loop_initializing
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            with _loop_lock:
                _background_loop = loop
                _background_loop_initializing = False
                _background_loop_ready.set()
            loop.run_forever()
        except Exception as e:
            # Ensure the event is set even if initialization fails
            with _loop_lock:
                _background_loop_initializing = False
                _background_loop_ready.set()
            raise RuntimeError(f"Background event loop failed: {e}") from e

    with _loop_lock:
        if _background_loop is None or not _background_loop.is_running():
            if not _background_loop_initializing:
                _background_loop_initializing = True
                _background_loop_ready.clear()
                thread = threading.Thread(target=run_loop, daemon=True)
                thread.start()
    
    # Wait with timeout to prevent indefinite blocking
    if not _background_loop_ready.wait(timeout=10):
        raise RuntimeError("Timeout waiting for background event loop to initialize")
    
    if _background_loop is None:
        raise RuntimeError("Background event loop failed to initialize")
    
    return _background_loop


def _run_async(coro, timeout: int = 60):
    """Run an async coroutine in the background loop synchronously."""
    loop = _get_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


@asynccontextmanager
async def _create_mcp_transport(
    url: str, transport: str, headers: Dict[str, str] = None
):
    """Create an MCP transport connection.

    Args:
        url: The MCP endpoint URL
        transport: Transport type: 'sse' or 'streamable_http'
        headers: Optional HTTP headers for requests

    Yields:
        Tuple of (read_stream, write_stream)
    """
    if transport == "sse":
        from mcp.client.sse import sse_client

        async with sse_client(url, headers=headers or None) as (
            read_stream,
            write_stream,
        ):
            yield read_stream, write_stream
    elif transport == "streamable_http":
        import httpx
        from mcp.client.streamable_http import streamable_http_client

        # streamable_http_client doesn't accept headers directly,
        # so we create a custom httpx client with headers configured
        if headers:
            async with httpx.AsyncClient(headers=headers) as http_client:
                async with streamable_http_client(url, http_client=http_client) as (
                    read_stream,
                    write_stream,
                    _,
                ):
                    yield read_stream, write_stream
        else:
            async with streamable_http_client(url) as (
                read_stream,
                write_stream,
                _,
            ):
                yield read_stream, write_stream
    else:
        raise ValueError(
            f"Unknown MCP transport: {transport}. Use 'sse' or 'streamable_http'."
        )


class MCPTool(EnvironmentTool):
    """MCP Tool that manages its own session to an MCP server.

    Each instance connects to an MCP server and exposes a single tool.
    The session is lazily initialized on first use and cleaned up when needed.
    """

    def __init__(
        self,
        url: str,
        mcp_tool_name: str,
        tool_name: str = None,
        description: str = None,
        input_schema: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
        transport: str = "sse",
        timeout: int = 60,
    ):
        """Initialize an MCP tool.

        Args:
            url: The MCP endpoint URL (e.g., 'http://localhost:8000/sse' or 'https://api.example.com/mcp')
            mcp_tool_name: Name of the tool on the MCP server
            tool_name: Name to expose this tool as (defaults to mcp_tool_name)
            description: Tool description (fetched from server if not provided)
            input_schema: Tool input schema (fetched from server if not provided)
            headers: Optional HTTP headers for requests
            transport: Transport type: 'sse' (default) or 'streamable_http'
            timeout: Timeout in seconds for MCP operations (default: 60)
        """
        super().__init__()
        self._url = url
        self._mcp_tool_name = mcp_tool_name
        self._headers = headers or {}
        self._transport = transport
        self._timeout = timeout

        # Set tool metadata (will be updated from server if not provided)
        self.name = tool_name or mcp_tool_name
        self.description = description or f"MCP tool: {mcp_tool_name}"
        self.arguments = self._convert_schema(input_schema or {})

    def use(self, environment, **kwargs) -> Observation:
        """Execute the MCP tool."""
        output = _run_async(self._call_tool_async(kwargs), timeout=self._timeout)
        return Observation(self.name, output)

    def _convert_schema(self, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MCP JSON Schema to EnvironmentTool arguments format."""
        arguments = {}
        properties = json_schema.get("properties", {})
        required_fields = set(json_schema.get("required", []))

        for prop_name, prop_def in properties.items():
            prop_type = prop_def.get("type", "string")
            prop_desc = prop_def.get("description", f"Parameter {prop_name}")
            # Normalize type to a flat list of JSON Schema types
            if isinstance(prop_type, list):
                type_list = list(prop_type)
            else:
                type_list = [prop_type]
            # For optional fields, ensure "null" is included once
            if prop_name not in required_fields and "null" not in type_list:
                type_list.append("null")
            arguments[prop_name] = {"type": type_list, "description": prop_desc}
            if "enum" in prop_def:
                arguments[prop_name]["enum"] = prop_def["enum"]

        return arguments

    async def _call_tool_async(self, arguments: Dict[str, Any]) -> str:
        """Call the MCP tool asynchronously.

        Creates a fresh connection for each call to avoid cross-task context
        manager issues with anyio task groups used by the MCP SDK.
        """
        from mcp import ClientSession

        async with _create_mcp_transport(self._url, self._transport, self._headers) as (
            read_stream,
            write_stream,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(self._mcp_tool_name, arguments)

                # Extract text content from result
                texts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        texts.append(item.text)
                return "\n".join(texts) if texts else str(result)


def discover_mcp_tools(
    url: str,
    headers: Dict[str, str] = None,
    tool_prefix: str = "",
    tool_filter: List[str] = None,
    transport: str = "sse",
    timeout: int = 60,
) -> List[MCPTool]:
    """Discover and create MCPTool instances for all tools on a server.

    Args:
        url: The MCP endpoint URL
        headers: Optional HTTP headers
        tool_prefix: Prefix to add to tool names
        tool_filter: Optional list of tool names to include (None = all tools)
        transport: Transport type: 'sse' (default) or 'streamable_http'
        timeout: Timeout in seconds for MCP operations (default: 60)

    Returns:
        List of MCPTool instances ready to be added to an environment
    """

    async def _discover():

        from mcp import ClientSession

        async with _create_mcp_transport(url, transport, headers) as (
            read_stream,
            write_stream,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()

                tools = []
                for tool in result.tools:
                    if tool_filter and tool.name not in tool_filter:
                        continue
                    tools.append(
                        MCPTool(
                            url=url,
                            mcp_tool_name=tool.name,
                            tool_name=f"{tool_prefix}{tool.name}",
                            description=tool.description or f"MCP tool: {tool.name}",
                            input_schema=tool.inputSchema or {},
                            headers=headers,
                            transport=transport,
                            timeout=timeout,
                        )
                    )
                return tools

    return _run_async(_discover(), timeout=timeout)
