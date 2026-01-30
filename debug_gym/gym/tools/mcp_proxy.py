"""MCP Proxy Tool - Each tool manages its own MCP session.

This module provides MCP tools where each tool instance holds its own session
to the MCP server, simplifying lifecycle management.
"""

import asyncio
import threading
import time
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool

# Background event loop for async MCP operations
_background_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_lock = threading.Lock()


def _get_background_loop() -> asyncio.AbstractEventLoop:
    """Get or create a background event loop for MCP operations."""
    global _background_loop
    with _loop_lock:
        if _background_loop is None or not _background_loop.is_running():

            def run_loop():
                global _background_loop
                _background_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_background_loop)
                _background_loop.run_forever()

            thread = threading.Thread(target=run_loop, daemon=True)
            thread.start()
            while _background_loop is None:
                time.sleep(0.01)
    return _background_loop


def _run_async(coro, timeout: int = 60):
    """Run an async coroutine in the background loop synchronously."""
    loop = _get_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


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
    ):
        """Initialize an MCP tool.

        Args:
            url: The SSE endpoint URL (e.g., 'http://localhost:8000/sse')
            mcp_tool_name: Name of the tool on the MCP server
            tool_name: Name to expose this tool as (defaults to mcp_tool_name)
            description: Tool description (fetched from server if not provided)
            input_schema: Tool input schema (fetched from server if not provided)
            headers: Optional HTTP headers for requests
        """
        super().__init__()
        self._url = url
        self._mcp_tool_name = mcp_tool_name
        self._headers = headers or {}
        self._session = None
        self._context_stack = None
        self._initialized = False

        # Set tool metadata (will be updated from server if not provided)
        self.name = tool_name or mcp_tool_name
        self.description = description or f"MCP tool: {mcp_tool_name}"
        self.arguments = self._convert_schema(input_schema or {})

    def _convert_schema(self, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MCP JSON Schema to EnvironmentTool arguments format."""
        arguments = {}
        properties = json_schema.get("properties", {})
        required_fields = set(json_schema.get("required", []))

        for prop_name, prop_def in properties.items():
            prop_type = prop_def.get("type", "string")
            prop_desc = prop_def.get("description", f"Parameter {prop_name}")
            type_list = [prop_type]
            if prop_name not in required_fields:
                type_list.append("null")
            arguments[prop_name] = {"type": type_list, "description": prop_desc}
            if "enum" in prop_def:
                arguments[prop_name]["enum"] = prop_def["enum"]

        return arguments

    async def _connect(self):
        """Connect to the MCP server and initialize session."""
        if self._session is not None:
            return

        from mcp import ClientSession
        from mcp.client.sse import sse_client

        self._context_stack = AsyncExitStack()
        read_stream, write_stream = await self._context_stack.enter_async_context(
            sse_client(self._url, headers=self._headers or None)
        )
        self._session = await self._context_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()
        self._initialized = True

    async def _disconnect(self):
        """Disconnect from the MCP server."""
        if self._context_stack:
            await self._context_stack.aclose()
            self._context_stack = None
            self._session = None
            self._initialized = False

    async def _call_tool_async(self, arguments: Dict[str, Any]) -> str:
        """Call the MCP tool asynchronously."""
        await self._connect()
        result = await self._session.call_tool(self._mcp_tool_name, arguments)

        # Extract text content from result
        texts = []
        for item in result.content:
            if hasattr(item, "text"):
                texts.append(item.text)
        return "\n".join(texts) if texts else str(result)

    def use(self, environment, **kwargs) -> Observation:
        """Execute the MCP tool."""
        output = _run_async(self._call_tool_async(kwargs))
        return Observation(self.name, output)


def discover_mcp_tools(
    url: str,
    headers: Dict[str, str] = None,
    tool_prefix: str = "",
    tool_filter: List[str] = None,
) -> List[MCPTool]:
    """Discover and create MCPTool instances for all tools on a server.

    Args:
        url: The SSE endpoint URL
        headers: Optional HTTP headers
        tool_prefix: Prefix to add to tool names
        tool_filter: Optional list of tool names to include (None = all tools)

    Returns:
        List of MCPTool instances ready to be added to an environment
    """

    async def _discover():

        from mcp import ClientSession
        from mcp.client.sse import sse_client

        async with AsyncExitStack() as stack:
            read_stream, write_stream = await stack.enter_async_context(
                sse_client(url, headers=headers or None)
            )
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
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
                    )
                )
            return tools

    return _run_async(_discover())
