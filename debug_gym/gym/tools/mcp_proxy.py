"""MCP Proxy Tool - Each tool manages its own MCP session.

This module provides MCP tools where each tool instance holds its own session
to the MCP server, simplifying lifecycle management.
"""

import asyncio
import copy
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict, List

from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool


def _run_async(coro, timeout: int = 60, loop: asyncio.AbstractEventLoop | None = None):
    """Run an async coroutine synchronously with timeout.

    If a loop is provided, the coroutine is executed on that loop.
    """
    if loop is None:
        return asyncio.run(asyncio.wait_for(coro, timeout=timeout))

    if loop.is_closed():
        raise RuntimeError("Event loop is closed.")
    if loop.is_running():
        raise RuntimeError("Event loop is already running.")

    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
    finally:
        asyncio.set_event_loop(None)


@dataclass
class _McpSessionState:
    """Session state bundle to keep loop, session, and cleanup together.

    The MCP client session is bound to the event loop it was created on, so this
    object owns the loop to ensure all calls use the same loop.
    """

    url: str
    headers: Dict[str, str]
    transport: str
    timeout: int
    session: Any = None
    exit_stack: AsyncExitStack | None = None
    loop: asyncio.AbstractEventLoop | None = None

    def without_runtime(self):
        return _McpSessionState(
            url=self.url,
            headers=self.headers,
            transport=self.transport,
            timeout=self.timeout,
        )

    def ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self.loop is None or self.loop.is_closed():
            self.loop = asyncio.new_event_loop()
        return self.loop

    def run(self, coro):
        loop = self.ensure_loop()
        return _run_async(coro, timeout=self.timeout, loop=loop)

    async def start_session_async(self):
        self.exit_stack = AsyncExitStack()
        await self.exit_stack.__aenter__()
        self.session = await _create_mcp_session(
            self.url, self.transport, self.headers, self.exit_stack
        )

    async def stop_session_async(self):
        self.session = None
        if self.exit_stack:
            await self.exit_stack.__aexit__(None, None, None)
            self.exit_stack = None

    def stop_session(self):
        if self.session is None:
            return
        try:
            self.run(self.stop_session_async())
        except Exception:
            self.session = None
            self.exit_stack = None
        finally:
            if self.loop is not None:
                self.loop.close()
                self.loop = None


async def _create_mcp_session(
    url: str, transport: str, headers: Dict[str, str], exit_stack: AsyncExitStack
):
    """Create and initialize an MCP session.

    Args:
        url: The MCP endpoint URL
        transport: 'sse' or 'streamable_http'
        headers: Optional HTTP headers
        exit_stack: AsyncExitStack for resource cleanup

    Returns:
        Initialized ClientSession
    """
    from mcp import ClientSession

    if transport == "sse":
        from mcp.client.sse import sse_client

        read, write = await exit_stack.enter_async_context(
            sse_client(url, headers=headers or None)
        )
    elif transport == "streamable_http":
        import httpx
        from mcp.client.streamable_http import streamable_http_client

        if headers:
            http_client = await exit_stack.enter_async_context(
                httpx.AsyncClient(headers=headers)
            )
            read, write, _ = await exit_stack.enter_async_context(
                streamable_http_client(url, http_client=http_client)
            )
        else:
            read, write, _ = await exit_stack.enter_async_context(
                streamable_http_client(url)
            )
    else:
        raise ValueError(
            f"Unknown transport: {transport}. Use 'sse' or 'streamable_http'."
        )

    session = await exit_stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


class MCPTool(EnvironmentTool):
    """MCP Tool that manages its own persistent session to an MCP server."""

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
        super().__init__()
        self._url = url
        self._mcp_tool_name = mcp_tool_name
        self._headers = headers or {}
        self._transport = transport
        self._timeout = timeout
        # Keep a dedicated loop so session and calls run on the same event loop.
        self._state = _McpSessionState(
            url=self._url,
            headers=self._headers,
            transport=self._transport,
            timeout=self._timeout,
        )

        self.name = tool_name or mcp_tool_name
        self.description = description or f"MCP tool: {mcp_tool_name}"
        self.arguments = self._convert_schema(input_schema or {})

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_state"] = self._state.without_runtime()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._state is None:
            self._state = _McpSessionState(
                url=self._url,
                headers=self._headers,
                transport=self._transport,
                timeout=self._timeout,
            )
        else:
            self._state = self._state.without_runtime()

    def __deepcopy__(self, memo):
        result = type(self).__new__(self.__class__)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_state":
                setattr(result, k, v.without_runtime())
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    @property
    def session_is_running(self) -> bool:
        return self._state.session is not None

    async def _start_session_async(self) -> str:
        await self._state.start_session_async()
        return f"MCP session started for tool '{self.name}'."

    async def _stop_session_async(self):
        await self._state.stop_session_async()

    def start_session(self) -> str:
        if self.session_is_running:
            return "Session already running."

        return self._state.run(self._start_session_async())

    def stop_session(self):
        self._state.stop_session()

    def on_env_reset(self, environment, **kwargs) -> Observation:
        super().on_env_reset(environment, **kwargs)
        self.stop_session()
        try:
            output = self.start_session()
        except Exception as e:
            output = f"Failed to start MCP session: {e}"
        return Observation(self.name, output)

    def use(self, environment, **kwargs) -> Observation:
        if not self.session_is_running:
            try:
                self.start_session()
            except Exception as e:
                return Observation(self.name, f"Failed to start MCP session: {e}")

        try:
            output = _run_async(
                self._state.session.call_tool(self._mcp_tool_name, kwargs),
                timeout=self._timeout,
                loop=self._state.ensure_loop(),
            )
            texts = [item.text for item in output.content if hasattr(item, "text")]
            return Observation(self.name, "\n".join(texts) if texts else str(output))
        except Exception:
            # Session may have died, retry once
            self.stop_session()
            try:
                self.start_session()
                output = _run_async(
                    self._state.session.call_tool(self._mcp_tool_name, kwargs),
                    timeout=self._timeout,
                    loop=self._state.ensure_loop(),
                )
                texts = [item.text for item in output.content if hasattr(item, "text")]
                return Observation(
                    self.name, "\n".join(texts) if texts else str(output)
                )
            except Exception as e:
                return Observation(self.name, f"MCP tool call failed: {e}")

    def _convert_schema(self, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        arguments = {}
        properties = json_schema.get("properties", {})
        required = set(json_schema.get("required", []))

        for name, prop in properties.items():
            prop_type = prop.get("type", "string")
            type_list = list(prop_type) if isinstance(prop_type, list) else [prop_type]
            if name not in required and "null" not in type_list:
                type_list.append("null")

            arguments[name] = {
                "type": type_list,
                "description": prop.get("description", f"Parameter {name}"),
            }
            if "enum" in prop:
                arguments[name]["enum"] = prop["enum"]

        return arguments


async def _discover_mcp_tools_async(
    url: str,
    headers: Dict[str, str] = None,
    tool_prefix: str = "",
    tool_filter: List[str] = None,
    transport: str = "sse",
    timeout: int = 60,
) -> List[MCPTool]:
    async with AsyncExitStack() as stack:
        session = await _create_mcp_session(url, transport, headers, stack)
        result = await session.list_tools()

        return [
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
            for tool in result.tools
            if not tool_filter or tool.name in tool_filter
        ]


def discover_mcp_tools(
    url: str,
    headers: Dict[str, str] = None,
    tool_prefix: str = "",
    tool_filter: List[str] = None,
    transport: str = "sse",
    timeout: int = 60,
) -> List[MCPTool]:
    """Discover and create MCPTool instances for all tools on a server."""

    return _run_async(
        _discover_mcp_tools_async(
            url=url,
            headers=headers,
            tool_prefix=tool_prefix,
            tool_filter=tool_filter,
            transport=transport,
            timeout=timeout,
        ),
        timeout=timeout,
    )
