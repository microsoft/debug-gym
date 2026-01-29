# MCP Proxy Tool

Dynamically register tools from any MCP (Model Context Protocol) server as native `EnvironmentTool` instances. Uses the official MCP Python SDK for HTTP+SSE transport.

## Usage

### Via Config (Recommended)

Add MCP servers to your experiment config. This is **multi-process safe** - each worker creates its own connection.

```yaml
mcp_servers:
  my-server:
    url: "http://localhost:8000/sse"
    tool_prefix: "mcp_"
  another-server:
    url: "http://localhost:9000/sse"
    headers:
      Authorization: "Bearer token"
    tool_prefix: "other_"

tools:
  - bash
  - view
  - edit
  - mcp_query      # MCP tools available after registration
  - mcp_search
```

### Programmatic Registration

```python
from debug_gym.gym.tools.mcp_proxy import register_mcp_server_sse

tools = register_mcp_server_sse(
    server_id="my-server",
    url="http://127.0.0.1:8000/sse",
    tool_prefix="mcp_"
)

for tool in tools:
    print(f"  - {tool.name}: {tool.description}")
```

## Architecture

```
MCP Server (Python/Node.js/etc.)
       │ HTTP + SSE
       ▼
MCPClientSSE (uses official MCP SDK)
       │
       ▼
MCPToolFactory (creates EnvironmentTool classes)
       │
       ▼
Toolbox (bash, view, edit, mcp_tool1, mcp_tool2, ...)
```

## Key Components

| Component | Description |
|-----------|-------------|
| `MCPClientSSE` | Connects via HTTP+SSE using official MCP SDK |
| `MCPToolFactory` | Converts MCP tool definitions to EnvironmentTool classes |
| `MCPToolRegistry` | Manages multiple server connections |
| `register_mcp_server_sse()` | Sync convenience function for registration |

## Troubleshooting

```python
# Test connection
import asyncio
from debug_gym.gym.tools.mcp_proxy import MCPClientSSE

async def test():
    client = MCPClientSSE("http://localhost:8000/sse")
    try:
        await client.start()
        result = await client.initialize()
        print(f"Server: {result['serverInfo']['name']}")
        tools = await client.list_tools()
        print(f"Tools: {[t['name'] for t in tools]}")
    finally:
        await client.stop()

asyncio.run(test())
```

## Resources

- [MCP Specification](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
