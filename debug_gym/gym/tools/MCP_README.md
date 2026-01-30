# MCP Proxy Tool

Dynamically register tools from any MCP (Model Context Protocol) server as native `EnvironmentTool` instances. Each tool manages its own session to the MCP server.

| Component | Description |
|-----------|-------------|
| `MCPTool` | Tool that manages its own MCP session |
| `discover_mcp_tools()` | Discovers and creates tools from an MCP server |


## Usage

### Via Config (Recommended)

Add MCP servers to your experiment config:

```yaml
mcp_servers:
  # SSE transport (default) - for servers with /sse endpoints
  my-server:
    url: "http://localhost:8000/sse"
    tool_prefix: "mcp_"
    tools:  # optional: list of tool names to include (omit to include all)
      - query
      - search
    # transport defaults to "sse"

  # Streamable HTTP transport - for REST-style MCP endpoints
  rest-api:
    url: "https://api.example.com/mcp"
    tool_prefix: "api_"
    transport: "streamable_http"

  # Long-running operations with custom timeout
  slow-server:
    url: "http://localhost:9000/sse"
    tool_prefix: "slow_"
    timeout: 300  # 5 minutes for long-running operations

tools:
  - bash
  - view
  - edit
  # MCP tools are added automatically from mcp_servers config
```

#### Config Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `url` | Yes | - | The MCP endpoint URL |
| `transport` | No | `sse` | Transport type: `sse` or `streamable_http` |
| `tool_prefix` | No | `""` | Prefix to add to tool names (e.g., `mcp_` → `mcp_query`) |
| `headers` | No | `{}` | HTTP headers for authentication |
| `tools` | No | all | List of tool names to include. If omitted, all tools from the server are registered |
| `timeout` | No | `60` | Timeout in seconds for MCP operations |

#### Transport Types

- **`sse`** (default): Server-Sent Events transport. Use for servers with `/sse` endpoints.
- **`streamable_http`**: HTTP-based transport. Use for REST-style MCP APIs.

### Programmatic Registration

```python
from debug_gym.gym.tools.mcp_proxy import discover_mcp_tools

# SSE transport (default)
tools = discover_mcp_tools(
    url="http://localhost:8000/sse",
    tool_prefix="mcp_",
)

# Streamable HTTP transport
tools = discover_mcp_tools(
    url="https://api.example.com/mcp",
    tool_prefix="api_",
    transport="streamable_http",
)

# Long-running operations with custom timeout
tools = discover_mcp_tools(
    url="http://localhost:9000/sse",
    tool_prefix="slow_",
    timeout=300,  # 5 minutes for long-running operations
)

for tool in tools:
    env.add_tool(tool)
    print(f"  - {tool.name}: {tool.description}")

# Filter specific tools
tools = discover_mcp_tools(
    url="http://localhost:8000/sse",
    tool_filter=["query", "search"],
)
```

### Direct Tool Creation

```python
from debug_gym.gym.tools.mcp_proxy import MCPTool

# Create a single tool directly
tool = MCPTool(
    url="http://localhost:8000/sse",
    mcp_tool_name="query",
    tool_name="my_query",  # optional custom name
    description="Query the database",
    input_schema={
        "properties": {"sql": {"type": "string"}},
        "required": ["sql"],
    },
    transport="sse",  # or "streamable_http"
    timeout=120,  # optional: custom timeout in seconds (default: 60)
)
env.add_tool(tool)
```

## Resources

- [MCP Specification](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
