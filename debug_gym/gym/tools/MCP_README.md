# MCP Proxy Tool

Dynamically register tools from any MCP (Model Context Protocol) server as native `EnvironmentTool` instances. Each tool manages its own session to the MCP server.

## Usage

### Via Config (Recommended)

Add MCP servers to your experiment config:

```yaml
mcp_servers:
  my-server:
    url: "http://localhost:8000/sse"
    tool_prefix: "mcp_"
    tools:  # optional: list of tool names to include (omit to include all)
      - query
      - search
  another-server:
    url: "http://localhost:9000/sse"
    headers:
      Authorization: "Bearer token"
    tool_prefix: "other_"

tools:
  - bash
  - view
  - edit
  # MCP tools are added automatically from mcp_servers config
```

#### Config Options

| Option | Required | Description |
|--------|----------|-------------|
| `url` | Yes | The SSE endpoint URL (e.g., `http://localhost:8000/sse`) |
| `tool_prefix` | No | Prefix to add to tool names (e.g., `mcp_` → `mcp_query`) |
| `headers` | No | HTTP headers for authentication |
| `tools` | No | List of tool names to include. If omitted, all tools from the server are registered |

### Programmatic Registration

```python
from debug_gym.gym.tools.mcp_proxy import discover_mcp_tools

# Discover all tools from a server
tools = discover_mcp_tools(
    url="http://localhost:8000/sse",
    tool_prefix="mcp_",
)

for tool in tools:
    env.add_tool(tool)
    print(f"  - {tool.name}: {tool.description}")

# Or filter specific tools
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
)
env.add_tool(tool)
```

## Architecture

```
MCP Server (Python/Node.js/etc.)
       │ HTTP + SSE
       ▼
MCPTool (each tool has its own session)
       │
       ▼
Environment (bash, view, edit, mcp_tool1, mcp_tool2, ...)
```

## Key Components

| Component | Description |
|-----------|-------------|
| `MCPTool` | Tool that manages its own MCP session (lazy-initialized) |
| `discover_mcp_tools()` | Discovers and creates tools from an MCP server |

## Resources

- [MCP Specification](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
