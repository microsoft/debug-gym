import datetime
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from debug_gym import version as dg_version
from debug_gym.gym.envs import select_env
from debug_gym.gym.terminals import select_terminal
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.logger import DebugGymLogger


def _validate_mcp_url(url: str, logger: DebugGymLogger) -> bool:
    """Validate MCP server URL for security concerns.
    
    Args:
        url: The URL to validate
        logger: Logger for warnings
        
    Returns:
        True if URL appears safe, False otherwise
        
    Security checks:
    - Validates URL can be parsed
    - Warns about localhost/internal IPs (potential SSRF)
    - Warns about non-HTTP(S) schemes
    """
    try:
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme not in ("http", "https"):
            logger.warning(
                f"MCP server URL uses non-HTTP(S) scheme '{parsed.scheme}': {url}. "
                "This may pose security risks."
            )
            
        # Check for localhost/internal addresses (SSRF risk)
        hostname = parsed.hostname or ""
        
        # Check for localhost
        if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
            logger.warning(
                f"MCP server URL points to localhost address: {url}. "
                "Ensure this is intentional and the server is trusted."
            )
        
        # Check for private IP ranges
        # 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
        if hostname.startswith("10.") or hostname.startswith("192.168."):
            logger.warning(
                f"MCP server URL points to internal/private IP address: {url}. "
                "Ensure this is intentional and the server is trusted."
            )
        
        # Check for 172.16.0.0/12 range (172.16.0.0 to 172.31.255.255)
        if hostname.startswith("172."):
            parts = hostname.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                second_octet = int(parts[1])
                if 16 <= second_octet <= 31:
                    logger.warning(
                        f"MCP server URL points to internal/private IP address: {url}. "
                        "Ensure this is intentional and the server is trusted."
                    )
            
        # Check for common cloud metadata endpoints (SSRF risk)
        if "169.254.169.254" in hostname:
            logger.warning(
                f"MCP server URL points to cloud metadata endpoint: {url}. "
                "This is a potential SSRF vulnerability and should not be used."
            )
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to parse MCP server URL '{url}': {e}")
        return False


def create_env(config: dict, task_data: dict, logger: DebugGymLogger):
    terminal = select_terminal(config.get("terminal"), logger, uuid=config["uuid"])

    env_class = select_env(task_data.get("env_type"))
    env = env_class(
        task_data=task_data,
        terminal=terminal,
        logger=logger,
        **config.get("env", {}),
    )

    # First add standard tools, then register MCP servers per-process so that
    # any name conflicts with existing tools are detected during MCP registration.
    add_tools(env, config, logger)
    register_mcp_servers(env, config, logger)
    return env


def register_mcp_servers(env, config: dict, logger: DebugGymLogger):
    """Register MCP servers from config and add their tools to env after standard tools.
    
    Security Note:
    - MCP server URLs and headers are loaded from configuration without authentication.
    - Connecting to untrusted MCP servers may expose your system to security risks.
    - URLs are validated for common SSRF patterns but additional security measures
      may be needed for production environments.
    - Only connect to trusted MCP servers and validate their authenticity.
    """
    mcp_servers = config.get("mcp_servers", {})
    if not mcp_servers:
        return

    from debug_gym.gym.tools.mcp_proxy import discover_mcp_tools

    for server_id, server_config in mcp_servers.items():
        url = server_config.get("url")
        if not url:
            logger.warning(f"Skipping MCP server '{server_id}': missing url")
            continue
            
        # Validate URL for security concerns
        if not _validate_mcp_url(url, logger):
            logger.error(f"Skipping MCP server '{server_id}': URL validation failed")
            continue
            
        # Validate headers if present
        headers = server_config.get("headers")
        if headers is not None and not isinstance(headers, dict):
            logger.warning(
                f"Invalid headers for MCP server '{server_id}': "
                f"expected dict, got {type(headers).__name__}; ignoring headers"
            )
            headers = None

        try:
            tools = discover_mcp_tools(
                url=url,
                headers=headers,
                tool_prefix=server_config.get("tool_prefix", ""),
                tool_filter=server_config.get("tools"),
                transport=server_config.get("transport", "sse"),
                timeout=server_config.get("timeout", 60),
            )
            for tool in tools:
                env.add_tool(tool)
                logger.info(f"Adding MCP tool: {tool.name}")
        except Exception as e:
            logger.error(f"Failed to register MCP server '{server_id}': {e}")


def add_tools(env, config: dict, logger: DebugGymLogger):
    """Add tools to the environment"""
    for tool in config.get("tools", []):
        tool_config = {}
        if isinstance(tool, dict):
            assert len(tool) == 1, "Tool dict must have exactly one key"
            tool, tool_config = list(tool.items())[0]
        if isinstance(config["tools"], dict) and isinstance(
            config["tools"][tool], dict
        ):
            tool_config.update(config["tools"][tool])

        tool_instantiated = Toolbox.get_tool(tool, **tool_config)
        env.add_tool(tool_instantiated)
        logger.debug(f"Adding tool to toolbox: {tool_instantiated.__class__.__name__}")


def dump_experiment_info(config: dict, args: dict):
    """Dump experiment information to a JSONL file.
    Each line is one experiment run with its metadata."""

    try:  # Get git commit hash
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
            )
            .decode()
            .strip()
        )
    except Exception:
        git_hash = ""

    try:  # Get git diff
        git_diff = subprocess.check_output(
            ["git", "diff"], cwd=os.path.dirname(__file__)
        ).decode()
    except Exception:
        git_diff = ""

    version_info = {
        "debug_gym_version": dg_version.__version__,
        "datetime": datetime.datetime.now().isoformat(),
        "git_hash": git_hash,
        "git_diff": git_diff,
        "config": config,
        "args": vars(args),
        "python_version": os.sys.version,
    }

    file = Path(config["output_path"]) / "experiment_info.jsonl"
    with open(file, "a") as f:
        f.write(f"{json.dumps(version_info)}\n")
