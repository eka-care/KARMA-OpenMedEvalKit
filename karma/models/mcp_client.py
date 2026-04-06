"""
Reusable MCP (Model Context Protocol) client utilities for KARMA models.

Provides tool discovery, schema conversion, and a connection-pooling client pool
that any model class can use to integrate MCP-hosted tools.

Requires: pip install 'karma-medeval[tools]'
"""

import logging
from typing import Callable, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# FastMCP optional import — only needed when tools= parameter is used
try:
    from fastmcp import Client as FastMCPClient

    FASTMCP_AVAILABLE = True
except ImportError:
    FastMCPClient = None
    FASTMCP_AVAILABLE = False


_SAFE_SCHEMA_KEYS = {
    "type",
    "properties",
    "required",
    "description",
    "enum",
    "items",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "pattern",
    "default",
}


def sanitise_schema(schema: Dict) -> Dict:
    """Recursively strip JSON Schema keywords unsupported by OpenAI's function-calling."""
    if not isinstance(schema, dict):
        return schema
    cleaned = {k: v for k, v in schema.items() if k in _SAFE_SCHEMA_KEYS}
    if "properties" in cleaned:
        cleaned["properties"] = {
            k: sanitise_schema(v) for k, v in cleaned["properties"].items()
        }
    if "items" in cleaned:
        cleaned["items"] = sanitise_schema(cleaned["items"])
    return cleaned


def mcp_tool_to_openai_schema(mcp_tool) -> Dict[str, Any]:
    """Convert a FastMCP Tool object to OpenAI function-calling format."""
    input_schema = mcp_tool.inputSchema or {}
    parameters: Dict[str, Any] = {
        "type": input_schema.get("type", "object"),
        "properties": {
            k: sanitise_schema(v) for k, v in input_schema.get("properties", {}).items()
        },
    }
    if input_schema.get("required"):
        parameters["required"] = input_schema["required"]

    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or f"Tool: {mcp_tool.name}",
            "parameters": parameters,
        },
    }


def mcp_tool_to_bedrock_schema(mcp_tool) -> Dict[str, Any]:
    """Convert a FastMCP Tool object to AWS Bedrock converse toolSpec format."""
    input_schema = mcp_tool.inputSchema or {"type": "object", "properties": {}}
    return {
        "toolSpec": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or f"Tool: {mcp_tool.name}",
            "inputSchema": {"json": sanitise_schema(input_schema)},
        },
    }


class MCPClientPool:
    """Manages one FastMCPClient per MCP server URL within a single async context for connection reuse."""

    def __init__(self, tool_server_map: Dict[str, str]):
        self._tool_server_map = tool_server_map  # tool_name -> server_url
        self._clients: Dict[str, Any] = {}  # server_url -> live FastMCPClient

    async def __aenter__(self):
        for server_url in set(self._tool_server_map.values()):
            client = FastMCPClient(server_url)
            await client.__aenter__()
            self._clients[server_url] = client
        return self

    async def __aexit__(self, *args):
        for client in self._clients.values():
            try:
                await client.__aexit__(*args)
            except Exception as e:
                logger.debug(f"Error closing MCP client: {e}")
        self._clients.clear()

    async def call_tool(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Call a tool by name, routing to the correct MCP server.

        Returns a list of content blocks, each a dict with a ``"type"`` key:
          - ``{"type": "text", "text": "..."}``
          - ``{"type": "image", "data": "<base64>", "mime_type": "image/jpeg"}``
        """
        server_url = self._tool_server_map.get(tool_name)
        if not server_url:
            raise ValueError(
                f"Unknown tool: '{tool_name}'. Not found in any registered MCP server."
            )
        client = self._clients[server_url]
        result = await client.call_tool(tool_name, tool_args)

        blocks: List[Dict[str, Any]] = []
        if hasattr(result, "content") and result.content:
            for item in result.content:
                content_type = getattr(item, "type", None)
                if content_type == "text":
                    blocks.append({"type": "text", "text": item.text})
                elif content_type == "image":
                    blocks.append({
                        "type": "image",
                        "data": item.data,
                        "mime_type": getattr(item, "mimeType", "image/jpeg"),
                    })
                else:
                    logger.warning(
                        "Unsupported MCP content type '%s', converting to text",
                        content_type,
                    )
                    blocks.append({"type": "text", "text": str(item)})
        elif hasattr(result, "data") and result.data is not None:
            blocks.append({"type": "text", "text": str(result.data)})
        else:
            blocks.append({"type": "text", "text": str(result)})

        return blocks


async def discover_mcp_tools(
    server_urls: List[str],
    schema_converter: Callable = mcp_tool_to_openai_schema,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Connect to each MCP server, list tools, and convert schemas.

    Args:
        server_urls: List of MCP server URLs to discover tools from.
        schema_converter: Function to convert MCP tool to provider-specific schema.
            Defaults to OpenAI format. Use mcp_tool_to_bedrock_schema for Bedrock.

    Returns:
        Tuple of (converted_tools_list, tool_name_to_server_url_map).
    """
    converted_tools = []
    tool_server_map = {}

    for server_url in server_urls:
        async with FastMCPClient(server_url) as mcp_client:
            tools = await mcp_client.list_tools()
            for tool in tools:
                if tool.name in tool_server_map:
                    logger.warning(
                        f"Tool name collision: '{tool.name}' found on multiple servers. "
                        f"Using first occurrence ({tool_server_map[tool.name]})."
                    )
                    continue
                converted_tools.append(schema_converter(tool))
                tool_server_map[tool.name] = server_url

    return converted_tools, tool_server_map


def blocks_to_text(blocks: List[Dict[str, Any]]) -> str:
    """Flatten structured content blocks to a single text string (for tracing/logging)."""
    parts = []
    for b in blocks:
        if b["type"] == "text":
            parts.append(b["text"])
        elif b["type"] == "image":
            parts.append(f"[Image: {b.get('mime_type', 'image/jpeg')}]")
    return "\n".join(parts)
