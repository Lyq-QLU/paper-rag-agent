"""Synchronous gateway used by LangGraph nodes to call the local MCP server."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolError(RuntimeError):
    pass


class MCPPaperToolGateway:
    """Discover, validate and call paper tools over MCP stdio with one reconnect."""

    def __init__(self, server_path: Path | None = None, schema_ttl_seconds: int = 300) -> None:
        self.server_path = server_path or Path(__file__).resolve().parents[1] / "mcp_server.py"
        self.schema_ttl_seconds = schema_ttl_seconds
        self._schemas: dict[str, dict] = {}
        self._schema_expires_at = 0.0

    def list_tools(self, force_refresh: bool = False) -> dict[str, dict]:
        if not force_refresh and self._schemas and time.monotonic() < self._schema_expires_at:
            return dict(self._schemas)
        tools = self._run(self._list_tools_async())
        self._schemas = tools
        self._schema_expires_at = time.monotonic() + self.schema_ttl_seconds
        return dict(tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict:
        schemas = self.list_tools()
        if name not in schemas:
            raise MCPToolError(f"MCP tool is not available: {name}")
        self._validate_required(name, arguments, schemas[name])
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return self._run(self._call_tool_async(name, arguments))
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    self.list_tools(force_refresh=True)
        detail = self._format_error(last_error) if last_error else "unknown error"
        raise MCPToolError(f"MCP tool call failed after reconnect: {detail}") from last_error

    def _server_parameters(self) -> StdioServerParameters:
        return StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_path)],
            cwd=str(self.server_path.parent),
            env={"OPENALEX_API_KEY": os.getenv("OPENALEX_API_KEY", "")},
        )

    async def _list_tools_async(self) -> dict[str, dict]:
        async with stdio_client(self._server_parameters()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await asyncio.wait_for(session.list_tools(), timeout=10.0)
                return {tool.name: dict(tool.inputSchema or {}) for tool in response.tools}

    async def _call_tool_async(self, name: str, arguments: dict[str, Any]) -> dict:
        async with stdio_client(self._server_parameters()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await asyncio.wait_for(
                    session.call_tool(name, arguments=arguments), timeout=60.0
                )
                if result.isError:
                    raise MCPToolError(self._text_content(result) or f"tool {name} returned an error")
                structured = getattr(result, "structuredContent", None)
                if isinstance(structured, dict):
                    return structured
                text = self._text_content(result)
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise MCPToolError(f"tool {name} returned invalid JSON") from exc
                if not isinstance(parsed, dict):
                    raise MCPToolError(f"tool {name} returned a non-object result")
                return parsed

    @staticmethod
    def _text_content(result) -> str:
        return "\n".join(
            str(getattr(item, "text", "")) for item in result.content if getattr(item, "text", "")
        )

    @staticmethod
    def _validate_required(name: str, arguments: dict[str, Any], schema: dict) -> None:
        missing = [field for field in schema.get("required", []) if field not in arguments]
        if missing:
            raise MCPToolError(f"MCP tool {name} missing required arguments: {', '.join(missing)}")

    @staticmethod
    def _format_error(error: BaseException) -> str:
        nested = getattr(error, "exceptions", None)
        if nested:
            messages = [MCPPaperToolGateway._format_error(item) for item in nested]
            return "; ".join(message for message in messages if message)
        return str(error)

    @staticmethod
    def _run(awaitable):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        raise MCPToolError("MCP synchronous gateway cannot run inside an active event loop")
