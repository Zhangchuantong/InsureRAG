# -*- coding: utf-8 -*-
"""Run the InsureRAG MCP server over stdio."""

from __future__ import annotations

from mcp_server.tools import mcp


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
