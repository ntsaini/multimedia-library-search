from mcp.server.fastmcp import FastMCP

from mcp_server.logging_config import configure_logging
from mcp_server.tools.library import register_tools

mcp = FastMCP("multimedia-library-search")
register_tools(mcp)


def main() -> None:
    configure_logging()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
