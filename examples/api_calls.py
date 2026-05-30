"""Small HTTP API smoke test.

This intentionally calls FastAPI directly; use MCP Inspector or an MCP-compatible
client for stdio transport and tool-discovery verification.
"""

import asyncio
import json
import os
import sys

import httpx


API_BASE_URL = os.getenv("MULTIMEDIA_API_BASE_URL", "http://127.0.0.1:8000")


async def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else ""

    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30) as client:
        health = (await client.get("/api/health")).json()
        stats = (await client.get("/api/stats")).json()
        people = (await client.get("/api/persons")).json()

        payload = {
            "health": health,
            "stats": stats,
            "people_sample": people[:5],
        }

        if name:
            payload["search"] = (await client.get("/api/search", params={"name": name})).json()

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
