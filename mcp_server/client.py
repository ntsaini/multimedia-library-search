from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

from mcp_server.config import settings


class ApiClient:
    def __init__(self) -> None:
        self.base_url = settings.api_base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=settings.http_timeout_sec,
        )

    async def __aenter__(self) -> "ApiClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    def absolute_url(self, path: str) -> str:
        return urljoin(self.base_url + "/", path.lstrip("/"))

    async def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        response = await self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    async def post_json(self, path: str, payload: dict[str, Any]) -> dict:
        response = await self._client.post(path, json=payload)
        response.raise_for_status()
        return response.json()

    async def post_file(self, path: str, file_path: str) -> dict:
        src = Path(file_path).expanduser()
        with src.open("rb") as f:
            files = {"file": (src.name, f, "application/octet-stream")}
            response = await self._client.post(path, files=files)
        response.raise_for_status()
        return response.json()
