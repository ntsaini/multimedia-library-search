import logging
from pathlib import Path
from typing import Literal

import httpx

from mcp_server.client import ApiClient
from mcp_server.config import settings
from mcp_server.models.schemas import SearchResult, ToolError, ToolResult, dump_model

logger = logging.getLogger(__name__)


def _error(exc: Exception) -> dict:
    if isinstance(exc, httpx.HTTPStatusError):
        detail = exc.response.text
        try:
            body = exc.response.json()
            detail = str(body.get("detail", body))
        except Exception:
            pass
        payload = ToolResult(
            ok=False,
            error=ToolError(
                type="http_error",
                message=detail,
                status_code=exc.response.status_code,
            ),
        )
    elif isinstance(exc, httpx.RequestError):
        payload = ToolResult(
            ok=False,
            error=ToolError(type="connection_error", message=str(exc)),
        )
    elif isinstance(exc, ValueError):
        payload = ToolResult(
            ok=False,
            error=ToolError(type="validation_error", message=str(exc)),
        )
    else:
        logger.exception("unexpected tool failure")
        payload = ToolResult(
            ok=False,
            error=ToolError(type="unexpected_error", message=str(exc)),
        )
    return dump_model(payload)


def _ok(data) -> dict:
    return dump_model(ToolResult(ok=True, data=data))


def _limit(value: int) -> int:
    return max(1, min(int(value), settings.search_limit_max))


def _filter_search(data: dict, limit: int, distance_threshold: float | None = None) -> dict:
    videos = data.get("videos") or []
    photos = data.get("photos") or []
    if distance_threshold is not None:
        videos = [h for h in videos if h.get("distance") is None or h["distance"] <= distance_threshold]
        photos = [h for h in photos if h.get("distance") is None or h["distance"] <= distance_threshold]
    return dump_model(SearchResult(videos=videos[:limit], photos=photos[:limit]))


def _add_download_url(client: ApiClient, job: dict, path: str) -> dict:
    if job.get("status") == "done":
        job = dict(job)
        job["download_url"] = client.absolute_url(path)
    return job


def register_tools(mcp) -> None:
    @mcp.tool()
    async def health_check() -> dict:
        """Verify the FastAPI app is reachable and storage is initialized."""
        try:
            async with ApiClient() as client:
                return _ok(await client.get("/api/health"))
        except Exception as exc:
            return _error(exc)

    @mcp.tool()
    async def get_library_stats() -> dict:
        """Return indexed video, photo, face, person, and labeled-person counts."""
        try:
            async with ApiClient() as client:
                return _ok(await client.get("/api/stats"))
        except Exception as exc:
            return _error(exc)

    @mcp.tool()
    async def list_people(
        include_unnamed: bool = False,
        limit: int = 100,
        name_query: str | None = None,
    ) -> dict:
        """List person clusters, optionally including unnamed clusters."""
        try:
            async with ApiClient() as client:
                people = await client.get("/api/persons")
            if not include_unnamed:
                people = [p for p in people if p.get("name")]
            if name_query:
                needle = name_query.casefold()
                people = [p for p in people if needle in (p.get("name") or "").casefold()]
            return _ok(people[:_limit(limit)])
        except Exception as exc:
            return _error(exc)

    @mcp.tool()
    async def get_person(person_id: str) -> dict:
        """Fetch a person record by ID, including representative sample thumbnails."""
        try:
            async with ApiClient() as client:
                return _ok(await client.get(f"/api/persons/{person_id}"))
        except Exception as exc:
            return _error(exc)

    @mcp.tool()
    async def search_by_name(name: str, limit: int = 50) -> dict:
        """Search appearances by labeled person name; limit applies per media type."""
        try:
            if not name.strip():
                raise ValueError("name is required")
            max_hits = _limit(limit)
            async with ApiClient() as client:
                data = await client.get("/api/search", params={"name": name})
            return _ok(_filter_search(data, max_hits))
        except Exception as exc:
            return _error(exc)

    @mcp.tool()
    async def search_by_photo(
        image_path: str,
        limit: int = 50,
        distance_threshold: float = 0.5,
    ) -> dict:
        """Search by a reference image path; limit applies per media type."""
        try:
            src = Path(image_path).expanduser()
            if not src.is_file():
                raise ValueError(f"image_path does not exist or is not a file: {image_path}")
            max_hits = _limit(limit)
            async with ApiClient() as client:
                data = await client.post_file("/api/search/photo", str(src))
            return _ok(_filter_search(data, max_hits, distance_threshold))
        except Exception as exc:
            return _error(exc)

    @mcp.tool()
    async def get_media_info(video_id: int | None = None, photo_id: int | None = None) -> dict:
        """Fetch metadata and local API paths for exactly one video or photo."""
        try:
            if (video_id is None) == (photo_id is None):
                raise ValueError("pass exactly one of video_id or photo_id")
            async with ApiClient() as client:
                if video_id is not None:
                    return _ok(await client.get(f"/api/video/{video_id}/info"))
                return _ok(await client.get(f"/api/photo/{photo_id}/info"))
        except Exception as exc:
            return _error(exc)

    @mcp.tool()
    async def compile_highlight_reel(
        person_id: str,
        clip_duration_sec: int = 30,
        merge_gap_sec: float = 30.0,
        max_clips_per_video: int = 5,
        order: Literal["asc", "desc", "random"] = "asc",
    ) -> dict:
        """Start a highlight reel compile job for a person."""
        try:
            payload = {
                "person_id": person_id,
                "clip_duration_sec": clip_duration_sec,
                "merge_gap_sec": merge_gap_sec,
                "max_clips_per_video": max_clips_per_video,
                "order": order,
            }
            async with ApiClient() as client:
                return _ok(await client.post_json("/api/compile", payload))
        except Exception as exc:
            return _error(exc)

    @mcp.tool()
    async def check_compile_status(job_id: str) -> dict:
        """Poll a highlight reel compile job."""
        try:
            async with ApiClient() as client:
                job = await client.get(f"/api/compile/{job_id}")
                job = _add_download_url(client, job, f"/api/compile/{job_id}/download")
            return _ok(job)
        except Exception as exc:
            return _error(exc)

    @mcp.tool()
    async def create_photo_collage(
        person_id: str,
        columns: int = 3,
        sort: Literal["asc", "desc", "random"] = "asc",
        captions: bool = True,
    ) -> dict:
        """Start a photo collage job for a person."""
        try:
            payload = {
                "person_id": person_id,
                "columns": columns,
                "sort": sort,
                "captions": captions,
            }
            async with ApiClient() as client:
                return _ok(await client.post_json("/api/collage", payload))
        except Exception as exc:
            return _error(exc)

    @mcp.tool()
    async def check_collage_status(job_id: str) -> dict:
        """Poll a photo collage job."""
        try:
            async with ApiClient() as client:
                job = await client.get(f"/api/collage/{job_id}")
                job = _add_download_url(client, job, f"/api/collage/{job_id}/download")
            return _ok(job)
        except Exception as exc:
            return _error(exc)
