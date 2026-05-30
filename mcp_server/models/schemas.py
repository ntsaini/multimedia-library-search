from typing import Any

from pydantic import BaseModel, Field


class ToolError(BaseModel):
    type: str
    message: str
    status_code: int | None = None


class ToolResult(BaseModel):
    ok: bool = True
    data: Any = None
    error: ToolError | None = None


class SearchResult(BaseModel):
    videos: list[dict] = Field(default_factory=list)
    photos: list[dict] = Field(default_factory=list)


def dump_model(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
