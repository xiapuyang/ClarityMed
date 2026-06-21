"""Pydantic request / response models for the web v1 API.

Every endpoint declares its request and response model here (or in a
per-router schema module). FastAPI's auto-generated ``/openapi.json``
becomes the published contract; the frontend hand-mirrors these shapes
in TypeScript until codegen lands with the first admin-endpoint PR.

Unit 1 ships only ``ErrorEnvelope`` — later units (auth / me / chat)
add their own request/response models alongside their routers.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ErrorEnvelope(BaseModel):
    """Common error body. FastAPI's default 4xx/5xx body uses ``{"detail"}``
    too — declaring it as a schema lets OpenAPI consumers pick it up.
    """

    model_config = ConfigDict(extra="forbid")

    detail: str
