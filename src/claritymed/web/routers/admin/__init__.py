"""Parent admin router — ``/api/v1/admin/*``.

All admin-only endpoints hang off a single :class:`APIRouter` with the
``Depends(require_admin)`` gate applied once, here. Sub-routers add their
endpoints to leaf modules in this package; this file imports each leaf
module's ``router`` object and includes it with no extra prefix (the
leaves declare their own).

The intent is that ``require_admin`` is enforced exactly once — at the
boundary — so a future contributor can't accidentally publish an admin
endpoint by forgetting the ``Depends`` on their handler.

The leaf modules ship as empty stubs in U1 and grow per-unit.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from claritymed.web.deps import require_admin

from claritymed.web.routers.admin import (
    audit,
    benchmark,
    configs,
    i18n,
    jobs,
    models,
    overview,
    rag,
    servers,
    users,
)

router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


@router.get("/_health")
async def _health() -> dict[str, str]:
    """Cheap admin-only health probe.

    Used by the admin SPA to confirm the user still holds an admin
    session after a long idle, before showing any data. A regular
    ``GET /health`` returns ``200`` even to anonymous callers, so the
    SPA can't use it to gate admin nav state.
    """
    return {"status": "ok"}


router.include_router(audit.router)
router.include_router(benchmark.router)
router.include_router(configs.router)
router.include_router(i18n.router)
router.include_router(jobs.router)
router.include_router(models.router)
router.include_router(overview.router)
router.include_router(rag.router)
router.include_router(servers.router)
router.include_router(users.router)
