"""Web routers — one APIRouter per business surface.

* :mod:`claritymed.web.routers.auth` — ``/auth/login`` + ``/auth/logout``.
* :mod:`claritymed.web.routers.me` — ``/api/v1/me`` (Unit 3).
* :mod:`claritymed.web.routers.chat` — ``/api/v1/sessions`` + SSE stream
  endpoint (Unit 4).

Routers are registered in :func:`claritymed.web.app.create_app`.
"""
