"""Admin-side helpers — secrets, YAML write, allowlists, jobs registry.

This package collects everything that the admin routers and admin UI lean
on but the public ``/api/v1/*`` surface does not. Importing from
``claritymed.web.admin`` should never side-effect — the parent admin
router does its own ``Depends(require_admin)`` wiring.
"""
