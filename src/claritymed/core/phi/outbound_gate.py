"""OutboundTextGate: scrub free-text PII before it leaves the process.

Mirrors the ``PhiScrubSpanProcessor`` pattern used for tracing: a pipeline
stage injected at construction time so individual service clients (embedder,
reranker, translation) stay PHI-unaware.

Usage
-----
- ``PhiOutboundGate`` wraps ``ScrubService`` and is activated for cloud
  service entries (``phi_kind="cloud"``).
- ``make_outbound_gate(phi_kind)`` returns a gate for cloud or ``None`` for
  local, so callers can use ``if self._scrub_gate`` without a null-check
  protocol.
- Local services receive ``scrub_gate=None`` — zero overhead, no scrubbing.

phi_kind resolution
-------------------
Explicit config always wins. When ``phi_kind`` is ``None`` (not set in YAML),
``resolve_phi_kind`` auto-detects from the service URL: hosts in
``_LOCAL_HOSTS`` are safe, everything else is treated as cloud (scrub).
This is the "secure by default" posture — a forgotten config on a remote
URL scrubs rather than leaks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlparse

if TYPE_CHECKING:
    from claritymed.core.scrub.service import ScrubService

_LOCAL_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})


class OutboundTextGate(Protocol):
    """Protocol for pre-egress text scrubbers injected into service clients."""

    def scrub(self, text: str) -> str: ...

    def scrub_batch(self, texts: list[str]) -> list[str]: ...


class PhiOutboundGate:
    """Scrubs free-text PII before text reaches an external cloud service.

    Delegates to ``ScrubService`` (regex + optional privacy-filter model).
    The ScrubReport is intentionally discarded — individual span-level audit
    of outbound scrubbing is handled by ``PhiScrubSpanProcessor`` in tracing.
    """

    def __init__(self, scrub_svc: "ScrubService") -> None:
        self._scrub = scrub_svc

    def scrub(self, text: str) -> str:
        scrubbed, _ = self._scrub.scrub(text)
        return scrubbed

    def scrub_batch(self, texts: list[str]) -> list[str]:
        return [self.scrub(t) for t in texts]


def resolve_phi_kind(phi_kind: str | None, base_url: str) -> str:
    """Return the effective phi kind for a service with the given URL.

    - Explicit ``"local"`` or ``"cloud"`` → returned as-is.
    - ``None`` (not configured in YAML): hostname in ``_LOCAL_HOSTS``
      → ``"local"``; any other host → ``"cloud"`` (secure by default).
    """
    if phi_kind is not None:
        return phi_kind
    hostname = urlparse(base_url).hostname or ""
    return "local" if hostname in _LOCAL_HOSTS else "cloud"


def make_outbound_gate(phi_kind: str) -> PhiOutboundGate | None:
    """Return a ``PhiOutboundGate`` for cloud providers, ``None`` for local.

    Expects an already-resolved kind string (``"local"`` or ``"cloud"``).
    Call ``resolve_phi_kind`` first when the raw config value may be ``None``.
    """
    if phi_kind != "cloud":
        return None
    from claritymed.core.scrub.service import ScrubService

    return PhiOutboundGate(ScrubService.from_config())
