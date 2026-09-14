"""PHI policy primitives (out-of-domain filter)."""

from claritymed.core.phi.guard import (
    ChunkFilterReport,
    PhiGuard,
    PhiHit,
    PhiRules,
)
from claritymed.core.phi.outbound_gate import (
    OutboundTextGate,
    PhiOutboundGate,
    make_outbound_gate,
    resolve_phi_kind,
)

__all__ = [
    "ChunkFilterReport",
    "OutboundTextGate",
    "PhiGuard",
    "PhiHit",
    "PhiOutboundGate",
    "PhiRules",
    "make_outbound_gate",
    "resolve_phi_kind",
]
