"""Static topology for the admin servers dependency graph.

Each :class:`ServerNode` describes a process (or a logical pseudo-node)
that the admin SPA renders in the Mermaid graph. The mapping between
node ``id`` and the actual subprocess listen port is hard-coded here —
the same defaults the production launch scripts use. Operators who
override a port via env var see the wrong node turn red; that's a
known limitation we document in the runbook.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

NodeKind = Literal["process", "logical"]


@dataclass(frozen=True)
class ServerNode:
    id: str
    kind: NodeKind
    label: str
    depends_on: tuple[str, ...] = ()
    host: str = "127.0.0.1"
    port: int = 0
    port_env: str | None = None
    health_path: str = "/health"

    def resolved_port(self) -> int | None:
        if self.port_env:
            raw = os.environ.get(self.port_env)
            if raw:
                try:
                    return int(raw)
                except ValueError:
                    return self.port or None
        return self.port or None


# Process nodes — one per subprocess server we launch.
NODES_PROCESS: tuple[ServerNode, ...] = (
    ServerNode(
        id="embedder",
        kind="process",
        label="embedder",
        port=8082,
        port_env="CLARITYMED_EMBEDDER_PORT",
    ),
    ServerNode(
        id="reranker",
        kind="process",
        label="reranker",
        port=8083,
        port_env="BGE_RERANKER_PORT",
        depends_on=("embedder",),
    ),
    ServerNode(
        id="symptoms",
        kind="process",
        label="symptoms",
        port=8084,
        port_env="CLARITYMED_SYMPTOMS_PORT",
    ),
    ServerNode(
        id="vision",
        kind="process",
        label="vision",
        port=8085,
        port_env="CLARITYMED_VISION_PORT",
    ),
    ServerNode(
        id="medical_clip",
        kind="process",
        label="medical_clip",
        port=8086,
        port_env="CLARITYMED_MEDICAL_CLIP_PORT",
    ),
    # Local MLX LLM server (OpenAI-compatible). Powers the chat agent
    # plus OCR vision-LLM and emergency-triage composer. Port matches
    # the ``omlx`` entry in configs/models.yaml.
    ServerNode(
        id="omlx",
        kind="process",
        label="omlx",
        port=8000,
        port_env="CLARITYMED_OMLX_PORT",
    ),
    # FastAPI backend (this very process when launched via
    # ``claritymed-web``). Has a real ``/health`` route.
    ServerNode(
        id="claritymed_web",
        kind="process",
        label="claritymed-web",
        port=8120,
        port_env="CLARITYMED_WEB_PORT",
    ),
    # User-facing Vite dev server (chat SPA). No ``/health`` route —
    # Vite's SPA fallback answers ``/`` with index.html, which is good
    # enough for liveness. In a baked deployment where the SPA is
    # served as static files behind a CDN, port 5173 is unused and
    # this node will read as down; that's a known limitation.
    ServerNode(
        id="claritymed_ui",
        kind="process",
        label="claritymed-ui",
        port=5173,
        port_env="CLARITYMED_UI_PORT",
        health_path="/",
        depends_on=("claritymed_web",),
    ),
    # Admin SPA Vite dev server. Same fallback caveat: in production
    # this is bundled into ``claritymed-web``'s ``/admin/*`` static mount
    # and port 5174 isn't listening — node will read as down.
    # Probe ``/admin/`` directly: vite is configured with
    # ``base: "/admin/"`` so a bare ``/`` returns 302, which our
    # 200-only probe would misread as down.
    ServerNode(
        id="admin_ui",
        kind="process",
        label="admin_ui",
        port=5174,
        port_env="CLARITYMED_ADMIN_UI_PORT",
        health_path="/admin/",
        depends_on=("claritymed_web",),
    ),
)

# Logical pseudo-nodes — operators see "what breaks if X dies".
NODES_LOGICAL: tuple[ServerNode, ...] = (
    ServerNode(
        id="rag",
        kind="logical",
        label="RAG",
        depends_on=("embedder", "reranker"),
    ),
    ServerNode(
        id="chat",
        kind="logical",
        label="Chat",
        # Chat agent loop dispatches to RAG retrieval, the LLM backend
        # (omlx), and to each tool. Listing them here puts a Chat → X
        # edge in the graph so operators see what fans out when chat
        # is broken.
        depends_on=(
            "rag",
            "omlx",
            "symptoms_tool",
            "vision_tool",
            "clip_tool",
        ),
    ),
    ServerNode(
        id="vision_tool",
        kind="logical",
        label="Vision tool",
        depends_on=("vision",),
    ),
    ServerNode(
        id="symptoms_tool",
        kind="logical",
        label="Symptoms tool",
        depends_on=("symptoms",),
    ),
    ServerNode(
        id="clip_tool",
        kind="logical",
        label="CLIP tool",
        depends_on=("medical_clip",),
    ),
)


def all_nodes() -> tuple[ServerNode, ...]:
    return NODES_PROCESS + NODES_LOGICAL


def edges() -> list[tuple[str, str]]:
    return [(n.id, dep) for n in all_nodes() for dep in n.depends_on]
