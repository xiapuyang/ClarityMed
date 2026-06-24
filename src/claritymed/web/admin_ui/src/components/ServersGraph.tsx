import { Box } from "@mantine/core";
import mermaid from "mermaid";
import { useEffect, useRef } from "react";

import type { ServerEdge, ServerNodeData } from "../hooks/useAdminServers";

mermaid.initialize({ startOnLoad: false, theme: "default", securityLevel: "loose" });

function buildSource(nodes: ServerNodeData[], edges: ServerEdge[]): string {
  const lines: string[] = ["graph LR"];
  for (const n of nodes) {
    const meta =
      n.pid !== undefined && n.port !== undefined
        ? `<br/>port ${n.port} · pid ${n.pid}`
        : n.port !== undefined
          ? `<br/>port ${n.port}`
          : "";
    const cls =
      n.status === "up"
        ? ":::up"
        : n.status === "down"
          ? ":::down"
          : n.status === "timeout"
            ? ":::timeout"
            : ":::logical";
    if (n.kind === "logical") {
      lines.push(`  ${n.id}([${n.label}])${cls}`);
    } else {
      lines.push(`  ${n.id}["${n.label}${meta}"]${cls}`);
    }
  }
  for (const e of edges) {
    lines.push(`  ${e.from} --> ${e.to}`);
  }
  lines.push(
    "  classDef up fill:#1f7a3a,stroke:#0e4e25,color:#fff",
    "  classDef down fill:#a4222b,stroke:#6e161c,color:#fff",
    "  classDef timeout fill:#a48b1f,stroke:#6e5a14,color:#fff",
    "  classDef logical fill:#374151,stroke:#1f2937,color:#e5e7eb,stroke-dasharray:4 2",
  );
  return lines.join("\n");
}

export function ServersGraph({
  nodes,
  edges,
}: {
  nodes: ServerNodeData[];
  edges: ServerEdge[];
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const src = buildSource(nodes, edges);
    let cancelled = false;
    const renderId = `servers-graph-${Date.now()}`;
    mermaid
      .render(renderId, src)
      .then(({ svg }) => {
        if (cancelled || !ref.current) return;
        ref.current.innerHTML = svg;
      })
      .catch((err) => {
        if (ref.current) {
          ref.current.innerText = `mermaid: ${(err as Error).message}`;
        }
      });
    return () => {
      cancelled = true;
    };
  }, [nodes, edges]);

  return <Box ref={ref} />;
}
