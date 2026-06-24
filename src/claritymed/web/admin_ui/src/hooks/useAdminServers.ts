import { useQuery } from "@tanstack/react-query";

import { adminFetch } from "../api/client";

export type ServerStatus = "up" | "down" | "timeout" | "n/a";

export interface ServerNodeData {
  id: string;
  kind: "process" | "logical";
  label: string;
  status: ServerStatus;
  port?: number;
  uptime_s?: number;
  pid?: number;
  manifest_sha?: string;
  detail?: string;
}

export interface ServerEdge {
  from: string;
  to: string;
}

export interface ServersPayload {
  nodes: ServerNodeData[];
  edges: ServerEdge[];
}

export function useAdminServers() {
  return useQuery<ServersPayload>({
    queryKey: ["admin", "servers"],
    queryFn: () => adminFetch<ServersPayload>("/api/v1/admin/servers"),
    refetchInterval: 10_000,
  });
}
