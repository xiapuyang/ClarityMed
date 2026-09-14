import { useQuery } from "@tanstack/react-query";

import { adminFetch } from "../api/client";
import type { JobSpec } from "./useAdminJobs";
import type { AuditEvent } from "./useAdminAudit";

export interface OverviewServerNode {
  id: string;
  label: string;
  status: "up" | "down" | "timeout" | "n/a";
}

export interface OverviewResponse {
  providers: { count: number; default: string | null };
  users: { count: number; admin_count: number };
  recent_jobs: { items: JobSpec[]; active: boolean };
  audit_tail: { items: AuditEvent[] };
  servers: { nodes: OverviewServerNode[]; ready: boolean };
}

export function useAdminOverview() {
  return useQuery<OverviewResponse>({
    queryKey: ["admin", "overview"],
    queryFn: () => adminFetch<OverviewResponse>("/api/v1/admin/overview"),
    refetchInterval: 10_000,
  });
}
