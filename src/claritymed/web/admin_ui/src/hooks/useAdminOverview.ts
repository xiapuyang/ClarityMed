import { useQuery } from "@tanstack/react-query";

import { adminFetch } from "../api/client";
import type { JobSpec } from "./useAdminJobs";
import type { AuditEvent } from "./useAdminAudit";

export interface OverviewResponse {
  providers: { count: number; default: string | null };
  users: { count: number; admin_count: number };
  recent_jobs: { items: JobSpec[]; active: boolean };
  audit_tail: { items: AuditEvent[] };
  servers: { nodes: unknown[]; edges: unknown[]; ready: boolean };
}

export function useAdminOverview() {
  return useQuery<OverviewResponse>({
    queryKey: ["admin", "overview"],
    queryFn: () => adminFetch<OverviewResponse>("/api/v1/admin/overview"),
    refetchInterval: 10_000,
  });
}
