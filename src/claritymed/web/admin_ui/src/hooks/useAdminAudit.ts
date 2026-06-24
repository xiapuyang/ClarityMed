import { useQuery } from "@tanstack/react-query";

import { adminFetch } from "../api/client";

export interface AuditEvent {
  kind: string;
  payload: Record<string, unknown>;
  request_id: string;
  user_id: string;
  language: string;
  created_at: string;
  trace_id: string | null;
  span_id: string | null;
}

export interface AuditPage {
  items: AuditEvent[];
  total_count: number;
  offset: number;
  limit: number;
  distinct_actors: string[];
}

export interface AuditQuery {
  offset?: number;
  limit?: number;
  kinds?: string[];
  actor?: string;
  request_id?: string;
  since?: string;
  until?: string;
}

function buildQuery(q: AuditQuery): string {
  const params = new URLSearchParams();
  if (q.offset !== undefined) params.set("offset", String(q.offset));
  if (q.limit !== undefined) params.set("limit", String(q.limit));
  if (q.kinds?.length) params.set("kind", q.kinds.join(","));
  if (q.actor) params.set("actor", q.actor);
  if (q.request_id) params.set("request_id", q.request_id);
  if (q.since) params.set("since", q.since);
  if (q.until) params.set("until", q.until);
  return params.toString();
}

export function useAdminAudit(query: AuditQuery) {
  const qs = buildQuery(query);
  return useQuery<AuditPage>({
    queryKey: ["admin", "audit", qs],
    queryFn: () => adminFetch<AuditPage>(`/api/v1/admin/audit?${qs}`),
  });
}
