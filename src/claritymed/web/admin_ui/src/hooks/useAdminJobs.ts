import { useQuery } from "@tanstack/react-query";

import { adminFetch } from "../api/client";

export type JobState =
  | "queued"
  | "running"
  | "done"
  | "failed"
  | "cancelled"
  | "crashed";

export type JobKind = "rag_ingest" | "benchmark_run";

export interface JobSpec {
  id: string;
  kind: JobKind;
  state: JobState;
  params: Record<string, unknown>;
  progress: string;
  stdout_tail: string[];
  started_at: number | null;
  finished_at: number | null;
  exit_code: number | null;
  error: string | null;
  cancellable: boolean;
  created_at: number;
}

interface ListJobsResponse {
  items: JobSpec[];
  total: number;
}

// Poll every 2s while any job is queued or running, otherwise stop. The
// chip cards and the Jobs page reuse this hook.
export function useAdminJobs() {
  return useQuery<ListJobsResponse>({
    queryKey: ["admin", "jobs"],
    queryFn: () => adminFetch<ListJobsResponse>("/api/v1/admin/jobs?limit=100"),
    refetchInterval: (q) => {
      const data = q.state.data;
      if (!data) return 2000;
      const active = data.items.some(
        (j) => j.state === "queued" || j.state === "running",
      );
      return active ? 2000 : false;
    },
  });
}

export async function cancelAdminJob(jobId: string): Promise<JobSpec> {
  return adminFetch<JobSpec>(`/api/v1/admin/jobs/${jobId}`, { method: "DELETE" });
}

const TERMINAL_STATES: ReadonlySet<JobState> = new Set([
  "done",
  "failed",
  "cancelled",
  "crashed",
]);

// Poll a single job until it reaches a terminal state. Used by the RAG
// upsert and bootstrap notifications so the operator gets a real
// outcome message instead of just "queued.". Caps the wait so a hung
// runner doesn't pin the notification open forever.
export async function pollJobUntilTerminal(
  jobId: string,
  { timeoutMs = 600_000, intervalMs = 1500 }: { timeoutMs?: number; intervalMs?: number } = {},
): Promise<JobSpec> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const spec = await adminFetch<JobSpec>(`/api/v1/admin/jobs/${jobId}`);
    if (TERMINAL_STATES.has(spec.state)) return spec;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  // Timed out — return the last-seen spec so the caller can still
  // render whatever progress was reported.
  return adminFetch<JobSpec>(`/api/v1/admin/jobs/${jobId}`);
}
