import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { adminFetch } from "../api/client";
import type { JobSpec } from "./useAdminJobs";

export interface BenchmarkRunSummary {
  run_id: string;
  complete: boolean;
  summary: Record<string, unknown> | null;
}

interface BenchmarkListResponse {
  items: BenchmarkRunSummary[];
  total_count: number;
}

export interface BenchmarkRunRequest {
  dataset?: string;
  provider_id?: string;
  sample_size?: number;
  runner?: string;
  extra_args?: string[];
}

export function useAdminBenchmarkRuns() {
  return useQuery<BenchmarkListResponse>({
    queryKey: ["admin", "benchmark", "runs"],
    queryFn: () =>
      adminFetch<BenchmarkListResponse>("/api/v1/admin/benchmark/runs"),
  });
}

export function useTriggerBenchmark() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (req: BenchmarkRunRequest) =>
      adminFetch<JobSpec>("/api/v1/admin/benchmark/runs", {
        method: "POST",
        body: req,
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["admin", "jobs"] }),
  });
}
