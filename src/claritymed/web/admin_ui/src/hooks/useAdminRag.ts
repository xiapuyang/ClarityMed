import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { adminFetch } from "../api/client";
import type { JobSpec } from "./useAdminJobs";

export interface RagCollection {
  name: string;
  language: string;
  authority_tier: number;
  topics: string[];
  license: string | null;
  chunk_count: number | null;
}

interface RagListResponse {
  items: RagCollection[];
  total_count: number;
}

export function useAdminRagCollections() {
  return useQuery<RagListResponse>({
    queryKey: ["admin", "rag", "collections"],
    queryFn: () =>
      adminFetch<RagListResponse>("/api/v1/admin/rag/collections"),
  });
}

export function useDeleteRagCollection() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (name: string) =>
      adminFetch<void>(`/api/v1/admin/rag/collections/${name}`, {
        method: "DELETE",
      }),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["admin", "rag", "collections"] }),
  });
}

export function useTriggerRagBootstrap() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async () =>
      adminFetch<JobSpec>("/api/v1/admin/rag/bootstrap", {
        method: "POST",
        body: { skip_existing: true },
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["admin", "jobs"] }),
  });
}
