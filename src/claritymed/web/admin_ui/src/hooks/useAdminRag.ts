import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { adminFetch, adminFetchMultipart } from "../api/client";
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

// Multipart upload + ingest. ``files`` is the list of File objects from
// the dropzone; ``metadata`` is the JSON blob the backend validates as
// RagUpsertMetadata. Returns the dispatched JobSpec.
export interface RagUpsertMetadata {
  name: string;
  topics?: string[];
  language?: string | null;
  cross_lingual?: boolean;
  authority_tier?: number | null;
  license?: string | null;
  dedupe_cosine_threshold?: number;
}

export function useUpsertRagCollection() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({
      files,
      metadata,
    }: {
      files: File[];
      metadata: RagUpsertMetadata;
    }) => {
      const formData = new FormData();
      formData.append("metadata", JSON.stringify(metadata));
      for (const file of files) {
        formData.append("files", file, file.name);
      }
      return adminFetchMultipart<JobSpec>(
        "/api/v1/admin/rag/collections/upsert",
        formData,
      );
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["admin", "jobs"] });
      qc.invalidateQueries({ queryKey: ["admin", "rag", "collections"] });
    },
  });
}
