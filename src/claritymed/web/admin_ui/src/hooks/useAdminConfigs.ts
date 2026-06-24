import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { adminFetch } from "../api/client";

export interface ConfigCatalog {
  configs: string[];
  editable_keys: Record<string, string[]>;
}

export interface ConfigDetail {
  name: string;
  data: Record<string, unknown>;
  editable_keys: string[];
}

export function useAdminConfigCatalog() {
  return useQuery<ConfigCatalog>({
    queryKey: ["admin", "configs"],
    queryFn: () => adminFetch<ConfigCatalog>("/api/v1/admin/configs"),
  });
}

export function useAdminConfig(name: string | null) {
  return useQuery<ConfigDetail>({
    queryKey: ["admin", "configs", name],
    queryFn: () =>
      adminFetch<ConfigDetail>(`/api/v1/admin/configs/${name}`),
    enabled: !!name,
  });
}

export function usePatchAdminConfig() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (args: { name: string; path: string; value: unknown }) =>
      adminFetch<ConfigDetail>(`/api/v1/admin/configs/${args.name}`, {
        method: "PATCH",
        body: { path: args.path, value: args.value },
      }),
    onSuccess: (_, vars) =>
      qc.invalidateQueries({ queryKey: ["admin", "configs", vars.name] }),
  });
}

export function getDotted(data: Record<string, unknown>, dotted: string): unknown {
  const parts = dotted.split(".");
  let cursor: unknown = data;
  for (const part of parts) {
    if (cursor && typeof cursor === "object" && part in (cursor as Record<string, unknown>)) {
      cursor = (cursor as Record<string, unknown>)[part];
    } else {
      return undefined;
    }
  }
  return cursor;
}
