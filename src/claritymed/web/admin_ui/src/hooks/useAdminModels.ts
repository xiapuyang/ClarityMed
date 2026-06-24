import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { adminFetch } from "../api/client";

export interface CatalogDetail {
  name: string;
  data: Record<string, unknown>;
}

export interface SecretRow {
  key: string;
  label: string;
  hint: string;
  category: string;
  required: boolean;
  is_set: boolean;
  source: "env" | "file" | "missing";
}

export interface SecretsPayload {
  manifest: Omit<SecretRow, "is_set" | "source">[];
  status: SecretRow[];
}

export function useAdminCatalogs() {
  return useQuery<{ catalogs: string[] }>({
    queryKey: ["admin", "models", "catalogs"],
    queryFn: () =>
      adminFetch<{ catalogs: string[] }>("/api/v1/admin/models/catalogs"),
  });
}

export function useAdminCatalog(name: string | null) {
  return useQuery<CatalogDetail>({
    queryKey: ["admin", "models", "catalogs", name],
    queryFn: () =>
      adminFetch<CatalogDetail>(`/api/v1/admin/models/catalogs/${name}`),
    enabled: !!name,
  });
}

export function usePatchAdminCatalog() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (args: { name: string; data: Record<string, unknown> }) =>
      adminFetch<CatalogDetail>(
        `/api/v1/admin/models/catalogs/${args.name}`,
        { method: "PATCH", body: { data: args.data } },
      ),
    onSuccess: (_, vars) =>
      qc.invalidateQueries({
        queryKey: ["admin", "models", "catalogs", vars.name],
      }),
  });
}

export function useAdminSecrets() {
  return useQuery<SecretsPayload>({
    queryKey: ["admin", "models", "secrets"],
    queryFn: () => adminFetch<SecretsPayload>("/api/v1/admin/models/secrets"),
  });
}

export function usePatchAdminSecrets() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (updates: Record<string, string>) =>
      adminFetch<{ keys_changed: string[]; restart_required: boolean }>(
        "/api/v1/admin/models/secrets",
        { method: "PATCH", body: { updates } },
      ),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["admin", "models", "secrets"] }),
  });
}
