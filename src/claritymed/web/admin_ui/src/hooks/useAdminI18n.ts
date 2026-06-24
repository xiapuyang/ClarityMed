import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { adminFetch } from "../api/client";

export type I18nSurface = "backend" | "admin";
export type I18nLang = "en" | "zh";

interface I18nReadResponse {
  lang: I18nLang;
  data: Record<string, unknown>;
}

interface I18nPatchResponse extends I18nReadResponse {
  needs_rebuild: boolean;
}

const SURFACE_PATH: Record<I18nSurface, string> = {
  backend: "backend-strings",
  admin: "admin-strings",
};

export function useAdminI18n(surface: I18nSurface, lang: I18nLang) {
  return useQuery<I18nReadResponse>({
    queryKey: ["admin", "i18n", surface, lang],
    queryFn: () =>
      adminFetch<I18nReadResponse>(
        `/api/v1/admin/i18n/${SURFACE_PATH[surface]}?lang=${lang}`,
      ),
  });
}

export function usePatchAdminI18n(surface: I18nSurface, lang: I18nLang) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (updates: Record<string, unknown>) =>
      adminFetch<I18nPatchResponse>(
        `/api/v1/admin/i18n/${SURFACE_PATH[surface]}?lang=${lang}`,
        { method: "PATCH", body: { updates } },
      ),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["admin", "i18n", surface, lang] }),
  });
}

// Reshape a nested dict into [{path, value}] tuples so the editor can
// render one row per leaf, regardless of how deep the keys go.
export function flattenStrings(
  data: Record<string, unknown>,
  prefix = "",
): { path: string; value: string }[] {
  const out: { path: string; value: string }[] = [];
  for (const [k, v] of Object.entries(data)) {
    const full = prefix ? `${prefix}.${k}` : k;
    if (v !== null && typeof v === "object" && !Array.isArray(v)) {
      out.push(...flattenStrings(v as Record<string, unknown>, full));
    } else {
      out.push({ path: full, value: String(v ?? "") });
    }
  }
  return out;
}

export function makeNested(path: string, value: string): Record<string, unknown> {
  const parts = path.split(".");
  const top: Record<string, unknown> = {};
  let cursor: Record<string, unknown> = top;
  for (let i = 0; i < parts.length - 1; i++) {
    const next: Record<string, unknown> = {};
    cursor[parts[i]] = next;
    cursor = next;
  }
  cursor[parts[parts.length - 1]] = value;
  return top;
}
