import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { adminFetch } from "../api/client";

export interface AdminUserSummary {
  user_id: string;
  display_name: string;
  role: "admin" | "user";
  language: "en" | "zh";
  provider_id: string | null;
  active_system_rag_collections: string[];
}

export interface AdminUserPatch {
  display_name?: string;
  role?: "admin" | "user";
  language?: "en" | "zh";
  provider_id?: string | null;
  active_system_rag_collections?: string[];
}

interface AdminUserListResponse {
  items: AdminUserSummary[];
  total_count: number;
  offset: number;
  limit: number;
}

export function useAdminUsers() {
  return useQuery<AdminUserListResponse>({
    queryKey: ["admin", "users"],
    queryFn: () => adminFetch<AdminUserListResponse>("/api/v1/admin/users"),
  });
}

export function usePatchAdminUser() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (args: { userId: string; patch: AdminUserPatch }) =>
      adminFetch<AdminUserSummary>(`/api/v1/admin/users/${args.userId}`, {
        method: "PATCH",
        body: args.patch,
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["admin", "users"] }),
  });
}

export function useResetAdminUserPassword() {
  return useMutation({
    mutationFn: async (args: { userId: string; newPassword: string }) =>
      adminFetch<void>(`/api/v1/admin/users/${args.userId}/reset-password`, {
        method: "POST",
        body: { new_password: args.newPassword },
      }),
  });
}
