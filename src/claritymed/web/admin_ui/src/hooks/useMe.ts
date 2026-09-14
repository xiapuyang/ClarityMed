import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { useTranslation } from "react-i18next";

import { adminFetch } from "../api/client";
import type { MeResponse } from "../api/types";

// Read /api/v1/me — the chat SPA's account endpoint. The admin SPA
// piggy-backs so language preference stays consistent across the two
// surfaces. The first response also switches i18n.
export function useMe() {
  const { i18n } = useTranslation();
  const query = useQuery<MeResponse>({
    queryKey: ["me"],
    queryFn: () => adminFetch<MeResponse>("/api/v1/me"),
  });
  useEffect(() => {
    if (query.data?.language && query.data.language !== i18n.language) {
      void i18n.changeLanguage(query.data.language);
    }
  }, [query.data?.language, i18n]);
  return query;
}
