import { readCsrfToken } from "./csrf";

export class AdminApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, detail: unknown, message?: string) {
    super(message ?? `admin api ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

type FetchInit = Omit<RequestInit, "body"> & { body?: unknown };

// Centralized fetch wrapper:
// - Adds CSRF header on mutations.
// - Includes credentials so the session cookie travels.
// - Throws AdminApiError on non-2xx with the parsed detail attached.
// - Redirects to the chat SPA login on 401 (admin SPA does not own auth).
// - Routes 403 to the local /admin/forbidden page so the operator sees
//   the role mismatch rather than a blank screen.
export async function adminFetch<T = unknown>(
  path: string,
  init: FetchInit = {},
): Promise<T> {
  const method = (init.method ?? "GET").toUpperCase();
  const headers = new Headers(init.headers);
  if (init.body !== undefined && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (method !== "GET" && method !== "HEAD") {
    const csrf = readCsrfToken();
    if (csrf) headers.set("X-CSRF-Token", csrf);
  }
  const body =
    init.body === undefined
      ? undefined
      : typeof init.body === "string"
        ? init.body
        : JSON.stringify(init.body);
  const res = await fetch(path, {
    ...init,
    method,
    headers,
    credentials: "include",
    body,
  });
  if (res.status === 401) {
    window.location.assign("/");
    throw new AdminApiError(401, null, "not_authenticated");
  }
  if (res.status === 403) {
    window.location.assign("/admin/forbidden");
    throw new AdminApiError(403, null, "forbidden");
  }
  if (!res.ok) {
    let detail: unknown = null;
    try {
      detail = await res.json();
    } catch {
      detail = await res.text();
    }
    throw new AdminApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}
