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
  if (!res.ok) {
    const text = await res.text();
    let detail: unknown = null;
    try {
      detail = JSON.parse(text);
    } catch {
      detail = text;
    }
    throw new AdminApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// Multipart upload variant. Same auth/CSRF/redirect posture as adminFetch
// but lets the browser set the multipart Content-Type with the right
// boundary (you can't set it by hand — fetch needs to compute it).
export async function adminFetchMultipart<T = unknown>(
  path: string,
  formData: FormData,
): Promise<T> {
  const headers = new Headers();
  const csrf = readCsrfToken();
  if (csrf) headers.set("X-CSRF-Token", csrf);
  const res = await fetch(path, {
    method: "POST",
    headers,
    credentials: "include",
    body: formData,
  });
  if (res.status === 401) {
    window.location.assign("/");
    throw new AdminApiError(401, null, "not_authenticated");
  }
  if (!res.ok) {
    const text = await res.text();
    let detail: unknown = null;
    try {
      detail = JSON.parse(text);
    } catch {
      detail = text;
    }
    throw new AdminApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}
