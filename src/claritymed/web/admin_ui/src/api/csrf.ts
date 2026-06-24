// Read the CSRF cookie set by the backend middleware (rotated on safe
// requests). The matching header `X-CSRF-Token` must appear on every
// state-mutating request.

const CSRF_COOKIE = "csrf_token";

export function readCsrfToken(): string | null {
  const raw = document.cookie || "";
  for (const part of raw.split(";")) {
    const [name, value] = part.trim().split("=");
    if (name === CSRF_COOKIE && value) {
      return decodeURIComponent(value);
    }
  }
  return null;
}
