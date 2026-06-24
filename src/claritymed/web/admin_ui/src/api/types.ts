// Hand-typed wire shapes shared across hooks/pages. Stays minimal in U1
// (only what the foundation needs) — each module unit appends its own
// types here.

export type AdminHealth = { status: "ok" };

export type MeResponse = {
  user_id: string;
  role: "admin" | "user";
  display_name: string;
  language: "en" | "zh";
  provider_id: string | null;
};
