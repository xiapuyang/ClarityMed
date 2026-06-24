import { MantineProvider } from "@mantine/core";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { MemoryRouter, Route, Routes } from "react-router-dom";

import { AdminShell } from "../../src/components/AdminShell";
import i18n from "../../src/i18n";

const NAV_LABELS_EN = [
  "Overview",
  "RAG Corpus",
  "Benchmark",
  "Models",
  "Servers",
  "Users",
  "System Config",
  "Localization",
  "Audit Log",
  "Jobs",
];

function renderShell() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <MantineProvider>
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={["/"]}>
          <Routes>
            <Route element={<AdminShell />}>
              <Route index element={<div>home</div>} />
            </Route>
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>
    </MantineProvider>,
  );
}

describe("AdminShell", () => {
  beforeEach(() => {
    void i18n.changeLanguage("en");
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({
          user_id: "admin",
          role: "admin",
          display_name: "Admin",
          language: "en",
          provider_id: null,
        }),
      }),
    );
  });

  it("renders the 10 nav links in locked order", () => {
    renderShell();
    const links = screen.getAllByRole("link");
    const labels = links.map((a) => a.textContent?.trim()).filter(Boolean);
    for (const expected of NAV_LABELS_EN) {
      expect(labels).toContain(expected);
    }
    // Order check — Overview appears before Jobs in the rendered DOM.
    const overviewIdx = labels.indexOf("Overview");
    const jobsIdx = labels.indexOf("Jobs");
    expect(overviewIdx).toBeLessThan(jobsIdx);
  });

  it("renders the app title", () => {
    renderShell();
    expect(screen.getByText("ClarityMed Admin")).toBeInTheDocument();
  });
});
