import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built into ./dist/, served by FastAPI's StaticFiles mount at /admin.
// `base: "/admin/"` makes the bundled asset URLs absolute under that
// prefix so deep-link refreshes (/admin/users, /admin/models) still
// resolve to the right script paths.
export default defineConfig({
  plugins: [react()],
  base: "/admin/",
  build: {
    outDir: "./dist",
    emptyOutDir: true,
    sourcemap: true,
  },
  server: {
    port: 5174,
    proxy: {
      "/api": "http://127.0.0.1:8120",
      "/auth": "http://127.0.0.1:8120",
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
  },
});
