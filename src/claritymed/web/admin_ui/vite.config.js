import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// Vite's dev server with `base: "/admin/"` answers a bare `/admin`
// (no trailing slash) with a public-base-URL error page instead of
// redirecting. Production is fine — Starlette's StaticFiles(html=True)
// at `app.mount("/admin", ...)` redirects bare /admin → /admin/. This
// middleware mirrors that behavior in dev so deep links and shared
// URLs work either way.
function redirectBareAdminPath() {
    return {
        name: "claritymed-redirect-bare-admin",
        configureServer: function (server) {
            server.middlewares.use(function (req, res, next) {
                var url = req.url;
                if (url === "/admin") {
                    res.statusCode = 301;
                    res.setHeader("Location", "/admin/");
                    res.end();
                    return;
                }
                next();
            });
        },
    };
}
// Built into ./dist/, served by FastAPI's StaticFiles mount at /admin.
// `base: "/admin/"` makes the bundled asset URLs absolute under that
// prefix so deep-link refreshes (/admin/users, /admin/models) still
// resolve to the right script paths.
export default defineConfig({
    plugins: [react(), redirectBareAdminPath()],
    base: "/admin/",
    build: {
        outDir: "./dist",
        emptyOutDir: true,
        sourcemap: true,
    },
    server: {
        host: "127.0.0.1",
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
