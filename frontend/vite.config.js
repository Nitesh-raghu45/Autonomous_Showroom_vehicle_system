import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Every backend path the UI talks to is proxied to uvicorn on :8000, so the
// browser only ever sees one origin and SSE is not a cross-origin request.
const backend = process.env.BACKEND_URL || "http://localhost:8000";
const proxied = ["/health", "/metrics", "/customers", "/customer", "/session", "/chat", "/inventory"];

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: Object.fromEntries(
      proxied.map((path) => [
        path,
        {
          target: backend,
          changeOrigin: true,
          // http-proxy buffers by default on some setups; this keeps the SSE
          // stream flowing chunk by chunk instead of arriving all at once.
          configure: (proxy) => {
            proxy.on("proxyRes", (proxyRes) => {
              proxyRes.headers["cache-control"] = "no-cache, no-transform";
            });
          },
        },
      ])
    ),
  },
  preview: {
    host: "0.0.0.0",
    port: 4173,
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
