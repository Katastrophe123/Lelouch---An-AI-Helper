import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// In development the API runs separately on :8000; these paths are proxied so
// the browser only ever talks to one origin (no CORS, cookies or tokens leak).
const API = process.env.API_BASE_URL ?? "http://localhost:8000";
const apiPaths = ["/auth", "/conversations", "/documents", "/agent", "/health"];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(apiPaths.map((path) => [path, { target: API, changeOrigin: true }])),
  },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 900 },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
