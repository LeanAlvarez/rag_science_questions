import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Build output lands in ../static so FastAPI can mount it in production.
// Dev server proxies /api/* to the FastAPI process on :8000 so the frontend
// code can pretend everything is same-origin.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "../static",
    emptyOutDir: true,
    // Small enough app that we don't need chunk splitting yet.
    sourcemap: true,
  },
});
