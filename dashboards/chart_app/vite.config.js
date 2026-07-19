import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

export default defineConfig({
  plugins: [react()],
  // Vite 8's native OXC minifier currently exits without a diagnostic on this
  // Windows setup. Keep builds reproducible; the dashboard is served locally.
  build: {
    minify: false,
    rollupOptions: {
      input: {
        chart: resolve(import.meta.dirname, "index.html"),
        dataManager: resolve(import.meta.dirname, "management.html"),
      },
    },
  },
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://127.0.0.1:8000",
        ws: true,
      },
    },
  },
});
