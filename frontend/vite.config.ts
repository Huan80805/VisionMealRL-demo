import fs from "node:fs";
import path from "node:path";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

const DEFAULT_AGENT_API_URL = "http://127.0.0.1:8000";

function readRootEnv(): Record<string, string> {
  const envPath = path.resolve(__dirname, "../.env");
  if (!fs.existsSync(envPath)) {
    return {};
  }

  return fs
    .readFileSync(envPath, "utf8")
    .split(/\r?\n/)
    .reduce<Record<string, string>>((acc, line) => {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) {
        return acc;
      }
      const sep = trimmed.indexOf("=");
      if (sep === -1) {
        return acc;
      }
      const key = trimmed.slice(0, sep).trim();
      const value = trimmed.slice(sep + 1).trim().replace(/^["']|["']$/g, "");
      acc[key] = value;
      return acc;
    }, {});
}

export default defineConfig(({ mode }) => {
  const frontendEnv = loadEnv(mode, process.cwd(), "");
  const rootEnv = readRootEnv();
  const hfToken = process.env.HF_TOKEN ?? frontendEnv.HF_TOKEN ?? rootEnv.HF_TOKEN;
  const hfSpaceUrl =
    frontendEnv.HF_SPACE_URL ??
    rootEnv.HF_SPACE_URL ??
    process.env.HF_SPACE_URL ??
    DEFAULT_AGENT_API_URL;

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/agent-api": {
          target: hfSpaceUrl,
          changeOrigin: true,
          secure: true,
          rewrite: (proxyPath) => proxyPath.replace(/^\/agent-api/, ""),
          headers: hfToken
            ? {
                Authorization: `Bearer ${hfToken}`,
              }
            : undefined,
        },
      },
    },
  };
});
