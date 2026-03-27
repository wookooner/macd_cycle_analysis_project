import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const baseDataDir = path.resolve(projectRoot, "data", "base_data");

function toNumber(value) {
  if (value == null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function toDateTimeLocal(value) {
  if (!value) return null;
  return String(value).replace(" ", "T").slice(0, 16);
}

function csvApiPlugin() {
  return {
    name: "base-data-api",
    configureServer(server) {
      server.middlewares.use("/api/base-data/files", async (_req, res) => {
        try {
          const entries = await fs.readdir(baseDataDir, { withFileTypes: true });
          const files = entries
            .filter((entry) => entry.isFile() && entry.name.endsWith(".csv"))
            .map((entry) => ({
              name: entry.name,
              timeframe: entry.name.replace(".csv", "").split("_").at(-1) ?? "",
            }))
            .sort((left, right) => left.name.localeCompare(right.name));

          const defaultFile = files.find((item) => item.name === "BTCUSD_1h.csv")?.name ?? files[0]?.name ?? "";
          res.setHeader("Content-Type", "application/json; charset=utf-8");
          res.end(JSON.stringify({ files, defaultFile }));
        } catch (error) {
          res.statusCode = 500;
          res.end(JSON.stringify({ error: error.message }));
        }
      });

      server.middlewares.use("/api/base-data/series", async (req, res) => {
        try {
          const requestUrl = new URL(req.originalUrl ?? req.url, "http://localhost");
          const fileName = requestUrl.searchParams.get("file");
          const limitParam = Number(requestUrl.searchParams.get("limit") || "0");
          const limit = Number.isFinite(limitParam) && limitParam > 0 ? Math.floor(limitParam) : 0;

          if (!fileName) {
            res.statusCode = 400;
            res.end(JSON.stringify({ error: "The file query parameter is required." }));
            return;
          }

          const resolvedPath = path.resolve(baseDataDir, fileName);
          if (!resolvedPath.startsWith(baseDataDir)) {
            res.statusCode = 400;
            res.end(JSON.stringify({ error: "The requested file path is not allowed." }));
            return;
          }

          const rawText = await fs.readFile(resolvedPath, "utf8");
          const lines = rawText.split(/\r?\n/).filter(Boolean);
          if (lines.length < 2) {
            res.statusCode = 404;
            res.end(JSON.stringify({ error: "The CSV file is empty." }));
            return;
          }

          const headers = lines[0].split(",").map((item) => item.trim());
          const dataLines = lines.slice(1);
          const slicedLines = limit > 0 ? dataLines.slice(-limit) : dataLines;
          const rows = slicedLines.map((line) => {
            const values = line.split(",");
            const row = {};

            headers.forEach((header, index) => {
              row[header] = values[index] ?? "";
            });

            return {
              unix: toNumber(row.unix),
              date: toDateTimeLocal(row.date),
              open: toNumber(row.open),
              high: toNumber(row.high),
              low: toNumber(row.low),
              close: toNumber(row.close),
              volumeUsd: toNumber(row["Volume USD"]),
                symbol: row.symbol || fileName.split("_")[0],
              macd: toNumber(row.macd),
              macd_signal: toNumber(row.macd_signal),
              macd_hist: toNumber(row.macd_hist),
              rsi: toNumber(row.rsi),
              volume: toNumber(row.volume),
              taker_buy_base: toNumber(row.taker_buy_base),
              volume_delta: toNumber(row.volume_delta),
              cvd: toNumber(row.cvd),
              cvd_rolling: toNumber(row.cvd_rolling),
              ppo: toNumber(row.ppo),
              ppo_signal: toNumber(row.ppo_signal),
              ppo_hist: toNumber(row.ppo_hist),
              delta: toNumber(row.delta),
              ma_7: toNumber(row.ma_7),
              ma_25: toNumber(row.ma_25),
              ma_99: toNumber(row.ma_99),
              oi: toNumber(row.oi),
              oi_usd: toNumber(row.oi_usd),
              funding_rate: toNumber(row.funding_rate),
            };
          });

          res.setHeader("Content-Type", "application/json; charset=utf-8");
          res.end(
            JSON.stringify({
              meta: {
                fileName,
                symbol: rows[0]?.symbol ?? "",
                timeframe: fileName.replace(".csv", "").split("_").at(-1) ?? "",
                rowCount: rows.length,
                totalRowCount: dataLines.length,
                startDate: rows[0]?.date ?? "",
                endDate: rows[rows.length - 1]?.date ?? "",
                columns: headers,
              },
              rows,
            }),
          );
        } catch (error) {
          res.statusCode = 500;
          res.end(JSON.stringify({ error: error.message }));
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), csvApiPlugin()],
});
