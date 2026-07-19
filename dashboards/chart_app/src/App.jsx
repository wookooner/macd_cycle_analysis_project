import { useEffect, useEffectEvent, useMemo, useRef, useState } from "react";
import {
  CandlestickSeries,
  ColorType,
  createChart,
  createTextWatermark,
  CrosshairMode,
  HistogramSeries,
  LineSeries,
  LineStyle,
} from "lightweight-charts";
import "./App.css";

const FILES_URL = "/api/base-data/files";
const SERIES_URL = "/api/base-data/series";
const CYCLE_FILES_URL = "/api/cycle-candles/files";
const CYCLE_SERIES_URL = "/api/cycle-candles/series";
const TIMEFRAME_CONTEXT_URL = "/api/timeframe-context/series";
const FOOTPRINT_DATES_URL = "/api/footprint/dates";
const FOOTPRINT_SURFACE_URL = "/api/footprint/surface";
// Keep this list in sync with the raw market-file names.  The file list is still
// the source of truth, so a button is only shown when that timeframe exists.
const TIMEFRAME_ORDER = ["1M", "1w", "1d", "4h", "1h", "30m", "15m", "5m", "1min"];
const TIMEFRAME_LABELS = {
  "1M": "1M",
  "1w": "1W",
  "1d": "1D",
  "4h": "4h",
  "1h": "1h",
  "30m": "30m",
  "15m": "15m",
  "5m": "5m",
  "1min": "1m",
};
const CONTEXT_RIBBON_TIMEFRAMES = {
  "15m": ["1d", "4h", "1h"],
  "1h": ["1d", "4h"],
  "4h": ["1d"],
};
const DEFAULT_TIMEFRAME = "4h";
const DEFAULT_SYMBOL = "BTCUSD";
const DEFAULT_CYCLE_ASSET = "btc";
const DEFAULT_OVERLAYS = ["ma_7", "ma_25"];
const NO_PRICE_OVERLAYS = [];
const DEFAULT_INDICATORS = ["macd", "volume", "rsi"];
const DEFAULT_CYCLE_INDICATORS = ["cycle_type_band", "cycle_ppo_range"];
const VIEW_MODES = {
  TIME: "time",
  CYCLE: "cycle",
};
const PARENT_OVERLAY_MODES = {
  OFF: "off",
  DAY: "1d",
  DAY_WEEK: "1d_1w",
};
const DEFAULT_CYCLE_PARENT_OVERLAY = PARENT_OVERLAY_MODES.DAY_WEEK;
const PARENT_OVERLAY_OPTIONS = [
  { key: PARENT_OVERLAY_MODES.OFF, label: "Off" },
  { key: PARENT_OVERLAY_MODES.DAY, label: "1D" },
  { key: PARENT_OVERLAY_MODES.DAY_WEEK, label: "1D + 1W" },
];
const PARENT_OVERLAY_TIMEFRAMES = ["1h", "4h", "1d"];
const INITIAL_LOAD_LIMIT = 1600;
// The latest 1h candles currently extend beyond the latest OI snapshot.  Load a
// little more history so the most recent available OI is not hidden by the tail
// window.  Other timeframes fit in the normal initial window.
const OI_HISTORY_LOAD_LIMITS = { "1h": 5000 };
const MAX_LOAD_LIMIT = Number.POSITIVE_INFINITY;
const INITIAL_VISIBLE_BARS = 240;
const TIME_SERIES_REFRESH_MS = 15000;
const CYCLE_SERIES_REFRESH_MS = 60000;
const TIME_CYCLE_CONTEXT_REFRESH_MS = 60000;
const TIMEFRAME_CONTEXT_REFRESH_MS = 60000;
const REALTIME_TIMEFRAMES = ["5m", "15m", "1h"];
const TIMEFRAME_SECONDS = {
  "1min": 60,
  "5m": 5 * 60,
  "15m": 15 * 60,
  "30m": 30 * 60,
  "1h": 60 * 60,
  "4h": 4 * 60 * 60,
  "1d": 24 * 60 * 60,
  "1w": 7 * 24 * 60 * 60,
  "1M": 30 * 24 * 60 * 60,
};
const STORAGE_KEY = "tv-dashboard-preferences-v1";

const PRICE_OVERLAYS = [
  { key: "ma_7", label: "MA 7", color: "#f59e0b" },
  { key: "ma_25", label: "MA 25", color: "#d946ef" },
  { key: "ma_99", label: "MA 99", color: "#67e8f9" },
];

const INDICATOR_DEFS = [
  {
    key: "volume",
    label: "Volume",
    stretch: 0.8,
    series: [{ key: "volume", label: "Volume", type: "histogram" }],
  },
  {
    key: "macd",
    label: "MACD 12 26 close 9 EMA",
    stretch: 1,
    referenceLines: [0],
    series: [
      { key: "macd_hist", label: "Hist", type: "histogram" },
      { key: "macd", label: "MACD", color: "#3b82f6", type: "line" },
      { key: "macd_signal", label: "Signal", color: "#f97316", type: "line" },
    ],
  },
  {
    key: "ppo",
    label: "PPO 12 26 close 9 EMA",
    stretch: 1,
    referenceLines: [0],
    series: [
      { key: "ppo_hist", label: "Hist", type: "histogram" },
      { key: "ppo", label: "PPO", color: "#f43f5e", type: "line" },
      { key: "ppo_signal", label: "Signal", color: "#2dd4bf", type: "line" },
    ],
  },
  {
    key: "rsi",
    label: "RSI 14",
    stretch: 1,
    referenceLines: [30, 50, 70],
    series: [{ key: "rsi", label: "RSI", color: "#8b5cf6", type: "line" }],
  },
  {
    key: "volume_delta",
    label: "Volume Delta",
    stretch: 0.9,
    referenceLines: [0],
    series: [{ key: "volume_delta", label: "Volume Delta", type: "histogram" }],
  },
  {
    key: "cvd",
    label: "CVD",
    stretch: 0.9,
    series: [{ key: "cvd", label: "CVD", color: "#22d3ee", type: "line" }],
  },
  {
    key: "cvd_rolling",
    label: "CVD Rolling",
    stretch: 0.9,
    series: [{ key: "cvd_rolling", label: "CVD Rolling", color: "#f472b6", type: "line" }],
  },
  {
    key: "oi",
    label: "OI",
    stretch: 0.9,
    series: [{ key: "oi", label: "OI", color: "#c084fc", type: "line" }],
  },
  {
    key: "oi_contracts",
    label: "OI Contracts",
    stretch: 0.9,
    series: [{ key: "oi_contracts", label: "OI Contracts", color: "#a78bfa", type: "line" }],
  },
  {
    key: "oi_usd",
    label: "OI USD",
    stretch: 0.9,
    series: [{ key: "oi_usd", label: "OI USD", color: "#818cf8", type: "line" }],
  },
  {
    key: "oi_notional",
    label: "OI Notional",
    stretch: 0.9,
    series: [{ key: "oi_notional", label: "OI Notional", color: "#38bdf8", type: "line" }],
  },
  {
    key: "funding_rate",
    label: "Funding Rate",
    stretch: 0.8,
    referenceLines: [0],
    series: [{ key: "funding_rate", label: "Funding Rate", color: "#facc15", type: "line" }],
  },
];

const CYCLE_INDICATOR_DEFS = [
  {
    key: "cycle_type_band",
    label: "Cycle Type",
    stretch: 0.45,
    referenceLines: [0],
    series: [{ key: "cycle_direction_value", label: "Type", type: "histogram" }],
  },
  {
    key: "cycle_ppo_range",
    label: "Cycle PPO Multi-Level",
    stretch: 1,
    referenceLines: [0],
    series: [
      { key: "end_ppo_hist", label: "End Hist", type: "histogram" },
      { key: "start_ppo", label: "Start PPO", color: "#93c5fd", type: "line" },
      { key: "end_ppo", label: "End PPO", color: "#2dd4bf", type: "line" },
      { key: "end_ppo_signal", label: "End Signal", color: "#f59e0b", type: "line" },
    ],
  },
];

function parseFileName(fileName) {
  const match = /^(?<symbol>[A-Z0-9]+)_(?<timeframe>1M|1w|1d|4h|1h|30m|15m|5m|1min)\.csv$/.exec(fileName);
  if (!match?.groups) return null;
  return { symbol: match.groups.symbol.toUpperCase(), timeframe: match.groups.timeframe, fileName };
}

function parseCycleFile(file) {
  if (!file?.asset || !file?.timeframe) return null;
  return { asset: file.asset, symbol: file.symbol ?? file.asset.toUpperCase(), timeframe: file.timeframe };
}

function formatNumber(value) {
  if (value == null || Number.isNaN(value)) return "-";
  const abs = Math.abs(value);
  if (abs >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)}B`;
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(2)}K`;
  if (abs >= 100) return value.toFixed(2);
  if (abs >= 1) return value.toFixed(4);
  return value.toFixed(6);
}

function formatDateLabel(value) {
  if (!value) return "-";
  return value.replace("T", " ").slice(0, 16);
}

function formatChartTime(value) {
  if (typeof value !== "number") return formatDateLabel(String(value));
  const date = new Date(value * 1000);
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const day = String(date.getUTCDate()).padStart(2, "0");
  const hour = String(date.getUTCHours()).padStart(2, "0");
  const minute = String(date.getUTCMinutes()).padStart(2, "0");
  return `${year}-${month}-${day} ${hour}:${minute}`;
}

function formatCycleRange(row) {
  if (!row?.start_date && !row?.date) return "-";
  const start = formatDateLabel(row.start_date ?? row.date);
  const end = formatDateLabel(row.end_date);
  return end && end !== "-" ? `${start} ~ ${end}` : start;
}

function formatCycleType(value) {
  return value ? String(value).toUpperCase() : "-";
}

function contextDirectionLabel(value) {
  if (!value || value === "gap") return "-";
  return String(value).toUpperCase();
}

function formatContextProgress(value) {
  if (value == null || Number.isNaN(value)) return "-";
  return `${Math.round(Number(value) * 100)}%`;
}

function contextRowForTime(rows, time) {
  if (!Array.isArray(rows) || !rows.length || time == null) return null;
  let low = 0;
  let high = rows.length - 1;
  let best = null;
  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const row = rows[mid];
    if ((row?.unix ?? 0) <= time) {
      best = row;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  return best;
}

function contextLookupTime(row, viewMode) {
  if (!row) return null;
  if (viewMode === VIEW_MODES.CYCLE) {
    return unixFromDateLabel(row.end_date) ?? row.unix;
  }
  return row.unix;
}

function cycleRowForTime(rows, time) {
  if (!Array.isArray(rows) || !rows.length || time == null) return null;
  return rows.find((cycle) => {
    const endTime = unixFromDateLabel(cycle.end_date) ?? cycle.unix;
    return cycle.unix <= time && time <= endTime;
  }) ?? null;
}

function formatContextSummary(row) {
  if (!row) return "Ctx -";
  const tfBits = ["1w", "1d", "4h", "1h"].map((timeframe) => {
    const cycleId = row[`${timeframe}_cycle_id`] ?? `key ${row[`${timeframe}_key`] ?? "-"}`;
    const endDate = formatDateLabel(row[`${timeframe}_cycle_end_date`]);
    return `${timeframe.toUpperCase()} ${cycleId} ${contextDirectionLabel(row[`${timeframe}_type`])} ${formatContextProgress(row[`${timeframe}_time_prog`])} -> ${endDate}`;
  });
  return `Ctx ${row.combo_4 ?? "-"} ${row.n_up_4 ?? "-"}/4 Late ${row.major_late_count ?? "-"} | ${tfBits.join(" | ")}`;
}

function parentFieldPrefix(timeframe) {
  if (timeframe === "15m") return "1h";
  if (timeframe === "1h") return "4h";
  if (timeframe === "4h") return "1d";
  if (timeframe === "1d") return "1w";
  return null;
}

function formatCycleSummary(row, timeframe) {
  if (!row) return `Cycle ${timeframe.toUpperCase()} -`;
  return `Cycle ${timeframe.toUpperCase()} ${row.cycle_id ?? "-"} ${formatCycleType(row.cycle_type)}`;
}

function formatImmediateParent(row, timeframe) {
  const prefix = parentFieldPrefix(timeframe);
  if (!prefix) return "Parent -";
  const label = prefix.toUpperCase();
  if (!row) return `${label} -`;
  return `${label} ${row[`parent_${prefix}_cycle_id`] ?? "-"} ${formatCycleType(row[`parent_${prefix}_cycle_type`])}`;
}

function formatWeeklyParent(row) {
  if (!row) return "1W -";
  return `1W ${row.parent_1w_cycle_id ?? "-"} ${formatCycleType(row.parent_1w_cycle_type)}`;
}

function immediateParentSpec(timeframe) {
  if (timeframe === "1h") return { prefix: "4h", label: "4H", className: "parent-cycle-four-hour", alpha: 0.24 };
  if (timeframe === "4h") return { prefix: "1d", label: "1D", className: "parent-cycle-band", alpha: 0.25 };
  if (timeframe === "1d") return { prefix: "1w", label: "1W", className: "parent-cycle-week-band", alpha: 0.28 };
  return null;
}

function unixFromDateLabel(value) {
  if (!value) return null;
  const raw = String(value).trim();
  const normalized = raw.includes("T") ? raw : raw.replace(" ", "T");
  const withTimezone = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(normalized) ? normalized : `${normalized}Z`;
  const parsed = Date.parse(withTimezone);
  return Number.isNaN(parsed) ? null : Math.floor(parsed / 1000);
}

function wsUrl(path) {
  if (typeof window === "undefined") return path;
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
}

function normalizeRealtimeCandle(event) {
  if (event?.type !== "candle_update" || event?.candleType !== "time" || !event.data) return null;
  const row = {
    ...event.data,
    unix: Number(event.data.unix),
    date: formatDateLabel(event.data.date),
    symbol: event.displaySymbol ?? event.symbol,
    volume: event.data.volume,
  };
  if ([row.unix, row.open, row.high, row.low, row.close].some((value) => value == null || Number.isNaN(value))) {
    return null;
  }
  return {
    event,
    row,
    subscriptionKey: `${event.displaySymbol ?? event.symbol}:${event.timeframe}`.toUpperCase(),
  };
}

function toggleValue(list, value) {
  return list.includes(value) ? list.filter((item) => item !== value) : [...list, value];
}

function loadStoredPreferences() {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function getInitialPreference(key, fallback) {
  const stored = loadStoredPreferences();
  if (!stored || !(key in stored)) return fallback;
  return stored[key];
}

function availableSeriesSignature(rows, definitions) {
  const keys = new Set(definitions.flatMap((definition) => definition.series.map((item) => item.key)));
  const available = new Set();
  rows.forEach((row) => {
    keys.forEach((key) => {
      if (row[key] != null) available.add(key);
    });
  });
  return [...available].sort().join("|");
}

function sanitizeRows(rows) {
  if (!Array.isArray(rows)) return [];

  const deduped = new Map();
  rows.forEach((row) => {
    if (row?.unix == null) return;
    deduped.set(row.unix, row);
  });

  return [...deduped.values()]
    .sort((left, right) => left.unix - right.unix)
    .filter((row) => {
      if ([row.open, row.high, row.low, row.close].some((value) => value == null || Number.isNaN(value))) {
        return false;
      }
      if (row.high < row.low) return false;
      if (row.open > row.high || row.open < row.low) return false;
      if (row.close > row.high || row.close < row.low) return false;
      return true;
    });
}

function makeLegend(container, items, variant = "price", title = "") {
  if (!container) return;
  container.replaceChildren();
  container.className = `pane-legend pane-legend--${variant}`;

  const valuesNode = document.createElement("div");
  valuesNode.className = "pane-legend__values";
  container.appendChild(valuesNode);

  if (title) {
    const titleNode = document.createElement("span");
    titleNode.className = "pane-legend__label";
    titleNode.textContent = title;
    valuesNode.appendChild(titleNode);
  }

  items.forEach((item) => {
    if (item.value == null || Number.isNaN(item.value)) return;
    const node = document.createElement("span");
    node.className = "pane-legend__item";
    node.style.color = item.color ?? "#cbd5e1";
    node.textContent = `${item.label} ${formatNumber(item.value)}`;
    valuesNode.appendChild(node);
  });
}

function histogramTone(panelKey, value, previousValue) {
  if (value == null) return undefined;
  const rising = previousValue == null ? true : value >= previousValue;

  if (panelKey === "cycle_type_band") {
    return value >= 0 ? "rgba(96, 165, 250, 0.62)" : "rgba(251, 146, 60, 0.62)";
  }

  if (panelKey === "duration_candles") {
    return "rgba(147, 197, 253, 0.64)";
  }

  if (panelKey === "macd" || panelKey === "ppo" || panelKey === "cycle_ppo_range" || panelKey === "cycle_momentum_delta") {
    if (value >= 0) {
      return rising ? "rgba(126, 239, 208, 0.98)" : "rgba(198, 249, 235, 0.92)";
    }
    return rising ? "rgba(255, 196, 196, 0.9)" : "rgba(255, 123, 123, 0.98)";
  }

  return value >= 0 ? "rgba(126, 239, 208, 0.92)" : "rgba(255, 123, 123, 0.92)";
}

function buildParentBands(rows, parentKey, parentTypeKey) {
  const bands = [];
  let active = null;

  rows.forEach((row) => {
    const parentId = row?.[parentKey];
    if (!parentId || row.unix == null) {
      if (active) {
        bands.push(active);
        active = null;
      }
      return;
    }

    if (!active || active.id !== parentId) {
      if (active) {
        active.endTime = row.unix;
        bands.push(active);
      }
      active = {
        id: parentId,
        type: row[parentTypeKey] ?? "",
        startTime: row.unix,
        endTime: row.unix,
        firstRow: row,
        lastRow: row,
      };
      return;
    }

    active.lastRow = row;
  });

  if (active) {
    active.endTime = unixFromDateLabel(active.lastRow?.end_date) ?? active.lastRow?.unix ?? active.startTime;
    bands.push(active);
  }

  return bands.filter((band) => band.endTime != null && band.endTime >= band.startTime);
}

function buildCycleBands(rows) {
  return rows
    .filter((row) => row?.cycle_id && row.unix != null)
    .map((row) => ({
      id: row.cycle_id,
      type: row.cycle_type ?? "",
      startTime: row.unix,
      endTime: unixFromDateLabel(row.end_date) ?? row.unix,
      firstRow: row,
      lastRow: row,
    }))
    .filter((band) => band.endTime >= band.startTime);
}

function bandPpoMetrics(band, parentPrefix = null) {
  if (!band) return null;
  const startKey = parentPrefix ? `parent_${parentPrefix}_start_ppo` : "start_ppo";
  const endKey = parentPrefix ? `parent_${parentPrefix}_end_ppo` : "end_ppo";
  const start = band.firstRow?.[startKey];
  const end = band.firstRow?.[endKey];
  const startValue = start == null || Number.isNaN(start) ? null : start;
  const endValue = end == null || Number.isNaN(end) ? null : end;
  const deltaValue = startValue == null || endValue == null ? null : endValue - startValue;
  return {
    start: startValue,
    end: endValue,
    delta: deltaValue,
  };
}

function scalarPpoTone(value, maxAbsLevel, alpha = 0.12) {
  if (value == null || Number.isNaN(value) || !maxAbsLevel) {
    return {
      fill: `rgba(148, 163, 184, ${alpha})`,
      accent: "rgba(148, 163, 184, 0.52)",
    };
  }

  const raw = Math.max(0, Math.min(1, Math.abs(value) / maxAbsLevel));
  const curved = Math.pow(raw, 0.58);
  const t = raw === 0 ? 0 : Math.max(0.22, curved);
  const start = [30, 41, 59];
  const end = value >= 0 ? [45, 212, 191] : [248, 113, 113];
  const fillRgb = start.map((channel, index) => Math.round(channel + (end[index] - channel) * t));
  const accentRgb = start.map((channel, index) => Math.round(channel + (end[index] - channel) * Math.max(0.48, t)));

  return {
    fill: `rgba(${fillRgb[0]}, ${fillRgb[1]}, ${fillRgb[2]}, ${alpha})`,
    accent: `rgba(${accentRgb[0]}, ${accentRgb[1]}, ${accentRgb[2]}, 0.82)`,
  };
}

function formatPpoTransition(metrics) {
  if (!metrics) return "-";
  return `${formatNumber(metrics.start)}→${formatNumber(metrics.end)} (${metrics.delta == null ? "-" : metrics.delta >= 0 ? "+" : ""}${formatNumber(metrics.delta)})`;
}

function ppoLevelToY(value, maxAbsLevel, minY = 3, maxY = 15) {
  if (value == null || Number.isNaN(value) || !maxAbsLevel) return (minY + maxY) / 2;
  const normalized = Math.max(-1, Math.min(1, value / maxAbsLevel));
  const t = (normalized + 1) / 2;
  return maxY - t * (maxY - minY);
}

function attachPpoTrendDecor(node, band, parentPrefix, maxAbsLevel, labelPrefix = "") {
  const metrics = bandPpoMetrics(band, parentPrefix);
  if (!metrics || metrics.start == null || metrics.end == null) return;

  const width = Number(node.dataset.bandWidth ?? 0);
  if (width >= 56) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 100 18");
    svg.setAttribute("preserveAspectRatio", "none");
    svg.classList.add("parent-band-sparkline");

    const line = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const startY = ppoLevelToY(metrics.start, maxAbsLevel);
    const endY = ppoLevelToY(metrics.end, maxAbsLevel);
    line.setAttribute("d", `M 6 ${startY} L 94 ${endY}`);
    line.setAttribute("class", "parent-band-sparkline__line");
    svg.appendChild(line);

    const startDot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    startDot.setAttribute("cx", "6");
    startDot.setAttribute("cy", String(startY));
    startDot.setAttribute("r", "1.8");
    startDot.setAttribute("class", "parent-band-sparkline__dot");
    svg.appendChild(startDot);

    const endDot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    endDot.setAttribute("cx", "94");
    endDot.setAttribute("cy", String(endY));
    endDot.setAttribute("r", "2.2");
    endDot.setAttribute("class", "parent-band-sparkline__dot parent-band-sparkline__dot--end");
    svg.appendChild(endDot);

    node.appendChild(svg);
  }

  if (width >= 94) {
    const badge = document.createElement("div");
    badge.className = "parent-band-badge";
    const deltaText = metrics.delta == null ? "-" : `${metrics.delta >= 0 ? "+" : ""}${formatNumber(metrics.delta)}`;
    badge.textContent = `${labelPrefix}${String(band.type).toUpperCase()} ${deltaText}`;
    node.appendChild(badge);
  }
}

function applyPpoBandTone(node, band, maxAbsLevel, parentPrefix, alpha = 0.13) {
  const metrics = bandPpoMetrics(band, parentPrefix);
  const startTone = scalarPpoTone(metrics?.start, maxAbsLevel, alpha);
  const endTone = scalarPpoTone(metrics?.end, maxAbsLevel, alpha);
  const directionAccent = String(band.type).toLowerCase() === "down" ? "rgba(248, 113, 113, 0.82)" : "rgba(45, 212, 191, 0.82)";
  node.style.background = `linear-gradient(90deg, ${startTone.fill} 0%, ${endTone.fill} 100%)`;
  node.style.boxShadow = `inset 0 0 0 1px ${directionAccent}, inset 0 1px 0 ${startTone.accent}, inset 0 -1px 0 ${endTone.accent}`;
  node.dataset.startPpo = metrics?.start == null ? "" : String(metrics.start);
  node.dataset.endPpo = metrics?.end == null ? "" : String(metrics.end);
  node.dataset.deltaPpo = metrics?.delta == null ? "" : String(metrics.delta);
  node.title = `${node.title} · PPO ${formatPpoTransition(metrics)}`;
}

function maxAbsPpoLevel(bands, parentPrefix = null) {
  return Math.max(
    0,
    ...bands
      .flatMap((band) => {
        const metrics = bandPpoMetrics(band, parentPrefix);
        return [metrics?.start, metrics?.end];
      })
      .filter((value) => value != null && !Number.isNaN(value))
      .map((value) => Math.abs(value)),
  );
}

function referenceLineOptions(panelKey, value) {
  if (panelKey === "rsi") {
    if (value === 50) {
      return {
        color: "rgba(255, 255, 255, 0.48)",
        lineWidth: 2,
        lineStyle: LineStyle.LargeDashed,
      };
    }
    return {
      color: "rgba(168, 85, 247, 0.44)",
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
    };
  }
  if (panelKey === "volume_delta" || panelKey === "funding_rate") {
    return {
      color: "rgba(255, 255, 255, 0.36)",
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
    };
  }
  return {
    color: "rgba(148, 163, 184, 0.38)",
    lineWidth: 1,
    lineStyle: LineStyle.Dashed,
  };
}

function rebuildPaneLegends(chart, paneLegendsRef) {
  paneLegendsRef.current.forEach((legend) => legend.remove());
  paneLegendsRef.current.clear();

  chart.panes().forEach((pane, index) => {
    const element = pane.getHTMLElement();
    if (!element) return;
    element.style.position = "relative";

    const legend = document.createElement("div");
    legend.className = "pane-legend";
    element.appendChild(legend);
    paneLegendsRef.current.set(index, legend);
  });
}

function ToolbarMenu({ label, children, wide = false }) {
  return (
    <div className={`toolbar-menu ${wide ? "toolbar-menu--wide" : ""}`}>
      <button type="button" className="toolbar-button">
        {label}
      </button>
      <div className="toolbar-menu__dropdown">{children}</div>
    </div>
  );
}

function TimeframeButton({ label, active, onClick }) {
  return (
    <button type="button" className={`timeframe-button ${active ? "timeframe-button--active" : ""}`} onClick={onClick}>
      {label}
    </button>
  );
}

function TradingChart({
  rows,
  symbol,
  timeframe,
  viewMode,
  parentOverlayMode,
  cycleContextRows,
  timeframeContextRows,
  priceOverlays,
  indicatorPanels,
  realtimeCandle,
  realtimeFootprint,
  historicalFootprint,
  showFootprint,
  onNearHistoryStart,
  onHoverRowChange,
}) {
  const hostRef = useRef(null);
  const tooltipRef = useRef(null);
  const parentOverlayRef = useRef(null);
  const footprintOverlayRef = useRef(null);
  const footprintDetailRef = useRef(null);
  const priceScaleOverlayRef = useRef(null);
  const contextRibbonRef = useRef(null);
  const chartRef = useRef(null);
  const candleSeriesRef = useRef(null);
  const overlaySeriesRef = useRef([]);
  const indicatorSeriesRef = useRef([]);
  const paneLegendsRef = useRef(new Map());
  const paneWatermarksRef = useRef([]);
  const rowByTimeRef = useRef(new Map());
  const latestRowsRef = useRef([]);
  const cycleContextRowsRef = useRef([]);
  const timeframeContextRowsRef = useRef([]);
  const visibleRangeRef = useRef(null);
  const visibleTimeRangeRef = useRef(null);
  const hasInitializedRangeRef = useRef(false);
  const parentOverlayFrameRef = useRef(null);
  const footprintSnapshotRef = useRef(null);
  const handleHoverRowChange = useEffectEvent((row) => {
    onHoverRowChange(row);
  });
  const handleNearHistoryStart = useEffectEvent(() => {
    onNearHistoryStart();
  });
  const syncLegends = useEffectEvent((row) => {
    const title =
      viewMode === VIEW_MODES.CYCLE && row
        ? `${row.cycle_id ?? "Cycle"} ${row.cycle_type ? row.cycle_type.toUpperCase() : ""}`
        : viewMode === VIEW_MODES.TIME && row?._cycleContext
          ? `${row._cycleContext.cycle_id ?? "Cycle"} ${row._cycleContext.cycle_type ? row._cycleContext.cycle_type.toUpperCase() : ""}`
        : "";
    const priceItems = [
      { label: "O", value: row?.open, color: "#cbd5e1" },
      { label: "H", value: row?.high, color: "#0ecb81" },
      { label: "L", value: row?.low, color: "#f6465d" },
      { label: "C", value: row?.close, color: row?.close >= row?.open ? "#0ecb81" : "#f6465d" },
      ...overlaySeriesRef.current.map((overlay) => ({
        label: overlay.label,
        value: row?.[overlay.key],
        color: overlay.color,
      })),
    ];
    if (viewMode === VIEW_MODES.CYCLE) {
      priceItems.push(
        { label: "Pct", value: row?.change_price_pct, color: (row?.change_price_pct ?? 0) >= 0 ? "#0ecb81" : "#f6465d" },
        { label: "Dur", value: row?.duration_candles, color: "#93c5fd" },
        { label: "PPO", value: row?.end_ppo, color: "#2dd4bf" },
      );
    }
    makeLegend(paneLegendsRef.current.get(0), priceItems, "price", title);

    indicatorSeriesRef.current.forEach((panel, index) => {
      const paneIndex = index + 1;
      const items = panel.series.map((seriesItem) => ({
        label: seriesItem.label,
        value: row?.[seriesItem.key],
        color: seriesItem.color ?? (seriesItem.type === "histogram" ? "#5bd6bc" : "#cbd5e1"),
      }));
      makeLegend(paneLegendsRef.current.get(paneIndex), [], "indicator");

      const watermarkEntry = paneWatermarksRef.current[index];
      if (watermarkEntry) {
        watermarkEntry.watermark.applyOptions({
          lines: [
            {
              text: watermarkEntry.title,
              color: "rgba(255, 255, 255, 0.96)",
              fontSize: 12,
              fontStyle: "bold",
              fontFamily: "Pretendard, Segoe UI, sans-serif",
            },
            ...items
              .filter((item) => item.value != null && !Number.isNaN(item.value))
              .map((item) => ({
                text: `${item.label} ${formatNumber(item.value)}`,
                color: item.color ?? "#cbd5e1",
                fontSize: 11,
                fontFamily: "Pretendard, Segoe UI, sans-serif",
              })),
          ],
        });
      }
    });

    if (row) handleHoverRowChange(row);
  });
  const renderContextRibbons = useEffectEvent(() => {
    const chart = chartRef.current;
    const layer = contextRibbonRef.current;
    if (!chart || !layer) return;

    layer.replaceChildren();
    const ribbonTimeframes = CONTEXT_RIBBON_TIMEFRAMES[timeframe] ?? [];
    if (!ribbonTimeframes.length) return;

    const contextRows = timeframeContextRowsRef.current;
    if (!contextRows.length) return;

    const timeScale = chart.timeScale();
    const viewportWidth = layer.clientWidth || hostRef.current?.clientWidth || 0;
    const rowNodes = [];

    ribbonTimeframes.forEach((ribbonTf) => {
      const rowNode = document.createElement("div");
      rowNode.className = "context-ribbon-row";

      const label = document.createElement("div");
      label.className = "context-ribbon-row__label";
      label.textContent = ribbonTf.toUpperCase();
      rowNode.appendChild(label);

      const track = document.createElement("div");
      track.className = "context-ribbon-row__track";
      rowNode.appendChild(track);

      let activeBand = null;
      const flushBand = () => {
        if (!activeBand) return;
        const startX = timeScale.timeToCoordinate(activeBand.startTime);
        const endX = timeScale.timeToCoordinate(activeBand.endTime);
        if (startX != null && endX != null) {
          const left = Math.min(startX, endX);
          const width = Math.max(4, Math.abs(endX - startX));
          if (!viewportWidth || !(left > viewportWidth + 40 || left + width < -40)) {
            const direction = activeBand.type === "down" ? "down" : activeBand.type === "up" ? "up" : "gap";
            const bandNode = document.createElement("div");
            bandNode.className = `context-ribbon-band context-ribbon-band--${direction}`;
            bandNode.style.left = `${left}px`;
            bandNode.style.width = `${width}px`;
            bandNode.title = `${ribbonTf.toUpperCase()} ${activeBand.cycleId ?? activeBand.key} ${contextDirectionLabel(activeBand.type)} ${formatContextProgress(activeBand.maxProgress)}`;

            const boundary = document.createElement("div");
            boundary.className = "context-ribbon-band__boundary";
            bandNode.appendChild(boundary);

            if (activeBand.maxProgress >= 0.8 && width >= 16) {
              const late = document.createElement("div");
              late.className = "context-ribbon-band__late";
              late.textContent = "!";
              bandNode.appendChild(late);
            }

            if (width >= 72) {
              const text = document.createElement("div");
              text.className = "context-ribbon-band__text";
              text.textContent = `${contextDirectionLabel(activeBand.type)} ${formatContextProgress(activeBand.maxProgress)}`;
              bandNode.appendChild(text);
            }

            track.appendChild(bandNode);
          }
        }
        activeBand = null;
      };

      contextRows.forEach((row, index) => {
        if (row?.unix == null) return;
        const key = row[`${ribbonTf}_key`] ?? 0;
        const type = row[`${ribbonTf}_type`] ?? "gap";
        const progress = Number(row[`${ribbonTf}_time_prog`] ?? 0);
        const nextRow = contextRows[index + 1];
        const fallbackEnd = nextRow?.unix ?? row.unix + (TIMEFRAME_SECONDS[timeframe] ?? 3600);
        const endTime = unixFromDateLabel(row[`${ribbonTf}_cycle_end_date`]) ?? fallbackEnd;

        if (!activeBand || activeBand.key !== key) {
          flushBand();
          activeBand = {
            key,
            type,
            cycleId: row[`${ribbonTf}_cycle_id`],
            startTime: row.unix,
            endTime,
            maxProgress: progress,
          };
          return;
        }

        activeBand.endTime = endTime;
        activeBand.maxProgress = Math.max(activeBand.maxProgress, progress);
      });
      flushBand();
      rowNodes.push(rowNode);
    });

    layer.replaceChildren(...rowNodes);
  });
  const renderParentOverlay = useEffectEvent(() => {
    const chart = chartRef.current;
    const layer = parentOverlayRef.current;
    if (!chart || !layer) return;

    layer.replaceChildren();
    const timeScale = chart.timeScale();
    const viewportWidth = layer.clientWidth || hostRef.current?.clientWidth || 0;
    const makeOverlayNode = (band, className, minWidth = 6) => {
      const startX = timeScale.timeToCoordinate(band.startTime);
      const endX = timeScale.timeToCoordinate(band.endTime) ?? timeScale.timeToCoordinate(band.lastRow?.unix);
      if (startX == null || endX == null) return null;

      const left = Math.min(startX, endX);
      const width = Math.max(minWidth, Math.abs(endX - startX));
      if (viewportWidth && (left > viewportWidth + 40 || left + width < -40)) return null;
      const direction = String(band.type).toLowerCase() === "down" ? "down" : "up";
      const node = document.createElement("div");
      node.className = `${className} ${className}--${direction}`;
      node.style.left = `${left}px`;
      node.style.width = `${width}px`;
      node.dataset.bandWidth = String(width);
      node.title = `${band.id} ${String(band.type).toUpperCase()}`;
      return node;
    };

    if (viewMode === VIEW_MODES.TIME) {
      const nodes = [];
      const currentCycleBands = buildContextCycleBands(timeframeContextRowsRef.current, timeframe);
      const fallbackCycleBands = currentCycleBands.length ? currentCycleBands : buildCycleBands(cycleContextRowsRef.current);
      fallbackCycleBands.forEach((band) => {
        const node = makeOverlayNode(band, "time-cycle-context", 4);
        if (node) nodes.push(node);
      });

      layer.replaceChildren(...nodes);
      return;
    }

    if (
      viewMode !== VIEW_MODES.CYCLE ||
      !PARENT_OVERLAY_TIMEFRAMES.includes(timeframe) ||
      parentOverlayMode === PARENT_OVERLAY_MODES.OFF ||
      !latestRowsRef.current.length
    ) {
      return;
    }
    const nodes = [];
    const spec = immediateParentSpec(timeframe);
    if (!spec) return;

    const parentBands = buildParentBands(latestRowsRef.current, `parent_${spec.prefix}_cycle_id`, `parent_${spec.prefix}_cycle_type`);
    const parentPpoMax = maxAbsPpoLevel(parentBands, spec.prefix);

    parentBands.forEach((band) => {
      const node = makeOverlayNode(band, spec.className);
      if (!node) return;
      applyPpoBandTone(node, band, parentPpoMax, spec.prefix, spec.alpha);
      attachPpoTrendDecor(node, band, spec.prefix, parentPpoMax, `${spec.label} `);
      const metrics = bandPpoMetrics(band, spec.prefix);

      if (spec.className === "parent-cycle-four-hour") {
        const tick = document.createElement("div");
        tick.className = "parent-cycle-four-hour__tick";
        tick.textContent = `${spec.label} ${String(band.type).toUpperCase()} ${metrics?.delta == null ? "-" : metrics.delta >= 0 ? "+" : ""}${formatNumber(metrics?.delta)}`;
        node.appendChild(tick);
      } else if (spec.className === "parent-cycle-week-band") {
        const label = document.createElement("div");
        label.className = "parent-cycle-week-band__label";
        label.textContent = `${spec.label} ${String(band.type).toUpperCase()} ${metrics?.delta == null ? "-" : metrics.delta >= 0 ? "+" : ""}${formatNumber(metrics?.delta)}`;
        node.appendChild(label);
      }

      nodes.push(node);
    });

    layer.replaceChildren(...nodes);
  });
  const renderFootprintSurface = useEffectEvent(() => {
    const chart = chartRef.current;
    const layer = footprintOverlayRef.current;
    const candleSeries = candleSeriesRef.current;
    const snapshot = footprintSnapshotRef.current;
    if (!chart || !layer || !candleSeries || !showFootprint || viewMode !== VIEW_MODES.TIME) return;

    layer.replaceChildren();
    const bars = Array.isArray(snapshot?.bars) ? snapshot.bars : [];
    if (!bars.length) return;

    const timeScale = chart.timeScale();
    const chartSeconds = TIMEFRAME_SECONDS[timeframe] ?? 60;
    const coordinateForTime = (unix) => {
      const exact = timeScale.timeToCoordinate(unix);
      if (exact != null) return exact;
      return timeScale.timeToCoordinate(Math.floor(unix / chartSeconds) * chartSeconds);
    };
    const maxVolume = Math.max(1, ...bars.flatMap((bar) => (bar.levels ?? []).map((level) => Number(level.totalVolume) || 0)));
    const nodes = [];

    bars.forEach((bar, index) => {
      const startUnix = Math.floor(Number(bar.barStartMs) / 1000);
      const endUnix = Math.floor(Number(bar.barEndMs) / 1000);
      const startX = coordinateForTime(startUnix);
      const endX = coordinateForTime(endUnix);
      if (startX == null) return;

      const followingBar = bars[index + 1];
      const followingX = followingBar ? coordinateForTime(Math.floor(Number(followingBar.barStartMs) / 1000)) : null;
      const width = Math.max(5, Math.min(84, Math.abs(endX ?? followingX ?? startX + 28 - startX) - 2 || 28));
      const showValues = width >= 24;

      (bar.levels ?? []).forEach((level) => {
        if (!(Number(level.totalVolume) > 0)) return;
        const price = Number(level.price);
        const y = candleSeries.priceToCoordinate(price);
        const nextY = candleSeries.priceToCoordinate(price + Number(snapshot.priceBinSize ?? 0));
        if (y == null) return;
        const height = Math.max(5, Math.min(30, Math.abs((nextY ?? y - 12) - y) - 1 || 12));
        const node = document.createElement("div");
        const direction = Number(level.delta) > 0 ? "flow-footprint-cell--buy" : Number(level.delta) < 0 ? "flow-footprint-cell--sell" : "";
        node.className = `flow-footprint-cell ${direction} ${level.isPoc ? "flow-footprint-cell--poc" : ""} ${level.buyImbalance ? "flow-footprint-cell--buy-imbalance" : ""} ${level.sellImbalance ? "flow-footprint-cell--sell-imbalance" : ""}`;
        node.style.left = `${startX - width / 2}px`;
        node.style.top = `${y - height / 2}px`;
        node.style.width = `${width}px`;
        node.style.height = `${height}px`;
        node.style.setProperty("--flow-intensity", String(Math.max(0.12, Number(level.totalVolume) / maxVolume)));
        node.title = `${formatFootprintBarTime(Number(bar.barStartMs))} · ${formatNumber(price)} | Sell ${formatNumber(level.sellVolume)} / Buy ${formatNumber(level.buyVolume)} / Δ ${formatNumber(level.delta)}`;
        if (showValues) {
          const sell = document.createElement("span");
          sell.textContent = formatNumber(level.sellVolume);
          const buy = document.createElement("b");
          buy.textContent = formatNumber(level.buyVolume);
          node.append(sell, buy);
        }
        nodes.push(node);
      });
    });
    layer.replaceChildren(...nodes);
  });
  const renderFootprintDetail = useEffectEvent((time, point) => {
    const layer = footprintDetailRef.current;
    const snapshot = footprintSnapshotRef.current;
    const host = hostRef.current;
    if (!layer || !host || !showFootprint || viewMode !== VIEW_MODES.TIME || !time || !point) {
      layer?.replaceChildren();
      if (layer) layer.style.opacity = "0";
      return;
    }

    const targetMs = Number(time) * 1000;
    const bars = Array.isArray(snapshot?.bars) ? snapshot.bars : [];
    const bar = bars.find((item) => Number(item.barStartMs) <= targetMs && targetMs < Number(item.barEndMs))
      ?? bars.reduce((closest, item) => {
        if (!closest) return item;
        return Math.abs(Number(item.barStartMs) - targetMs) < Math.abs(Number(closest.barStartMs) - targetMs) ? item : closest;
      }, null);
    if (!bar || Math.abs(Number(bar.barStartMs) - targetMs) > 60_000) {
      layer.replaceChildren();
      layer.style.opacity = "0";
      return;
    }
    const levels = (bar?.levels ?? []).filter((level) => Number(level.totalVolume) > 0).sort((left, right) => Number(right.price) - Number(left.price));
    if (!levels.length) {
      layer.replaceChildren();
      layer.style.opacity = "0";
      return;
    }

    const maxSideVolume = Math.max(1, ...levels.flatMap((level) => [Number(level.sellVolume) || 0, Number(level.buyVolume) || 0]));
    const profile = document.createElement("div");
    profile.className = "footprint-hover-profile__card";
    const header = document.createElement("div");
    header.className = "footprint-hover-profile__header";
    header.textContent = `${formatFootprintBarTime(Number(bar.barStartMs))} UTC · Sell / Price / Buy`;
    profile.appendChild(header);

    const rows = document.createElement("div");
    rows.className = "footprint-hover-profile__rows";
    levels.forEach((level) => {
      const row = document.createElement("div");
      row.className = `footprint-hover-profile__row ${level.isPoc ? "footprint-hover-profile__row--poc" : ""} ${level.buyImbalance ? "footprint-hover-profile__row--buy-imbalance" : ""} ${level.sellImbalance ? "footprint-hover-profile__row--sell-imbalance" : ""}`;
      const sell = document.createElement("i");
      sell.className = "footprint-hover-profile__sell";
      sell.style.setProperty("--profile-width", `${(Number(level.sellVolume || 0) / maxSideVolume) * 100}%`);
      sell.textContent = formatNumber(Number(level.sellVolume));
      sell.title = `Sell ${formatNumber(level.sellVolume)}`;
      const price = document.createElement("b");
      price.textContent = formatNumber(Number(level.price));
      const buy = document.createElement("i");
      buy.className = "footprint-hover-profile__buy";
      buy.style.setProperty("--profile-width", `${(Number(level.buyVolume || 0) / maxSideVolume) * 100}%`);
      buy.textContent = formatNumber(Number(level.buyVolume));
      buy.title = `Buy ${formatNumber(level.buyVolume)}`;
      row.append(sell, price, buy);
      rows.appendChild(row);
    });
    profile.appendChild(rows);
    const footer = document.createElement("div");
    footer.className = "footprint-hover-profile__footer";
    footer.textContent = `POC ${formatNumber(levels.find((level) => level.isPoc)?.price)} · Δ ${formatNumber(bar.delta)}`;
    profile.appendChild(footer);
    layer.replaceChildren(profile);

    const panelWidth = 310;
    const panelHeight = Math.min(510, 58 + levels.length * 16);
    const left = point.x + panelWidth + 20 <= host.clientWidth ? point.x + 16 : Math.max(8, point.x - panelWidth - 16);
    const top = Math.max(8, Math.min(point.y - panelHeight / 2, host.clientHeight - panelHeight - 8));
    layer.style.left = `${left}px`;
    layer.style.top = `${top}px`;
    layer.style.opacity = "1";
  });
  const renderChartPriceScale = useEffectEvent(() => {
    const layer = priceScaleOverlayRef.current;
    const candleSeries = candleSeriesRef.current;
    const range = visibleRangeRef.current;
    const sourceRows = latestRowsRef.current;
    if (!layer || !candleSeries || !sourceRows.length) return;

    layer.replaceChildren();
    const from = Math.max(0, Math.floor(range?.from ?? Math.max(0, sourceRows.length - INITIAL_VISIBLE_BARS)));
    const to = Math.min(sourceRows.length, Math.ceil(range?.to ?? sourceRows.length));
    const visibleRows = sourceRows.slice(from, to).filter((row) => Number.isFinite(row?.high) && Number.isFinite(row?.low));
    if (!visibleRows.length) return;
    const low = Math.min(...visibleRows.map((row) => row.low));
    const high = Math.max(...visibleRows.map((row) => row.high));
    const span = Math.max(high - low, Math.abs(high) * 0.0001, 1);
    const roughStep = span / 7;
    const power = 10 ** Math.floor(Math.log10(roughStep));
    const normalized = roughStep / power;
    const step = (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10) * power;
    const labels = [];
    for (let price = Math.ceil(low / step) * step; price <= high + step * 0.001; price += step) {
      const y = candleSeries.priceToCoordinate(price);
      if (y == null) continue;
      const label = document.createElement("span");
      label.className = "chart-price-axis__tick";
      label.textContent = formatNumber(price);
      label.style.top = `${y}px`;
      labels.push(label);
    }
    const latest = sourceRows.at(-1);
    const latestY = candleSeries.priceToCoordinate(latest?.close);
    if (latestY != null) {
      const lastLabel = document.createElement("span");
      lastLabel.className = `chart-price-axis__last ${latest.close >= latest.open ? "chart-price-axis__last--up" : "chart-price-axis__last--down"}`;
      lastLabel.textContent = formatNumber(latest.close);
      lastLabel.style.top = `${latestY}px`;
      labels.push(lastLabel);
    }
    layer.replaceChildren(...labels);
  });
  const findCycleContextForTime = useEffectEvent((time) => {
    if (time == null) return null;
    return cycleContextRowsRef.current.find((cycle) => {
      const endTime = unixFromDateLabel(cycle.end_date) ?? cycle.unix;
      return cycle.unix <= time && time <= endTime;
    }) ?? null;
  });
  const findTimeframeContextForTime = useEffectEvent((time) => contextRowForTime(timeframeContextRowsRef.current, time));
  const scheduleParentOverlayRender = useEffectEvent(() => {
    if (typeof window === "undefined") {
      renderParentOverlay();
      renderContextRibbons();
      renderFootprintSurface();
      renderChartPriceScale();
      return;
    }
    if (parentOverlayFrameRef.current != null) return;
    parentOverlayFrameRef.current = window.requestAnimationFrame(() => {
      parentOverlayFrameRef.current = null;
      renderParentOverlay();
      renderContextRibbons();
      renderFootprintSurface();
      renderChartPriceScale();
    });
  });
  const applySeriesData = useEffectEvent((rowsToApply) => {
    if (!chartRef.current || !candleSeriesRef.current || !rowsToApply.length) return;

    rowByTimeRef.current = new Map(rowsToApply.filter((row) => row.unix != null).map((row) => [row.unix, row]));

    const candleData = rowsToApply
      .filter((row) => row.unix != null && row.open != null && row.high != null && row.low != null && row.close != null)
      .map((row) => {
        const priceUp = row.change_price_pct != null ? row.change_price_pct >= 0 : row.close >= row.open;
        const tone = priceUp ? "#0ecb81" : "#f6465d";
        return {
          time: row.unix,
          open: row.open,
          high: row.high,
          low: row.low,
          close: row.close,
          color: tone,
          wickColor: tone,
          borderColor: tone,
        };
      });
    candleSeriesRef.current.setData(candleData);

    overlaySeriesRef.current.forEach((overlay) => {
      overlay.series.setData(
        rowsToApply
          .filter((row) => row.unix != null && row[overlay.key] != null)
          .map((row) => ({
            time: row.unix,
            value: row[overlay.key],
          })),
      );
    });

    indicatorSeriesRef.current.forEach((panel) => {
      panel.series.forEach((seriesItem) => {
        const data = rowsToApply
          .filter((row) => row.unix != null && row[seriesItem.key] != null)
          .map((row, dataIndex, filteredRows) => {
            const previousRow = dataIndex > 0 ? filteredRows[dataIndex - 1] : null;
            return {
              time: row.unix,
              value: row[seriesItem.key],
              color:
                seriesItem.type === "histogram"
                  ? panel.key === "volume"
                    ? row.close >= row.open
                      ? "rgba(14, 203, 129, 0.65)"
                      : "rgba(246, 70, 93, 0.65)"
                    : histogramTone(panel.key, row[seriesItem.key], previousRow?.[seriesItem.key] ?? null)
                  : undefined,
            };
          });
        seriesItem.series.setData(data);
      });
    });

    const baseLatestRow = rowsToApply[rowsToApply.length - 1] ?? null;
    const latestCycleContext = viewMode === VIEW_MODES.TIME ? findCycleContextForTime(baseLatestRow?.unix) : null;
    const latestTimeframeContext = viewMode === VIEW_MODES.TIME || viewMode === VIEW_MODES.CYCLE ? findTimeframeContextForTime(baseLatestRow?.unix) : null;
    const latestRow =
      latestCycleContext || latestTimeframeContext
        ? { ...baseLatestRow, _cycleContext: latestCycleContext, _timeframeContext: latestTimeframeContext }
        : baseLatestRow;
    syncLegends(latestRow);

    if (candleData.length && !hasInitializedRangeRef.current) {
      const from = Math.max(0, rowsToApply.length - INITIAL_VISIBLE_BARS);
      const to = rowsToApply.length + 12;
      chartRef.current.timeScale().setVisibleLogicalRange({ from, to });
      visibleRangeRef.current = { from, to };
      visibleTimeRangeRef.current = chartRef.current.timeScale().getVisibleRange?.() ?? null;
      hasInitializedRangeRef.current = true;
    }

    scheduleParentOverlayRender();
  });
  const handleCrosshairMove = useEffectEvent((param) => {
    const latestRows = latestRowsRef.current;
    if (!param.time) {
      const baseLatestRow = latestRows[latestRows.length - 1] ?? null;
      const latestContext = viewMode === VIEW_MODES.TIME ? findCycleContextForTime(baseLatestRow?.unix) : null;
      const latestTimeframeContext = viewMode === VIEW_MODES.TIME || viewMode === VIEW_MODES.CYCLE ? findTimeframeContextForTime(baseLatestRow?.unix) : null;
      const latestRow =
        latestContext || latestTimeframeContext
          ? { ...baseLatestRow, _cycleContext: latestContext, _timeframeContext: latestTimeframeContext }
          : baseLatestRow;
      syncLegends(latestRow);
      if (tooltipRef.current) tooltipRef.current.style.opacity = "0";
      renderFootprintDetail(null, null);
      return;
    }

    const baseRow = rowByTimeRef.current.get(param.time) ?? latestRows[latestRows.length - 1] ?? null;
    const cycleContext = viewMode === VIEW_MODES.TIME ? findCycleContextForTime(param.time) : null;
    const timeframeContext = viewMode === VIEW_MODES.TIME || viewMode === VIEW_MODES.CYCLE ? findTimeframeContextForTime(param.time) : null;
    const nextRow =
      cycleContext || timeframeContext
        ? { ...baseRow, _cycleContext: cycleContext, _timeframeContext: timeframeContext }
        : baseRow;
    syncLegends(nextRow);

    if (tooltipRef.current && param.point && hostRef.current) {
      const label =
        viewMode === VIEW_MODES.CYCLE
          ? `${formatCycleRange(nextRow)} | ${nextRow?.cycle_id ?? ""}`
          : `${formatDateLabel(nextRow?.date ?? formatChartTime(param.time))}${nextRow?._cycleContext ? ` | ${nextRow._cycleContext.cycle_id}` : ""}`;
      tooltipRef.current.textContent = label;
      tooltipRef.current.style.opacity = "1";
      tooltipRef.current.style.transform = `translate(${Math.min(param.point.x + 14, hostRef.current.clientWidth - 260)}px, ${Math.max(param.point.y - 34, 8)}px)`;
    }
    renderFootprintDetail(param.time, param.point);
  });
  const handleVisibleRangeChange = useEffectEvent((range) => {
    if (!range) return;
    visibleRangeRef.current = range;
    visibleTimeRangeRef.current = chartRef.current?.timeScale().getVisibleRange?.() ?? visibleTimeRangeRef.current;
    if (viewMode === VIEW_MODES.TIME && range.from < 80) handleNearHistoryStart();
    scheduleParentOverlayRender();
  });
  const applyRealtimeCandle = useEffectEvent((row) => {
    if (!chartRef.current || !candleSeriesRef.current || !row?.unix) return;

    const currentRows = latestRowsRef.current;
    const lastRow = currentRows[currentRows.length - 1] ?? null;
    const mergedRow = { ...(lastRow?.unix === row.unix ? lastRow : {}), ...row };
    const priceUp = mergedRow.change_price_pct != null ? mergedRow.change_price_pct >= 0 : mergedRow.close >= mergedRow.open;
    const tone = priceUp ? "#0ecb81" : "#f6465d";

    candleSeriesRef.current.update({
      time: mergedRow.unix,
      open: mergedRow.open,
      high: mergedRow.high,
      low: mergedRow.low,
      close: mergedRow.close,
      color: tone,
      wickColor: tone,
      borderColor: tone,
    });

    const nextRows =
      lastRow?.unix === mergedRow.unix
        ? [...currentRows.slice(0, -1), mergedRow]
        : [...currentRows, mergedRow];
    latestRowsRef.current = nextRows;
    rowByTimeRef.current.set(mergedRow.unix, mergedRow);
    syncLegends(mergedRow);
  });

  useEffect(() => {
    latestRowsRef.current = rows;
  }, [rows]);

  useEffect(() => {
    cycleContextRowsRef.current = cycleContextRows ?? [];
    scheduleParentOverlayRender();
  }, [cycleContextRows]);

  useEffect(() => {
    timeframeContextRowsRef.current = timeframeContextRows ?? [];
    scheduleParentOverlayRender();
  }, [timeframeContextRows]);

  useEffect(() => {
    // Historical aggTrades are preferred whenever the user selected a window.
    // The realtime stream remains a fallback while a historical request loads.
    footprintSnapshotRef.current = historicalFootprint ?? realtimeFootprint;
    scheduleParentOverlayRender();
  }, [historicalFootprint, realtimeFootprint, showFootprint]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.applyOptions({ timeScale: { barSpacing: 8 } });
    scheduleParentOverlayRender();
  }, [showFootprint]);

  useEffect(() => {
    hasInitializedRangeRef.current = false;
    visibleRangeRef.current = null;
    visibleTimeRangeRef.current = null;
  }, [symbol, timeframe, viewMode]);

  useEffect(() => {
    if (!hostRef.current) return undefined;

    const chart = createChart(hostRef.current, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: "#131722" },
        textColor: "#8b93a6",
        attributionLogo: false,
        panes: {
          enableResize: true,
          separatorColor: "rgba(148, 163, 184, 0.18)",
          separatorHoverColor: "rgba(148, 163, 184, 0.28)",
        },
      },
      localization: {
        timeFormatter: formatChartTime,
      },
      grid: {
        vertLines: { color: "rgba(66, 77, 102, 0.38)" },
        horzLines: { color: "rgba(66, 77, 102, 0.38)" },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "rgba(154, 164, 183, 0.72)", width: 1, style: LineStyle.LargeDashed, labelVisible: true },
        horzLine: { color: "rgba(154, 164, 183, 0.28)", width: 1, style: LineStyle.LargeDashed, labelVisible: true },
      },
      rightPriceScale: {
        visible: true,
        autoScale: true,
        alignLabels: true,
        entireTextOnly: true,
        ticksVisible: true,
        minimumWidth: 76,
        borderVisible: true,
        borderColor: "rgba(66, 77, 102, 0.58)",
      },
      timeScale: {
        borderColor: "rgba(66, 77, 102, 0.58)",
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 12,
        barSpacing: 8,
        minBarSpacing: 0.01,
        shiftVisibleRangeOnNewBar: false,
        rightBarStaysOnScroll: false,
        lockVisibleTimeRangeOnResize: true,
        tickMarkFormatter: formatChartTime,
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
      handleScale: {
        mouseWheel: true,
        pinch: true,
        axisPressedMouseMove: {
          time: true,
          price: true,
        },
      },
    });
    chartRef.current = chart;

    const candleSeries = chart.addSeries(
      CandlestickSeries,
      {
        upColor: "#0ecb81",
        downColor: "#f6465d",
        borderVisible: false,
        wickUpColor: "#0ecb81",
        wickDownColor: "#f6465d",
        priceLineVisible: true,
        lastValueVisible: true,
        priceFormat: {
          type: "price",
          precision: 2,
          minMove: 0.01,
        },
      },
      0,
    );
    candleSeriesRef.current = candleSeries;

    candleSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.06, bottom: 0.02 },
      autoScale: true,
    });

    rebuildPaneLegends(chart, paneLegendsRef);

    chart.subscribeCrosshairMove(handleCrosshairMove);
    chart.timeScale().subscribeVisibleLogicalRangeChange(handleVisibleRangeChange);
    const resizeObserver =
      typeof ResizeObserver !== "undefined"
        ? new ResizeObserver(() => {
            scheduleParentOverlayRender();
          })
        : null;
    if (resizeObserver && hostRef.current) resizeObserver.observe(hostRef.current);
    if (resizeObserver && contextRibbonRef.current) resizeObserver.observe(contextRibbonRef.current);
    const legendRefs = paneLegendsRef.current;

    return () => {
      if (parentOverlayFrameRef.current != null) {
        window.cancelAnimationFrame(parentOverlayFrameRef.current);
        parentOverlayFrameRef.current = null;
      }
      resizeObserver?.disconnect();
      chart.unsubscribeCrosshairMove(handleCrosshairMove);
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(handleVisibleRangeChange);
      paneWatermarksRef.current.forEach((entry) => entry.watermark.detach());
      paneWatermarksRef.current = [];
      legendRefs.clear();
      overlaySeriesRef.current = [];
      indicatorSeriesRef.current = [];
      candleSeriesRef.current = null;
      visibleRangeRef.current = null;
      visibleTimeRangeRef.current = null;
      hasInitializedRangeRef.current = false;
      chartRef.current = null;
      chart.remove();
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !candleSeriesRef.current) return;

    overlaySeriesRef.current.forEach((overlay) => {
      chart.removeSeries(overlay.series);
    });
    overlaySeriesRef.current = [];

    paneWatermarksRef.current.forEach((entry) => entry.watermark.detach());
    paneWatermarksRef.current = [];

    indicatorSeriesRef.current.forEach((panel) => {
      panel.series.forEach((seriesItem) => {
        chart.removeSeries(seriesItem.series);
      });
    });
    indicatorSeriesRef.current = [];

    for (let paneIndex = chart.panes().length - 1; paneIndex >= 1; paneIndex -= 1) {
      chart.removePane(paneIndex);
    }

    priceOverlays
      .filter((key) => PRICE_OVERLAYS.some((item) => item.key === key))
      .forEach((key) => {
        const overlay = PRICE_OVERLAYS.find((item) => item.key === key);
        if (!overlay) return;

        const lineSeries = chart.addSeries(
          LineSeries,
          {
            color: overlay.color,
            lineWidth: 1,
            crosshairMarkerVisible: false,
            lastValueVisible: true,
            priceLineVisible: false,
          },
          0,
        );
        overlaySeriesRef.current.push({ ...overlay, series: lineSeries });
      });

    indicatorPanels.forEach((panel, index) => {
      const paneIndex = index + 1;
      const firstSeries = [];
      const panelSeries = [];

      panel.series.forEach((seriesItem) => {
        const color = seriesItem.color ?? "#22c55e";
        const series =
          seriesItem.type === "histogram"
            ? chart.addSeries(
                HistogramSeries,
                {
                  color,
                  priceLineVisible: false,
                  lastValueVisible: true,
                  priceFormat: panel.key === "volume" ? { type: "volume" } : { type: "price", precision: 4, minMove: 0.0001 },
                },
                paneIndex,
              )
            : chart.addSeries(
                LineSeries,
                {
                  color,
                  lineWidth: seriesItem.lineWidth ?? 2,
                  lineStyle: seriesItem.lineStyle,
                  lineType: seriesItem.lineType,
                  crosshairMarkerVisible: false,
                  priceLineVisible: false,
                  lastValueVisible: true,
                  priceFormat: { type: "price", precision: 4, minMove: 0.0001 },
                },
                paneIndex,
              );
        firstSeries.push(series);
        panelSeries.push({ ...seriesItem, color, series });
      });
      indicatorSeriesRef.current.push({ ...panel, series: panelSeries });

      if (panel.referenceLines?.length && firstSeries[0]) {
        panel.referenceLines.forEach((value) => {
          const lineOptions = referenceLineOptions(panel.key, value);
          firstSeries[0].createPriceLine({
            price: value,
            color: lineOptions.color,
            lineWidth: lineOptions.lineWidth,
            lineStyle: lineOptions.lineStyle,
            axisLabelVisible: false,
            title: "",
          });
        });
      }

      const paneApi = chart.panes()[paneIndex];
      if (paneApi) {
        paneApi.setStretchFactor(panel.stretch ?? 1);
        paneWatermarksRef.current.push({
          title: panel.label,
          watermark: createTextWatermark(paneApi, {
            horzAlign: "left",
            vertAlign: "top",
            lines: [
              {
                text: panel.label,
                color: "rgba(255, 255, 255, 0.96)",
                fontSize: 12,
                fontStyle: "bold",
                fontFamily: "Pretendard, Segoe UI, sans-serif",
              },
            ],
          }),
        });
      }
    });

    if (chart.panes()[0]) {
      chart.panes()[0].setStretchFactor(Math.max(2.8, 2 + indicatorPanels.length * 0.42));
    }

    rebuildPaneLegends(chart, paneLegendsRef);
    applySeriesData(latestRowsRef.current);
  }, [indicatorPanels, priceOverlays]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    chart.applyOptions({
      timeScale: {
        rightOffset: viewMode === VIEW_MODES.CYCLE ? 2 : 12,
        minBarSpacing: 0.01,
        rightBarStaysOnScroll: false,
        shiftVisibleRangeOnNewBar: false,
      },
      handleScale: {
        mouseWheel: true,
        pinch: true,
        axisPressedMouseMove: {
          time: true,
          price: true,
        },
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
    });
  }, [viewMode]);

  useEffect(() => {
    applySeriesData(rows);
  }, [rows]);

  useEffect(() => {
    scheduleParentOverlayRender();
  }, [rows, parentOverlayMode, timeframe, viewMode, cycleContextRows, timeframeContextRows]);

  useEffect(() => {
    if (realtimeCandle?.row) {
      applyRealtimeCandle(realtimeCandle.row);
    }
  }, [realtimeCandle]);

  return (
    <div className="tv-chart-frame">
      <div ref={contextRibbonRef} className="context-ribbon-panel" />
      <div className="chart-canvas-area">
        <div ref={hostRef} className="tv-chart-host" />
        <div ref={parentOverlayRef} className="parent-cycle-overlay" />
        <div ref={footprintOverlayRef} className="flow-footprint-overlay" />
        <div ref={footprintDetailRef} className="footprint-hover-profile" />
        <div ref={priceScaleOverlayRef} className="chart-price-axis" />
        <div ref={tooltipRef} className="chart-hover-tooltip" />
      </div>
    </div>
  );
}

function normalizeRealtimeFootprint(event) {
  if (event?.type !== "footprint_update" || !event.data || !Array.isArray(event.data.levels)) return null;
  const normalizeLevels = (rawLevels) => rawLevels
    .map((level) => ({
      price: Number(level.price),
      buyVolume: Number(level.buyVolume),
      sellVolume: Number(level.sellVolume),
      delta: Number(level.delta),
      totalVolume: Number(level.totalVolume),
      isPoc: Boolean(level.isPoc),
      buyImbalance: Boolean(level.buyImbalance),
      sellImbalance: Boolean(level.sellImbalance),
    }))
    .filter((level) => [level.price, level.buyVolume, level.sellVolume, level.delta, level.totalVolume].every(Number.isFinite));
  const levels = normalizeLevels(event.data.levels);
  if (!levels.length) return null;
  const bars = (Array.isArray(event.data.bars) ? event.data.bars : [event.data])
    .map((bar) => ({
      ...bar,
      barStartMs: Number(bar.barStartMs),
      barEndMs: Number(bar.barEndMs),
      delta: Number(bar.delta ?? 0),
      totalVolume: Number(bar.totalVolume ?? 0),
      levels: normalizeLevels(bar.levels ?? []),
    }))
    .filter((bar) => Number.isFinite(bar.barStartMs) && bar.levels.length);
  return {
    data: { ...event.data, levels, bars },
    subscriptionKey: `${event.displaySymbol ?? event.symbol}:${event.timeframe}`.toUpperCase(),
  };
}

function formatFootprintBarTime(value) {
  if (!Number.isFinite(value)) return "-";
  return new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
}

function FootprintPanel({ snapshot, visible }) {
  if (!visible) return null;
  const bars = snapshot?.bars?.length ? snapshot.bars : snapshot?.levels?.length ? [snapshot] : [];
  const levels = [...new Set(bars.flatMap((bar) => bar.levels.map((level) => level.price)))].sort((left, right) => right - left);
  const maxVolume = Math.max(1, ...bars.flatMap((bar) => bar.levels.map((level) => level.totalVolume)));
  const levelFor = (bar, price) => bar.levels.find((level) => level.price === price);
  const gridStyle = { gridTemplateColumns: `64px repeat(${Math.max(1, bars.length)}, minmax(92px, 1fr))` };
  return (
    <aside className="footprint-panel" aria-label="Realtime footprint surface">
      <div className="footprint-panel__header">
        <span>Footprint Surface · {snapshot?.priceBinSize ?? "-"} USDT bins</span>
        <span>{snapshot ? `${snapshot.tradeCount} trades · POC ${formatNumber(bars.at(-1)?.pocPrice)}` : "Waiting for trades"}</span>
      </div>
      <div className="footprint-surface-scroll">
        <div className="footprint-surface" style={gridStyle}>
          <div className="footprint-surface__corner">Price</div>
          {bars.map((bar) => <div className="footprint-surface__bar-header" key={bar.barStartMs}>{formatFootprintBarTime(bar.barStartMs)}<small>Δ {formatNumber(bar.delta)}</small></div>)}
          {levels.map((price) => [
            <div className="footprint-surface__price" key={`price-${price}`}>{formatNumber(price)}</div>,
            ...bars.map((bar) => {
              const level = levelFor(bar, price);
              const intensity = level ? Math.max(0.08, level.totalVolume / maxVolume) : 0;
              const direction = level?.delta > 0 ? "footprint-cell--buy" : level?.delta < 0 ? "footprint-cell--sell" : "";
              return (
                <div
                  className={`footprint-cell ${direction} ${level?.isPoc ? "footprint-cell--poc" : ""} ${level?.buyImbalance ? "footprint-cell--buy-imbalance" : ""} ${level?.sellImbalance ? "footprint-cell--sell-imbalance" : ""}`}
                  key={`${bar.barStartMs}-${price}`}
                  style={{ "--footprint-intensity": intensity }}
                  title={level ? `${formatFootprintBarTime(bar.barStartMs)} · ${formatNumber(price)} | Sell ${formatNumber(level.sellVolume)} / Buy ${formatNumber(level.buyVolume)} / Δ ${formatNumber(level.delta)}` : undefined}
                >
                  {level ? <><span>{formatNumber(level.sellVolume)}</span><b>{formatNumber(level.buyVolume)}</b></> : null}
                </div>
              );
            }),
          ])}
        </div>
      </div>
    </aside>
  );
}

function footprintSurfaceToSnapshot(surface) {
  if (!surface?.meta || !Array.isArray(surface.timeSlots) || !Array.isArray(surface.cells)) return null;
  const cellsByTime = new Map();
  surface.cells.forEach((cell) => {
    const index = Number(cell.timeIndex);
    if (!cellsByTime.has(index)) cellsByTime.set(index, []);
    cellsByTime.get(index).push(cell);
  });
  const bars = surface.timeSlots.map((slot, index) => {
    const levels = (cellsByTime.get(index) ?? []).sort((left, right) => Number(right.price) - Number(left.price));
    return {
      barStartMs: Date.parse(slot),
      barEndMs: Date.parse(slot) + 60_000,
      levels,
      totalVolume: levels.reduce((sum, level) => sum + Number(level.totalVolume || 0), 0),
      delta: levels.reduce((sum, level) => sum + Number(level.delta || 0), 0),
    };
  });
  return {
    priceBinSize: Number(surface.meta.tick),
    bars,
    tradeCount: Number(surface.meta.windowTradeCount),
  };
}

function FootprintWorkspace({ visible, symbol, onSurfaceChange }) {
  const aggTradeSymbol = symbol === "BTCUSD" ? "BTCUSDT" : symbol;
  const [availableDates, setAvailableDates] = useState([]);
  const [sessionDate, setSessionDate] = useState("");
  const [start, setStart] = useState("00:00");
  const [minutes, setMinutes] = useState(20);
  const [tick, setTick] = useState(5);
  const [surface, setSurface] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    if (!visible) return undefined;
    let cancelled = false;
    async function loadDates() {
      try {
        const response = await fetch(`${FOOTPRINT_DATES_URL}?symbol=${encodeURIComponent(aggTradeSymbol)}`);
        if (!response.ok) throw new Error("Footprint 날짜 목록을 불러오지 못했습니다.");
        const payload = await response.json();
        const dates = payload.dates ?? [];
        if (cancelled) return;
        setAvailableDates(dates);
        setSessionDate((current) => {
          if (dates.includes(current)) return current;
          setStart(payload.latestStart ?? "00:00");
          return dates.at(-1) ?? "";
        });
      } catch (loadError) {
        if (!cancelled) setError(loadError.message);
      }
    }
    loadDates();
    return () => { cancelled = true; };
  }, [aggTradeSymbol, visible]);

  useEffect(() => {
    if (!visible || !sessionDate) return undefined;
    let cancelled = false;
    async function loadSurface() {
      setLoading(true);
      setError("");
      try {
        const params = new URLSearchParams({
          symbol: aggTradeSymbol,
          date: sessionDate,
          start,
          minutes: String(minutes),
          tick: String(tick),
        });
        const response = await fetch(`${FOOTPRINT_SURFACE_URL}?${params}`);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail ?? "Footprint 데이터를 불러오지 못했습니다.");
        if (!cancelled) {
          setSurface(payload);
          onSurfaceChange(footprintSurfaceToSnapshot(payload));
        }
      } catch (loadError) {
        if (!cancelled) {
          setSurface(null);
          onSurfaceChange(null);
          setError(loadError.message);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    loadSurface();
    return () => { cancelled = true; };
  }, [aggTradeSymbol, minutes, onSurfaceChange, refreshKey, sessionDate, start, tick, visible]);

  function shiftWindow(direction) {
    if (!sessionDate) return;
    const windowStart = new Date(`${sessionDate}T${start}:00Z`);
    windowStart.setUTCMinutes(windowStart.getUTCMinutes() + direction * Number(minutes));
    setSessionDate(windowStart.toISOString().slice(0, 10));
    setStart(windowStart.toISOString().slice(11, 16));
  }

  if (!visible) return null;
  const quality = surface?.meta?.quality;
  return (
    <aside className="footprint-chart-controls" aria-label="Footprint controls">
      <strong>Footprint</strong>
      <span>red sell · blue buy · gold POC</span>
      <label>Date<input type="date" list="footprint-dates" value={sessionDate} onChange={(event) => setSessionDate(event.target.value)} /></label>
      <datalist id="footprint-dates">{availableDates.map((item) => <option value={item} key={item} />)}</datalist>
      <label>UTC<input type="time" step="60" value={start} onChange={(event) => setStart(event.target.value)} /></label>
      <label>Min<input type="number" min="1" max="240" value={minutes} onChange={(event) => setMinutes(Math.max(1, Math.min(240, Number(event.target.value) || 1)))} /></label>
      <label>Tick<input type="number" min="0.01" step="1" value={tick} onChange={(event) => setTick(Math.max(0.01, Number(event.target.value) || 5))} /></label>
      <button type="button" onClick={() => shiftWindow(-1)}>←</button>
      <button type="button" onClick={() => setRefreshKey((value) => value + 1)}>Go</button>
      <button type="button" onClick={() => shiftWindow(1)}>→</button>
      <small>{loading ? "Loading…" : error || (surface ? `${surface.meta.windowTradeCount.toLocaleString()} trades · buy ${((quality?.buyRatioMin ?? 0) * 100).toFixed(0)}–${((quality?.buyRatioMax ?? 0) * 100).toFixed(0)}%` : "Select a window")}</small>
    </aside>
  );
}

function buildContextCycleBands(rows, timeframe) {
  if (!Array.isArray(rows) || !timeframe) return [];
  const bands = [];
  let active = null;

  rows.forEach((row, index) => {
    const key = row?.[`${timeframe}_key`];
    if (!key || row.unix == null) {
      if (active) {
        bands.push(active);
        active = null;
      }
      return;
    }

    const nextRow = rows[index + 1];
    const endTime = unixFromDateLabel(row[`${timeframe}_cycle_end_date`]) ?? nextRow?.unix ?? row.unix + (TIMEFRAME_SECONDS[timeframe] ?? 3600);
    if (!active || active.id !== key) {
      if (active) bands.push(active);
      active = {
        id: key,
        type: row[`${timeframe}_type`] ?? row[`${timeframe}_cycle_type`] ?? "",
        startTime: row.unix,
        endTime,
        firstRow: row,
        lastRow: row,
      };
      return;
    }

    active.endTime = endTime;
    active.lastRow = row;
  });

  if (active) bands.push(active);
  return bands.filter((band) => band.endTime != null && band.endTime >= band.startTime);
}

export default function App() {
  const [availableFiles, setAvailableFiles] = useState([]);
  const [availableCycleFiles, setAvailableCycleFiles] = useState([]);
  const [loadingFiles, setLoadingFiles] = useState(true);
  const [loadingSeries, setLoadingSeries] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [dataset, setDataset] = useState(null);
  const [timeCycleContextRows, setTimeCycleContextRows] = useState([]);
  const [timeframeContextRows, setTimeframeContextRows] = useState([]);
  const [viewMode, setViewMode] = useState(() => getInitialPreference("viewMode", VIEW_MODES.TIME));
  const [selectedSymbol, setSelectedSymbol] = useState(() => getInitialPreference("selectedSymbol", DEFAULT_SYMBOL));
  const [selectedTimeframe, setSelectedTimeframe] = useState(() => getInitialPreference("selectedTimeframe", DEFAULT_TIMEFRAME));
  const [priceOverlays, setPriceOverlays] = useState(() => getInitialPreference("priceOverlays", DEFAULT_OVERLAYS));
  const [selectedIndicators, setSelectedIndicators] = useState(() => getInitialPreference("selectedIndicators", DEFAULT_INDICATORS));
  const [selectedCycleIndicators, setSelectedCycleIndicators] = useState(() => getInitialPreference("selectedCycleIndicators", DEFAULT_CYCLE_INDICATORS));
  const [cycleParentOverlayMode, setCycleParentOverlayMode] = useState(() => getInitialPreference("cycleParentOverlayMode", DEFAULT_CYCLE_PARENT_OVERLAY));
  const [loadLimit, setLoadLimit] = useState(INITIAL_LOAD_LIMIT);
  const [hoveredRow, setHoveredRow] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [realtimeCandle, setRealtimeCandle] = useState(null);
  const [realtimeFootprint, setRealtimeFootprint] = useState(null);
  const [historicalFootprint, setHistoricalFootprint] = useState(null);
  const [showFootprint, setShowFootprint] = useState(false);
  const [realtimeStatus, setRealtimeStatus] = useState("idle");
  const [reloadNonce, setReloadNonce] = useState(0);
  const frameRef = useRef(null);
  const hasLoadedSeriesRef = useRef(false);
  const latestRequestIdRef = useRef(0);
  const datasetRef = useRef(null);
  const latestRealtimeRowRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    async function loadFiles() {
      try {
        setLoadingFiles(true);
        const [baseResponse, cycleResponse] = await Promise.all([fetch(FILES_URL), fetch(CYCLE_FILES_URL)]);
        if (!baseResponse.ok) throw new Error("Failed to load the CSV file list.");
        if (!cycleResponse.ok) throw new Error("Failed to load the cycle file list.");
        const basePayload = await baseResponse.json();
        const cyclePayload = await cycleResponse.json();
        if (!cancelled) {
          setAvailableFiles(basePayload.files ?? []);
          setAvailableCycleFiles(cyclePayload.files ?? []);
        }
      } catch (error) {
        if (!cancelled) setErrorMessage(error.message);
      } finally {
        if (!cancelled) setLoadingFiles(false);
      }
    }

    loadFiles();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    function syncFullscreen() {
      setIsFullscreen(document.fullscreenElement === frameRef.current);
    }

    document.addEventListener("fullscreenchange", syncFullscreen);
    return () => document.removeEventListener("fullscreenchange", syncFullscreen);
  }, []);

  const candleFiles = useMemo(
    () =>
      availableFiles
        .map((file) => parseFileName(file.name))
        .filter(Boolean)
        .sort((left, right) => TIMEFRAME_ORDER.indexOf(left.timeframe) - TIMEFRAME_ORDER.indexOf(right.timeframe)),
    [availableFiles],
  );
  const cycleFiles = useMemo(
    () =>
      availableCycleFiles
        .map(parseCycleFile)
        .filter(Boolean)
        .sort((left, right) => TIMEFRAME_ORDER.indexOf(left.timeframe) - TIMEFRAME_ORDER.indexOf(right.timeframe)),
    [availableCycleFiles],
  );

  const availableSymbols = useMemo(() => [...new Set(candleFiles.map((file) => file.symbol))], [candleFiles]);

  useEffect(() => {
    if (!availableSymbols.length) return;
    if (!availableSymbols.includes(selectedSymbol)) {
      setSelectedSymbol(availableSymbols.includes(DEFAULT_SYMBOL) ? DEFAULT_SYMBOL : availableSymbols[0]);
    }
  }, [availableSymbols, selectedSymbol]);

  const timeAvailableTimeframes = useMemo(
    () => TIMEFRAME_ORDER.filter((timeframe) => candleFiles.some((file) => file.symbol === selectedSymbol && file.timeframe === timeframe)),
    [candleFiles, selectedSymbol],
  );
  const cycleAvailableTimeframes = useMemo(
    () => TIMEFRAME_ORDER.filter((timeframe) => cycleFiles.some((file) => file.asset === DEFAULT_CYCLE_ASSET && file.timeframe === timeframe)),
    [cycleFiles],
  );
  const availableTimeframes = viewMode === VIEW_MODES.CYCLE ? cycleAvailableTimeframes : timeAvailableTimeframes;

  useEffect(() => {
    if (!availableTimeframes.length) return;
    if (!availableTimeframes.includes(selectedTimeframe)) {
      setSelectedTimeframe(availableTimeframes.includes(DEFAULT_TIMEFRAME) ? DEFAULT_TIMEFRAME : availableTimeframes[0]);
    }
  }, [availableTimeframes, selectedTimeframe]);

  useEffect(() => {
    setLoadLimit(INITIAL_LOAD_LIMIT);
    hasLoadedSeriesRef.current = false;
  }, [selectedSymbol, selectedTimeframe, viewMode]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        selectedSymbol,
        selectedTimeframe,
        priceOverlays,
        selectedIndicators,
        selectedCycleIndicators,
        cycleParentOverlayMode,
        viewMode,
        showFootprint,
      }),
    );
  }, [cycleParentOverlayMode, priceOverlays, selectedCycleIndicators, selectedIndicators, selectedSymbol, selectedTimeframe, showFootprint, viewMode]);

  useEffect(() => {
    const selectedFile =
      viewMode === VIEW_MODES.CYCLE
        ? cycleFiles.find((file) => file.asset === DEFAULT_CYCLE_ASSET && file.timeframe === selectedTimeframe)
        : candleFiles.find((file) => file.symbol === selectedSymbol && file.timeframe === selectedTimeframe);
    if (!selectedFile) {
      setLoadingSeries(false);
      setDataset(null);
      return;
    }

    let cancelled = false;

    async function loadSeries() {
      const requestId = latestRequestIdRef.current + 1;
      latestRequestIdRef.current = requestId;

      try {
        setLoadingSeries(!hasLoadedSeriesRef.current);
        setErrorMessage("");
        const params =
          viewMode === VIEW_MODES.CYCLE
            ? new URLSearchParams({ asset: selectedFile.asset, timeframe: selectedFile.timeframe, limit: String(loadLimit), refresh: String(Date.now()) })
            : new URLSearchParams({
                file: selectedFile.fileName,
                limit: String(Math.max(loadLimit, OI_HISTORY_LOAD_LIMITS[selectedTimeframe] ?? 0)),
                refresh: String(Date.now()),
              });
        const url = viewMode === VIEW_MODES.CYCLE ? CYCLE_SERIES_URL : SERIES_URL;
        const response = await fetch(`${url}?${params.toString()}`);
        if (!response.ok) {
          const payload = await response.json().catch(() => null);
          throw new Error(payload?.detail ?? `Failed to load ${viewMode === VIEW_MODES.CYCLE ? `${selectedFile.asset}_${selectedFile.timeframe}` : selectedFile.fileName}`);
        }
        const payload = await response.json();
        if (!cancelled && latestRequestIdRef.current === requestId) {
          setDataset(payload);
          setHoveredRow(payload.rows?.at(-1) ?? null);
          hasLoadedSeriesRef.current = true;
        }
      } catch (error) {
        if (!cancelled && latestRequestIdRef.current === requestId) {
          setErrorMessage(error.message);
        }
      } finally {
        if (!cancelled && latestRequestIdRef.current === requestId) setLoadingSeries(false);
      }
    }

    loadSeries();
    const refreshMs = viewMode === VIEW_MODES.CYCLE ? CYCLE_SERIES_REFRESH_MS : TIME_SERIES_REFRESH_MS;
    const intervalId = window.setInterval(loadSeries, refreshMs);
    return () => {
      cancelled = true;
      latestRequestIdRef.current += 1;
      window.clearInterval(intervalId);
    };
  }, [candleFiles, cycleFiles, loadLimit, reloadNonce, selectedSymbol, selectedTimeframe, viewMode]);

  const rows = useMemo(() => sanitizeRows(dataset?.rows ?? []), [dataset]);
  useEffect(() => {
    if (viewMode !== VIEW_MODES.TIME || !cycleAvailableTimeframes.includes(selectedTimeframe)) {
      setTimeCycleContextRows([]);
      return undefined;
    }

    let cancelled = false;
    async function loadTimeCycleContext() {
      try {
        const params = new URLSearchParams({
          asset: DEFAULT_CYCLE_ASSET,
          timeframe: selectedTimeframe,
          limit: String(Math.max(INITIAL_LOAD_LIMIT, Math.min(loadLimit, 4000))),
        });
        const response = await fetch(`${CYCLE_SERIES_URL}?${params.toString()}`);
        if (!response.ok) {
          setTimeCycleContextRows([]);
          return;
        }
        const payload = await response.json();
        if (!cancelled) setTimeCycleContextRows(sanitizeRows(payload.rows ?? []));
      } catch {
        if (!cancelled) setTimeCycleContextRows([]);
      }
    }

    loadTimeCycleContext();
    const intervalId = window.setInterval(loadTimeCycleContext, TIME_CYCLE_CONTEXT_REFRESH_MS);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [cycleAvailableTimeframes, loadLimit, selectedTimeframe, viewMode]);

  useEffect(() => {
    if (!(selectedTimeframe in CONTEXT_RIBBON_TIMEFRAMES)) {
      setTimeframeContextRows([]);
      return undefined;
    }

    let cancelled = false;
    async function loadTimeframeContext() {
      try {
        const params = new URLSearchParams({
          asset: DEFAULT_CYCLE_ASSET,
          resolution: selectedTimeframe === "15m" ? "15m" : "1h",
          limit: String(Math.max(INITIAL_LOAD_LIMIT, Math.min(loadLimit, 12000))),
          refresh: String(Date.now()),
        });
        const response = await fetch(`${TIMEFRAME_CONTEXT_URL}?${params.toString()}`);
        if (!response.ok) {
          if (!cancelled) setTimeframeContextRows([]);
          return;
        }
        const payload = await response.json();
        const contextRows = Array.isArray(payload.rows)
          ? [...payload.rows].filter((row) => row?.unix != null).sort((left, right) => left.unix - right.unix)
          : [];
        if (!cancelled) setTimeframeContextRows(contextRows);
      } catch {
        if (!cancelled) setTimeframeContextRows([]);
      }
    }

    loadTimeframeContext();
    const intervalId = window.setInterval(loadTimeframeContext, TIMEFRAME_CONTEXT_REFRESH_MS);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [loadLimit, selectedTimeframe]);

  useEffect(() => {
    datasetRef.current = dataset;
    latestRealtimeRowRef.current = rows.at(-1) ?? null;
  }, [dataset, rows]);

  useEffect(() => {
    if (viewMode !== VIEW_MODES.TIME || !REALTIME_TIMEFRAMES.includes(selectedTimeframe) || typeof window === "undefined") {
      setRealtimeStatus("idle");
      setRealtimeFootprint(null);
      return undefined;
    }

    const selectedKey = `${selectedSymbol}:${selectedTimeframe}`.toUpperCase();
    latestRealtimeRowRef.current = null;
    setRealtimeFootprint(null);
    let closed = false;
    let reconnectTimer = null;
    let retry = 0;
    let socket = null;

    const reloadFromRest = () => {
      setReloadNonce((current) => current + 1);
    };

    const connect = () => {
      if (closed) return;
      const params = new URLSearchParams({ symbol: selectedSymbol, timeframe: selectedTimeframe });
      socket = new WebSocket(wsUrl(`/ws/market?${params.toString()}`));
      setRealtimeStatus("connecting");

      socket.onopen = () => {
        retry = 0;
        setRealtimeStatus("connected");
      };

      socket.onmessage = (message) => {
        let payload = null;
        try {
          payload = JSON.parse(message.data);
        } catch {
          return;
        }

        if (payload?.type === "error") {
          setRealtimeStatus("error");
          closed = true;
          socket?.close();
          setErrorMessage(payload.message ?? "Realtime connection failed.");
          return;
        }

        const normalizedFootprint = normalizeRealtimeFootprint(payload);
        if (normalizedFootprint?.subscriptionKey === selectedKey) {
          setRealtimeFootprint(normalizedFootprint.data);
          return;
        }

        const normalized = normalizeRealtimeCandle(payload);
        if (!normalized || normalized.subscriptionKey !== selectedKey) return;

        const latest = latestRealtimeRowRef.current;
        const timeframeSeconds = TIMEFRAME_SECONDS[selectedTimeframe];
        if (latest?.unix && timeframeSeconds) {
          const expectedNextUnix = latest.unix + timeframeSeconds;
          if (normalized.row.unix < latest.unix) return;
          if (normalized.row.unix > expectedNextUnix) {
            setRealtimeStatus("resyncing");
            reloadFromRest();
            return;
          }
          latestRealtimeRowRef.current =
            normalized.row.unix === latest.unix ? { ...latest, ...normalized.row } : normalized.row;
        } else {
          latestRealtimeRowRef.current = normalized.row;
        }

        setRealtimeCandle({ key: `${selectedKey}:${normalized.row.unix}:${Date.now()}`, row: latestRealtimeRowRef.current });
        setHoveredRow(latestRealtimeRowRef.current);
        setRealtimeStatus("connected");

        if (normalized.row.isClosed) {
          window.setTimeout(() => {
            if (!closed) reloadFromRest();
          }, 1500);
        }
      };

      socket.onclose = () => {
        if (closed) return;
        setRealtimeStatus("disconnected");
        const delay = Math.min(1000 * 2 ** retry, 30000) + Math.random() * 500;
        retry += 1;
        reconnectTimer = window.setTimeout(connect, delay);
      };

      socket.onerror = () => {
        setRealtimeStatus("disconnected");
        socket?.close();
      };
    };

    connect();
    return () => {
      closed = true;
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [selectedSymbol, selectedTimeframe, viewMode]);

  const totalRowCount = dataset?.meta?.totalRowCount ?? rows.length;
  const hoveredCycleContext =
    viewMode === VIEW_MODES.TIME
      ? hoveredRow?._cycleContext ?? cycleRowForTime(timeCycleContextRows, hoveredRow?.unix)
      : null;
  const hoveredContextTime = contextLookupTime(hoveredRow, viewMode);
  const hoveredTimeframeContext =
    viewMode === VIEW_MODES.TIME || viewMode === VIEW_MODES.CYCLE
      ? hoveredRow?._timeframeContext ?? contextRowForTime(timeframeContextRows, hoveredContextTime)
      : null;
  const cycleInfoRow = viewMode === VIEW_MODES.TIME ? hoveredCycleContext : hoveredRow;
  const dateInfo = viewMode === VIEW_MODES.CYCLE ? formatCycleRange(hoveredRow) : hoveredRow?.date ? formatDateLabel(hoveredRow.date) : "-";
  const cycleInfo = formatCycleSummary(cycleInfoRow, selectedTimeframe);
  const parentInfo = formatImmediateParent(cycleInfoRow, selectedTimeframe);
  const weeklyInfo = formatWeeklyParent(cycleInfoRow);
  const contextInfo = viewMode === VIEW_MODES.TIME || viewMode === VIEW_MODES.CYCLE ? formatContextSummary(hoveredTimeframeContext) : "";
  const contextRibbonSupported = selectedTimeframe in CONTEXT_RIBBON_TIMEFRAMES;
  const parentOverlaySupported = viewMode === VIEW_MODES.CYCLE && PARENT_OVERLAY_TIMEFRAMES.includes(selectedTimeframe) && !contextRibbonSupported;
  const effectiveParentOverlayMode =
    parentOverlaySupported ? cycleParentOverlayMode : PARENT_OVERLAY_MODES.OFF;
  const immediateParent = immediateParentSpec(selectedTimeframe);
  const parentOverlayLabel =
    effectiveParentOverlayMode === PARENT_OVERLAY_MODES.OFF ? "Off" : immediateParent?.label ?? "Parent";
  const staleCycleMessage =
    viewMode === VIEW_MODES.CYCLE && dataset?.meta?.isStale
      ? `Cycle source is ${formatNumber(dataset.meta.sourceLagHours)}h behind raw data (${formatDateLabel(dataset.meta.endDate)} / raw ${formatDateLabel(dataset.meta.rawEndDate)}).`
      : "";
  const indicatorAvailability = useMemo(
    () => availableSeriesSignature(rows, viewMode === VIEW_MODES.CYCLE ? CYCLE_INDICATOR_DEFS : INDICATOR_DEFS),
    [rows, viewMode],
  );

  const indicatorPanels = useMemo(
    () => {
      const availableKeys = new Set(indicatorAvailability ? indicatorAvailability.split("|") : []);
      const hasAvailableSeries = (definition) => definition.series.some((item) => availableKeys.has(item.key));
      if (viewMode === VIEW_MODES.CYCLE) {
        return CYCLE_INDICATOR_DEFS.filter(
          (definition) => selectedCycleIndicators.includes(definition.key) && hasAvailableSeries(definition),
        );
      }
      return INDICATOR_DEFS.filter(
        (definition) => selectedIndicators.includes(definition.key) && hasAvailableSeries(definition),
      );
    },
    [indicatorAvailability, selectedCycleIndicators, selectedIndicators, viewMode],
  );

  async function toggleFullscreen() {
    if (!frameRef.current) return;
    if (document.fullscreenElement === frameRef.current) {
      await document.exitFullscreen();
    } else {
      await frameRef.current.requestFullscreen();
    }
  }

  return (
    <div className="app-shell">
      {errorMessage ? <div className="notice notice--error">{errorMessage}</div> : null}
      <section ref={frameRef} className={`trading-workspace ${isFullscreen ? "trading-workspace--fullscreen" : ""}`}>
        <header className="workspace-toolbar">
          <div className="workspace-toolbar__left">
            <div className="symbol-badge">{selectedSymbol}</div>

            <ToolbarMenu label={loadingFiles ? "Symbols..." : "Symbol"}>
              <div className="toolbar-grid">
                {availableSymbols.map((symbol) => (
                  <button key={symbol} type="button" className={`toolbar-option ${selectedSymbol === symbol ? "toolbar-option--active" : ""}`} onClick={() => setSelectedSymbol(symbol)}>
                    {symbol}
                  </button>
                ))}
              </div>
            </ToolbarMenu>

            <div className="timeframe-strip">
              {TIMEFRAME_ORDER.filter((timeframe) => availableTimeframes.includes(timeframe)).map((timeframe) => (
                <TimeframeButton key={timeframe} label={TIMEFRAME_LABELS[timeframe] ?? timeframe} active={selectedTimeframe === timeframe} onClick={() => setSelectedTimeframe(timeframe)} />
              ))}
            </div>

            <div className="view-mode-strip">
              <button type="button" className={`view-mode-button ${viewMode === VIEW_MODES.TIME ? "view-mode-button--active" : ""}`} onClick={() => setViewMode(VIEW_MODES.TIME)}>
                Time Candles
              </button>
              <button type="button" className={`view-mode-button ${viewMode === VIEW_MODES.CYCLE ? "view-mode-button--active" : ""}`} onClick={() => setViewMode(VIEW_MODES.CYCLE)}>
                Cycle Candles
              </button>
            </div>

            {viewMode === VIEW_MODES.TIME ? (
              <ToolbarMenu label="Overlays" wide>
                <div className="toolbar-list">
                  {PRICE_OVERLAYS.map((overlay) => (
                    <label key={overlay.key} className="toolbar-check">
                      <input type="checkbox" checked={priceOverlays.includes(overlay.key)} onChange={() => setPriceOverlays((current) => toggleValue(current, overlay.key))} />
                      <span style={{ color: overlay.color }}>{overlay.label}</span>
                    </label>
                  ))}
                </div>
              </ToolbarMenu>
            ) : null}

            {viewMode === VIEW_MODES.CYCLE ? (
              <ToolbarMenu label={parentOverlaySupported ? "Parent Overlay" : "Parent Overlay · 1H/4H only"} wide>
                <div className="toolbar-list">
                  {PARENT_OVERLAY_OPTIONS.map((option) => (
                    <label key={option.key} className={`toolbar-check ${!parentOverlaySupported ? "toolbar-check--disabled" : ""}`}>
                      <input
                        type="radio"
                        name="parent-overlay-mode"
                        disabled={!parentOverlaySupported}
                        checked={cycleParentOverlayMode === option.key}
                        onChange={() => setCycleParentOverlayMode(option.key)}
                      />
                      <span>{option.label}</span>
                    </label>
                  ))}
                </div>
              </ToolbarMenu>
            ) : null}

            {viewMode === VIEW_MODES.TIME ? (
              <label className="toolbar-check toolbar-check--inline">
                <input
                  type="checkbox"
                  checked={showFootprint}
                  onChange={() => setShowFootprint((current) => !current)}
                />
                <span>Footprint</span>
              </label>
            ) : null}

            <ToolbarMenu label="Indicators" wide>
              <div className="toolbar-list">
                {(viewMode === VIEW_MODES.CYCLE ? CYCLE_INDICATOR_DEFS : INDICATOR_DEFS).map((indicator) => {
                  const selected = viewMode === VIEW_MODES.CYCLE ? selectedCycleIndicators : selectedIndicators;
                  const setSelected = viewMode === VIEW_MODES.CYCLE ? setSelectedCycleIndicators : setSelectedIndicators;
                  return (
                    <label key={indicator.key} className="toolbar-check">
                      <input type="checkbox" checked={selected.includes(indicator.key)} onChange={() => setSelected((current) => toggleValue(current, indicator.key))} />
                      <span>{indicator.label}</span>
                    </label>
                  );
                })}
              </div>
            </ToolbarMenu>
          </div>

          <div className="workspace-toolbar__right">
            <div className="ohlc-strip">
              <span className="ohlc-strip__date">{dateInfo}</span>
              <span className="ohlc-strip__cycle">{cycleInfo}</span>
              <span className="ohlc-strip__parent">{parentInfo}</span>
              <span className="ohlc-strip__weekly">{weeklyInfo}</span>
              {viewMode === VIEW_MODES.TIME || viewMode === VIEW_MODES.CYCLE ? <span className="ohlc-strip__context">{contextInfo}</span> : null}
              {viewMode === VIEW_MODES.CYCLE ? <span>Overlay {parentOverlaySupported ? parentOverlayLabel : "1H/4H only"}</span> : null}
              <span>Pct {formatNumber(cycleInfoRow?.change_price_pct)}%</span>
              <span>Dur {formatNumber(cycleInfoRow?.duration_candles)}</span>
              <span>O {formatNumber(hoveredRow?.open)}</span>
              <span>H {formatNumber(hoveredRow?.high)}</span>
              <span>L {formatNumber(hoveredRow?.low)}</span>
              <span>C {formatNumber(hoveredRow?.close)}</span>
              {viewMode === VIEW_MODES.CYCLE ? <span>PPO {formatNumber(hoveredRow?.start_ppo)}-&gt;{formatNumber(hoveredRow?.end_ppo)}</span> : null}
              {viewMode === VIEW_MODES.TIME && REALTIME_TIMEFRAMES.includes(selectedTimeframe) ? <span>WS {realtimeStatus}</span> : null}
              {viewMode === VIEW_MODES.TIME && showFootprint ? <span>FP heatmap</span> : null}
              <span>Rows {rows.length}/{totalRowCount}</span>
            </div>
            <button type="button" className="toolbar-button" onClick={toggleFullscreen}>
              {isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
            </button>
          </div>
        </header>

        <div className="chart-shell">
          {loadingSeries ? <div className="chart-status">Loading chart data...</div> : null}
          {staleCycleMessage ? <div className="chart-status chart-status--warning">{staleCycleMessage}</div> : null}
          {rows.length ? (
            <TradingChart
              rows={rows}
              symbol={selectedSymbol}
              timeframe={selectedTimeframe}
              viewMode={viewMode}
              parentOverlayMode={effectiveParentOverlayMode}
              cycleContextRows={timeCycleContextRows}
              timeframeContextRows={timeframeContextRows}
              priceOverlays={viewMode === VIEW_MODES.TIME ? priceOverlays : NO_PRICE_OVERLAYS}
              indicatorPanels={indicatorPanels}
              realtimeCandle={realtimeCandle}
              realtimeFootprint={realtimeFootprint}
              historicalFootprint={historicalFootprint}
              showFootprint={showFootprint}
              onHoverRowChange={setHoveredRow}
              onNearHistoryStart={() => {
                if (rows.length < totalRowCount && loadLimit < MAX_LOAD_LIMIT) {
                  setLoadLimit((current) => Math.min(current * 2, totalRowCount, MAX_LOAD_LIMIT));
                }
              }}
            />
          ) : (
            <div className="chart-status">No chart data</div>
          )}
          <FootprintWorkspace visible={viewMode === VIEW_MODES.TIME && showFootprint} symbol={selectedSymbol} onSurfaceChange={setHistoricalFootprint} />
        </div>
      </section>
    </div>
  );
}
