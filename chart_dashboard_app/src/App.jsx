import { useDeferredValue, useEffect, useEffectEvent, useMemo, useRef, useState } from "react";
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
const TIMEFRAME_ORDER = ["1w", "1d", "4h", "1h"];
const DEFAULT_TIMEFRAME = "4h";
const DEFAULT_SYMBOL = "BTCUSD";
const DEFAULT_OVERLAYS = ["ma_7", "ma_25"];
const DEFAULT_INDICATORS = ["macd", "volume", "rsi"];
const INITIAL_LOAD_LIMIT = 1600;
const MAX_LOAD_LIMIT = 12000;
const INITIAL_VISIBLE_BARS = 240;
const REFRESH_INTERVAL_MS = 15000;
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

function parseFileName(fileName) {
  const match = /^(?<symbol>[A-Z0-9]+)_(?<timeframe>1w|1d|4h|1h)\.csv$/i.exec(fileName);
  if (!match?.groups) return null;
  return { symbol: match.groups.symbol.toUpperCase(), timeframe: match.groups.timeframe, fileName };
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

function seriesAvailable(rows, key) {
  return rows.some((row) => row[key] != null);
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

  if (panelKey === "macd" || panelKey === "ppo") {
    if (value >= 0) {
      return rising ? "rgba(126, 239, 208, 0.98)" : "rgba(198, 249, 235, 0.92)";
    }
    return rising ? "rgba(255, 196, 196, 0.9)" : "rgba(255, 123, 123, 0.98)";
  }

  return value >= 0 ? "rgba(126, 239, 208, 0.92)" : "rgba(255, 123, 123, 0.92)";
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
  priceOverlays,
  indicatorPanels,
  onNearHistoryStart,
  onHoverRowChange,
}) {
  const hostRef = useRef(null);
  const deferredRows = useDeferredValue(rows);
  const chartRef = useRef(null);
  const candleSeriesRef = useRef(null);
  const overlaySeriesRef = useRef([]);
  const indicatorSeriesRef = useRef([]);
  const paneLegendsRef = useRef(new Map());
  const paneWatermarksRef = useRef([]);
  const rowByTimeRef = useRef(new Map());
  const latestRowsRef = useRef([]);
  const visibleRangeRef = useRef(null);
  const hasInitializedRangeRef = useRef(false);
  const handleHoverRowChange = useEffectEvent((row) => {
    onHoverRowChange(row);
  });
  const handleNearHistoryStart = useEffectEvent(() => {
    onNearHistoryStart();
  });
  const syncLegends = useEffectEvent((row) => {
    makeLegend(paneLegendsRef.current.get(0), [
      { label: "O", value: row?.open, color: "#cbd5e1" },
      { label: "H", value: row?.high, color: "#0ecb81" },
      { label: "L", value: row?.low, color: "#f6465d" },
      { label: "C", value: row?.close, color: row?.close >= row?.open ? "#0ecb81" : "#f6465d" },
      ...overlaySeriesRef.current.map((overlay) => ({
        label: overlay.label,
        value: row?.[overlay.key],
        color: overlay.color,
      })),
    ], "price");

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
  const applySeriesData = useEffectEvent((rowsToApply, preserveRange = true) => {
    if (!chartRef.current || !candleSeriesRef.current || !rowsToApply.length) return;

    const previousRange = preserveRange ? visibleRangeRef.current : null;
    rowByTimeRef.current = new Map(rowsToApply.filter((row) => row.unix != null).map((row) => [row.unix, row]));

    const candleData = rowsToApply
      .filter((row) => row.unix != null && row.open != null && row.high != null && row.low != null && row.close != null)
      .map((row) => ({
        time: row.unix,
        open: row.open,
        high: row.high,
        low: row.low,
        close: row.close,
      }));
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

    const latestRow = rowsToApply[rowsToApply.length - 1] ?? null;
    syncLegends(latestRow);

    if (candleData.length && !hasInitializedRangeRef.current) {
      const from = Math.max(0, rowsToApply.length - INITIAL_VISIBLE_BARS);
      const to = rowsToApply.length + 12;
      chartRef.current.timeScale().setVisibleLogicalRange({ from, to });
      visibleRangeRef.current = { from, to };
      hasInitializedRangeRef.current = true;
    } else if (previousRange) {
      chartRef.current.timeScale().setVisibleLogicalRange(previousRange);
    }
  });

  useEffect(() => {
    latestRowsRef.current = deferredRows;
  }, [deferredRows]);

  useEffect(() => {
    if (!hostRef.current) return undefined;

    if (chartRef.current) {
      paneWatermarksRef.current.forEach((entry) => entry.watermark.detach());
      paneWatermarksRef.current = [];
      paneLegendsRef.current.clear();
      overlaySeriesRef.current = [];
      indicatorSeriesRef.current = [];
      candleSeriesRef.current = null;
      chartRef.current.remove();
      chartRef.current = null;
      visibleRangeRef.current = null;
      hasInitializedRangeRef.current = false;
    }

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
        borderColor: "rgba(66, 77, 102, 0.58)",
      },
      timeScale: {
        borderColor: "rgba(66, 77, 102, 0.58)",
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 12,
        barSpacing: 8,
        minBarSpacing: 2,
        shiftVisibleRangeOnNewBar: false,
        rightBarStaysOnScroll: true,
        lockVisibleTimeRangeOnResize: true,
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
          price: false,
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

    const crosshairHandler = (param) => {
      const latestRows = latestRowsRef.current;
      if (!param.time) {
        const latestRow = latestRows[latestRows.length - 1] ?? null;
        syncLegends(latestRow);
        return;
      }

      const nextRow = rowByTimeRef.current.get(param.time) ?? latestRows[latestRows.length - 1] ?? null;
      syncLegends(nextRow);
    };

    const visibleRangeHandler = (range) => {
      if (!range) return;
      visibleRangeRef.current = range;
      if (range.from < 80) handleNearHistoryStart();
    };

    chart.subscribeCrosshairMove(crosshairHandler);
    chart.timeScale().subscribeVisibleLogicalRangeChange(visibleRangeHandler);

    const watermarkRefs = paneWatermarksRef.current;
    const legendRefs = paneLegendsRef.current;

    return () => {
      chart.unsubscribeCrosshairMove(crosshairHandler);
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(visibleRangeHandler);
      watermarkRefs.forEach((entry) => entry.watermark.detach());
      paneWatermarksRef.current = [];
      legendRefs.clear();
      overlaySeriesRef.current = [];
      indicatorSeriesRef.current = [];
      candleSeriesRef.current = null;
      visibleRangeRef.current = null;
      hasInitializedRangeRef.current = false;
      chartRef.current = null;
      chart.remove();
    };
  }, [symbol, timeframe]);

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
                  lineWidth: 2,
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
    applySeriesData(latestRowsRef.current, true);
  }, [indicatorPanels, priceOverlays]);

  useEffect(() => {
    applySeriesData(deferredRows, true);
  }, [deferredRows]);

  return <div ref={hostRef} className="tv-chart-host" />;
}

export default function App() {
  const [availableFiles, setAvailableFiles] = useState([]);
  const [loadingFiles, setLoadingFiles] = useState(true);
  const [loadingSeries, setLoadingSeries] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [dataset, setDataset] = useState(null);
  const [selectedSymbol, setSelectedSymbol] = useState(() => getInitialPreference("selectedSymbol", DEFAULT_SYMBOL));
  const [selectedTimeframe, setSelectedTimeframe] = useState(() => getInitialPreference("selectedTimeframe", DEFAULT_TIMEFRAME));
  const [priceOverlays, setPriceOverlays] = useState(() => getInitialPreference("priceOverlays", DEFAULT_OVERLAYS));
  const [selectedIndicators, setSelectedIndicators] = useState(() => getInitialPreference("selectedIndicators", DEFAULT_INDICATORS));
  const [loadLimit, setLoadLimit] = useState(INITIAL_LOAD_LIMIT);
  const [hoveredRow, setHoveredRow] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const frameRef = useRef(null);
  const hasLoadedSeriesRef = useRef(false);
  const latestRequestIdRef = useRef(0);

  useEffect(() => {
    let cancelled = false;

    async function loadFiles() {
      try {
        setLoadingFiles(true);
        const response = await fetch(FILES_URL);
        if (!response.ok) throw new Error("Failed to load the CSV file list.");
        const payload = await response.json();
        if (!cancelled) setAvailableFiles(payload.files ?? []);
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

  const availableSymbols = useMemo(() => [...new Set(candleFiles.map((file) => file.symbol))], [candleFiles]);

  useEffect(() => {
    if (!availableSymbols.length) return;
    if (!availableSymbols.includes(selectedSymbol)) {
      setSelectedSymbol(availableSymbols.includes(DEFAULT_SYMBOL) ? DEFAULT_SYMBOL : availableSymbols[0]);
    }
  }, [availableSymbols, selectedSymbol]);

  const availableTimeframes = useMemo(
    () => TIMEFRAME_ORDER.filter((timeframe) => candleFiles.some((file) => file.symbol === selectedSymbol && file.timeframe === timeframe)),
    [candleFiles, selectedSymbol],
  );

  useEffect(() => {
    if (!availableTimeframes.length) return;
    if (!availableTimeframes.includes(selectedTimeframe)) {
      setSelectedTimeframe(availableTimeframes.includes(DEFAULT_TIMEFRAME) ? DEFAULT_TIMEFRAME : availableTimeframes[0]);
    }
  }, [availableTimeframes, selectedTimeframe]);

  useEffect(() => {
    setLoadLimit(INITIAL_LOAD_LIMIT);
    hasLoadedSeriesRef.current = false;
  }, [selectedSymbol, selectedTimeframe]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        selectedSymbol,
        selectedTimeframe,
        priceOverlays,
        selectedIndicators,
      }),
    );
  }, [priceOverlays, selectedIndicators, selectedSymbol, selectedTimeframe]);

  useEffect(() => {
    const selectedFile = candleFiles.find((file) => file.symbol === selectedSymbol && file.timeframe === selectedTimeframe);
    if (!selectedFile) {
      setLoadingSeries(false);
      return;
    }

    let cancelled = false;

    async function loadSeries() {
      const requestId = latestRequestIdRef.current + 1;
      latestRequestIdRef.current = requestId;

      try {
        setLoadingSeries(!hasLoadedSeriesRef.current);
        setErrorMessage("");
        const params = new URLSearchParams({ file: selectedFile.fileName, limit: String(loadLimit), refresh: String(Date.now()) });
        const response = await fetch(`${SERIES_URL}?${params.toString()}`);
        if (!response.ok) throw new Error(`Failed to load ${selectedFile.fileName}`);
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
    const intervalId = window.setInterval(loadSeries, REFRESH_INTERVAL_MS);
    return () => {
      cancelled = true;
      latestRequestIdRef.current += 1;
      window.clearInterval(intervalId);
    };
  }, [candleFiles, loadLimit, selectedSymbol, selectedTimeframe]);

  const rows = useMemo(() => sanitizeRows(dataset?.rows ?? []), [dataset]);
  const totalRowCount = dataset?.meta?.totalRowCount ?? rows.length;

  const indicatorPanels = useMemo(
    () =>
      INDICATOR_DEFS.filter(
        (definition) => selectedIndicators.includes(definition.key) && definition.series.some((item) => seriesAvailable(rows, item.key)),
      ),
    [rows, selectedIndicators],
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
                <TimeframeButton key={timeframe} label={timeframe.toUpperCase()} active={selectedTimeframe === timeframe} onClick={() => setSelectedTimeframe(timeframe)} />
              ))}
            </div>

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

            <ToolbarMenu label="Indicators" wide>
              <div className="toolbar-list">
                {INDICATOR_DEFS.map((indicator) => (
                  <label key={indicator.key} className="toolbar-check">
                    <input type="checkbox" checked={selectedIndicators.includes(indicator.key)} onChange={() => setSelectedIndicators((current) => toggleValue(current, indicator.key))} />
                    <span>{indicator.label}</span>
                  </label>
                ))}
              </div>
            </ToolbarMenu>
          </div>

          <div className="workspace-toolbar__right">
            <div className="ohlc-strip">
              <span>{hoveredRow?.date ? formatDateLabel(hoveredRow.date) : "-"}</span>
              <span>O {formatNumber(hoveredRow?.open)}</span>
              <span>H {formatNumber(hoveredRow?.high)}</span>
              <span>L {formatNumber(hoveredRow?.low)}</span>
              <span>C {formatNumber(hoveredRow?.close)}</span>
              <span>Rows {rows.length}/{totalRowCount}</span>
            </div>
            <button type="button" className="toolbar-button" onClick={toggleFullscreen}>
              {isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
            </button>
          </div>
        </header>

        <div className="chart-shell">
          {loadingSeries ? <div className="chart-status">Loading chart data...</div> : null}
          {rows.length ? (
            <TradingChart
              rows={rows}
              symbol={selectedSymbol}
              timeframe={selectedTimeframe}
              priceOverlays={priceOverlays}
              indicatorPanels={indicatorPanels}
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
        </div>
      </section>
    </div>
  );
}
