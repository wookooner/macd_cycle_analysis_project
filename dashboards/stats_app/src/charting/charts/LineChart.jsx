import EChartPanel from "../echarts/EChartPanel.jsx";
import { withCartesianInteractions } from "../echarts/interactiveOptions.js";

function groupRows(rows, colorField) {
  if (!colorField) {
    return [{ name: "series", rows }];
  }

  const groups = new Map();
  rows.forEach((row) => {
    const key = String(row[colorField] ?? "unknown");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  });
  return [...groups.entries()].map(([name, entries]) => ({ name, rows: entries }));
}

export default function LineChart({ chart }) {
  const rows = chart.rows ?? [];
  if (!rows.length) {
    return <div className="chart-empty">No line data available.</div>;
  }

  const datetimeAxis = chart.x_type === "datetime";
  const seriesGroups = groupRows(rows, chart.color);
  const option = withCartesianInteractions({
    animationDuration: 250,
    tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
    grid: { left: 60, right: 28, top: 36, bottom: 48 },
    legend: chart.color ? { top: 6 } : undefined,
    xAxis: {
      type: datetimeAxis ? "time" : "value",
      name: chart.x,
      axisLine: { onZero: !datetimeAxis },
    },
    yAxis: {
      type: "value",
      name: chart.y,
      axisLine: { onZero: true },
    },
    series: seriesGroups.map((group) => ({
      name: group.name,
      type: "line",
      showSymbol: group.rows.length < 180,
      smooth: false,
      emphasis: { focus: "series" },
      data: group.rows.map((row) => [row[chart.x], Number(row[chart.y] ?? 0)]),
    })),
  }, { zoom: true, brush: false });

  return <EChartPanel option={option} height={360} />;
}
