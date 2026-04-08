import EChartPanel from "../echarts/EChartPanel.jsx";
import { withCartesianInteractions } from "../echarts/interactiveOptions.js";

export default function HistogramChart({ chart }) {
  const rows = chart.rows ?? [];
  if (!rows.length) {
    return <div className="chart-empty">No histogram data available.</div>;
  }

  const option = withCartesianInteractions({
    animationDuration: 250,
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    grid: { left: 56, right: 24, top: 36, bottom: 96 },
    xAxis: {
      type: "category",
      name: chart.x,
      data: rows.map((row) => row.bucket),
      axisLabel: { interval: 0, rotate: rows.length > 8 ? 32 : 0, fontSize: 10 },
    },
    yAxis: { type: "value", name: "count" },
    series: [
      {
        type: "bar",
        barCategoryGap: "8%",
        itemStyle: { color: "#0f766e" },
        data: rows.map((row) => Number(row.count ?? 0)),
      },
    ],
  }, { zoom: true, brush: false });

  return <EChartPanel option={option} height={380} />;
}
