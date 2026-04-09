import EChartPanel from "../echarts/EChartPanel.jsx";
import { withCartesianInteractions } from "../echarts/interactiveOptions.js";

export default function BoxplotChart({ chart }) {
  const rows = chart.rows ?? [];
  if (!rows.length) {
    return <div className="chart-empty">No boxplot data available.</div>;
  }

  const option = withCartesianInteractions({
    animationDuration: 250,
    tooltip: { trigger: "item" },
    grid: { left: 56, right: 24, top: 36, bottom: 56 },
    xAxis: {
      type: "category",
      data: rows.map((row) => row[chart.group_by]),
      axisLabel: { interval: 0, rotate: rows.length > 6 ? 24 : 0 },
    },
    yAxis: {
      type: "value",
      name: chart.y,
    },
    series: [
      {
        type: "boxplot",
        itemStyle: {
          color: "rgba(15,118,110,0.18)",
          borderColor: "#0f766e",
          borderWidth: 1.5,
        },
        emphasis: {
          itemStyle: {
            borderColor: "#c2410c",
          },
        },
        data: rows.map((row) => [row.min, row.q1, row.median, row.q3, row.max]),
      },
    ],
  }, { zoom: true, brush: false });

  return <EChartPanel option={option} height={380} />;
}
