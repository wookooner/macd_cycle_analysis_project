import EChartPanel from "../echarts/EChartPanel.jsx";
import { withCartesianInteractions } from "../echarts/interactiveOptions.js";

export default function BarChart({ chart }) {
  const rows = chart.rows ?? [];
  if (!rows.length) {
    return <div className="chart-empty">No bar data available.</div>;
  }

  const labelKey = chart.group_by;
  const option = withCartesianInteractions({
    animationDuration: 250,
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    grid: { left: 56, right: 24, top: 36, bottom: 64 },
    xAxis: {
      type: "category",
      data: rows.map((row) => row[labelKey]),
      axisLabel: { interval: 0, rotate: rows.length > 8 ? 24 : 0 },
    },
    yAxis: { type: "value" },
    series: [
      {
        type: "bar",
        barMaxWidth: 36,
        itemStyle: { color: "#c2410c", borderRadius: [8, 8, 0, 0] },
        data: rows.map((row) => Number(row.value ?? 0)),
      },
    ],
  }, { zoom: true, brush: false });

  return <EChartPanel option={option} height={360} />;
}
