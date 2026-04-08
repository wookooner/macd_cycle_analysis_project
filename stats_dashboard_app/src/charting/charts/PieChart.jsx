import EChartPanel from "../echarts/EChartPanel.jsx";
import { withItemInteractions } from "../echarts/interactiveOptions.js";

export default function PieChart({ chart }) {
  const rows = chart.rows ?? [];
  if (!rows.length) {
    return <div className="chart-empty">No pie data available.</div>;
  }

  const option = withItemInteractions({
    animationDuration: 250,
    tooltip: { trigger: "item" },
    legend: { bottom: 0, left: "center" },
    series: [
      {
        type: "pie",
        radius: ["36%", "68%"],
        center: ["50%", "44%"],
        avoidLabelOverlap: true,
        itemStyle: { borderRadius: 8, borderColor: "#fff8f2", borderWidth: 2 },
        label: { formatter: "{b}: {d}%" },
        data: rows.map((row) => ({
          name: row[chart.group_by],
          value: Number(row.value ?? 0),
        })),
      },
    ],
  });

  return <EChartPanel option={option} height={400} />;
}
