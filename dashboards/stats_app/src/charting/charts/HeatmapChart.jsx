import EChartPanel from "../echarts/EChartPanel.jsx";
import { withCartesianInteractions } from "../echarts/interactiveOptions.js";

export default function HeatmapChart({ chart }) {
  const rows = chart.rows ?? [];
  if (!rows.length) {
    return <div className="chart-empty">No heatmap data available.</div>;
  }

  const xBins = [...new Set(rows.map((row) => row.x_bin))];
  const yBins = [...new Set(rows.map((row) => row.y_bin))];

  const option = withCartesianInteractions({
    animationDuration: 200,
    tooltip: {
      position: "top",
      formatter: (params) => `${chart.x}: ${params.value[0]}<br/>${chart.y}: ${params.value[1]}<br/>count: ${params.value[2]}`,
    },
    grid: { left: 92, right: 24, top: 42, bottom: 70 },
    xAxis: {
      type: "category",
      name: chart.x,
      data: xBins,
      splitArea: { show: true },
      axisLabel: { interval: 0, rotate: xBins.length > 8 ? 28 : 0, fontSize: 10 },
    },
    yAxis: {
      type: "category",
      name: chart.y,
      data: yBins,
      splitArea: { show: true },
      axisLabel: { fontSize: 10 },
    },
    visualMap: {
      min: 0,
      max: Math.max(...rows.map((row) => Number(row.value ?? 0)), 1),
      calculable: true,
      orient: "horizontal",
      left: "center",
      bottom: 8,
      inRange: { color: ["#f4efe8", "#9bd1c6", "#0f766e"] },
    },
    series: [
      {
        type: "heatmap",
        data: rows.map((row) => [row.x_bin, row.y_bin, Number(row.value ?? 0)]),
        label: { show: false },
        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: "rgba(0,0,0,0.25)" } },
      },
    ],
    graphic: chart.source_row_count
      ? [
          {
            type: "text",
            right: 18,
            top: 12,
            style: {
              text: `${chart.source_row_count} rows aggregated into ${chart.rendered_row_count} cells`,
              fill: "#8a755f",
              fontSize: 11,
            },
          },
        ]
      : undefined,
  }, { zoom: true, brush: true });

  return <EChartPanel option={option} height={420} />;
}
