import BarChart from "./charts/BarChart.jsx";
import BoxplotChart from "./charts/BoxplotChart.jsx";
import HeatmapChart from "./charts/HeatmapChart.jsx";
import HistogramChart from "./charts/HistogramChart.jsx";
import LineChart from "./charts/LineChart.jsx";
import PieChart from "./charts/PieChart.jsx";
import ScatterChart from "./charts/ScatterChart.jsx";
import TableChart from "./charts/TableChart.jsx";

export const CHART_REGISTRY = {
  histogram: {
    label: "Histogram",
    description: "Single numeric feature distribution",
    required: ["x"],
    component: HistogramChart,
    defaultMetric: "count",
  },
  scatter: {
    label: "Scatter",
    description: "Relationship between two numeric features",
    required: ["x", "y"],
    component: ScatterChart,
    defaultMetric: "count",
  },
  bar: {
    label: "Bar",
    description: "Category-based count or aggregated value",
    required: ["group_by"],
    component: BarChart,
    defaultMetric: "count",
  },
  boxplot: {
    label: "Boxplot",
    description: "Distribution comparison by group",
    required: ["group_by", "y"],
    component: BoxplotChart,
    defaultMetric: "count",
  },
  line: {
    label: "Line",
    description: "Trend over ordered or time-based x axis",
    required: ["x", "y"],
    component: LineChart,
    defaultMetric: "count",
  },
  pie: {
    label: "Pie",
    description: "Category composition by count or aggregate",
    required: ["group_by"],
    component: PieChart,
    defaultMetric: "count",
  },
  heatmap: {
    label: "Heatmap",
    description: "Density map for two numeric features",
    required: ["x", "y"],
    component: HeatmapChart,
    defaultMetric: "count",
  },
  table: {
    label: "Table",
    description: "Raw filtered rows",
    required: [],
    component: TableChart,
    defaultMetric: "count",
  },
};

export const CHART_OPTIONS = Object.entries(CHART_REGISTRY).map(([key, value]) => ({
  key,
  ...value,
}));
