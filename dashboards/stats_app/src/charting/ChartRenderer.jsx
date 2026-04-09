import { CHART_REGISTRY } from "./chartRegistry.js";
import { uniqueLegendItems } from "./colorUtils.js";

function EmptyState({ message }) {
  return <div className="chart-empty">{message}</div>;
}

function ChartLegend({ chart }) {
  const items = uniqueLegendItems(chart);
  if (!items.length) return null;

  return (
    <div className="chart-legend" aria-label="chart legend">
      {items.map((item) => (
        <div key={item.value} className="chart-legend__item">
          <span className="chart-legend__swatch" style={{ background: item.color }} />
          <span className="chart-legend__label">{item.value}</span>
        </div>
      ))}
    </div>
  );
}

export default function ChartRenderer({ chart }) {
  if (!chart) {
    return <EmptyState message="No chart data yet." />;
  }

  const definition = CHART_REGISTRY[chart.type];
  if (!definition) {
    return <EmptyState message={`Unsupported chart type: ${chart.type}`} />;
  }

  const Component = definition.component;
  return (
    <div className="chart-renderer">
      <Component chart={chart} />
      <ChartLegend chart={chart} />
    </div>
  );
}
