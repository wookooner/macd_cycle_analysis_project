import ReactECharts from "echarts-for-react";

export default function EChartPanel({ option, height = 360 }) {
  return (
    <div className="echart-panel" style={{ height }}>
      <ReactECharts
        option={option}
        style={{ height: "100%", width: "100%" }}
        opts={{ renderer: "canvas" }}
        notMerge
        lazyUpdate
      />
    </div>
  );
}
