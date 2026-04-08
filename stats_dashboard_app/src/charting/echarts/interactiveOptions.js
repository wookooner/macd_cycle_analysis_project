export function withCartesianInteractions(option, { zoom = true, brush = false } = {}) {
  return {
    ...option,
    axisPointer: {
      link: [{ xAxisIndex: "all" }],
      snap: false,
      label: { backgroundColor: "#7e6037" },
    },
    toolbox: {
      right: 10,
      top: 8,
      itemSize: 14,
      feature: {
        dataZoom: zoom ? { yAxisIndex: "none" } : undefined,
        restore: {},
        saveAsImage: {},
      },
    },
    dataZoom: zoom
      ? [
          {
            type: "inside",
            xAxisIndex: 0,
            filterMode: "none",
            moveOnMouseMove: true,
            zoomOnMouseWheel: true,
            moveOnMouseWheel: false,
          },
          {
            type: "slider",
            xAxisIndex: 0,
            height: 18,
            bottom: 10,
          },
        ]
      : undefined,
    brush: brush
      ? {
          toolbox: ["rect", "polygon", "clear"],
          xAxisIndex: "all",
          yAxisIndex: "all",
        }
      : undefined,
  };
}

export function withItemInteractions(option) {
  return {
    ...option,
    toolbox: {
      right: 10,
      top: 8,
      itemSize: 14,
      feature: {
        restore: {},
        saveAsImage: {},
      },
    },
  };
}
