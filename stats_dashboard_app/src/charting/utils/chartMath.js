export function isDatetimeAxis(chart) {
  return chart?.x_type === "datetime";
}

export function toAxisNumber(value, datetimeAxis = false) {
  if (datetimeAxis) {
    const parsed = new Date(value).getTime();
    return Number.isFinite(parsed) ? parsed : NaN;
  }
  return Number(value);
}

export function extent(values) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return [0, 1];
  }
  return [min, max];
}

export function makeTicks(min, max, count = 5) {
  return Array.from({ length: count }, (_, index) => min + ((max - min) * index) / (count - 1 || 1));
}

export function formatTick(value, datetimeAxis = false) {
  if (datetimeAxis) {
    const date = new Date(value);
    return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
  }
  return Number(value).toFixed(2);
}
