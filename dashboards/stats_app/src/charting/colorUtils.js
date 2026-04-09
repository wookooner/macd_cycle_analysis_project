const PALETTE = ["#0f766e", "#c2410c", "#2563eb", "#7c3aed", "#be123c", "#4d7c0f", "#b45309"];

export function colorForCategory(value) {
  const str = String(value ?? "");
  let hash = 0;
  for (let index = 0; index < str.length; index += 1) {
    hash = (hash + str.charCodeAt(index)) % PALETTE.length;
  }
  return PALETTE[hash];
}

export function uniqueLegendItems(chart) {
  const key = chart?.color;
  if (!key || !Array.isArray(chart?.rows)) return [];

  const seen = new Set();
  return chart.rows
    .map((row) => row?.[key])
    .filter((value) => value != null && value !== "")
    .filter((value) => {
      const token = String(value);
      if (seen.has(token)) return false;
      seen.add(token);
      return true;
    })
    .map((value) => ({
      value: String(value),
      color: colorForCategory(value),
    }));
}
