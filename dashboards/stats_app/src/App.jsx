import { useEffect, useState } from "react";
import ChartRenderer from "./charting/ChartRenderer.jsx";
import { CHART_OPTIONS, CHART_REGISTRY } from "./charting/chartRegistry.js";
import VirtualizedTable from "./components/VirtualizedTable.jsx";

const DATASETS_URL = "/api/dashboard/datasets";
const FEATURES_URL = (dataset) => `/api/dashboard/features/${dataset}`;
const PREVIEW_URL = "/api/dashboard/preview";
const CHART_URL = "/api/dashboard/chart";
const TOOLBAR_ITEMS = [
  { key: "dataset", label: "Dataset" },
  { key: "filters", label: "Filters" },
  { key: "analysis", label: "Analysis" },
  { key: "chart", label: "Chart" },
];
function defaultValueForField(field) {
  if (field.filter_type === "range") {
    return { min: field.min ?? "", max: field.max ?? "" };
  }
  if (field.filter_type === "date_range") {
    return { start: field.min?.slice(0, 10) ?? "", end: field.max?.slice(0, 10) ?? "" };
  }
  if (field.filter_type === "select") {
    return Array.isArray(field.options) ? [...field.options] : [];
  }
  if (field.filter_type === "boolean") {
    return "";
  }
  return "";
}

function compactLabel(value) {
  return String(value ?? "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function groupFieldsForSelect(fields) {
  return Object.values(
    fields.reduce((acc, field) => {
      const key = `${field.field_group}:${field.category}`;
      if (!acc[key]) {
        acc[key] = {
          key,
          label: `${compactLabel(field.field_group)} / ${compactLabel(field.category)}`,
          fields: [],
        };
      }
      acc[key].fields.push(field);
      return acc;
    }, {}),
  );
}

function FieldOptions({ groups }) {
  return groups.map((group) => (
    <optgroup key={group.key} label={group.label}>
      {group.fields.map((field) => (
        <option key={field.field} value={field.field}>
          {field.label}
        </option>
      ))}
    </optgroup>
  ));
}

function scopeLabel(scope) {
  if (scope === "current") return "Current";
  return compactLabel(scope);
}

function fieldScope(field, currentTimeframe) {
  if (field.relation_scope === "parent" && field.relation_timeframe) {
    return field.relation_timeframe;
  }
  return currentTimeframe || "current";
}

function buildFiltersPayload(featureMetaMap, filterValues) {
  return Object.entries(filterValues)
    .map(([field, value]) => {
      const meta = featureMetaMap[field];
      if (!meta) return null;

      if (meta.filter_type === "range") {
        const min = value?.min;
        const max = value?.max;
        if (min === "" || max === "" || min == null || max == null) return null;
        return { field, op: "between", value: [Number(min), Number(max)] };
      }

      if (meta.filter_type === "date_range") {
        const start = value?.start;
        const end = value?.end;
        if (!start || !end) return null;
        return { field, op: "between", value: [start, end] };
      }

      if (meta.filter_type === "select") {
        if (!Array.isArray(value) || !value.length) return null;
        return { field, op: "in", value };
      }

      if (meta.filter_type === "boolean") {
        if (value === "") return null;
        return { field, op: "eq", value: value === "true" };
      }

      return null;
    })
    .filter(Boolean);
}

function summarizeChart(chart) {
  if (!chart) return "No chart data yet.";
  if (chart.type === "scatter") return `${chart.rows?.length ?? 0} points ready`;
  if (chart.type === "histogram") return `${chart.rows?.length ?? 0} buckets ready`;
  if (chart.type === "bar") return `${chart.rows?.length ?? 0} bars ready`;
  if (chart.type === "boxplot") return `${chart.rows?.length ?? 0} groups ready`;
  if (chart.type === "line") return `${chart.rows?.length ?? 0} points ready`;
  if (chart.type === "pie") return `${chart.rows?.length ?? 0} slices ready`;
  if (chart.type === "heatmap") return `${chart.rows?.length ?? 0} cells ready`;
  if (chart.type === "table") return `${chart.rows?.length ?? 0} rows ready`;
  return "Chart data ready";
}

function getChartConfig(chartType) {
  return CHART_REGISTRY[chartType] ?? CHART_REGISTRY.histogram;
}

function Toolbar({ activeTool, onChange }) {
  return (
    <div className="toolbar-shell">
      {TOOLBAR_ITEMS.map((item) => (
        <button
          key={item.key}
          type="button"
          className={`toolbar-pill ${activeTool === item.key ? "toolbar-pill--active" : ""}`}
          onClick={() => onChange(activeTool === item.key ? null : item.key)}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

function FilterField({ field, value, onChange }) {
  if (field.filter_type === "range") {
    return (
      <div className="filter-field">
        <div className="filter-field__header">
          <strong>{field.label}</strong>
          <span>range</span>
        </div>
        <div className="range-row">
          <input
            type="number"
            value={value?.min ?? ""}
            placeholder={field.min ?? "min"}
            onChange={(event) => onChange(field.field, { ...value, min: event.target.value })}
          />
          <input
            type="number"
            value={value?.max ?? ""}
            placeholder={field.max ?? "max"}
            onChange={(event) => onChange(field.field, { ...value, max: event.target.value })}
          />
        </div>
      </div>
    );
  }

  if (field.filter_type === "date_range") {
    return (
      <div className="filter-field">
        <div className="filter-field__header">
          <strong>{field.label}</strong>
          <span>date</span>
        </div>
        <div className="range-row">
          <input
            type="date"
            value={value?.start ?? ""}
            onChange={(event) => onChange(field.field, { ...value, start: event.target.value })}
          />
          <input
            type="date"
            value={value?.end ?? ""}
            onChange={(event) => onChange(field.field, { ...value, end: event.target.value })}
          />
        </div>
      </div>
    );
  }

  if (field.filter_type === "select") {
    return (
      <div className="filter-field">
        <div className="filter-field__header">
          <strong>{field.label}</strong>
          <span>select</span>
        </div>
        <select
          multiple
          value={value ?? []}
          onChange={(event) =>
            onChange(
              field.field,
              [...event.target.selectedOptions].map((option) => option.value),
            )
          }
        >
          {(field.options ?? []).map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </div>
    );
  }

  if (field.filter_type === "boolean") {
    return (
      <div className="filter-field">
        <div className="filter-field__header">
          <strong>{field.label}</strong>
          <span>boolean</span>
        </div>
        <select value={value ?? ""} onChange={(event) => onChange(field.field, event.target.value)}>
          <option value="">All</option>
          <option value="true">True</option>
          <option value="false">False</option>
        </select>
      </div>
    );
  }

  return null;
}

function App() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [featureMeta, setFeatureMeta] = useState(null);
  const [filterValues, setFilterValues] = useState({});
  const [activeTool, setActiveTool] = useState("dataset");
  const [loading, setLoading] = useState(false);
  const [previewResult, setPreviewResult] = useState(null);
  const [chartResult, setChartResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [chartType, setChartType] = useState("histogram");
  const [xField, setXField] = useState("");
  const [yField, setYField] = useState("");
  const [colorField, setColorField] = useState("");
  const [groupField, setGroupField] = useState("");
  const [metric, setMetric] = useState("count");
  const [showPreviewPanel, setShowPreviewPanel] = useState(false);
  const [showChartPayload, setShowChartPayload] = useState(false);
  const [selectedFilterScope, setSelectedFilterScope] = useState("current");

  useEffect(() => {
    let cancelled = false;

    async function loadDatasets() {
      try {
        setLoading(true);
        const response = await fetch(DATASETS_URL);
        if (!response.ok) throw new Error("Failed to load datasets.");
        const payload = await response.json();
        if (cancelled) return;
        const list = payload.datasets ?? [];
        setDatasets(list);
        if (list.length && !selectedDataset) {
          setSelectedDataset(list[0]);
        }
      } catch (error) {
        if (!cancelled) setErrorMessage(error.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadDatasets();
    return () => {
      cancelled = true;
    };
  }, [selectedDataset]);

  useEffect(() => {
    if (!selectedDataset) return;
    let cancelled = false;

    async function loadFeatures() {
      try {
        setLoading(true);
        setErrorMessage("");
        const response = await fetch(FEATURES_URL(selectedDataset));
        if (!response.ok) throw new Error("Failed to load feature metadata.");
        const payload = await response.json();
        if (cancelled) return;
        setFeatureMeta(payload);

        const numericField = (payload.fields ?? []).find((field) => field.data_type === "number")?.field ?? "";
        const axisField =
          (payload.fields ?? []).find((field) => field.data_type === "datetime")?.field ??
          numericField;
        const categoricalField = (payload.fields ?? []).find((field) => field.data_type === "string")?.field ?? "";
        setXField((current) => current || axisField);
        setYField((current) => current || numericField);
        setColorField((current) => current || categoricalField);
        setGroupField((current) => current || categoricalField);
        setSelectedFilterScope(payload.timeframe || "current");
      } catch (error) {
        if (!cancelled) setErrorMessage(error.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadFeatures();
    return () => {
      cancelled = true;
    };
  }, [selectedDataset]);

  const fields = featureMeta?.fields ?? [];
  const fieldMap = Object.fromEntries(fields.map((field) => [field.field, field]));
  const filterableFields = fields.filter((field) => field.filterable);
  const numericFields = fields.filter((field) => field.data_type === "number");
  const axisFields = fields.filter((field) => field.data_type === "number" || field.data_type === "datetime");
  const categoricalFields = fields.filter((field) => field.data_type === "string");
  const axisFieldGroups = groupFieldsForSelect(axisFields);
  const numericFieldGroups = groupFieldsForSelect(numericFields);
  const categoricalFieldGroups = groupFieldsForSelect(categoricalFields);
  const filterScopeOrder = [
    featureMeta?.timeframe,
    ...(featureMeta?.available_parent_timeframes ?? []),
  ].filter(Boolean);
  const groupedFilterFields = Object.values(
    filterableFields.reduce((acc, field) => {
      const scope = fieldScope(field, featureMeta?.timeframe);
      const key = `${scope}:${field.field_group}:${field.category}`;
      if (!acc[key]) {
        acc[key] = {
          key,
          scope,
          title: field.category,
          fields: [],
        };
      }
      acc[key].fields.push(field);
      return acc;
    }, {}),
  );
  const scopedFilterGroups = groupedFilterFields.filter((group) => group.scope === selectedFilterScope);
  const activeFilterEntries = Object.keys(filterValues).map((field) => ({
    field,
    meta: fieldMap[field],
    value: filterValues[field],
  }));
  const activeChart = getChartConfig(chartType);
  const previewColumns = previewResult?.rows?.[0] ? Object.keys(previewResult.rows[0]).slice(0, 6) : ["field", "value"];

  function handleFilterFieldToggle(field) {
    setFilterValues((current) => {
      if (field.field in current) {
        const next = { ...current };
        delete next[field.field];
        return next;
      }
      return {
        ...current,
        [field.field]: defaultValueForField(field),
      };
    });
  }

  function handleFilterValueChange(field, nextValue) {
    setFilterValues((current) => ({ ...current, [field]: nextValue }));
  }

  async function runPreview() {
    try {
      setLoading(true);
      setErrorMessage("");
      const response = await fetch(PREVIEW_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset: selectedDataset,
          filters: buildFiltersPayload(fieldMap, filterValues),
          limit: 8,
        }),
      });
      if (!response.ok) throw new Error("Failed to run preview.");
      const payload = await response.json();
      setPreviewResult(payload);
    } catch (error) {
      setErrorMessage(error.message);
    } finally {
      setLoading(false);
    }
  }

  async function runChart() {
    try {
      setLoading(true);
      setErrorMessage("");
      const response = await fetch(CHART_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset: selectedDataset,
          filters: buildFiltersPayload(fieldMap, filterValues),
          chart_type: chartType,
          x: xField || null,
          y: yField || null,
          color: colorField || null,
          group_by: groupField || null,
          metric,
          bins: 20,
        }),
      });
      if (!response.ok) throw new Error("Failed to build chart data.");
      const payload = await response.json();
      setChartResult(payload);
    } catch (error) {
      setErrorMessage(error.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!selectedDataset || !featureMeta) return;

    const timer = window.setTimeout(() => {
      runChart();
    }, 180);

    return () => window.clearTimeout(timer);
  }, [selectedDataset, featureMeta, chartType, xField, yField, colorField, groupField, metric, filterValues]);

  return (
    <div className="app-shell">
      <section className="top-strip">
        <div className="dataset-compact">
          <div className="dataset-compact__main">
            <span className="dataset-compact__label">Dataset</span>
            <strong>{selectedDataset || "None"}</strong>
          </div>
          <div className="dataset-compact__meta">
            <span>{featureMeta?.field_count ?? 0} fields</span>
            <span>{featureMeta?.row_count ?? 0} rows</span>
            {(featureMeta?.available_parent_timeframes ?? []).length ? (
              <span>parents: {(featureMeta?.available_parent_timeframes ?? []).join(", ")}</span>
            ) : null}
          </div>
        </div>
      </section>

      <Toolbar activeTool={activeTool} onChange={setActiveTool} />

      <main className="workspace workspace--stacked">
        <section className={`control-drawer control-drawer--top ${activeTool ? "control-drawer--open" : ""}`}>
          <div className="control-drawer__content">
            {activeTool === "dataset" ? (
              <section className="drawer-panel">
                <h2>Dataset</h2>
                <select value={selectedDataset} onChange={(event) => setSelectedDataset(event.target.value)}>
                  {datasets.map((dataset) => (
                    <option key={dataset} value={dataset}>
                      {dataset}
                    </option>
                  ))}
                </select>
              </section>
            ) : null}

            {activeTool === "filters" ? (
              <section className="drawer-panel">
                <h2>Filters</h2>
                {(featureMeta?.available_parent_timeframes ?? []).length ? (
                  <div className="relation-note">
                    Target timeframe <strong>{featureMeta?.timeframe}</strong> can be filtered with parent cycles from{" "}
                    <strong>{(featureMeta?.available_parent_timeframes ?? []).join(", ")}</strong>.
                  </div>
                ) : null}
                <div className="filters-workbench filters-workbench--stacked">
                  <div className="filters-library">
                    <div className="scope-tabs">
                      {filterScopeOrder.map((scope) => (
                        <button
                          key={scope}
                          type="button"
                          className={`scope-tab ${selectedFilterScope === scope ? "scope-tab--active" : ""}`}
                          onClick={() => setSelectedFilterScope(scope)}
                        >
                          <span>{scopeLabel(scope)}</span>
                          <small>{scope === featureMeta?.timeframe ? "target" : "parent"}</small>
                        </button>
                      ))}
                    </div>
                    <div className="category-stack category-stack--wide">
                      {scopedFilterGroups.map((group) => (
                        <details key={group.key} className="category-block" open>
                          <summary>
                            <span>{compactLabel(group.title)}</span>
                            <strong>{group.fields.length}</strong>
                          </summary>
                          <div className="chip-grid">
                            {group.fields.map((field) => (
                              <button
                                key={field.field}
                                type="button"
                                className={`field-chip ${field.field in filterValues ? "field-chip--active" : ""}`}
                                onClick={() => handleFilterFieldToggle(field)}
                              >
                                <span>{field.label}</span>
                                <small>{field.data_type}</small>
                              </button>
                            ))}
                          </div>
                        </details>
                      ))}
                      {!scopedFilterGroups.length ? (
                        <div className="empty-scope">No filterable fields in this timeframe scope.</div>
                      ) : null}
                    </div>
                  </div>
                  <div className="filters-config">
                    <div className="active-filter-bar">
                      {activeFilterEntries.length ? (
                        activeFilterEntries.map((entry) => (
                          <button
                            key={entry.field}
                            type="button"
                            className="active-filter-pill"
                            onClick={() => handleFilterFieldToggle(entry.meta)}
                          >
                            {entry.meta?.label}
                          </button>
                        ))
                      ) : (
                        <span className="empty-hint">Choose fields above to start building conditions.</span>
                      )}
                    </div>
                    <div className="filter-stack filter-stack--grid">
                      {activeFilterEntries.map((entry) => (
                        <FilterField
                          key={entry.field}
                          field={entry.meta}
                          value={entry.value}
                          onChange={handleFilterValueChange}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </section>
            ) : null}

            {activeTool === "analysis" ? (
              <section className="drawer-panel">
                <h2>Analysis</h2>
                <div className="analysis-list analysis-list--row">
                  <div className="analysis-card analysis-card--active">
                    <strong>Preview</strong>
                    <span>필터 결과 row 미리보기</span>
                  </div>
                  <div className="analysis-card">
                    <strong>Distribution</strong>
                    <span>분포형 차트로 확장 예정</span>
                  </div>
                  <div className="analysis-card">
                    <strong>Group Compare</strong>
                    <span>범주별 비교 분석 예정</span>
                  </div>
                  <div className="analysis-card">
                    <strong>Correlation</strong>
                    <span>두 feature 관계 분석 예정</span>
                  </div>
                </div>
              </section>
            ) : null}

            {activeTool === "chart" ? (
              <section className="drawer-panel">
                <h2>Chart Builder</h2>
                <div className="form-grid form-grid--wide">
                  <label>
                    Chart Type
                    <select value={chartType} onChange={(event) => setChartType(event.target.value)}>
                      {CHART_OPTIONS.map((type) => (
                        <option key={type.key} value={type.key}>
                          {type.label}
                        </option>
                      ))}
                    </select>
                  </label>

                  {activeChart.required.includes("x") ? (
                    <label>
                      X Field
                      <select value={xField} onChange={(event) => setXField(event.target.value)}>
                        <option value="">None</option>
                        <FieldOptions groups={axisFieldGroups} />
                      </select>
                    </label>
                  ) : null}

                  {activeChart.required.includes("y") ? (
                    <label>
                      Y Field
                      <select value={yField} onChange={(event) => setYField(event.target.value)}>
                        <option value="">None</option>
                        <FieldOptions groups={numericFieldGroups} />
                      </select>
                    </label>
                  ) : null}

                  <label>
                    Color Field
                    <select value={colorField} onChange={(event) => setColorField(event.target.value)}>
                      <option value="">None</option>
                      <FieldOptions groups={categoricalFieldGroups} />
                    </select>
                  </label>

                  {activeChart.required.includes("group_by") || chartType === "bar" ? (
                    <label>
                      Group By
                      <select value={groupField} onChange={(event) => setGroupField(event.target.value)}>
                        <option value="">None</option>
                        <FieldOptions groups={categoricalFieldGroups} />
                      </select>
                    </label>
                  ) : null}

                  <label>
                    Metric
                    <select value={metric} onChange={(event) => setMetric(event.target.value)}>
                      <option value="count">count</option>
                      <option value="mean">mean</option>
                      <option value="median">median</option>
                      <option value="sum">sum</option>
                    </select>
                  </label>

                  {chartType === "table" ? (
                    <label>
                      Table Mode
                      <input value="Filtered rows preview" readOnly />
                    </label>
                  ) : null}
                </div>
                <div className="chart-meta-card">
                  <strong>{activeChart.label}</strong>
                  <p>{activeChart.description}</p>
                  <span>Required: {activeChart.required.join(", ") || "none"}</span>
                </div>
              </section>
            ) : null}
          </div>
        </section>

        <section className="canvas canvas--results">
          {errorMessage ? <div className="notice notice--error">{errorMessage}</div> : null}

          <div className="action-bar action-bar--results">
            <button type="button" className="action-button" onClick={runPreview} disabled={loading || !selectedDataset}>
              Preview Rows
            </button>
            <button type="button" className="action-button action-button--strong" onClick={runChart} disabled={loading || !selectedDataset}>
              Refresh Chart
            </button>
            <button
              type="button"
              className="action-button"
              onClick={() => setShowPreviewPanel((current) => !current)}
            >
              {showPreviewPanel ? "Hide Preview" : "Show Preview"}
            </button>
          </div>

          {showPreviewPanel ? (
            <section className="result-panel result-panel--compact">
              <div className="panel-heading">
                <h3>Preview</h3>
                <span>{previewResult ? `${previewResult.filtered_rows} rows matched` : "Not run yet"}</span>
              </div>
              <div className="result-table-wrap">
                <VirtualizedTable
                  rows={previewResult?.rows ?? []}
                  columns={previewColumns}
                  height={280}
                  emptyMessage="Run preview to inspect matched rows."
                />
              </div>
            </section>
          ) : null}

          <section className="result-panel result-panel--highlight result-panel--canvas">
            <div className="panel-heading">
              <h3>Chart Payload</h3>
              <span>{summarizeChart(chartResult?.chart)}</span>
            </div>
            <div className="chart-placeholder">
              <div className="chart-placeholder__badge">{chartType}</div>
              <ChartRenderer chart={chartResult?.chart} />
              <button
                type="button"
                className="payload-toggle"
                onClick={() => setShowChartPayload((current) => !current)}
              >
                {showChartPayload ? "Hide Raw Payload" : "Show Raw Payload"}
              </button>
              {showChartPayload ? (
                <pre>{JSON.stringify(chartResult?.chart?.rows?.slice(0, 6) ?? [], null, 2)}</pre>
              ) : null}
            </div>
          </section>
        </section>
      </main>
    </div>
  );
}

export default App;
