import VirtualizedTable from "../../components/VirtualizedTable.jsx";

export default function TableChart({ chart }) {
  const rows = chart.rows ?? [];
  if (!rows.length) {
    return <div className="chart-empty">No table rows available.</div>;
  }

  const columns = Object.keys(rows[0]).slice(0, 8);
  return (
    <div className="table-chart-wrap">
      <VirtualizedTable rows={rows} columns={columns} height={360} emptyMessage="No table rows available." />
    </div>
  );
}
