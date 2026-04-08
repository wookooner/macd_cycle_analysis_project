import { FixedSizeList as List } from "react-window";

const ROW_HEIGHT = 42;

function cellValue(value) {
  if (value == null) return "";
  return String(value);
}

export default function VirtualizedTable({
  rows,
  columns,
  height = 320,
  rowHeight = ROW_HEIGHT,
  emptyMessage = "No rows available.",
}) {
  if (!rows?.length || !columns?.length) {
    return <div className="chart-empty">{emptyMessage}</div>;
  }

  const gridTemplateColumns = `repeat(${columns.length}, minmax(140px, 1fr))`;

  return (
    <div className="virtual-table">
      <div className="virtual-table__header" style={{ gridTemplateColumns }}>
        {columns.map((column) => (
          <div key={column} className="virtual-table__cell virtual-table__cell--head">
            {column}
          </div>
        ))}
      </div>
      <div className="virtual-table__body">
        <List
          height={height}
          itemCount={rows.length}
          itemSize={rowHeight}
          width="100%"
          itemData={{ rows, columns, gridTemplateColumns }}
        >
          {({ index, style, data }) => {
            const row = data.rows[index];
            return (
              <div
                style={style}
                className={`virtual-table__row ${index % 2 === 0 ? "virtual-table__row--even" : ""}`}
              >
                <div className="virtual-table__row-grid" style={{ gridTemplateColumns }}>
                  {data.columns.map((column) => (
                    <div key={`${index}-${column}`} className="virtual-table__cell">
                      {cellValue(row[column])}
                    </div>
                  ))}
                </div>
              </div>
            );
          }}
        </List>
      </div>
    </div>
  );
}
