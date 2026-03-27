import { useState, useMemo, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend } from "recharts";
const API_URL = "http://localhost:8000/api/dashboard";
const avg = a => a.length ? (a.reduce((s, v) => s + v, 0) / a.length) : 0;
function pct_q(arr) {
  if (!arr.length) return { p10: "-", p25: "-", p50: "-", p75: "-", p90: "-" };
  const s = [...arr].sort((a, b) => a - b);
  const p = q => { const i = (s.length - 1) * q; const lo = Math.floor(i); return +(s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (i - lo)).toFixed(2) };
  return { p10: p(0.1), p25: p(0.25), p50: p(0.5), p75: p(0.75), p90: p(0.9) };
}
function durBin(d) { if (d <= 4) return "3-4"; if (d <= 7) return "5-7"; if (d <= 12) return "8-12"; if (d <= 20) return "13-20"; return "21+"; }
function rpBin(r) { if (r <= 20) return "0-20%"; if (r <= 40) return "20-40%"; if (r <= 60) return "40-60%"; if (r <= 80) return "60-80%"; return "80-100%"; }

const S = { bg: "#0b1120", card: "#111827", bdr: "#1e293b", t1: "#e2e8f0", t2: "#94a3b8", t3: "#64748b", up: "#22c55e", dn: "#ef4444", acc: "#06b6d4", warn: "#eab308" };

function Card({ title, sub, children }) {
  return <div style={{ background: S.card, border: "1px solid " + S.bdr, borderRadius: 8, padding: 12, marginBottom: 10 }}>
    {title && <div style={{ fontWeight: 700, fontSize: 13, color: S.t1, marginBottom: sub ? 2 : 8 }}>{title}</div>}
    {sub && <div style={{ fontSize: 10, color: S.t3, marginBottom: 6 }}>{sub}</div>}
    {children}
  </div>;
}

function TBL({ rows, cols }) {
  if (!rows.length) return <div style={{ color: S.t3, fontSize: 11 }}>{"\ub370\uc774\ud130 \uc5c6\uc74c"}</div>;
  return <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
    <thead><tr>{cols.map((c, i) => <th key={i} style={{ padding: "4px 5px", borderBottom: "1px solid " + S.bdr, color: S.t3, fontWeight: 600, textAlign: "center", whiteSpace: "nowrap", position: "sticky", top: 0, background: S.card }}>{c.h}</th>)}</tr></thead>
    <tbody>{rows.map((r, ri) => <tr key={ri}>
      {cols.map((c, ci) => { const v = typeof c.k === "function" ? c.k(r) : r[c.k]; const clr = c.clr ? c.clr(v, r) : S.t2;
        return <td key={ci} style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: clr, textAlign: "center", whiteSpace: "nowrap", fontWeight: c.b ? 600 : 400 }}>{c.fmt ? c.fmt(v, r) : v}</td> })}
    </tr>)}</tbody>
  </table></div>;
}

const FIELDS = [
  ["pct", "\uac00\uaca9%"], ["dur", "Duration"], ["rsi", "\uc2dc\uc791RSI"], ["ppo", "\uc2dc\uc791PPO%"],
  ["ppoh", "\uc2dc\uc791PPO_Hist"], ["cvd", "\uc2dc\uc791CVD"], ["ersi", "\uc885\ub8ccRSI"],
  ["eppo", "\uc885\ub8ccPPO%"], ["eppoh", "\uc885\ub8ccPPO_Hist"], ["ecvd", "\uc885\ub8ccCVD"],
  ["fr", "\ud380\ub529\ube44"], ["frsl", "FR\uae30\uc6b8\uae30"], ["tbr", "\ub9e4\uc218\ube44\uc728"],
  ["oi_chg", "OI\ubcc0\ud654%"], ["apph", "PPO_Hist\uba74\uc801"],
];

function DetailPanel({ items, onClose, label }) {
  const [field, setField] = useState("pct");
  if (!items || !items.length) return null;
  const rawVals = items.map(d => d[field]).filter(v => v != null && !isNaN(v));
  if (!rawVals.length) return <div style={{ marginTop: 8, padding: 10, background: "#0d1525", borderRadius: 6, border: "1px solid " + S.bdr }}>
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
      <span style={{ fontSize: 12, fontWeight: 600, color: S.t1 }}>{label}</span>
      <button onClick={onClose} style={{ fontSize: 10, color: S.dn, background: "none", border: "1px solid " + S.bdr, borderRadius: 3, padding: "2px 6px", cursor: "pointer" }}>{"\ub2eb\uae30"}</button>
    </div>
    <div style={{ display: "flex", gap: 3, marginBottom: 6, flexWrap: "wrap" }}>
      {FIELDS.map(([fk, fl]) => <button key={fk} onClick={() => setField(fk)} style={{ padding: "2px 8px", borderRadius: 3, fontSize: 10, cursor: "pointer", border: "1px solid " + (field === fk ? S.acc : S.bdr), background: field === fk ? S.acc + "18" : "transparent", color: field === fk ? S.acc : S.t3 }}>{fl}</button>)}
    </div>
    <div style={{ color: S.t3, fontSize: 11 }}>{"\uc774 \ud544\ub4dc\uc5d0 \ub370\uc774\ud130 \uc5c6\uc74c"}</div>
  </div>;

  const mn = Math.min(...rawVals), mx = Math.max(...rawVals);
  const a = avg(rawVals);
  const sd = Math.sqrt(rawVals.reduce((s, v) => s + (v - a) ** 2, 0) / rawVals.length);

  let chartData;
  if (field === "dur") {
    const freq = {}; rawVals.forEach(v => { const k = Math.round(v); freq[k] = (freq[k] || 0) + 1 });
    chartData = Object.entries(freq).map(([k, n]) => ({ x: +k, n })).sort((a, b) => a.x - b.x);
  } else {
    const nb = Math.min(25, Math.max(8, Math.ceil(Math.sqrt(rawVals.length))));
    const step = (mx - mn) / nb || 1;
    chartData = Array.from({ length: nb }, (_, i) => {
      const lo = mn + step * i, hi = mn + step * (i + 1);
      const n = rawVals.filter(v => v >= lo && (i === nb - 1 ? v <= hi : v < hi)).length;
      return { x: +(lo + step / 2).toFixed(3), n, lo: +lo.toFixed(3), hi: +hi.toFixed(3) };
    });
  }

  return <div style={{ marginTop: 8, padding: 10, background: "#0d1525", borderRadius: 6, border: "1px solid " + S.bdr }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6, flexWrap: "wrap", gap: 4 }}>
      <span style={{ fontSize: 12, fontWeight: 600, color: S.t1 }}>{label + " \u2192 " + FIELDS.find(f => f[0] === field)?.[1] + " (n=" + rawVals.length + ")"}</span>
      <button onClick={onClose} style={{ fontSize: 10, color: S.dn, background: "none", border: "1px solid " + S.bdr, borderRadius: 3, padding: "2px 6px", cursor: "pointer" }}>{"\ub2eb\uae30"}</button>
    </div>
    <div style={{ display: "flex", gap: 3, marginBottom: 6, flexWrap: "wrap" }}>
      {FIELDS.map(([fk, fl]) => <button key={fk} onClick={() => setField(fk)} style={{ padding: "2px 8px", borderRadius: 3, fontSize: 10, cursor: "pointer", border: "1px solid " + (field === fk ? S.acc : S.bdr), background: field === fk ? S.acc + "18" : "transparent", color: field === fk ? S.acc : S.t3, fontWeight: field === fk ? 600 : 400 }}>{fl}</button>)}
    </div>
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={chartData}><CartesianGrid strokeDasharray="3 3" stroke={S.bdr} />
        <XAxis dataKey="x" stroke={S.t2} tick={{ fontSize: 9, fill: S.t2 }} />
        <YAxis stroke={S.t3} tick={{ fontSize: 9, fill: S.t2 }} allowDecimals={false} label={{ value: "\uac74\uc218", angle: -90, position: "insideLeft", fill: S.t3, fontSize: 10 }} />
        <Tooltip contentStyle={{ background: S.card, border: "1px solid " + S.bdr, borderRadius: 6, color: S.t1, fontSize: 11 }} labelStyle={{ color: S.t1 }} itemStyle={{ color: S.t1 }}
          formatter={(v, n, p) => { const d = p.payload; return field === "dur" ? [v + "\uac1c", d.x + "\uce94\ub4e4"] : [v + "\uac1c", (d.lo || d.x) + "~" + (d.hi || d.x)] }} />
        <Bar dataKey="n" radius={[2, 2, 0, 0]}>{chartData.map((d, i) => <Cell key={i} fill={field === "pct" ? (d.x > 0 ? S.up : d.x < 0 ? S.dn : S.acc) : S.acc} />)}</Bar>
      </BarChart>
    </ResponsiveContainer>
    <div style={{ display: "flex", gap: 14, justifyContent: "center", fontSize: 11, marginTop: 4 }}>
      <span style={{ color: S.t2 }}>{"\ucd5c\uc18c: "}<span style={{ color: S.t1 }}>{field === "dur" ? Math.round(mn) : mn.toFixed(3)}</span></span>
      <span style={{ color: S.t2 }}>{"\ud3c9\uade0: "}<span style={{ color: field === "pct" ? (a > 0 ? S.up : S.dn) : S.t1 }}>{field === "dur" ? Math.round(a) : a.toFixed(4)}</span></span>
      <span style={{ color: S.t2 }}>{"\ucd5c\ub300: "}<span style={{ color: S.t1 }}>{field === "dur" ? Math.round(mx) : mx.toFixed(3)}</span></span>
      <span style={{ color: S.t2 }}>{"std: "}<span style={{ color: S.t1 }}>{sd.toFixed(4)}</span></span>
    </div>
    <div style={{ maxHeight: 200, overflowY: "auto", marginTop: 6 }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
        <thead><tr>
          {["#", "U/D", "pct%", "dur", "sRSI", "sPPO", "sCVD", "eRSI", "ePPO", "eCVD", "FR", "TBR", "OI%"].map((h, i) =>
            <th key={i} style={{ padding: "2px 4px", borderBottom: "1px solid " + S.bdr, color: S.t3, textAlign: "center", position: "sticky", top: 0, background: "#0d1525" }}>{h}</th>)}
        </tr></thead>
        <tbody>{items.map((d, i) =>
          <tr key={i}>
            <td style={{ padding: "2px 4px", borderBottom: "1px solid #1a2035", color: S.t1, textAlign: "center" }}>{i + 1}</td>
            <td style={{ padding: "2px 4px", borderBottom: "1px solid #1a2035", color: d.ct === "U" ? S.up : S.dn, textAlign: "center", fontWeight: 600 }}>{d.ct}</td>
            <td style={{ padding: "2px 4px", borderBottom: "1px solid #1a2035", color: d.pct > 0 ? S.up : d.pct < 0 ? S.dn : S.warn, textAlign: "center" }}>{d.pct}</td>
            <td style={{ padding: "2px 4px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{d.dur}</td>
            <td style={{ padding: "2px 4px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{d.rsi}</td>
            <td style={{ padding: "2px 4px", borderBottom: "1px solid #1a2035", color: d.ppo > 0 ? S.up : d.ppo < 0 ? S.dn : S.t2, textAlign: "center" }}>{d.ppo}</td>
            <td style={{ padding: "2px 4px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{d.cvd}</td>
            <td style={{ padding: "2px 4px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{d.ersi}</td>
            <td style={{ padding: "2px 4px", borderBottom: "1px solid #1a2035", color: d.eppo > 0 ? S.up : d.eppo < 0 ? S.dn : S.t2, textAlign: "center" }}>{d.eppo}</td>
            <td style={{ padding: "2px 4px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{d.ecvd}</td>
            <td style={{ padding: "2px 4px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{d.fr ?? "-"}</td>
            <td style={{ padding: "2px 4px", borderBottom: "1px solid #1a2035", color: d.tbr > 0.5 ? S.up : d.tbr < 0.5 ? S.dn : S.t2, textAlign: "center" }}>{d.tbr != null ? d.tbr.toFixed(3) : "-"}</td>
            <td style={{ padding: "2px 4px", borderBottom: "1px solid #1a2035", color: d.oi_chg > 0 ? S.up : d.oi_chg < 0 ? S.dn : S.t2, textAlign: "center" }}>{d.oi_chg ?? "-"}</td>
          </tr>
        )}</tbody>
      </table>
    </div>
  </div>;
}

function DirBtn({ value, current, onChange, label }) {
  const active = value === current;
  const clr = value === "U" ? S.up : value === "D" ? S.dn : S.acc;
  return <button onClick={() => onChange(value)} style={{ padding: "3px 10px", borderRadius: 4, fontSize: 11, cursor: "pointer", fontWeight: active ? 600 : 400,
    border: "1px solid " + (active ? clr : S.bdr), background: active ? clr + "18" : "transparent", color: active ? clr : S.t3 }}>{label}</button>;
}

function PosSection({ data, posKey, rpKey, label, dirFilter, useRel }) {
  const [sel, setSel] = useState(null);
  const binFn = useRel ? rpBin : null;
  const grouped = {}; const raw = {};
  data.forEach(d => {
    const rawK = d[posKey]; if (!rawK && rawK !== 0) return;
    const k = binFn ? binFn(d[rpKey]) : rawK; const ct = d.ct; const gk = k + "_" + ct;
    if (!grouped[gk]) grouped[gk] = { k, ct, sum: 0, n: 0, wins: 0, durs: [], rsis: [], ppos: [], cvds: [], ersis: [], eppos: [], ecvds: [], frs: [] };
    if (!raw[gk]) raw[gk] = []; raw[gk].push(d);
    const g = grouped[gk]; g.sum += d.pct; g.n++;
    if ((ct === "U" && d.pct > 0) || (ct === "D" && d.pct < 0)) g.wins++;
    g.durs.push(d.dur); g.rsis.push(d.rsi); g.ppos.push(d.ppo); g.cvds.push(d.cvd);
    g.ersis.push(d.ersi); g.eppos.push(d.eppo); g.ecvds.push(d.ecvd);
    if (d.fr != null) g.frs.push(d.fr);
  });
  const rows = Object.values(grouped).map(v => ({
    gk: v.k + "_" + v.ct, pos: v.k, ct: v.ct, avg: +(v.sum / v.n).toFixed(3), n: v.n, wr: +(100 * v.wins / v.n).toFixed(1),
    dur: Math.round(avg(v.durs)),
    sR: +avg(v.rsis).toFixed(1), sP: +avg(v.ppos).toFixed(4), sC: +avg(v.cvds).toFixed(1),
    eR: +avg(v.ersis).toFixed(1), eP: +avg(v.eppos).toFixed(4), eC: +avg(v.ecvds).toFixed(1),
    fr: v.frs.length ? +avg(v.frs).toFixed(6) : "-",
  })).filter(d => d.n >= 2);
  const sortKey = useRel ? { "0-20%": 0, "20-40%": 1, "40-60%": 2, "60-80%": 3, "80-100%": 4 } : null;
  rows.sort((a, b) => { const pa = sortKey ? sortKey[a.pos] || 0 : +a.pos; const pb = sortKey ? sortKey[b.pos] || 0 : +b.pos; if (pa !== pb) return pa - pb; return a.ct === "U" ? -1 : 1; });
  if (!rows.length) return null;
  const chartMap = {};
  rows.forEach(r => { if (!chartMap[r.pos]) chartMap[r.pos] = { pos: r.pos }; chartMap[r.pos][r.ct === "U" ? "avgU" : "avgD"] = r.avg; });
  const chartData = Object.values(chartMap).sort((a, b) => { const pa = sortKey ? sortKey[a.pos] || 0 : +a.pos; const pb = sortKey ? sortKey[b.pos] || 0 : +b.pos; return pa - pb }).slice(0, 25);

  return <Card title={label} sub={useRel ? "\uc0c1\ub300\uc704\uce58: 0%=\ucd08\ubc18, 100%=\ub9c8\uc9c0\ub9c9" : "\uc808\ub300 \uc21c\ubc88 \u2014 \ud589 \ud074\ub9ad\uc2dc \ubd84\ud3ec"}>
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={chartData}><CartesianGrid strokeDasharray="3 3" stroke={S.bdr} />
        <XAxis dataKey="pos" stroke={S.t3} tick={{ fontSize: 9 }} />
        <YAxis stroke={S.t3} tick={{ fontSize: 10 }} />
        <Tooltip contentStyle={{ background: S.card, border: "1px solid " + S.bdr, borderRadius: 6, color: S.t1, fontSize: 11 }} labelStyle={{ color: S.t1 }} itemStyle={{ color: S.t1 }} />
        <Bar dataKey="avgU" name="UP \ud3c9\uade0%" fill={S.up} radius={[3, 3, 0, 0]} hide={dirFilter === "D"} />
        <Bar dataKey="avgD" name="DOWN \ud3c9\uade0%" fill={S.dn} radius={[3, 3, 0, 0]} hide={dirFilter === "U"} />
        <Legend wrapperStyle={{ color: S.t2, fontSize: 11 }} iconSize={10} />
      </BarChart>
    </ResponsiveContainer>
    <div style={{ maxHeight: 300, overflowY: "auto", marginTop: 6 }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
        <thead><tr>
          {[useRel ? "\uc704\uce58" : "#", "U/D", "n", "\ud3c9\uade0%", "\uc2b9\ub960", "\ud3c9\uade0dur", "\ud3c9\uade0sRSI", "\ud3c9\uade0sPPO", "\ud3c9\uade0sCVD", "\ud3c9\uade0eRSI", "\ud3c9\uade0ePPO", "\ud3c9\uade0eCVD", "\ud3c9\uade0FR"].map((h, i) =>
            <th key={i} style={{ padding: "4px 5px", borderBottom: "1px solid " + S.bdr, color: S.t3, fontWeight: 600, textAlign: "center", whiteSpace: "nowrap", position: "sticky", top: 0, background: S.card }}>{h}</th>
          )}
        </tr></thead>
        <tbody>{rows.map((r, ri) => {
          const isSel = sel === r.gk;
          return <tr key={ri} onClick={() => setSel(isSel ? null : r.gk)} style={{ cursor: "pointer", background: isSel ? S.acc + "15" : "transparent" }}>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: S.t1, fontWeight: 600, textAlign: "center" }}>{r.pos}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: r.ct === "U" ? S.up : S.dn, fontWeight: 600, textAlign: "center" }}>{r.ct}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{r.n}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: r.avg > 0 ? S.up : r.avg < 0 ? S.dn : S.warn, textAlign: "center" }}>{r.avg}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: r.wr >= 70 ? S.up : r.wr >= 50 ? S.warn : S.dn, textAlign: "center" }}>{r.wr + "%"}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{r.dur}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{r.sR}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: +r.sP > 0 ? S.up : +r.sP < 0 ? S.dn : S.t2, textAlign: "center" }}>{r.sP}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{r.sC}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{r.eR}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: +r.eP > 0 ? S.up : +r.eP < 0 ? S.dn : S.t2, textAlign: "center" }}>{r.eP}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{r.eC}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{r.fr}</td>
          </tr>
        })}</tbody>
      </table>
    </div>
    {sel && raw[sel] && <DetailPanel items={raw[sel]} onClose={() => setSel(null)} label={sel.replace("_", " ")} />}
  </Card>;
}

function GroupTable({ data, labelKey, title, sub }) {
  const [sel, setSel] = useState(null);
  const groups = {}; const raw = {};
  data.forEach(d => {
    const k = d[labelKey]; if (!k) return; const ct = d.ct; const gk = k + "_" + ct;
    if (!groups[gk]) groups[gk] = { lbl: k, pDir: k[0], ct, n: 0, sum: 0, wins: 0, durs: [], rsis: [], ppos: [], cvds: [], ersis: [], eppos: [], ecvds: [], frs: [] };
    if (!raw[gk]) raw[gk] = []; raw[gk].push(d);
    const g = groups[gk]; g.n++; g.sum += d.pct;
    if ((ct === "U" && d.pct > 0) || (ct === "D" && d.pct < 0)) g.wins++;
    g.durs.push(d.dur); g.rsis.push(d.rsi); g.ppos.push(d.ppo); g.cvds.push(d.cvd);
    g.ersis.push(d.ersi); g.eppos.push(d.eppo); g.ecvds.push(d.ecvd);
    if (d.fr != null) g.frs.push(d.fr);
  });
  const rows = Object.values(groups).map(v => ({
    gk: v.lbl + "_" + v.ct, lbl: v.lbl, pDir: v.pDir, ct: v.ct, n: v.n, wr: +(100 * v.wins / v.n).toFixed(1),
    avg: +(v.sum / v.n).toFixed(3), dur: Math.round(avg(v.durs)),
    sR: +avg(v.rsis).toFixed(1), sP: +avg(v.ppos).toFixed(4), sC: +avg(v.cvds).toFixed(1),
    eR: +avg(v.ersis).toFixed(1), eP: +avg(v.eppos).toFixed(4), eC: +avg(v.ecvds).toFixed(1),
    fr: v.frs.length ? +avg(v.frs).toFixed(6) : "-",
  })).filter(d => d.n >= 2).sort((a, b) => {
    if (a.pDir !== b.pDir) return a.pDir === "U" ? -1 : 1;
    const na = +a.lbl.slice(1), nb = +b.lbl.slice(1);
    if (na !== nb) return na - nb; return a.ct === "U" ? -1 : 1;
  });
  if (!rows.length) return null;
  return <Card title={title} sub={sub}>
    <div style={{ maxHeight: 350, overflowY: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
        <thead><tr>
          {["\ubd80\ubaa8", "", "n", "\uc2b9\ub960", "\ud3c9\uade0%", "\ud3c9\uade0dur", "\ud3c9\uade0sRSI", "\ud3c9\uade0sPPO", "\ud3c9\uade0sCVD", "\ud3c9\uade0eRSI", "\ud3c9\uade0ePPO", "\ud3c9\uade0eCVD", "\ud3c9\uade0FR"].map((h, i) =>
            <th key={i} style={{ padding: "4px 5px", borderBottom: "1px solid " + S.bdr, color: S.t3, fontWeight: 600, textAlign: "center", whiteSpace: "nowrap", position: "sticky", top: 0, background: S.card }}>{h}</th>
          )}
        </tr></thead>
        <tbody>{rows.map((r, ri) => {
          const isSel = sel === r.gk;
          return <tr key={ri} onClick={() => setSel(isSel ? null : r.gk)} style={{ cursor: "pointer", background: isSel ? S.acc + "15" : "transparent" }}>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: r.pDir === "U" ? S.up : S.dn, fontWeight: 600, textAlign: "center" }}>{r.lbl}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: r.ct === "U" ? S.up : S.dn, fontWeight: 600, textAlign: "center" }}>{r.ct}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{r.n}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: r.wr >= 70 ? S.up : r.wr >= 50 ? S.warn : S.dn, textAlign: "center" }}>{r.wr + "%"}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: r.avg > 0 ? S.up : r.avg < 0 ? S.dn : S.warn, textAlign: "center" }}>{(r.avg > 0 ? "+" : "") + r.avg + "%"}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{r.dur}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{r.sR}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: +r.sP > 0 ? S.up : +r.sP < 0 ? S.dn : S.t2, textAlign: "center" }}>{r.sP}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{r.sC}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{r.eR}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: +r.eP > 0 ? S.up : +r.eP < 0 ? S.dn : S.t2, textAlign: "center" }}>{r.eP}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{r.eC}</td>
            <td style={{ padding: "3px 5px", borderBottom: "1px solid #1a2035", color: S.t2, textAlign: "center" }}>{r.fr}</td>
          </tr>
        })}</tbody>
      </table>
    </div>
    {sel && raw[sel] && <DetailPanel items={raw[sel]} onClose={() => setSel(null)} label={sel.replace("_", " ")} />}
  </Card>;
}

function Hist({ data, field, label }) {
  const vals = data.map(d => d[field]).filter(v => v != null && !isNaN(v));
  if (!vals.length) return null;
  const mn = Math.min(...vals), mx = Math.max(...vals); const nb = 20; const step = (mx - mn) / nb || 1;
  const buckets = Array.from({ length: nb }, (_, i) => ({ x: +(mn + step * (i + 0.5)).toFixed(2), n: 0 }));
  vals.forEach(v => { let idx = Math.floor((v - mn) / step); if (idx >= nb) idx = nb - 1; if (idx < 0) idx = 0; buckets[idx].n++ });
  const a = avg(vals);
  return <Card title={label}>
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={buckets}><CartesianGrid strokeDasharray="3 3" stroke={S.bdr} />
        <XAxis dataKey="x" stroke={S.t2} tick={{ fontSize: 9, fill: S.t2 }} />
        <YAxis stroke={S.t3} tick={{ fontSize: 9, fill: S.t2 }} />
        <Tooltip contentStyle={{ background: S.card, border: "1px solid " + S.bdr, borderRadius: 6, color: S.t1, fontSize: 11 }} labelStyle={{ color: S.t1 }} itemStyle={{ color: S.t1 }} />
        <Bar dataKey="n" radius={[2, 2, 0, 0]}>{buckets.map((d, i) => <Cell key={i} fill={field === "pct" ? (d.x > 0 ? S.up : d.x < 0 ? S.dn : S.acc) : S.acc} />)}</Bar>
      </BarChart>
    </ResponsiveContainer>
    {field === "pct" && <div style={{ display: "flex", gap: 12, justifyContent: "center", marginTop: 4, fontSize: 11 }}>
      <span style={{ color: S.t2 }}>{"min: "}<span style={{ color: S.dn }}>{mn.toFixed(2) + "%"}</span></span>
      <span style={{ color: S.t2 }}>{"\ud3c9\uade0: "}<span style={{ color: a > 0 ? S.up : S.dn }}>{a.toFixed(3) + "%"}</span></span>
      <span style={{ color: S.t2 }}>{"max: "}<span style={{ color: S.up }}>{mx.toFixed(2) + "%"}</span></span>
    </div>}
  </Card>;
}

export default function CycleDashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [wDir, setWDir] = useState("U");
  const [tf, setTf] = useState("1D");
  const [dir, setDir] = useState("all");
  const [dDir, setDDir] = useState("all");
  const [hDir, setHDir] = useState("all");
  const [posBase, setPosBase] = useState("1w");
  const [posMode, setPosMode] = useState("rel");
  const [filters, setFilters] = useState({});

  useEffect(() => {
    fetch(API_URL)
      .then(r => r.json())
      .then(d => { console.log("API data:", {h1: d.h1?.length, h4: d.h4?.length, d1: d.d1?.length}); console.log("Sample h1[0]:", d.h1?.[0]); console.log("w values:", [...new Set(d.h1?.map(x=>x.w))]); setData(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const _clean = arr => (arr || []).map(d => ({
    ...d,
    pct: +(d.pct || 0), dur: +(d.dur || 0),
    rsi: +(d.rsi || 50), ppo: +(d.ppo || 0), ppoh: +(d.ppoh || 0), cvd: +(d.cvd || 0),
    ersi: +(d.ersi || 50), eppo: +(d.eppo || 0), eppoh: +(d.eppoh || 0), ecvd: +(d.ecvd || 0),
    o4h: +(d.o4h || 0), o1d: +(d.o1d || 0), o1w: +(d.o1w || 0),
    r4h: +(d.r4h || 0), r1d: +(d.r1d || 0), r1w: +(d.r1w || 0),
    o4h1d: +(d.o4h1d || 0), r4h1d: +(d.r4h1d || 0),
    o1d1w: +(d.o1d1w || 0), r1d1w: +(d.r1d1w || 0),
    ct: d.ct || "U", w: d.w || "U", d: d.d || "U", h: d.h || "U",
  }));
  const H1 = _clean(data?.h1), H4 = _clean(data?.h4), D1 = _clean(data?.d1);
  console.log("Cleaned:", {H1: H1.length, H4: H4.length, D1: D1.length, wU_h1: H1.filter(x=>x.w==="U").length, wD_h1: H1.filter(x=>x.w==="D").length});

  const setF = (k, v) => setFilters(p => ({ ...p, [k]: v === "" ? undefined : +v }));
  const resetF = () => { setFilters({}); setDir("all"); setDDir("all"); setHDir("all"); };

  const filtered = useMemo(() => {
    let src;
    if (tf === "1H") {
      src = H1.filter(d => d.w === wDir);
      if (dDir !== "all") src = src.filter(d => d.d === dDir);
      if (hDir !== "all") src = src.filter(d => d.h === hDir);
      if (dir !== "all") src = src.filter(d => d.ct === dir);
    } else if (tf === "4H") {
      src = H4.filter(d => d.w === wDir);
      if (dDir !== "all") src = src.filter(d => d.d === dDir);
      if (dir !== "all") src = src.filter(d => d.ct === dir);
    } else {
      src = D1.filter(d => d.w === wDir);
      if (dir !== "all") src = src.filter(d => d.ct === dir);
    }
    const f = filters;
    if (f.rsiMin != null) src = src.filter(d => d.rsi >= f.rsiMin);
    if (f.rsiMax != null) src = src.filter(d => d.rsi <= f.rsiMax);
    if (f.ppoMin != null) src = src.filter(d => d.ppo >= f.ppoMin);
    if (f.ppoMax != null) src = src.filter(d => d.ppo <= f.ppoMax);
    if (f.durMin != null) src = src.filter(d => d.dur >= f.durMin);
    if (f.durMax != null) src = src.filter(d => d.dur <= f.durMax);
    return src;
  }, [wDir, tf, dir, dDir, hDir, filters, data]);

  const stats = useMemo(() => {
    const d = filtered; if (!d.length) return null;
    const pcts = d.map(x => x.pct); const ups = d.filter(x => x.ct === "U"); const dns = d.filter(x => x.ct === "D");
    const a = avg(pcts);
    const uwr = ups.length ? +(100 * ups.filter(x => x.pct > 0).length / ups.length).toFixed(1) : "-";
    const dwr = dns.length ? +(100 * dns.filter(x => x.pct < 0).length / dns.length).toFixed(1) : "-";
    return { n: d.length, avg: a.toFixed(3), med: ([...pcts].sort((a, b) => a - b)[Math.floor(pcts.length / 2)] || 0).toFixed(3), un: ups.length, dn: dns.length, uwr, dwr };
  }, [filtered]);

  const indRows = useMemo(() => {
    const d = filtered; if (!d.length) return [];
    return [["\uc2dc\uc791RSI", "rsi"], ["\uc2dc\uc791PPO%", "ppo"], ["\uc2dc\uc791PPO_H", "ppoh"], ["\uc2dc\uc791CVD", "cvd"],
    ["\uc885\ub8ccRSI", "ersi"], ["\uc885\ub8ccPPO%", "eppo"], ["\uc885\ub8ccPPO_H", "eppoh"], ["\uc885\ub8ccCVD", "ecvd"],
    ["\ud380\ub529\ube44", "fr"], ["FR\uae30\uc6b8\uae30", "frsl"], ["\ub9e4\uc218\ube44\uc728", "tbr"], ["OI\ubcc0\ud654%", "oi_chg"]
    ].map(([label, key]) => {
      const vals = d.map(x => x[key]).filter(x => x != null && !isNaN(x));
      return vals.length ? { label, ...pct_q(vals) } : { label, p10: "-", p25: "-", p50: "-", p75: "-", p90: "-" };
    });
  }, [filtered]);

  const durBins = useMemo(() => {
    const d = filtered; if (!d.length) return [];
    const bins = { "3-4": [], "5-7": [], "8-12": [], "13-20": [], "21+": [] };
    d.forEach(x => bins[durBin(x.dur)]?.push(x));
    return Object.entries(bins).map(([b, arr]) => {
      if (!arr.length) return { bin: b, n: 0, avg: "-", uwr: "-", dwr: "-" };
      const ups = arr.filter(x => x.ct === "U"); const dns = arr.filter(x => x.ct === "D");
      return { bin: b, n: arr.length, avg: avg(arr.map(x => x.pct)).toFixed(3),
        uwr: ups.length ? (100 * ups.filter(x => x.pct > 0).length / ups.length).toFixed(1) : "-",
        dwr: dns.length ? (100 * dns.filter(x => x.pct < 0).length / dns.length).toFixed(1) : "-" };
    });
  }, [filtered]);

  const pctRows = useMemo(() => {
    const d = filtered; if (!d.length) return [];
    const all = d.map(x => x.pct); const up = all.filter(x => x > 0); const dn = all.filter(x => x < 0);
    return [{ label: "\uc804\uccb4(" + all.length + ")", ...pct_q(all) },
      { label: "\uc0c1\uc2b9(" + up.length + ")", ...(up.length ? pct_q(up) : { p10: "-", p25: "-", p50: "-", p75: "-", p90: "-" }) },
      { label: "\ud558\ub77d(" + dn.length + ")", ...(dn.length ? pct_q(dn) : { p10: "-", p25: "-", p50: "-", p75: "-", p90: "-" }) }];
  }, [filtered]);

  const posKey = tf === "1H" ? (posBase === "4h" ? "o4h" : posBase === "1d" ? "o1d" : "o1w") : (tf === "4H" ? (posBase === "1w" ? "o1w" : "o1d") : "o1w");
  const rpKey = tf === "1H" ? (posBase === "4h" ? "r4h" : posBase === "1d" ? "r1d" : "r1w") : (tf === "4H" ? (posBase === "1w" ? "r1w" : "r1d") : "r1w");
  const posLabel = tf === "1H" ? (posBase === "4h" ? "4H\ub0b4 1H" : posBase === "1d" ? "1D\ub0b4 1H" : "1W\ub0b4 1H") : (tf === "4H" ? (posBase === "1w" ? "1W\ub0b4 4H" : "1D\ub0b4 4H") : "1W\ub0b4 1D");
  const wLabel = wDir === "U" ? "1W UP" : "1W DOWN";
  const dirLabel = dir === "U" ? tf + " UP" : dir === "D" ? tf + " DOWN" : tf + " \uc804\uccb4";
  const filterDesc = [wLabel, dDir !== "all" && tf !== "1D" ? "1D=" + dDir : "", hDir !== "all" && tf === "1H" ? "4H=" + hDir : "", dirLabel].filter(Boolean).join(" | ");

  if (loading) return <div style={{ background: "#0b1120", minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", color: "#06b6d4", fontSize: 16 }}>{"\ub370\uc774\ud130 \ub85c\ub529 \uc911..."}</div>;
  if (error) return <div style={{ background: "#0b1120", minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", color: "#ef4444", fontSize: 14 }}>{"\uc624\ub958: " + error + " (API \uc11c\ubc84 \ud655\uc778)"}</div>;
  if (!data) return null;

  return <div style={{ background: S.bg, minHeight: "100vh", color: S.t1, fontFamily: "'SF Mono','JetBrains Mono',monospace", padding: 12 }}>
    <div style={{ maxWidth: 1100, margin: "0 auto" }}>

      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10, flexWrap: "wrap" }}>
        <span style={{ fontWeight: 800, fontSize: 15, color: S.acc, letterSpacing: 1 }}>{"\uc0ac\uc774\ud074 \ubd84\uc11d"}</span>
        <span style={{ fontSize: 11, color: S.t3 }}>{"1H:" + H1.length + " | 4H:" + H4.length + " | 1D:" + D1.length + " | PPO/CVD/FR/OI"}</span>
      </div>

      <Card>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <span style={{ fontSize: 12, color: S.t3, fontWeight: 600, marginRight: 4 }}>{"1W \ubc29\ud5a5"}</span>
          <button onClick={() => setWDir("U")} style={{ padding: "8px 24px", borderRadius: 6, border: wDir === "U" ? "2px solid " + S.up : "1px solid " + S.bdr, background: wDir === "U" ? S.up + "22" : S.card, color: wDir === "U" ? S.up : S.t3, fontSize: 14, fontWeight: 700, cursor: "pointer" }}>{"UP"}</button>
          <button onClick={() => setWDir("D")} style={{ padding: "8px 24px", borderRadius: 6, border: wDir === "D" ? "2px solid " + S.dn : "1px solid " + S.bdr, background: wDir === "D" ? S.dn + "22" : S.card, color: wDir === "D" ? S.dn : S.t3, fontSize: 14, fontWeight: 700, cursor: "pointer" }}>{"DOWN"}</button>
        </div>
      </Card>

      <div style={{ display: "flex", gap: 4, marginBottom: 8 }}>
        {["1D", "4H", "1H"].map(t => <button key={t} onClick={() => { setTf(t); if (t === "1D") setPosBase("1w"); if (t === "4H" && posBase === "4h") setPosBase("1d") }} style={{
          padding: "6px 20px", borderRadius: 5, border: "none", cursor: "pointer", fontSize: 13, fontWeight: tf === t ? 700 : 500,
          background: tf === t ? S.acc + "22" : "transparent", color: tf === t ? S.acc : S.t3,
          borderBottom: tf === t ? "2px solid " + S.acc : "2px solid transparent" }}>{t}</button>)}
      </div>

      <div style={{ display: "flex", gap: 4, marginBottom: 8, flexWrap: "wrap", alignItems: "center" }}>
        {tf !== "1D" && <><span style={{ fontSize: 11, color: S.t3, fontWeight: 600 }}>{"1D:"}</span>
          {[["all", "\uc804\uccb4"], ["U", "UP"], ["D", "DN"]].map(([k, lb]) => <DirBtn key={"d" + k} value={k} current={dDir} onChange={setDDir} label={lb} />)}<span style={{ width: 8 }} /></>}
        {tf === "1H" && <><span style={{ fontSize: 11, color: S.t3, fontWeight: 600 }}>{"4H:"}</span>
          {[["all", "\uc804\uccb4"], ["U", "UP"], ["D", "DN"]].map(([k, lb]) => <DirBtn key={"h" + k} value={k} current={hDir} onChange={setHDir} label={lb} />)}<span style={{ width: 8 }} /></>}
        <span style={{ fontSize: 11, color: S.t3, fontWeight: 600 }}>{tf + ":"}</span>
        {[["all", "\uc804\uccb4"], ["U", "UP"], ["D", "DN"]].map(([k, lb]) => <DirBtn key={"o" + k} value={k} current={dir} onChange={setDir} label={lb} />)}
      </div>

      <div style={{ display: "flex", gap: 4, marginBottom: 8, flexWrap: "wrap" }}>
        {tf === "1H" && [["4h", "4H\ub0b4"], ["1d", "1D\ub0b4"], ["1w", "1W\ub0b4"]].map(([k, lb]) =>
          <button key={k} onClick={() => setPosBase(k)} style={{ padding: "3px 10px", borderRadius: 4, border: "1px solid " + (posBase === k ? S.acc : S.bdr), background: posBase === k ? S.acc + "18" : "transparent", color: posBase === k ? S.acc : S.t3, fontSize: 11, cursor: "pointer", fontWeight: posBase === k ? 600 : 400 }}>{lb}</button>
        )}
        {tf === "4H" && [["1d", "1D\ub0b4"], ["1w", "1W\ub0b4"]].map(([k, lb]) =>
          <button key={k} onClick={() => setPosBase(k)} style={{ padding: "3px 10px", borderRadius: 4, border: "1px solid " + (posBase === k ? S.acc : S.bdr), background: posBase === k ? S.acc + "18" : "transparent", color: posBase === k ? S.acc : S.t3, fontSize: 11, cursor: "pointer", fontWeight: posBase === k ? 600 : 400 }}>{lb}</button>
        )}
        <span style={{ width: 6 }} />
        {[["rel", "\uc0c1\ub300\uc704\uce58(%)"], ["abs", "\uc808\ub300\uc21c\ubc88"]].map(([k, lb]) =>
          <button key={k} onClick={() => setPosMode(k)} style={{ padding: "3px 10px", borderRadius: 4, border: "1px solid " + (posMode === k ? "#f59e0b" : S.bdr), background: posMode === k ? "#f59e0b18" : "transparent", color: posMode === k ? "#f59e0b" : S.t3, fontSize: 11, cursor: "pointer", fontWeight: posMode === k ? 600 : 400 }}>{lb}</button>
        )}
      </div>

      <Card title={"\ud544\ud130: " + filterDesc}>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
          {[["RSI", "rsiMin", "rsiMax"], ["PPO%", "ppoMin", "ppoMax"], ["\uae30\uac04", "durMin", "durMax"]].map(([lb, mn, mx]) =>
            <div key={lb} style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 11 }}>
              <span style={{ color: S.t3, width: 35 }}>{lb}</span>
              <input placeholder="min" value={filters[mn] != null ? filters[mn] : ""} onChange={e => setF(mn, e.target.value)} style={{ width: 48, padding: "2px 4px", background: "#1a2035", border: "1px solid " + S.bdr, borderRadius: 3, color: S.t1, fontSize: 11 }} />
              <span style={{ color: S.t3 }}>{"~"}</span>
              <input placeholder="max" value={filters[mx] != null ? filters[mx] : ""} onChange={e => setF(mx, e.target.value)} style={{ width: 48, padding: "2px 4px", background: "#1a2035", border: "1px solid " + S.bdr, borderRadius: 3, color: S.t1, fontSize: 11 }} />
            </div>
          )}
          <button onClick={resetF} style={{ padding: "3px 10px", borderRadius: 4, border: "1px solid " + S.bdr, background: S.card, color: S.dn, fontSize: 11, cursor: "pointer" }}>{"\ucd08\uae30\ud654"}</button>
        </div>
      </Card>

      {stats && <div style={{ display: "flex", gap: 6, marginBottom: 10, flexWrap: "wrap" }}>
        {[["\uac74\uc218", stats.n, ""], ["UP", stats.un, S.up], ["DN", stats.dn, S.dn],
        ["\ud3c9\uade0", stats.avg + "%", +stats.avg > 0 ? S.up : +stats.avg < 0 ? S.dn : S.warn],
        ["\uc911\uac04\uac12", stats.med + "%", ""],
        ["UP\uc2b9\ub960", stats.uwr === "-" ? "-" : stats.uwr + "%", typeof stats.uwr === "number" ? (stats.uwr >= 70 ? S.up : stats.uwr >= 50 ? S.warn : S.dn) : ""],
        ["DN\uc2b9\ub960", stats.dwr === "-" ? "-" : stats.dwr + "%", typeof stats.dwr === "number" ? (stats.dwr >= 70 ? S.up : stats.dwr >= 50 ? S.warn : S.dn) : ""],
        ].map(([lb, v, clr]) =>
          <div key={lb} style={{ background: S.card, border: "1px solid " + S.bdr, borderRadius: 6, padding: "5px 10px", textAlign: "center" }}>
            <div style={{ fontSize: 10, color: S.t3 }}>{lb}</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: clr || S.t1 }}>{v}</div>
          </div>
        )}
      </div>}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 10 }}>
        <Hist data={filtered} field="pct" label={"\uac00\uaca9\ubcc0\ud654\uc728 \ubd84\ud3ec"} />
        <Hist data={filtered} field="dur" label={"Duration \ubd84\ud3ec"} />
      </div>

      <PosSection data={filtered} posKey={posKey} rpKey={rpKey} label={posLabel} dirFilter={dir} useRel={posMode === "rel"} />
      {tf === "1H" && <PosSection data={filtered} posKey="o4h1d" rpKey="r4h1d" label={"\ubd80\ubaa8 4H\uc758 1D\ub0b4 \uc21c\ubc88\ubcc4"} dirFilter={dir} useRel={posMode === "rel"} />}
      {tf === "1H" && posBase !== "4h" && <PosSection data={filtered} posKey="o4h" rpKey="r4h" label={"4H\ub0b4 1H (\ucc38\uace0)"} dirFilter={dir} useRel={posMode === "rel"} />}
      {tf === "4H" && <PosSection data={filtered} posKey="o1d1w" rpKey="r1d1w" label={"\ubd80\ubaa8 1D\uc758 1W\ub0b4 \uc21c\ubc88\ubcc4 4H \uc131\uacfc"} dirFilter={dir} useRel={posMode === "rel"} />}

      {tf === "1H" && <GroupTable data={filtered} labelKey="p4l" title={"1D\ub0b4 4H\uc0ac\uc774\ud074\ubcc4 \u2192 \uc18c\uc18d 1H"} sub={"U1=\uccab\ubc88\uc9f8 4H UP, D2=\ub450\ubc88\uc9f8 4H DOWN \u2014 \ud074\ub9ad\uc2dc \ubd84\ud3ec"} />}
      {tf === "1H" && <GroupTable data={filtered} labelKey="p1dl" title={"1W\ub0b4 1D\uc0ac\uc774\ud074\ubcc4 \u2192 \uc18c\uc18d 1H"} sub={"U1=\uccab\ubc88\uc9f8 1D UP, D1=\uccab\ubc88\uc9f8 1D DOWN"} />}
      {tf === "4H" && <GroupTable data={filtered} labelKey="p1dl" title={"1W\ub0b4 1D\uc0ac\uc774\ud074\ubcc4 \u2192 \uc18c\uc18d 4H"} sub={"U1=\uccab\ubc88\uc9f8 1D UP, D1=\uccab\ubc88\uc9f8 1D DOWN"} />}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 10 }}>
        <Card title={"\uc9c0\ud45c \ubd84\uc704\uc218 (PPO/CVD/FR/OI \ud3ec\ud568)"}>
          <TBL rows={indRows} cols={[
            { h: "", k: "label", b: true, clr: () => S.t1 }, { h: "P10", k: "p10", clr: () => S.t2 }, { h: "P25", k: "p25", clr: () => S.t2 },
            { h: "P50", k: "p50", b: true, clr: () => S.t1 }, { h: "P75", k: "p75", clr: () => S.t2 }, { h: "P90", k: "p90", clr: () => S.t2 },
          ]} />
        </Card>
        <Card title={"Duration \uad6c\uac04\ubcc4"}>
          <TBL rows={durBins} cols={[
            { h: "\uad6c\uac04", k: "bin", b: true, clr: () => S.t1 }, { h: "N", k: "n", clr: () => S.t2 },
            { h: "\ud3c9\uade0%", k: "avg", clr: v => v === "-" ? S.t3 : +v > 0 ? S.up : +v < 0 ? S.dn : S.warn },
            { h: "UP\uc2b9\ub960", k: "uwr", clr: v => v === "-" ? S.t3 : +v >= 70 ? S.up : +v >= 50 ? S.warn : S.dn },
            { h: "DN\uc2b9\ub960", k: "dwr", clr: v => v === "-" ? S.t3 : +v >= 70 ? S.up : +v >= 50 ? S.warn : S.dn },
          ]} />
        </Card>
      </div>

      <Card title={"\uac00\uaca9\ubcc0\ud654\uc728 \ubd84\uc704\uc218"}>
        <TBL rows={pctRows} cols={[
          { h: "", k: "label", b: true, clr: () => S.t1 },
          { h: "P10", k: "p10", clr: v => v === "-" ? S.t3 : +v > 0 ? S.up : +v < 0 ? S.dn : S.warn },
          { h: "P25", k: "p25", clr: v => v === "-" ? S.t3 : +v > 0 ? S.up : +v < 0 ? S.dn : S.warn },
          { h: "P50", k: "p50", b: true, clr: v => v === "-" ? S.t3 : +v > 0 ? S.up : +v < 0 ? S.dn : S.warn },
          { h: "P75", k: "p75", clr: v => v === "-" ? S.t3 : +v > 0 ? S.up : +v < 0 ? S.dn : S.warn },
          { h: "P90", k: "p90", clr: v => v === "-" ? S.t3 : +v > 0 ? S.up : +v < 0 ? S.dn : S.warn },
        ]} />
      </Card>

    </div>
  </div>;
}