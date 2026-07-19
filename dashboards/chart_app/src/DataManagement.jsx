import { useCallback, useEffect, useState } from "react";

const API_ROOT = "/api/data-management";

function formatBytes(value) {
  const bytes = Number(value ?? 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index >= 3 ? 2 : 1)} ${units[index]}`;
}

function formatTime(value) {
  return value ? new Date(value).toLocaleString() : "-";
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail ?? "요청을 처리하지 못했습니다.");
  return payload;
}

export default function DataManagement({ onBack }) {
  const [storage, setStorage] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [timeframe, setTimeframe] = useState("1min");
  const [keepFiles, setKeepFiles] = useState(0);
  const [loading, setLoading] = useState(true);
  const [actionLabel, setActionLabel] = useState("");
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      setError("");
      const [storagePayload, jobsPayload] = await Promise.all([
        requestJson(`${API_ROOT}/storage`),
        requestJson(`${API_ROOT}/jobs`),
      ]);
      setStorage(storagePayload);
      setJobs(jobsPayload.jobs ?? []);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const intervalId = window.setInterval(refresh, 10000);
    return () => window.clearInterval(intervalId);
  }, [refresh]);

  async function startJob(payload, label) {
    if (payload.task === "microstructure_live" && !window.confirm("실시간 미시구조 수집은 중지할 때까지 계속 실행됩니다. 시작할까요?")) return;
    try {
      setActionLabel(label);
      setError("");
      await requestJson(`${API_ROOT}/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      await refresh();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setActionLabel("");
    }
  }

  async function refreshStorage() {
    try {
      await requestJson(`${API_ROOT}/storage/refresh`, { method: "POST" });
      await refresh();
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  async function stopJob(jobId) {
    try {
      setActionLabel(jobId);
      await requestJson(`${API_ROOT}/jobs/${jobId}/stop`, { method: "POST" });
      await refresh();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setActionLabel("");
    }
  }

  return (
    <main className="data-management">
      <header className="data-management__header">
        <div>
          <p className="data-management__eyebrow">LOCAL PIPELINE CONSOLE</p>
          <h1>데이터 수집 · 관리</h1>
          <p>{storage?.dataRoot ?? "데이터 루트를 확인하는 중입니다."}</p>
        </div>
        <div className="data-management__actions">
          <button type="button" className="management-button management-button--secondary" onClick={refreshStorage}>새로고침</button>
          {onBack ? <button type="button" className="management-button" onClick={onBack}>차트로 돌아가기</button> : null}
        </div>
      </header>

      {error ? <div className="management-notice management-notice--error">{error}</div> : null}

      <section className="management-summary-grid">
        <article className="management-card management-card--metric">
          <span>전체 데이터</span>
          <strong>{storage?.scanning ? "집계 중…" : formatBytes(storage?.totalBytes)}</strong>
          <small>{Number(storage?.totalFiles ?? 0).toLocaleString()} files</small>
        </article>
        <article className="management-card management-card--metric">
          <span>정리 검토 대상</span>
          <strong>{storage?.scanning ? "집계 중…" : formatBytes((storage?.cleanupCandidates ?? []).reduce((sum, item) => sum + Number(item.bytes ?? 0), 0))}</strong>
          <small>백업 · 임시 · recovery · archive</small>
        </article>
        <article className="management-card management-card--metric">
          <span>실행 중 작업</span>
          <strong>{jobs.filter((job) => job.status === "running").length}</strong>
          <small>작업 상태는 10초마다 갱신됩니다.</small>
        </article>
      </section>

      <section className="management-section">
        <div className="management-section__heading">
          <div><h2>파이프라인 실행</h2><p>허용된 작업만 별도 Python 프로세스로 실행합니다.</p></div>
        </div>
        <div className="management-command-grid">
          <button type="button" className="management-command" disabled={Boolean(actionLabel)} onClick={() => startJob({ task: "pipeline", asset: "btc", steps: [1, 2, 3, 5] }, "btc-pipeline")}>
            <b>BTC 전체 갱신</b><span>수집 → 지표 → 사이클 → 컨텍스트</span>
          </button>
          <button type="button" className="management-command" disabled={Boolean(actionLabel)} onClick={() => startJob({ task: "pipeline", asset: "gold", steps: [1, 2, 3, 5] }, "gold-pipeline")}>
            <b>GOLD 전체 갱신</b><span>API 키가 설정된 경우 실행됩니다.</span>
          </button>
          <button type="button" className="management-command" disabled={Boolean(actionLabel)} onClick={() => startJob({ task: "pipeline", asset: "btc", steps: [1] }, "btc-collect")}>
            <b>BTC 수집만 실행</b><span>OHLCV와 선물 데이터만 갱신합니다.</span>
          </button>
        </div>
        <div className="management-form-row">
          <label>심볼<input value={symbol} onChange={(event) => setSymbol(event.target.value.toUpperCase())} /></label>
          <label>Feature 주기<select value={timeframe} onChange={(event) => setTimeframe(event.target.value)}><option value="1min">1min</option><option value="5min">5min</option><option value="15min">15min</option></select></label>
          <button type="button" className="management-button" disabled={Boolean(actionLabel)} onClick={() => startJob({ task: "microstructure_features", symbol, timeframe }, "microstructure-features")}>미시구조 Feature 생성</button>
        </div>
        <div className="management-form-row management-form-row--live">
          <label>Raw 파일 보존 수<input type="number" min="0" value={keepFiles} onChange={(event) => setKeepFiles(Math.max(0, Number(event.target.value)))} /></label>
          <span>0은 보존 제한 없음입니다.</span>
          <button type="button" className="management-button management-button--warning" disabled={Boolean(actionLabel)} onClick={() => startJob({ task: "microstructure_live", symbol, timeframe, keep_files: keepFiles }, "microstructure-live")}>실시간 미시구조 수집 시작</button>
        </div>
      </section>

      <section className="management-section">
        <div className="management-section__heading"><div><h2>작업 상태 · 로그</h2><p>서버 재시작 전까지 최근 실행 작업을 유지합니다.</p></div></div>
        <div className="management-job-list">
          {jobs.length ? jobs.map((job) => (
            <article className="management-job" key={job.id}>
              <div className="management-job__header"><div><b>{job.label}</b><span>{job.status} · PID {job.pid} · {formatTime(job.startedAt)}</span></div>{job.status === "running" ? <button type="button" className="management-button management-button--danger" disabled={actionLabel === job.id} onClick={() => stopJob(job.id)}>중지</button> : null}</div>
              <code>{job.command.join(" ")}</code>
              <pre>{job.logs?.length ? job.logs.join("\n") : "출력을 기다리는 중입니다..."}</pre>
            </article>
          )) : <p className="management-empty">아직 관리 콘솔에서 시작한 작업이 없습니다.</p>}
        </div>
      </section>

      <section className="management-two-column">
        <article className="management-section">
          <div className="management-section__heading"><div><h2>저장소 용량</h2><p>{storage?.scanning || loading ? "백그라운드에서 집계 중..." : `마지막 집계: ${formatTime(storage?.scannedAt)}`}</p></div></div>
          <div className="management-table">{(storage?.categories ?? []).map((item) => <div key={item.name}><span>{item.name}</span><span>{item.files.toLocaleString()} files</span><b>{formatBytes(item.bytes)}</b></div>)}</div>
        </article>
        <article className="management-section">
          <div className="management-section__heading"><div><h2>정리 검토 대상</h2><p>삭제하지 않습니다. 항목 확인용 목록입니다.</p></div></div>
          <div className="management-table">{(storage?.cleanupCandidates ?? []).map((item) => <div key={item.path}><span title={item.path}>{item.path}</span><span>{item.files.toLocaleString()} files</span><b>{formatBytes(item.bytes)}</b></div>)}</div>
        </article>
      </section>

      <section className="management-section">
        <div className="management-section__heading"><div><h2>가장 큰 파일</h2><p>원본과 중복 파일을 판단하는 데 사용하세요.</p></div></div>
        <div className="management-table management-table--large">{(storage?.largestFiles ?? []).map((item) => <div key={item.path}><span title={item.path}>{item.path}</span><span>{formatTime(item.modifiedAt)}</span><b>{formatBytes(item.bytes)}</b></div>)}</div>
      </section>
    </main>
  );
}
