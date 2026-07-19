from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.common.paths import PROJECT_PATHS


LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/api/data-management", tags=["data-management"])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_PATHS.data_root))
    except ValueError:
        return str(path)


def _directory_size_summary() -> dict:
    """Build a bounded filesystem inventory rooted only at MACD_DATA_ROOT."""
    root = PROJECT_PATHS.data_root
    categories: dict[str, dict[str, int]] = {}
    cleanup: dict[str, dict[str, int]] = {}
    largest: list[tuple[int, Path, float]] = []
    total_bytes = 0
    total_files = 0

    if root.exists():
        pending = [root]
        while pending:
            directory = pending.pop()
            try:
                entries = list(os.scandir(directory))
            except OSError:
                continue
            for entry in entries:
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                try:
                    stat = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                size = stat.st_size
                total_bytes += size
                total_files += 1
                relative = path.relative_to(root)
                category = relative.parts[0] if relative.parts else "other"
                bucket = categories.setdefault(category, {"bytes": 0, "files": 0})
                bucket["bytes"] += size
                bucket["files"] += 1

                is_cleanup_candidate = (
                    path.suffix.lower() in {".tmp", ".recovery"}
                    or ".backup_" in path.name
                    or "backup_data" in relative.parts
                    or "archive" in relative.parts
                )
                if is_cleanup_candidate:
                    label = str(Path(*relative.parts[: min(4, len(relative.parts))]))
                    cleanup_bucket = cleanup.setdefault(label, {"bytes": 0, "files": 0})
                    cleanup_bucket["bytes"] += size
                    cleanup_bucket["files"] += 1

                largest.append((size, path, stat.st_mtime))

    largest.sort(key=lambda item: item[0], reverse=True)
    return {
        "dataRoot": str(root),
        "totalBytes": total_bytes,
        "totalFiles": total_files,
        "categories": [
            {"name": name, **values}
            for name, values in sorted(categories.items(), key=lambda item: item[1]["bytes"], reverse=True)
        ],
        "cleanupCandidates": [
            {"path": name, **values}
            for name, values in sorted(cleanup.items(), key=lambda item: item[1]["bytes"], reverse=True)[:20]
        ],
        "largestFiles": [
            {
                "path": _relative(path),
                "bytes": size,
                "modifiedAt": datetime.fromtimestamp(modified_at, tz=timezone.utc).isoformat(),
            }
            for size, path, modified_at in largest[:20]
        ],
        "scannedAt": _utc_now(),
    }


class StorageInventory:
    """Serve the most recent filesystem inventory without blocking API requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._summary: dict | None = None
        self._scanning = False
        self._last_error: str | None = None

    def snapshot(self, force_refresh: bool = False) -> dict:
        with self._lock:
            if force_refresh or self._summary is None:
                self._start_refresh_locked()
            summary = dict(self._summary or {"dataRoot": str(PROJECT_PATHS.data_root), "totalBytes": 0, "totalFiles": 0, "categories": [], "cleanupCandidates": [], "largestFiles": []})
            summary["scanning"] = self._scanning
            summary["scanError"] = self._last_error
            return summary

    def _start_refresh_locked(self) -> None:
        if self._scanning:
            return
        self._scanning = True
        self._last_error = None
        threading.Thread(target=self._refresh, daemon=True, name="data-storage-inventory").start()

    def _refresh(self) -> None:
        try:
            summary = _directory_size_summary()
        except Exception as exc:  # Keep the previous successful snapshot visible.
            LOGGER.exception("Storage inventory scan failed")
            with self._lock:
                self._last_error = str(exc)
                self._scanning = False
            return
        with self._lock:
            self._summary = summary
            self._scanning = False


STORAGE_INVENTORY = StorageInventory()


@dataclass
class ManagedJob:
    id: str
    label: str
    command: list[str]
    started_at: str
    process: subprocess.Popen[str]
    log_path: Path
    status: str = "running"
    finished_at: str | None = None
    return_code: int | None = None
    stop_requested: bool = False
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=250))

    def summary(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "command": self.command,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "status": self.status,
            "returnCode": self.return_code,
            "pid": self.process.pid,
            "logPath": _relative(self.log_path),
            "logs": list(self.logs),
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, ManagedJob] = {}
        self._lock = threading.Lock()

    def start(self, label: str, command: list[str]) -> ManagedJob:
        PROJECT_PATHS.logs_root.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex[:12]
        log_path = PROJECT_PATHS.logs_root / f"data-management-{job_id}.log"
        process = subprocess.Popen(
            command,
            cwd=PROJECT_PATHS.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        job = ManagedJob(job_id, label, command, _utc_now(), process, log_path)
        with self._lock:
            self._jobs[job_id] = job
        threading.Thread(target=self._capture_output, args=(job,), daemon=True, name=f"data-job-{job_id}").start()
        return job

    def _capture_output(self, job: ManagedJob) -> None:
        with job.log_path.open("a", encoding="utf-8") as log_file:
            if job.process.stdout:
                for line in iter(job.process.stdout.readline, ""):
                    log_file.write(line)
                    log_file.flush()
                    with self._lock:
                        job.logs.append(line.rstrip())
            return_code = job.process.wait()
        with self._lock:
            job.return_code = return_code
            job.finished_at = _utc_now()
            job.status = "stopped" if job.stop_requested else "completed" if return_code == 0 else "failed"

    def list(self) -> list[dict]:
        with self._lock:
            jobs = [job.summary() for job in self._jobs.values()]
        return sorted(jobs, key=lambda job: job["startedAt"], reverse=True)

    def stop(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.process.poll() is None:
                job.stop_requested = True
                job.process.terminate()
                job.logs.append("Stop requested from data management console.")
        return job.summary()


JOB_MANAGER = JobManager()


class StartJobRequest(BaseModel):
    task: Literal["pipeline", "microstructure_features", "microstructure_live"]
    asset: Literal["btc", "gold"] = "btc"
    steps: list[int] = Field(default_factory=lambda: [1, 2, 3, 5])
    symbol: str = "BTCUSDT"
    timeframe: str = "1min"
    force: bool = False
    collect_futures: bool = True
    keep_files: int = Field(default=0, ge=0)


def _job_command(request: StartJobRequest) -> tuple[str, list[str]]:
    if request.task == "pipeline":
        allowed_steps = {1, 2, 3, 4, 5}
        steps = sorted(set(request.steps))
        if not steps or any(step not in allowed_steps for step in steps):
            raise HTTPException(status_code=400, detail="Pipeline steps must be selected from 1 through 5.")
        command = [sys.executable, "-m", "data_pipeline.pipeline_runner", "--asset", request.asset, "--steps", *(str(step) for step in steps)]
        if request.force:
            command.append("--force")
        if not request.collect_futures:
            command.append("--no-futures")
        return f"{request.asset.upper()} pipeline ({', '.join(map(str, steps))})", command

    if request.task == "microstructure_features":
        return (
            f"Microstructure features {request.symbol.upper()} {request.timeframe}",
            [
                sys.executable,
                "-m",
                "data_pipeline.microstructure.features",
                "--symbol",
                request.symbol.upper(),
                "--timeframe",
                request.timeframe,
            ],
        )

    return (
        f"Microstructure live {request.symbol.upper()}",
        [
            sys.executable,
            "-m",
            "data_pipeline.pipeline_runner",
            "--asset",
            "btc",
            "--steps",
            "6",
            "--microstructure-symbol",
            request.symbol.upper(),
            "--microstructure-timeframe",
            request.timeframe,
            "--microstructure-keep-files",
            str(request.keep_files),
        ],
    )


@router.get("/storage")
def storage_summary() -> dict:
    return STORAGE_INVENTORY.snapshot()


@router.post("/storage/refresh")
def refresh_storage_summary() -> dict:
    return STORAGE_INVENTORY.snapshot(force_refresh=True)


@router.get("/jobs")
def list_jobs() -> dict:
    return {"jobs": JOB_MANAGER.list()}


@router.post("/jobs")
def start_job(request: StartJobRequest) -> dict:
    label, command = _job_command(request)
    try:
        job = JOB_MANAGER.start(label, command)
    except OSError as exc:
        LOGGER.exception("Failed to start managed job")
        raise HTTPException(status_code=500, detail=f"Could not start job: {exc}") from exc
    return {"job": job.summary()}


@router.post("/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict:
    try:
        job = JOB_MANAGER.stop(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    return {"job": job}
