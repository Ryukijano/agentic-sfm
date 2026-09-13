#!/usr/bin/env python3
"""Real-time GPU monitoring dashboard for AIRE cluster.

Polls squeue/sinfo for job/node status and nvidia-smi on running jobs
for GPU utilization. Serves a web dashboard on http://localhost:8080.

Usage:
    python scripts/gpu_monitor.py [--port 8080]

Requirements: fastapi, uvicorn (pip install fastapi uvicorn)
"""

import asyncio
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _run(cmd: str, timeout: int = 10) -> str:
    """Run a shell command and return stdout."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return ""
    except Exception:
        return ""


def get_gpu_nodes() -> list[dict[str, Any]]:
    """Get GPU node status from sinfo."""
    out = _run("sinfo -p gpu --format='%N %G %D %t %f' --noheader")
    nodes = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        # Expand node ranges like gpu[001-003] into individual nodes
        node_str = parts[0]
        if "[" in node_str:
            # Parse range: gpu[001-003,005] -> gpu001, gpu002, gpu003, gpu005
            base = node_str.split("[")[0]
            ranges = node_str.split("[")[1].rstrip("]").split(",")
            for r in ranges:
                if "-" in r:
                    start, end = r.split("-")
                    for i in range(int(start), int(end) + 1):
                        nodes.append({
                            "name": f"{base}{i:03d}",
                            "gres": parts[1],
                            "nodes": int(parts[2]),
                            "state": parts[3],
                            "features": parts[4] if len(parts) > 4 else "",
                        })
                else:
                    nodes.append({
                        "name": f"{base}{r}",
                        "gres": parts[1],
                        "nodes": int(parts[2]),
                        "state": parts[3],
                        "features": parts[4] if len(parts) > 4 else "",
                    })
        else:
            nodes.append({
                "name": node_str,
                "gres": parts[1],
                "nodes": int(parts[2]),
                "state": parts[3],
                "features": parts[4] if len(parts) > 4 else "",
            })
    return nodes


def get_my_jobs() -> list[dict[str, Any]]:
    """Get user's Slurm jobs."""
    out = _run("squeue --me --format='%i|%P|%j|%u|%T|%M|%D|%R|%b|%S|%e' --noheader")
    jobs = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 8:
            continue
        jobs.append({
            "job_id": parts[0],
            "partition": parts[1],
            "name": parts[2],
            "user": parts[3],
            "state": parts[4],
            "time": parts[5],
            "nodes": int(parts[6]),
            "nodelist": parts[7] if len(parts) > 7 else "",
            "gres": parts[8] if len(parts) > 8 else "",
            "start_time": parts[9] if len(parts) > 9 else "",
            "end_time": parts[10] if len(parts) > 10 else "",
        })
    return jobs


def get_gpu_utilization(job_id: str) -> list[dict[str, Any]]:
    """Get GPU utilization for a running job via srun."""
    out = _run(
        f"srun --jobid={job_id} --overlap nvidia-smi "
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw "
        "--format=csv,noheader,nounits",
        timeout=15,
    )
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "utilization": int(parts[2]),
            "memory_used_mb": int(parts[3]),
            "memory_total_mb": int(parts[4]),
            "memory_pct": round(int(parts[3]) / max(int(parts[4]), 1) * 100, 1),
            "temperature_c": int(parts[5]),
            "power_w": float(parts[6]),
        })
    return gpus


def get_node_gpu_info(node: str) -> list[dict[str, Any]]:
    """Get GPU info for a specific node via srun."""
    out = _run(
        f"srun --nodelist={node} --nodes=1 nvidia-smi "
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw "
        "--format=csv,noheader,nounits",
        timeout=15,
    )
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "utilization": int(parts[2]),
            "memory_used_mb": int(parts[3]),
            "memory_total_mb": int(parts[4]),
            "memory_pct": round(int(parts[3]) / max(int(parts[4]), 1) * 100, 1),
            "temperature_c": int(parts[5]),
            "power_w": float(parts[6]),
        })
    return gpus


def get_job_log_tail(job_id: str, n_lines: int = 20) -> str:
    """Get the last N lines of a job's log file."""
    log_dir = Path("/scratch/kcwp264/logs/agentic-sfm")
    for pattern in [f"*_{job_id}.out", f"*_{job_id}.err"]:
        files = list(log_dir.glob(pattern))
        if files:
            try:
                lines = files[0].read_text().splitlines()
                return "\n".join(lines[-n_lines:])
            except Exception:
                pass
    return ""


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<title>AIRE GPU Monitor</title>
<meta charset="utf-8">
<meta http-equiv="refresh" content="10">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }
  h1 { color: #58a6ff; margin-bottom: 20px; font-size: 24px; }
  h2 { color: #8b949e; margin: 20px 0 10px; font-size: 16px; text-transform: uppercase; letter-spacing: 1px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }
  .card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 16px; }
  .card-title { color: #58a6ff; font-size: 14px; font-weight: bold; margin-bottom: 8px; }
  .gpu-bar { background: #21262d; border-radius: 4px; height: 20px; margin: 6px 0; position: relative; overflow: hidden; }
  .gpu-bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
  .gpu-bar-fill.low { background: #238636; }
  .gpu-bar-fill.med { background: #d29922; }
  .gpu-bar-fill.high { background: #f85149; }
  .gpu-bar-label { position: absolute; top: 0; left: 8px; line-height: 20px; font-size: 11px; color: #fff; }
  .stat { display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px; }
  .stat-label { color: #8b949e; }
  .stat-value { color: #c9d1d9; font-weight: bold; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; }
  .badge.running { background: #238636; color: #fff; }
  .badge.pending { background: #d29922; color: #000; }
  .badge.failed { background: #f85149; color: #fff; }
  .badge.completed { background: #58a6ff; color: #fff; }
  .badge.alloc { background: #238636; color: #fff; }
  .badge.mix { background: #d29922; color: #000; }
  .badge.idle { background: #484f58; color: #fff; }
  .badge.drain, .badge.down { background: #f85149; color: #fff; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: #8b949e; padding: 8px; border-bottom: 1px solid #21262d; font-weight: normal; }
  td { padding: 8px; border-bottom: 1px solid #21262d; }
  .log-box { background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 12px; font-size: 12px; white-space: pre-wrap; max-height: 300px; overflow-y: auto; color: #8b949e; }
  .refresh { color: #484f58; font-size: 12px; margin-top: 20px; }
</style>
</head>
<body>
<h1>🖥️ AIRE GPU Monitor</h1>
<div id="timestamp" style="color:#484f58;font-size:12px;margin-bottom:16px"></div>

<h2>My Jobs</h2>
<div id="jobs"></div>

<h2>GPU Nodes</h2>
<div class="grid" id="nodes"></div>

<h2>Running Job GPU Details</h2>
<div id="gpu-details"></div>

<h2>Latest Logs</h2>
<div id="logs"></div>

<div class="refresh">Auto-refreshes every 10 seconds | <a href="/api/status" style="color:#58a6ff">JSON API</a></div>

<script>
fetch('/api/status').then(r => r.json()).then(data => {
  document.getElementById('timestamp').textContent = 'Last update: ' + data.timestamp;

  // Jobs table
  let jobsHtml = '<table><tr><th>ID</th><th>Name</th><th>State</th><th>Time</th><th>Node</th><th>GPUs</th></tr>';
  for (const j of data.jobs) {
    const badge = j.state === 'RUNNING' ? 'running' : j.state === 'PENDING' ? 'pending' : 'failed';
    jobsHtml += `<tr><td>${j.job_id}</td><td>${j.name}</td><td><span class="badge ${badge}">${j.state}</span></td><td>${j.time}</td><td>${j.nodelist}</td><td>${j.gres}</td></tr>`;
  }
  jobsHtml += '</table>';
  document.getElementById('jobs').innerHTML = jobsHtml;

  // GPU nodes
  let nodesHtml = '';
  for (const n of data.nodes) {
    const stateClass = n.state.replace(/[^a-z]/g, '');
    nodesHtml += `<div class="card"><div class="card-title">${n.name} <span class="badge ${stateClass}">${n.state}</span></div><div class="stat"><span class="stat-label">GPUs</span><span class="stat-value">${n.gres}</span></div></div>`;
  }
  document.getElementById('nodes').innerHTML = nodesHtml;

  // GPU details for running jobs
  let gpuHtml = '';
  for (const job of data.jobs) {
    if (job.state !== 'RUNNING' || !job.gpu_details) continue;
    gpuHtml += `<div class="card" style="margin-bottom:12px"><div class="card-title">Job ${job.job_id} — ${job.name} (${job.nodelist})</div>`;
    for (const gpu of job.gpu_details) {
      const pct = gpu.memory_pct;
      const cls = pct < 50 ? 'low' : pct < 80 ? 'med' : 'high';
      gpuHtml += `<div class="stat"><span class="stat-label">GPU ${gpu.index}: ${gpu.name}</span><span class="stat-value">${gpu.utilization}% util | ${gpu.memory_used_mb}/${gpu.memory_total_mb}MB | ${gpu.temperature_c}°C | ${gpu.power_w}W</span></div>`;
      gpuHtml += `<div class="gpu-bar"><div class="gpu-bar-fill ${cls}" style="width:${pct}%"></div><div class="gpu-bar-label">VRAM ${pct}%</div></div>`;
    }
    gpuHtml += '</div>';
  }
  document.getElementById('gpu-details').innerHTML = gpuHtml || '<div class="card">No running jobs</div>';

  // Logs
  let logsHtml = '';
  for (const job of data.jobs) {
    if (job.log_tail) {
      logsHtml += `<h3 style="color:#8b949e;margin:8px 0 4px;font-size:13px">Job ${job.job_id} (${job.name})</h3><div class="log-box">${job.log_tail}</div>`;
    }
  }
  document.getElementById('logs').innerHTML = logsHtml;
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app():
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError:
        print("Install fastapi and uvicorn: pip install fastapi uvicorn")
        raise

    app = FastAPI(title="AIRE GPU Monitor")

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return DASHBOARD_HTML

    @app.get("/api/status")
    def api_status():
        nodes = get_gpu_nodes()
        jobs = get_my_jobs()

        # Get GPU details for running jobs
        for job in jobs:
            if job["state"] == "RUNNING":
                job["gpu_details"] = get_gpu_utilization(job["job_id"])
                job["log_tail"] = get_job_log_tail(job["job_id"])

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "nodes": nodes,
            "jobs": jobs,
        }

    return app


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    app = create_app()

    import uvicorn
    print(f"GPU Monitor: http://{args.host}:{args.port}")
    print(f"API: http://{args.host}:{args.port}/api/status")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
