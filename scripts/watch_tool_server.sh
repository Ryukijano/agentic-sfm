#!/usr/bin/env bash
# Watch a FastAPI tool server /health endpoint.
#
# If health fails for FAIL_THRESHOLD consecutive probes (or the process dies),
# restart the server. If it cannot be restored within MAX_RESTARTS, kill
# PROTECT_PID (the training process) so the Slurm job fails instead of
# spinning for hours on Connection refused.
#
# Job 7882190: pycolmap SIGBUS in FAISS FeatureMatcherWorker killed uvicorn
# at 08:17; training kept rolling until wall-time with 0 registered images.
set -u

usage() {
    cat <<'EOF'
Usage: watch_tool_server.sh --port PORT --project-root DIR --protect-pid PID
                            [--pid-file FILE] [--gpu N] [--interval SEC]
                            [--fail-threshold N] [--max-restarts N]
EOF
    exit 2
}

PORT=""
PROJECT_ROOT=""
PROTECT_PID=""
PID_FILE=""
GPU=2
INTERVAL=10
FAIL_THRESHOLD=3
MAX_RESTARTS=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --project-root) PROJECT_ROOT="$2"; shift 2 ;;
        --protect-pid) PROTECT_PID="$2"; shift 2 ;;
        --pid-file) PID_FILE="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --fail-threshold) FAIL_THRESHOLD="$2"; shift 2 ;;
        --max-restarts) MAX_RESTARTS="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1"; usage ;;
    esac
done

[[ -n "${PORT}" && -n "${PROJECT_ROOT}" && -n "${PROTECT_PID}" ]] || usage
PID_FILE="${PID_FILE:-${PROJECT_ROOT}/outputs/.tool_server_${PORT}.pid}"

log() { echo "$(date -Is) [tool-watchdog] $*"; }

read_pid() {
    if [[ -f "${PID_FILE}" ]]; then
        cat "${PID_FILE}" 2>/dev/null || true
    fi
}

server_alive() {
    local pid
    pid="$(read_pid)"
    [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

health_ok() {
    curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1
}

start_server() {
    CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        python "${PROJECT_ROOT}/tools_server/server.py" --port "${PORT}" &
    local pid=$!
    echo "${pid}" > "${PID_FILE}"
    log "started tool server pid=${pid} port=${PORT} gpu=${GPU}"
    local i
    for i in $(seq 1 60); do
        if health_ok; then
            log "tool server healthy after $((i))s"
            return 0
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            log "tool server pid=${pid} died before /health bound"
            return 1
        fi
        sleep 1
    done
    log "tool server pid=${pid} did not become healthy in 60s"
    return 1
}

stop_server() {
    local pid
    pid="$(read_pid)"
    if [[ -n "${pid}" ]]; then
        log "stopping tool server pid=${pid}"
        kill "${pid}" 2>/dev/null || true
        local i
        for i in $(seq 1 20); do
            kill -0 "${pid}" 2>/dev/null || break
            sleep 0.5
        done
        kill -9 "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    fi
}

fails=0
restarts=0
log "watching :${PORT} protect_pid=${PROTECT_PID} interval=${INTERVAL}s fail_threshold=${FAIL_THRESHOLD} max_restarts=${MAX_RESTARTS}"

while kill -0 "${PROTECT_PID}" 2>/dev/null; do
    if health_ok; then
        fails=0
    else
        fails=$((fails + 1))
        alive="dead"
        server_alive && alive="alive-but-unhealthy"
        log "health fail ${fails}/${FAIL_THRESHOLD} (server ${alive} pid=$(read_pid))"
        if [[ ${fails} -ge ${FAIL_THRESHOLD} ]]; then
            restarts=$((restarts + 1))
            if [[ ${restarts} -gt ${MAX_RESTARTS} ]]; then
                log "unrestorable after ${MAX_RESTARTS} restarts — killing training pid=${PROTECT_PID}"
                kill -TERM "${PROTECT_PID}" 2>/dev/null || true
                sleep 5
                kill -KILL "${PROTECT_PID}" 2>/dev/null || true
                exit 2
            fi
            log "restarting tool server (${restarts}/${MAX_RESTARTS})"
            stop_server
            if start_server; then
                fails=0
            else
                log "restart failed"
            fi
        fi
    fi
    sleep "${INTERVAL}"
done

log "protect_pid=${PROTECT_PID} exited — watchdog stopping"
exit 0
