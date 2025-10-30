#!/usr/bin/env bash

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables before running)
# ---------------------------------------------------------------------------
COMPOSE_BIN="${COMPOSE_BIN:-docker compose}"
STREAM_PATH="${STREAM_PATH:-/iris}"
RTSP_HOST="${RTSP_HOST:-localhost}"
RTSP_URL="rtsp://${RTSP_HOST}:8554${STREAM_PATH}"
# avfoundation device indices
AVFOUNDATION_VIDEO="${AVFOUNDATION_VIDEO:-0}"
AVFOUNDATION_AUDIO="${AVFOUNDATION_AUDIO:-0}"
AVFOUNDATION_SOURCE="${AVFOUNDATION_SOURCE:-${AVFOUNDATION_VIDEO}:${AVFOUNDATION_AUDIO}}"
FRAME_RATE="${FRAME_RATE:-30.0}"
VIDEO_SIZE="${VIDEO_SIZE:-640x480}"
VIDEO_BITRATE="${VIDEO_BITRATE:-1500k}"
AUDIO_BITRATE="${AUDIO_BITRATE:-128k}"
AUDIO_RATE="${AUDIO_RATE:-44100}"
RTSP_BUFSIZE="${RTSP_BUFSIZE:-1M}"
RTSP_MAX_DELAY="${RTSP_MAX_DELAY:-0}"
VIDEO_TUNE="${VIDEO_TUNE:-zerolatency}"
VIDEO_PRESET="${VIDEO_PRESET:-ultrafast}"

FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"

# Optional flags
UP_BUILD=""
if [[ "${1:-}" == "--build" ]]; then
    UP_BUILD="--build"
    shift
fi

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
cleanup() {
    local exit_code=$?
    echo "[run_iris] Cleaning up (exit code ${exit_code})"
    if [[ -n "${PUBLISH_PID:-}" ]] && kill -0 "${PUBLISH_PID}" 2>/dev/null; then
        kill "${PUBLISH_PID}" || true
        wait "${PUBLISH_PID}" || true
    fi
    ${COMPOSE_BIN} down --remove-orphans
}
trap cleanup EXIT

echo "[run_iris] Starting RTSP relay (mediamtx)…"
if [[ -n "${UP_BUILD}" ]]; then
    ${COMPOSE_BIN} up ${UP_BUILD} -d rtsp
else
    ${COMPOSE_BIN} up -d rtsp
fi

echo "[run_iris] Launching ffmpeg publisher from avfoundation device ${AVFOUNDATION_SOURCE} → ${RTSP_URL}"
${FFMPEG_BIN} \
    -f avfoundation \
    -framerate "${FRAME_RATE}" \
    -pixel_format uyvy422 \
    -video_size "${VIDEO_SIZE}" \
    -i "${AVFOUNDATION_SOURCE}" \
    -c:v h264_videotoolbox \
    -b:v "${VIDEO_BITRATE}" \
    -profile:v baseline \
    -tune "${VIDEO_TUNE}" \
    -preset "${VIDEO_PRESET}" \
    -pix_fmt yuv420p \
    -c:a aac \
    -b:a "${AUDIO_BITRATE}" \
    -ar "${AUDIO_RATE}" \
    -ac 1 \
    -bufsize "${RTSP_BUFSIZE}" \
    -max_delay "${RTSP_MAX_DELAY}" \
    -f rtsp \
    -rtsp_transport tcp \
    "${RTSP_URL}" &
PUBLISH_PID=$!

sleep 2

echo "[run_iris] Starting Iris container…"
if [[ -n "${UP_BUILD}" ]]; then
    ${COMPOSE_BIN} up ${UP_BUILD} iris
else
    ${COMPOSE_BIN} up iris
fi
