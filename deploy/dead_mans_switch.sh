#!/bin/bash
# dead_mans_switch.sh — Telegram dead-man's-switch heartbeat monitor.
#
# Checks whether the quant_execution container has logged recently.
# If no log output in the last STALE_THRESHOLD_SECS, sends a Telegram
# alert to the admin chat.
#
# Install via cron (every 30 min):
#   */30 * * * * /usr/local/bin/dead_mans_switch.sh
#
# Requires: TELEGRAM_TOKEN and ADMIN_ID env vars (or set below).
# Can also be run as a GitHub Actions scheduled workflow or Lambda.

set -euo pipefail

CONTAINER_NAME="${QUANT_CONTAINER_NAME:-quant_execution}"
STALE_THRESHOLD_SECS="${DMS_STALE_THRESHOLD:-7200}"  # 2 hours
TELEGRAM_TOKEN="${TELEGRAM_TOKEN:-}"
ADMIN_CHAT_ID="${ADMIN_ID:-}"

if [ -z "$TELEGRAM_TOKEN" ] || [ -z "$ADMIN_CHAT_ID" ]; then
    echo "[dead-mans-switch] TELEGRAM_TOKEN or ADMIN_ID not set — skipping."
    exit 0
fi

send_alert() {
    local msg="$1"
    curl -sf -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
        -d "chat_id=${ADMIN_CHAT_ID}" \
        -d "text=${msg}" \
        -d "parse_mode=HTML" >/dev/null 2>&1 || true
}

# Check if container exists and is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    send_alert "🚨 <b>Dead-Man's Switch</b>: Container <code>${CONTAINER_NAME}</code> is NOT running!"
    exit 1
fi

# Get the timestamp of the last log line
LAST_LOG_TS=$(docker logs --tail=1 --timestamps "$CONTAINER_NAME" 2>&1 | head -1 | grep -oP '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}' || echo "")

if [ -z "$LAST_LOG_TS" ]; then
    send_alert "⚠️ <b>Dead-Man's Switch</b>: Cannot read logs from <code>${CONTAINER_NAME}</code>."
    exit 1
fi

# Convert to epoch
LAST_EPOCH=$(date -d "$LAST_LOG_TS" +%s 2>/dev/null || echo 0)
NOW_EPOCH=$(date +%s)
AGE=$((NOW_EPOCH - LAST_EPOCH))

if [ "$AGE" -gt "$STALE_THRESHOLD_SECS" ]; then
    AGE_HOURS=$((AGE / 3600))
    AGE_MINS=$(((AGE % 3600) / 60))
    send_alert "🚨 <b>Dead-Man's Switch</b>: <code>${CONTAINER_NAME}</code> last log was <b>${AGE_HOURS}h ${AGE_MINS}m ago</b>. Possible hang or crash."
    exit 1
fi

echo "[dead-mans-switch] OK — last log ${AGE}s ago (threshold: ${STALE_THRESHOLD_SECS}s)"
exit 0
