#!/bin/bash
# flatten_on_shutdown.sh — Graceful position flatten triggered on EC2 instance shutdown.
#
# Install as a systemd service so it runs before Docker containers are stopped:
#   sudo cp deploy/flatten_on_shutdown.sh /usr/local/bin/
#   sudo cp deploy/quant_bot_flatten.service /etc/systemd/system/
#   sudo systemctl enable quant_bot_flatten
#
# The script sends a SIGTERM to the quant_bot container and waits for it to
# emit a "positions_flat" log line (written by the bot's shutdown handler).
# If the bot doesn't flatten within FLATTEN_TIMEOUT_SECS, the script logs a
# warning and allows the shutdown to proceed anyway.

set -euo pipefail

CONTAINER_NAME="quant_bot"
FLATTEN_TIMEOUT_SECS=120      # max wait for position closure
POLL_INTERVAL_SECS=5
LOG_FILE="/var/log/quant_bot_shutdown.log"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG_FILE"; }

log "=== Flatten-on-shutdown triggered ==="
log "Container: $CONTAINER_NAME | Timeout: ${FLATTEN_TIMEOUT_SECS}s"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log "Container not running — nothing to flatten."
    exit 0
fi

# Send SIGTERM to trigger graceful shutdown in the bot
log "Sending SIGTERM to $CONTAINER_NAME..."
docker kill --signal=SIGTERM "$CONTAINER_NAME" 2>/dev/null || true

# Poll for "positions_flat" log entry (bot writes this after closing all positions)
elapsed=0
while [ $elapsed -lt $FLATTEN_TIMEOUT_SECS ]; do
    if docker logs --tail=50 "$CONTAINER_NAME" 2>&1 | grep -q "positions_flat"; then
        log "Bot confirmed positions flat after ${elapsed}s."
        break
    fi

    # Also check if container exited cleanly
    STATUS=$(docker inspect --format='{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "gone")
    if [ "$STATUS" = "exited" ] || [ "$STATUS" = "gone" ]; then
        log "Container exited (status=$STATUS) after ${elapsed}s."
        break
    fi

    sleep $POLL_INTERVAL_SECS
    elapsed=$((elapsed + POLL_INTERVAL_SECS))
done

if [ $elapsed -ge $FLATTEN_TIMEOUT_SECS ]; then
    log "WARNING: Flatten timeout after ${FLATTEN_TIMEOUT_SECS}s — forcing container stop."
    docker stop --time=10 "$CONTAINER_NAME" 2>/dev/null || true
fi

log "=== Flatten-on-shutdown complete ==="
exit 0
