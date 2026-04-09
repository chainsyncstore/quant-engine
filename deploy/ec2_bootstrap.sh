#!/bin/bash
# ec2_bootstrap.sh — Run once on a fresh EC2 c7i-flex.large instance to:
#   1. Set up 2GB swap file
#   2. Install & configure CloudWatch agent
#   3. Install flatten-on-shutdown systemd service
#   4. Install dead-man's-switch cron
#
# Usage (from the project root, after git clone):
#   sudo bash deploy/ec2_bootstrap.sh
#
# Prerequisites: AWS CLI configured with an IAM role that has CloudWatchAgentServerPolicy.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== EC2 Bootstrap starting ==="

# --- 1. Swap ---
echo "[1/4] Setting up swap..."
bash "$SCRIPT_DIR/setup_swap.sh"

# --- 2. CloudWatch ---
echo "[2/4] Setting up CloudWatch..."
bash "$SCRIPT_DIR/setup_cloudwatch.sh"

# --- 3. Flatten-on-shutdown ---
echo "[3/4] Installing flatten-on-shutdown service..."
cp "$SCRIPT_DIR/flatten_on_shutdown.sh" /usr/local/bin/flatten_on_shutdown.sh
chmod +x /usr/local/bin/flatten_on_shutdown.sh
cp "$SCRIPT_DIR/quant_bot_flatten.service" /etc/systemd/system/quant_bot_flatten.service
systemctl daemon-reload
systemctl enable quant_bot_flatten

# --- 4. Dead-man's switch ---
echo "[4/4] Installing dead-man's-switch cron..."
cp "$SCRIPT_DIR/dead_mans_switch.sh" /usr/local/bin/dead_mans_switch.sh
chmod +x /usr/local/bin/dead_mans_switch.sh
(crontab -l 2>/dev/null | grep -v dead_mans_switch; echo "*/30 * * * * /usr/local/bin/dead_mans_switch.sh") | crontab -

echo ""
echo "=== EC2 Bootstrap complete ==="
echo "  Swap:              $(free -h | awk '/^Swap:/{print $2}')"
echo "  CloudWatch agent:  $(systemctl is-active amazon-cloudwatch-agent 2>/dev/null || echo 'check manually')"
echo "  Flatten service:   $(systemctl is-enabled quant_bot_flatten 2>/dev/null || echo 'check manually')"
echo "  Dead-man's switch: cron installed (every 30 min)"
echo ""
echo "Next: docker compose -f docker-compose.prod.yml up -d"
