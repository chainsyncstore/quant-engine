#!/bin/bash
# setup_cloudwatch.sh — Install CloudWatch agent on EC2 and push:
#   - Container CPU/mem (docker stats)
#   - System swap usage
#   - Custom metric: quant_bot heartbeat (written by the bot every cycle)
# Usage: sudo bash deploy/setup_cloudwatch.sh
set -euo pipefail

AWS_REGION="${AWS_REGION:-eu-north-1}"
CW_NAMESPACE="QuantBot"
INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "local")

echo "[cloudwatch] Region: $AWS_REGION  Instance: $INSTANCE_ID"

# --- Install CloudWatch agent if needed ---
if ! command -v amazon-cloudwatch-agent-ctl &>/dev/null; then
    echo "[cloudwatch] Installing CloudWatch agent..."
    wget -q https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
    dpkg -i amazon-cloudwatch-agent.deb
    rm -f amazon-cloudwatch-agent.deb
fi

# --- Write agent config ---
mkdir -p /opt/aws/amazon-cloudwatch-agent/etc/
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'EOF'
{
  "agent": {
    "metrics_collection_interval": 60,
    "run_as_user": "root"
  },
  "metrics": {
    "namespace": "QuantBot",
    "append_dimensions": {
      "InstanceId": "${aws:InstanceId}"
    },
    "metrics_collected": {
      "mem": {
        "measurement": ["mem_used_percent", "mem_available_percent"],
        "metrics_collection_interval": 60
      },
      "swap": {
        "measurement": ["swap_used_percent"],
        "metrics_collection_interval": 60
      },
      "cpu": {
        "measurement": ["cpu_usage_idle", "cpu_usage_user", "cpu_usage_system"],
        "metrics_collection_interval": 60,
        "totalcpu": true
      },
      "disk": {
        "measurement": ["disk_used_percent"],
        "resources": ["/"],
        "metrics_collection_interval": 300
      }
    }
  },
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/app/logs/quant_bot.log",
            "log_group_name": "/quant_bot/app",
            "log_stream_name": "{instance_id}",
            "timezone": "UTC"
          }
        ]
      }
    }
  }
}
EOF

# --- Start agent ---
amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
  -s

echo "[cloudwatch] Agent started. Sending metrics to $CW_NAMESPACE in $AWS_REGION."

# --- Install custom heartbeat cron (writes to CW every 5 min) ---
HEARTBEAT_SCRIPT="/usr/local/bin/quant_bot_heartbeat.sh"
cat > "$HEARTBEAT_SCRIPT" << HEARTBEAT
#!/bin/bash
REGION="${AWS_REGION}"
NAMESPACE="QuantBot"
INSTANCE_ID=\$(curl -sf http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "local")

# Check if quant_bot container is running
if docker ps --format '{{.Names}}' | grep -q quant_bot; then
    STATUS=1
else
    STATUS=0
fi

aws cloudwatch put-metric-data \
  --region "\$REGION" \
  --namespace "\$NAMESPACE" \
  --metric-name "ContainerHeartbeat" \
  --value "\$STATUS" \
  --dimensions "InstanceId=\$INSTANCE_ID" \
  --unit Count 2>/dev/null || true
HEARTBEAT

chmod +x "$HEARTBEAT_SCRIPT"

# Add cron entry if not already present
(crontab -l 2>/dev/null | grep -v quant_bot_heartbeat; echo "*/5 * * * * $HEARTBEAT_SCRIPT") | crontab -

echo "[cloudwatch] Heartbeat cron installed (every 5 min)."
echo "[cloudwatch] Setup complete."
