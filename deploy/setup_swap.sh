#!/bin/bash
# setup_swap.sh — Create a 2 GB swap file on EC2 c7i-flex.large (2 vCPU, 4 GB RAM)
# Run once after instance launch (or re-run; idempotent).
# Usage: sudo bash deploy/setup_swap.sh
set -euo pipefail

SWAP_FILE="/swapfile"
SWAP_SIZE_MB=2048  # 2 GB

if swapon --show | grep -q "$SWAP_FILE"; then
    echo "[swap] Swap already active at $SWAP_FILE — skipping."
    swapon --show
    exit 0
fi

echo "[swap] Creating ${SWAP_SIZE_MB}MB swap at $SWAP_FILE..."
fallocate -l "${SWAP_SIZE_MB}M" "$SWAP_FILE"
chmod 600 "$SWAP_FILE"
mkswap "$SWAP_FILE"
swapon "$SWAP_FILE"

# Persist across reboots
if ! grep -q "$SWAP_FILE" /etc/fstab; then
    echo "$SWAP_FILE none swap sw 0 0" >> /etc/fstab
fi

# Reduce swappiness: only use swap under memory pressure
sysctl vm.swappiness=10
echo "vm.swappiness=10" >> /etc/sysctl.conf

echo "[swap] Done. Current swap:"
swapon --show
free -h
