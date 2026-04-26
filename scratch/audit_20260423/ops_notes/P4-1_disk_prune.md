# P4-1 Disk Reclamation Ops Notes

**Status**: Awaiting user approval for remote execution  
**Host**: ubuntu@13.48.85.88 (EC2 t3.medium)  
**SSH Key**: `C:\Users\HP\Downloads\hypothesis-research-engine\quant-key.pem`

## Pre-Check Commands (Safe to Run)

These commands inspect state without making changes:

```bash
# Check disk usage
ssh -i ./quant-key.pem ubuntu@13.48.85.88 "df -h /"

# Check Docker reclaimable space
ssh -i ./quant-key.pem ubuntu@13.48.85.88 "docker system df"

# List running containers
ssh -i ./quant-key.pem ubuntu@13.48.85.88 "docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'"
```

## Destructive Commands (Require Explicit User Approval)

**DANGEROUS**: The following commands will remove containers and images.

### 1. Stop and Remove Execution Container (P3-3)

```bash
ssh -i ./quant-key.pem ubuntu@13.48.85.88 "docker stop quant_execution && docker rm quant_execution"
```

### 2. Prune Docker Images (P4-1)

Expected reclaim: ~19.86 GB per audit.

```bash
# Remove all unused images
ssh -i ./quant-key.pem ubuntu@13.48.85.88 "docker image prune -a -f"

# Prune builder cache
ssh -i ./quant-key.pem ubuntu@13.48.85.88 "docker builder prune -f"
```

### 3. SQLite WAL Checkpoint (P4-2)

```bash
ssh -i ./quant-key.pem ubuntu@13.48.85.88 \
  "sqlite3 /home/ubuntu/quant_bot/state/quant_bot.db 'PRAGMA wal_checkpoint(TRUNCATE); VACUUM;'"
```

## Verification Commands (Safe to Run)

```bash
# Verify compose no longer lists execution_engine
ssh -i ./quant-key.pem ubuntu@13.48.85.88 "cd /home/ubuntu/quant_bot && docker compose config --services"

# Post-prune disk status
ssh -i ./quant-key.pem ubuntu@13.48.85.88 "df -h /"
ssh -i ./quant-key.pem ubuntu@13.48.85.88 "docker system df"

# Verify expected containers still running
ssh -i ./quant-key.pem ubuntu@13.48.85.88 "docker ps -a"
```

## Recommended Weekly Cron (Not Implemented)

```
0 3 * * 0 /usr/bin/docker image prune -a -f 2>&1 | logger -t docker-prune
```

---

**Execute each command only after explicit user approval.**
