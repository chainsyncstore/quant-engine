# ☁️ Cloud Deployment Guide

Your **Multi-User Trading Bot** is ready for the cloud. Follow these steps to deploy it on a VPS (Virtual Private Server) like DigitalOcean, Linode, or AWS.

## 1. Prerequisites
- A VPS running **Ubuntu 22.04 LTS** (recommended).
- **Docker** and **Docker Compose** installed on the VPS.
- A **Telegram Bot Token** (from [@BotFather](https://t.me/BotFather)).
- Your **Telegram User ID** (from [@userinfobot](https://t.me/userinfobot)).

## 2. Server Setup (One-Time)
SSH into your VPS and install Docker:
```bash
# Update and install Docker
sudo apt update && sudo apt upgrade -y
sudo apt install docker.io docker-compose -y
sudo usermod -aG docker $USER
# (Log out and back in for permissions to take effect)
```

## 3. Deployment Steps

### A. Copy Files
Upload the project files to your VPS. You can use `scp` or `git`.
Key files needed:
- `Dockerfile`
- `docker-compose.yml`
- `quant/` (The entire source code folder)
- `models/production/` (Your trained models)
- `datasets/snapshots/` (If needed, though api fetches fresh data)

### B. Configure Environment
Create a `.env` file or export variables:

```bash
export TELEGRAM_TOKEN="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
export ADMIN_ID="123456789"
# Generate a secure key: python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
export BOT_MASTER_KEY="YOUR_GENERATED_FERNET_KEY_HERE"
```

### C. Launch
Run the bot in the background:
```bash
docker-compose up -d --build
```

### D. Verify
Check logs:
```bash
docker-compose logs -f
```
You should see: `Bot is polling...`

## 4. User Manual

### Admin Setup
1.  Open your bot in Telegram.
2.  Click **Start**.
3.  You (Admin) will be auto-approved or can approve yourself via console if needed (logic handles admin override).

### Adding Users
1.  Friend sends `/start`.
2.  You get a notification: "New User: JohnDoe [ID: 99999]".
3.  You send: `/approve 99999`.
4.  Friend gets notification: "Access Granted".

### Configuring Keys
1.  User sends: `/setup <email> <api_key> <api_pass>`.
2.  Bot replies: "Credentials saved securely".
3.  User sends: `/start_trading`.

### Monitoring
- `/status` to check engine health.
- `/mode live` to switch to real money.
