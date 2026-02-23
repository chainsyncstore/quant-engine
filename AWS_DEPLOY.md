# ☁️ AWS Free Tier Deployment Guide

Follow these steps to deploy your **Multi-User Trading Bot** on AWS for **Free** (12 months).

## 1. Create an AWS Account
1.  Go to [aws.amazon.com/free](https://aws.amazon.com/free).
2.  Sign up (Credit card required for verification, but you won't be charged if you stick to Free Tier).

## 2. Launch an EC2 Instance (Virtual Server)
1.  Search for **EC2** in the top bar.
2.  Click **Launch Instance** (Orange button).
3.  **Name:** `QuantBot`
4.  **OS Image:** Select **Ubuntu** (Ubuntu Server 22.04 LTS).
5.  **Instance Type:** Select `t2.micro` or `t3.micro` (Look for "Free tier eligible" label).
6.  **Key Pair:**
    - Click "Create new key pair".
    - Name: `quant-key`.
    - Format: `.pem`.
    - **Download the file** (Save it safely!).
7.  **Network Settings:** Check "Allow SSH traffic from Anywhere" (0.0.0.0/0).
8.  **Storage:** 30 GB gp2/gp3 (Free tier limit is 30GB).
9.  Click **Launch Instance**.

## 3. Connect to Your Server
1.  Click on the instance ID to see details.
2.  Copy the **Public IPv4 address** (e.g., `54.123.45.67`).
3.  Open your localized terminal (PowerShell for Windows):
    ```powershell
    # Go to where you saved the key
    cd Downloads
    # Fix permissions (Linux/Mac only), Windows skip this step.
    
    # Connect
    ssh -i "quant-key.pem" ubuntu@54.123.45.67
    # (Type 'yes' if asked)
    ```

## 4. Install Docker (One-Time Setup)
Copy-paste these commands into the AWS terminal:

```bash
# Update
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install docker.io docker-compose -y

# Allow 'ubuntu' user to run docker
sudo usermod -aG docker $USER

# Apply changes (exit and reconnect)
exit
```
**Reconnect:** `ssh -i "quant-key.pem" ubuntu@YOUR_IP`

## 5. Upload Code (The Easiest Way)
Since `scp` can be tricky on Windows, the easiest way involves `git`.

### Option A: Zip and Upload (Advanced)
Use `scp`:
```powershell
scp -i "quant-key.pem" -r "C:\Users\HP\Downloads\hypothesis-research-engine" ubuntu@YOUR_IP:~/quant_bot
```

### Option B: Git (Recommended)
1.  Push your code to a private GitHub repo.
2.  Clone it on the server:
    ```bash
    git clone https://github.com/your-username/your-repo.git quant_bot
    cd quant_bot
    ```

## 6. Configure & Run
1.  **Enter the folder:** `cd quant_bot` (or `cd hypothesis-research-engine` if using scp).
2.  **Create .env:**
    ```bash
    nano .env
    ```
    Paste your variables:
    ```
    TELEGRAM_TOKEN=123...
    ADMIN_ID=987...
    BOT_MASTER_KEY=...
    ```
    (Press `Ctrl+X`, then `Y`, then `Enter` to save).

3.  **Run:**
    ```bash
    docker-compose up -d --build
    ```

4.  **Check Logs:**
    ```bash
    docker-compose logs -f
    ```

5.  **Verify:**
    - Open Telegram -> Start Bot.
    - Check `/status` and `/stats`.
    - Setup Binance credentials using `/setup BINANCE_API_KEY BINANCE_API_SECRET`.

If you see **"Bot is polling..."**, you are LIVE! 🚀
