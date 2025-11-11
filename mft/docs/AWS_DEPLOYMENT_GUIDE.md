# 🚀 TGT QR Trading System - AWS Deployment Guide

## ⚠️ IMPORTANT SAFETY NOTICE
**Before proceeding:**
1. ✅ **Use Testnet First**: Set `BYBIT_TESTNET=true` in `.env`
2. ✅ **Small Position Sizes**: Start with $0.50 per trade
3. ✅ **Monitor Closely**: Check dashboard frequently
4. ✅ **Emergency Stop**: Use dashboard shutdown button if needed

---

## 📋 Step 1: AWS Account Setup

### 1.1 Login to AWS Console
1. Go to https://aws.amazon.com/console/
2. Sign in with your credentials
3. Select your preferred region (e.g., `us-east-1`)

### 1.2 Create EC2 Instance
1. **Navigate to EC2 Dashboard**:
   - Click "Services" → "EC2" → "Instances"

2. **Launch Instance**:
   - Click "Launch Instance"
   - Choose "Ubuntu Server 22.04 LTS" (free tier eligible)
   - Instance type: `t3.micro` (free tier)
   - Click "Next: Configure Instance Details"

3. **Configure Instance**:
   - Number of instances: `1`
   - Network: Default VPC
   - Subnet: Default subnet
   - Auto-assign Public IP: `Enable`
   - Click "Next: Add Storage"

4. **Storage Configuration**:
   - Keep default 8GB SSD
   - Click "Next: Add Tags"

5. **Add Tags** (Optional):
   ```
   Key: Name
   Value: tgt-qr-trading
   ```
   - Click "Next: Configure Security Group"

6. **Security Group**:
   - Create new security group
   - Add rule for SSH (port 22)
   - Add rule for HTTP (port 5000) for dashboard
   - Source: `0.0.0.0/0` (or restrict to your IP)
   - Click "Review and Launch"

7. **Launch Instance**:
   - Select existing key pair or create new one
   - Download the `.pem` key file (keep it safe!)
   - Click "Launch Instances"

### 1.3 Get Instance Details
1. **Note your Instance ID** (e.g., `i-1234567890abcdef0`)
2. **Get Public IP Address** from the instances list
3. **Download your key pair** (`.pem` file)

---

## 🖥️ Step 2: Connect to Your Server

### 2.1 Set Key Permissions
```bash
# On your local machine (Mac/Linux)
chmod 400 your-key-pair.pem
```

### 2.2 SSH into Your Server
```bash
# Replace with your actual instance IP and key file
ssh -i your-key-pair.pem ubuntu@your-instance-public-ip
```

**Expected output:**
```
Welcome to Ubuntu 22.04.3 LTS
...
ubuntu@ip-xxx-xxx-xxx-xxx:~$
```

---

## 🚀 Step 3: Deploy Trading System

### 3.1 Update System
```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3 python3-pip python3-venv git curl ufw
```

### 3.2 Clone Your Repository
```bash
# Clone your trading system
git clone https://github.com/echen347/TGT-QR.git
cd TGT-QR

# Verify files are there
ls -la
```

### 3.3 Setup Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Create necessary directories
mkdir -p logs data
```

### 3.4 Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit environment file (CRITICAL!)
nano .env
```

**In `.env` file, set:**
```bash
# Bybit API Configuration
BYBIT_TESTNET=true                    # Start with testnet!
BYBIT_API_KEY=your_testnet_api_key    # Your testnet API key
BYBIT_API_SECRET=your_testnet_secret  # Your testnet secret

# Risk Management (Conservative)
MAX_POSITION_USDT=0.50               # 50¢ per trade
MAX_DAILY_LOSS_USDT=5.00             # $5 daily limit
STOP_LOSS_PCT=0.02                   # 2% stop loss
TAKE_PROFIT_PCT=0.04                 # 4% take profit
LEVERAGE=10                          # 10x leverage
```

### 3.5 Setup Firewall
```bash
# Enable firewall and allow dashboard port
sudo ufw enable
sudo ufw allow 5000
sudo ufw allow 22
sudo ufw status
```

### 3.6 Setup Systemd Service
```bash
# Create service file
sudo tee /etc/systemd/system/tgt-trading.service > /dev/null <<EOF
[Unit]
Description=TGT QR Trading System Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/TGT-QR
ExecStart=/home/ubuntu/TGT-QR/venv/bin/python3 src/dashboard.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment=PATH=/home/ubuntu/TGT-QR/venv/bin

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable tgt-trading.service
sudo systemctl start tgt-trading.service
```

### 3.7 Verify Dashboard is Running
```bash
# Check service status
sudo systemctl status tgt-trading.service

# Check logs
sudo journalctl -u tgt-trading.service -f
```

---

## 🌐 Step 4: Access Your Dashboard

### 4.1 Get Your Server IP
```bash
# On your server
curl -s ifconfig.me
# This will show your public IP address
```

### 4.2 Access Dashboard
Open your browser and go to:
```
http://your-server-ip:5000
```

**What you'll see:**
- 📈 **Real-time P&L Charts**
- 📊 **Position Monitoring**
- 📡 **Trading Signals**
- 🛡️ **Risk Status Dashboard**
- ⚙️ **Emergency Controls**

---

## ⚠️ Step 5: Safety First - Test Everything

### 5.1 Test Dashboard Functionality
1. **Check Risk Status**: Should show conservative limits
2. **Test Emergency Stop**: Click the red button
3. **Verify No Trading**: Dashboard should show no active positions

### 5.2 Enable Live Trading (When Ready)
```bash
# Edit .env file on your server
nano /home/ubuntu/TGT-QR/.env

# Change these settings:
BYBIT_TESTNET=false              # Switch to live trading
MAX_POSITION_USDT=0.50          # Keep conservative
MAX_DAILY_LOSS_USDT=5.00        # Keep conservative

# Restart the service
sudo systemctl restart tgt-trading.service
```

### 5.3 Monitor Closely
- **Check dashboard every 15 minutes initially**
- **Monitor AWS costs** (should be ~$10/month for t3.micro)
- **Watch logs**: `sudo journalctl -u tgt-trading.service -f`
- **Use emergency stop** if anything looks wrong

---

## 🔧 Useful Commands

### System Management
```bash
# Check service status
sudo systemctl status tgt-trading.service

# View logs
sudo journalctl -u tgt-trading.service -f

# Restart service
sudo systemctl restart tgt-trading.service

# Stop service
sudo systemctl stop tgt-trading.service

# Check system resources
htop
```

### File Management
```bash
# Navigate to your app
cd /home/ubuntu/TGT-QR

# Check logs
tail -f logs/trading_system.log

# Check database
ls -la data/
```

### Security
```bash
# Check firewall
sudo ufw status

# Update system
sudo apt update && sudo apt upgrade -y

# Check for suspicious activity
sudo journalctl --since "1 hour ago" | grep -i "error\|fail\|exception"
```

---

## 🚨 Emergency Procedures

### If Something Goes Wrong:

1. **Immediate Stop**:
   ```bash
   # Stop the service
   sudo systemctl stop tgt-trading.service
   ```

2. **Check Dashboard**:
   - Go to `http://your-server-ip:5000`
   - Click "🛑 Emergency Stop" button

3. **Check Logs**:
   ```bash
   sudo journalctl -u tgt-trading.service --since "10 minutes ago"
   ```

4. **Manual Position Close** (if needed):
   - Login to Bybit
   - Manually close any open positions

5. **Contact Support**:
   - Check AWS console for any issues
   - Review logs for error messages

---

## 💰 Cost Management

### Free Tier Eligible Setup:
- **EC2 t3.micro**: ~$8.50/month after free tier
- **EBS Storage**: 8GB SSD included
- **Data Transfer**: 100GB/month free

### Monitor Costs:
1. Go to AWS Console → "Billing Dashboard"
2. Set up billing alerts
3. Use "Cost Explorer" to track usage

---

## 🎯 Next Steps

1. ✅ **Complete this setup**
2. ⏳ **Monitor dashboard for 24-48 hours**
3. ⏳ **Test with small live trades** (when comfortable)
4. ⏳ **Consider upgrading to full trading system**

---

## 📞 Support

- **Dashboard**: `http://your-server-ip:5000`
- **Logs**: `sudo journalctl -u tgt-trading.service -f`
- **AWS Console**: https://console.aws.amazon.com
- **Your Repository**: https://github.com/echen347/TGT-QR

---

**🎉 Congratulations! Your trading system is now running on AWS with ultra-conservative risk management!**
