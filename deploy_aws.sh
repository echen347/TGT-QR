#!/bin/bash

# TGT QR Trading System AWS Deployment Script
# Run this script on your AWS server after SSH access

echo "🚀 Deploying TGT QR Trading System to AWS..."
echo "📊 Risk Management: 50¢ per trade, $5 daily limit, 10x leverage"
echo ""

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3 and pip
sudo apt install -y python3 python3-pip python3-venv git

# Install system dependencies
sudo apt install -y build-essential libssl-dev libffi-dev

# Create trading user (for security)
sudo useradd -m -s /bin/bash trading
sudo usermod -aG sudo trading

# Clone repository
git clone https://github.com/echen347/TGT-QR.git /home/trading/tgt-qr
cd /home/trading/tgt-qr

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip3 install -r requirements.txt

# Create necessary directories
mkdir -p logs data

# Set proper permissions
chmod +x src/run_trading_system.py
chmod +x deploy_aws.sh

# Create .env file from example (you'll need to add your API keys)
cp .env.example .env

# Create systemd service file for auto-start
sudo tee /etc/systemd/system/tgt-trading.service > /dev/null <<EOF
[Unit]
Description=TGT QR Trading System - 50¢ per trade, 10x leverage
After=network.target

[Service]
Type=simple
User=trading
WorkingDirectory=/home/trading/tgt-qr
ExecStart=/home/trading/tgt-qr/venv/bin/python3 src/run_trading_system.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment=PATH=/home/trading/tgt-qr/venv/bin

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable tgt-trading.service
sudo systemctl start tgt-trading.service

# Setup log rotation
sudo tee /etc/logrotate.d/tgt-trading > /dev/null <<EOF
/home/trading/tgt-qr/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 trading trading
}
EOF

# Setup firewall for dashboard access
sudo ufw allow 5000

echo "✅ Deployment completed!"
echo ""
echo "📊 Dashboard will be available at: http://$(curl -s ifconfig.me):5000"
echo ""
echo "🔧 SETUP REQUIRED:"
echo "  1. Edit /home/trading/tgt-qr/.env file"
echo "     - Add your BYBIT_API_KEY"
echo "     - Add your BYBIT_API_SECRET"
echo "     - Set BYBIT_TESTNET=false for live trading"
echo ""
echo "🔧 USEFUL COMMANDS:"
echo "  sudo systemctl status tgt-trading.service"
echo "  sudo systemctl stop tgt-trading.service"
echo "  sudo systemctl restart tgt-trading.service"
echo "  sudo journalctl -u tgt-trading.service -f"
echo ""
echo "⚠️  IMPORTANT REMINDERS:"
echo "  1. Monitor the system closely for the first few days"
echo "  2. Use the dashboard shutdown button in emergencies"
echo "  3. Check logs regularly: sudo journalctl -u tgt-trading.service"
echo "  4. Risk limits: 50¢ per trade, $5 daily limit, $10 total limit"
echo ""
echo "🎯 System is now running with ultra-conservative risk management!"
