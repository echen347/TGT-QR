#!/bin/bash

# TGT QR Trading System AWS Deployment Script
# Run this script on your AWS server after SSH access

echo "🚀 Deploying TGT QR Trading System to AWS..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3 and pip
sudo apt install -y python3 python3-pip python3-venv git

# Install system dependencies
sudo apt install -y build-essential libssl-dev libffi-dev

# Create trading user (for security)
sudo useradd -m -s /bin/bash trading
sudo usermod -aG sudo trading

# Clone repository (replace with your actual repo URL)
# git clone https://github.com/your-username/tgt-qr-trading.git /home/trading/tgt-qr
# cd /home/trading/tgt-qr

# For now, we'll work in the current directory
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

# Create systemd service file for auto-start
sudo tee /etc/systemd/system/tgt-trading.service > /dev/null <<EOF
[Unit]
Description=TGT QR Trading System
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

echo "✅ Deployment completed!"
echo ""
echo "📊 Dashboard will be available at: http://$(curl -s ifconfig.me):5000"
echo ""
echo "🔧 Useful commands:"
echo "  sudo systemctl status tgt-trading.service"
echo "  sudo systemctl stop tgt-trading.service"
echo "  sudo systemctl restart tgt-trading.service"
echo "  sudo journalctl -u tgt-trading.service -f"
echo ""
echo "⚠️  Remember to:"
echo "  1. Add your API keys to /home/trading/tgt-qr/.env"
echo "  2. Set BYBIT_TESTNET=false for live trading"
echo "  3. Monitor the system closely initially"
echo "  4. Setup firewall rules for port 5000 if needed"
