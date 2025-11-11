#!/bin/bash
# Quick reference commands for AWS EC2 instance management
# Usage: source aws_commands.sh or copy individual commands

# Configuration
KEY_FILE="tgt-qr-key-oct-9.pem"
EC2_HOST="ubuntu@16.171.196.141"

# Check service status
check_status() {
    ssh -i "$KEY_FILE" "$EC2_HOST" "sudo systemctl status tgt-trading.service --no-pager"
}

# View recent logs (last 20 lines)
view_logs() {
    ssh -i "$KEY_FILE" "$EC2_HOST" "sleep 2 && sudo journalctl -u tgt-trading.service -n 30 --no-pager | tail -n 20"
}

# Follow logs in real-time
follow_logs() {
    ssh -i "$KEY_FILE" "$EC2_HOST" "sudo journalctl -u tgt-trading.service -f"
}

# Restart service
restart_service() {
    ssh -i "$KEY_FILE" "$EC2_HOST" "sudo systemctl restart tgt-trading.service"
}

# Stop service
stop_service() {
    ssh -i "$KEY_FILE" "$EC2_HOST" "sudo systemctl stop tgt-trading.service"
}

# Start service
start_service() {
    ssh -i "$KEY_FILE" "$EC2_HOST" "sudo systemctl start tgt-trading.service"
}

# Pull latest code and restart
update_and_restart() {
    ssh -i "$KEY_FILE" "$EC2_HOST" "cd /home/trading/tgt-qr && git pull && sudo systemctl restart tgt-trading.service"
}

# Run database migration
run_migration() {
    ssh -i "$KEY_FILE" "$EC2_HOST" "cd /home/trading/tgt-qr/mft && source venv/bin/activate && python3 tools/migrate_database.py"
}

# Check if service is running
is_running() {
    ssh -i "$KEY_FILE" "$EC2_HOST" "sudo systemctl is-active tgt-trading.service"
}

echo "Available commands:"
echo "  check_status       - Check service status"
echo "  view_logs          - View last 20 log lines"
echo "  follow_logs        - Follow logs in real-time"
echo "  restart_service    - Restart the trading service"
echo "  stop_service       - Stop the trading service"
echo "  start_service      - Start the trading service"
echo "  update_and_restart - Pull latest code and restart"
echo "  run_migration      - Run database migration"
echo "  is_running         - Check if service is active"

