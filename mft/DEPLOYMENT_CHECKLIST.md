# Deployment Checklist

## Pre-Deployment

- [ ] **Backtest Complete**: Phase 1 tested with OOS validation
- [ ] **Results Verified**: 7.67 trades/day, 52.2% win rate, 112.93% return
- [ ] **Code Review**: All Phase 1 changes implemented correctly
- [ ] **Config Check**: `config/config.py` has correct Phase 1 parameters
  - [ ] `MIN_TREND_STRENGTH = 0.0002`
  - [ ] Thresholds: 0.2%/0.05%/0.03% (High/Normal/Low vol)
  - [ ] MA slope requirement removed in `strategy.py`

## Deployment Steps

### 1. Commit Changes
```bash
cd mft
git add .
git commit -m "Deploy Phase 1 Alpha Improvements - 7.67 trades/day target"
git push origin master
```

### 2. Deploy to AWS
```bash
# SSH into AWS instance
ssh -i "tgt-qr-key-oct-9.pem" ubuntu@16.171.196.141

# Pull latest code
cd /home/ubuntu/TGT-QR/mft
git stash  # Save any local changes
git pull origin master

# Restart services
sudo systemctl restart tgt-trading.service
sudo systemctl restart tgt-dashboard.service
```

### 3. Verify Deployment
```bash
# Check trading service status
sudo systemctl status tgt-trading.service

# Check recent logs
sudo journalctl -u tgt-trading.service -n 50 --no-pager

# Verify Phase 1 banner appears in logs
sudo journalctl -u tgt-trading.service -n 20 --no-pager | grep "Phase 1"
```

## Post-Deployment Monitoring

### First 24 Hours
- [ ] **Signal Generation**: Check logs for signal frequency
- [ ] **Trade Execution**: Verify trades are being placed
- [ ] **Win Rate**: Monitor initial win rate (target: ≥50%)
- [ ] **Error Check**: No API errors or connection issues

### First Week
- [ ] **Trades/Day**: Should average ~7-8 trades/day
- [ ] **Win Rate**: Should maintain ≥50%
- [ ] **Returns**: Compare to backtest predictions
- [ ] **Risk Limits**: Verify daily/total loss limits working

### Monitoring Commands
```bash
# Check recent signals
sudo journalctl -u tgt-trading.service -n 100 --no-pager | grep "signal generated"

# Check trade frequency
sudo journalctl -u tgt-trading.service --since "24 hours ago" | grep "Order placed"

# Check for errors
sudo journalctl -u tgt-trading.service --since "24 hours ago" | grep -i error
```

## Rollback Plan

If issues occur:
1. **Revert to Baseline**: Change `MIN_TREND_STRENGTH` back to 0.0005
2. **Restore MA Slope**: Re-add slope checks in `strategy.py`
3. **Restore Thresholds**: Change back to 0.3%/0.1%/0.05%
4. **Restart Services**: `sudo systemctl restart tgt-trading.service`

## Success Criteria

- ✅ Trades/day ≥ 1.0 (target: ~7-8)
- ✅ Win rate ≥ 50%
- ✅ No overfitting (live performance matches backtest)
- ✅ Risk limits functioning correctly

