# HFT Alpha Analysis - Quick Reference

## Key Results

| Metric | XGBoost | Lasso |
|--------|---------|-------|
| **Out-of-Sample R²** | 0.0248 | **0.0265** ✅ |
| Train R² | 0.0496 | 0.0327 |
| Validation R² | 0.0381 | 0.0364 |
| Overfitting | Minimal | None |
| Features Used | 34 | 16 |

## Top 5 Predictive Features

1. **ema_price_10** - 10-period EMA of price
2. **ema_price_20** - 20-period EMA of price  
3. **ask_levels** - Number of ask price levels
4. **ema_price_5** - 5-period EMA of price
5. **bid_levels** - Number of bid price levels

## Dataset Stats

- **Snapshots:** 183,029
- **Valid Samples:** 76,424
- **Features:** 34
- **Train/Val/Test:** 60%/20%/20%

## Model Recommendation

**Use Lasso Model** - Better OOS performance, more stable, interpretable

## Files

- `ANALYSIS_SUMMARY.md` - Full detailed report
- `hft_features_ethusdt_full.csv` - All features (55 MB)
- `hft_features_model_summary.txt` - Latest model results
- `hft_alpha.py` - Analysis pipeline

## Usage

```bash
# Train models with proper validation
python3 hft_alpha.py hft_features_ethusdt_full.csv --symbol ETHUSDT --train
```

## Key Insights

- ✅ Models show genuine predictive power (positive OOS R²)
- ✅ Minimal overfitting with proper validation
- ✅ Price momentum (EMA) is most important feature
- ✅ Order book depth provides signal
- ✅ Maker ratios have subtle predictive value
