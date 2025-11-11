# HFT Alpha Analysis

High-frequency trading alpha signal analysis pipeline for order book data.

## Features

The pipeline extracts the following features from order book data:

### Order Book Features
- **Book Pressure**: Bid/ask imbalance ratio (-1 to +1)
- **VWAP**: Volume-weighted average price for bid, ask, and mid
- **Spread**: Bid-ask spread in absolute and basis points
- **Maker Ratios**: Ratio of orders >100 size (likely makers) on bid/ask
- **Depth Imbalance**: Volume imbalance at 5, 10, and 20 levels
- **Order Book Stats**: Number of levels, best bid/ask, top 10 volume

### Time Series Features
- **EMA**: Exponential moving averages of price, spread, and pressure (windows: 5, 10, 20, 50)
- **Price Momentum**: Percentage change over different windows

### Target Variable
- **Future Return**: Price return N periods ahead (default: 5 snapshots)

## Usage

### Basic Feature Extraction

```bash
# Extract features from CSV or Parquet file
python3 hft_alpha.py input_file.csv --output features.csv

# Filter by symbol
python3 hft_alpha.py input_file.csv --symbol ETHUSDT --output ethusdt_features.csv
```

### Feature Extraction + Model Training

```bash
# Extract features and train models
python3 hft_alpha.py input_file.csv --symbol ETHUSDT --output features.csv --train
```

This will:
1. Extract all features from order book data
2. Train XGBoost and Lasso regression models
3. Output feature importance rankings
4. Save model summary to `{output}_model_summary.txt`

## Models

### XGBoost (Gradient Boosting)
- Tree-based model for non-linear relationships
- Good for capturing complex patterns in order book dynamics
- May overfit on small datasets - use regularization

### Lasso Regression
- Linear model with L1 regularization
- Good for feature selection and interpretability
- More stable on smaller datasets

## Example Output

```
Training XGBoost Model
Train R²: 0.7226, RMSE: 0.262981
Test R²: -0.1180, RMSE: 0.526248

Top 10 Most Important Features:
         feature  importance
        depth_20    0.036755
    ema_price_20    0.036320
    ema_price_10    0.036245
 ema_pressure_20    0.035266
bid_volume_top10    0.034893
```

## Requirements

```bash
pip3 install pandas numpy xgboost scikit-learn
```

## Notes

- **Maker Size Threshold**: Default is 100.0 (orders >= 100 size are considered makers)
- **Forward Periods**: Default is 5 snapshots for target calculation
- **Data Cleaning**: Automatically removes NaN, infinity, and extreme returns (>100%)

## Next Steps

1. **Hyperparameter Tuning**: Adjust XGBoost/Lasso parameters for your data
2. **Feature Engineering**: Add more domain-specific features (e.g., order flow, trade intensity)
3. **Cross-Validation**: Use time-series cross-validation instead of simple train/test split
4. **Ensemble Methods**: Combine multiple models for better predictions
5. **Backtesting**: Test signals on historical data with realistic execution assumptions

