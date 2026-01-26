#!/usr/bin/env python3
"""
ML-Based Trading Strategy
Uses machine learning to predict price direction with careful overfitting prevention

IMPORTANT: Model caching is used to prevent retraining every bar.
Call clear_ml_cache() between train/test runs to ensure proper OOD validation.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Global model cache - prevents retraining every bar
_model_cache = {}
_scaler_cache = {}
_last_train_len = {}
_feature_importance_cache = {}

def clear_ml_cache():
    """Clear model cache - MUST be called between train and test runs for proper OOD validation"""
    global _model_cache, _scaler_cache, _last_train_len, _feature_importance_cache
    _model_cache.clear()
    _scaler_cache.clear()
    _last_train_len.clear()
    _feature_importance_cache.clear()

def get_feature_importance(model_key=None):
    """Get feature importance for ML analysis dashboard"""
    if model_key and model_key in _feature_importance_cache:
        return _feature_importance_cache[model_key]
    elif _feature_importance_cache:
        # Return most recent if no key specified
        return list(_feature_importance_cache.values())[-1]
    return None

FEATURE_NAMES = [
    'price_zscore', 'price_range_position', 'trend_strength',
    'ma5_ratio', 'ma10_ratio', 'ma20_ratio', 'ma5_ma10_ratio', 'ma10_ma20_ratio',
    'volatility_std', 'volatility_abs_mean',
    'volume_ratio', 'volume_std_ratio',
    'rsi_like', 'atr_like'
]

def calculate_features(df, lookback=60):
    """Calculate technical features for ML model"""
    closes = df['close'].values
    volumes = df['volume'].values
    highs = df['high'].values
    lows = df['low'].values
    
    features = []
    
    for i in range(lookback, len(df)):
        window_closes = closes[i-lookback:i]
        window_volumes = volumes[i-lookback:i]
        window_highs = highs[i-lookback:i]
        window_lows = lows[i-lookback:i]
        
        # Price features
        price_features = [
            (closes[i] - np.mean(window_closes)) / np.std(window_closes) if np.std(window_closes) > 0 else 0,  # Z-score
            (closes[i] - np.min(window_closes)) / (np.max(window_closes) - np.min(window_closes)) if np.max(window_closes) != np.min(window_closes) else 0,  # Price position in range
            np.mean(np.diff(window_closes)) / np.std(window_closes) if np.std(window_closes) > 0 else 0,  # Trend strength
        ]
        
        # Moving averages
        ma5 = np.mean(window_closes[-5:])
        ma10 = np.mean(window_closes[-10:])
        ma20 = np.mean(window_closes[-20:])
        ma_features = [
            (closes[i] - ma5) / ma5 if ma5 > 0 else 0,
            (closes[i] - ma10) / ma10 if ma10 > 0 else 0,
            (closes[i] - ma20) / ma20 if ma20 > 0 else 0,
            (ma5 - ma10) / ma10 if ma10 > 0 else 0,
            (ma10 - ma20) / ma20 if ma20 > 0 else 0,
        ]
        
        # Volatility features
        returns = np.diff(window_closes) / window_closes[:-1]
        volatility_features = [
            np.std(returns) if len(returns) > 0 else 0,
            np.mean(np.abs(returns)) if len(returns) > 0 else 0,
        ]
        
        # Volume features
        avg_volume = np.mean(window_volumes)
        volume_features = [
            (volumes[i] - avg_volume) / avg_volume if avg_volume > 0 else 0,
            np.std(window_volumes) / avg_volume if avg_volume > 0 else 0,
        ]
        
        # RSI-like feature
        gains = returns[returns > 0]
        losses = -returns[returns < 0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi_like = (rs / (1 + rs)) - 0.5  # Normalized to [-0.5, 0.5]
        
        # ATR-like feature
        tr = np.max([
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]) if i > 0 else 0,
            abs(lows[i] - closes[i-1]) if i > 0 else 0
        ])
        atr_like = tr / closes[i] if closes[i] > 0 else 0
        
        feature_vector = price_features + ma_features + volatility_features + volume_features + [rsi_like, atr_like]
        features.append(feature_vector)
    
    return np.array(features)

def create_labels(df, lookback=60, forward_bars=5):
    """Create labels: 1 if price goes up, -1 if down, 0 if neutral"""
    closes = df['close'].values
    labels = []
    
    for i in range(lookback, len(df) - forward_bars):
        current_price = closes[i]
        future_price = closes[i + forward_bars]
        
        # Label: 1 if price increases > 0.5%, -1 if decreases > 0.5%, else 0
        pct_change = (future_price - current_price) / current_price
        if pct_change > 0.005:  # 0.5% threshold
            labels.append(1)
        elif pct_change < -0.005:
            labels.append(-1)
        else:
            labels.append(0)
    
    return np.array(labels)

def ml_strategy_signal(historical_prices, model_type='rf', retrain_interval=500,
                        train_end_idx=None, force_retrain=False):
    """
    ML-based trading strategy with proper model caching

    Args:
        historical_prices: DataFrame with OHLCV data
        model_type: 'rf' (Random Forest) or 'lr' (Logistic Regression)
        retrain_interval: Retrain model every N bars (default 500)
        train_end_idx: If set, only use data up to this index for training
                      (CRITICAL for proper train/test split in backtesting)
        force_retrain: Force model retraining even if cached

    Returns:
        signal: -1, 0, or 1
    """
    global _model_cache, _scaler_cache, _last_train_len, _feature_importance_cache

    if len(historical_prices) < 200:  # Need enough data
        return 0

    # Create cache key based on model type
    cache_key = f"{model_type}"
    current_len = len(historical_prices)

    # Determine if we need to retrain
    should_retrain = (
        force_retrain or
        cache_key not in _model_cache or
        current_len - _last_train_len.get(cache_key, 0) >= retrain_interval
    )

    if should_retrain:
        # Determine training data boundaries
        if train_end_idx is not None:
            # Use only data up to train_end_idx (for proper OOD validation)
            train_data = historical_prices.iloc[:train_end_idx].tail(500).copy()
        else:
            # Default behavior: use last 500 bars up to current point
            # Leave a gap to avoid lookahead bias
            train_end = len(historical_prices) - 10  # 10 bar gap
            train_data = historical_prices.iloc[:train_end].tail(500).copy()

        # Calculate features
        features = calculate_features(train_data, lookback=60)

        if len(features) < 50:  # Need minimum samples
            return 0

        # Create labels
        labels = create_labels(train_data, lookback=60, forward_bars=5)

        # Align features and labels
        min_len = min(len(features), len(labels))
        features = features[:min_len]
        labels = labels[:min_len]

        if len(features) < 50:
            return 0

        # Remove neutral labels for training (focus on directional predictions)
        non_neutral_mask = labels != 0
        if np.sum(non_neutral_mask) < 20:  # Need at least 20 non-neutral samples
            return 0

        X_train = features[non_neutral_mask]
        y_train = labels[non_neutral_mask]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train model with regularization to prevent overfitting
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=50,  # Smaller forest to reduce overfitting
                max_depth=5,  # Limit depth
                min_samples_split=20,  # Require more samples to split
                min_samples_leaf=10,  # Require more samples per leaf
                random_state=42,
                class_weight='balanced'  # Handle class imbalance
            )
        else:  # Logistic Regression
            model = LogisticRegression(
                C=0.1,  # Strong regularization
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )

        try:
            model.fit(X_train_scaled, y_train)
        except:
            return 0

        # Cache model, scaler, and training length
        _model_cache[cache_key] = model
        _scaler_cache[cache_key] = scaler
        _last_train_len[cache_key] = current_len

        # Store feature importance for dashboard
        if model_type == 'rf' and hasattr(model, 'feature_importances_'):
            _feature_importance_cache[cache_key] = {
                'features': FEATURE_NAMES,
                'importances': model.feature_importances_.tolist()
            }
        elif model_type == 'lr' and hasattr(model, 'coef_'):
            # For logistic regression, use absolute coefficients as importance proxy
            _feature_importance_cache[cache_key] = {
                'features': FEATURE_NAMES,
                'importances': np.abs(model.coef_[0]).tolist()
            }
    else:
        # Use cached model and scaler
        model = _model_cache[cache_key]
        scaler = _scaler_cache[cache_key]

    # Predict on most recent data point
    recent_features = calculate_features(historical_prices.tail(100), lookback=60)
    if len(recent_features) == 0:
        return 0
    
    recent_feature = recent_features[-1].reshape(1, -1)
    recent_feature_scaled = scaler.transform(recent_feature)
    
    try:
        prediction = model.predict(recent_feature_scaled)[0]
        # Get probability for confidence
        proba = model.predict_proba(recent_feature_scaled)[0]
        max_proba = np.max(proba)
        
        # Only trade if model is confident (>50% probability - lowered from 60% to get more trades)
        if max_proba < 0.5:
            return 0
        
        return int(prediction)
    except:
        return 0

if __name__ == "__main__":
    # Test the ML strategy
    print("ML Trading Strategy Module")
    print("Features: Price, MA, Volatility, Volume, RSI-like, ATR-like")
    print("Models: Random Forest or Logistic Regression")
    print("Overfitting prevention: Regularization, train/test split, confidence threshold")

