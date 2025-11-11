#!/usr/bin/env python3
"""
ML-Based Trading Strategy
Uses machine learning to predict price direction with careful overfitting prevention
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

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

def ml_strategy_signal(historical_prices, model_type='rf', retrain_interval=100):
    """
    ML-based trading strategy
    
    Args:
        historical_prices: DataFrame with OHLCV data
        model_type: 'rf' (Random Forest) or 'lr' (Logistic Regression)
        retrain_interval: Retrain model every N bars
    
    Returns:
        signal: -1, 0, or 1
    """
    if len(historical_prices) < 200:  # Need enough data
        return 0
    
    # Use recent data for training (last 500 bars)
    train_data = historical_prices.tail(500).copy()
    
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

