# ML-Based Strategy Ideas

## ⚠️ Overfitting Warning

**You're absolutely right about bias-variance tradeoff!**

More complex models = Higher risk of overfitting:
- **Simple models** (Logistic Regression): Low variance, high bias → Less overfitting risk
- **Complex models** (Deep Neural Networks): High variance, low bias → Higher overfitting risk

**Recommendation**: Start with **simplest possible ML** (Phase 3A with Logistic Regression), only add complexity if needed.

### Anti-Overfitting Principles:
1. ✅ **Prefer simple models** - Logistic Regression > Random Forest > Neural Networks
2. ✅ **Feature selection** - Use only statistically significant features
3. ✅ **Regularization** - L1/L2 penalties to prevent overfitting
4. ✅ **Out-of-sample testing** - Always validate on unseen data
5. ✅ **Walk-forward validation** - Test on rolling windows
6. ✅ **Live monitoring** - Compare live vs backtest performance

**If ML doesn't improve performance, stick with simpler rule-based systems!**

---

## Current Status

**Active System**: Phase 1 (Original) + Phase 2B (Adaptive Parameters)
- **Backtested**: ✅ Phase 1 extensively tested (7.67 trades/day, 52.2% win rate, 112.93% return)
- **Adaptive System**: ⚠️ Backtest exists (`backtest_adaptive.py`) but needs validation
- **Deployed**: ✅ Yes - `ENABLE_ADAPTIVE_PARAMS = True` in production

## ML-Based Strategy Improvements

### 1. Feature Engineering for Signal Quality

**Idea**: Use ML to predict signal quality before entering trades

**Features**:
- Price deviation from MA (current)
- Trend strength (current)
- Volatility (current)
- Volume profile (24h average vs current)
- Time-of-day patterns (crypto markets have intraday patterns)
- Cross-symbol correlation (e.g., BTC leading ETH)
- Market regime indicators (trending vs ranging)

**Model**: Binary classifier (profitable trade vs unprofitable)
- **Input**: Features at signal time
- **Output**: Probability of profitable trade
- **Threshold**: Only enter if probability > 60%

**Implementation**:
```python
# New module: mft/src/ml_signal_filter.py
class MLSignalFilter:
    def __init__(self):
        self.model = self.load_or_train_model()
    
    def predict_signal_quality(self, features):
        """Return probability that signal will be profitable"""
        return self.model.predict_proba(features)[0][1]
    
    def should_trade(self, signal, features):
        """Filter signals using ML prediction"""
        if signal == 0:  # NEUTRAL
            return False
        prob = self.predict_signal_quality(features)
        return prob > 0.60  # Only trade high-probability signals
```

**Backtesting**: Train on historical data, test on out-of-sample period

---

### 2. Dynamic Threshold Optimization

**Idea**: Use reinforcement learning to optimize thresholds dynamically

**Current**: Fixed thresholds (0.2%/0.05%/0.03%) based on volatility
**ML Approach**: Learn optimal thresholds per symbol/market regime

**Features**:
- Recent win rate
- Recent profit factor
- Market volatility regime
- Time since last trade
- Symbol-specific characteristics

**Model**: Multi-armed bandit or Q-learning
- **State**: Current market conditions + recent performance
- **Action**: Adjust thresholds (tighten/loosen)
- **Reward**: Trade PnL

**Implementation**:
```python
# New module: mft/src/ml_threshold_optimizer.py
class MLThresholdOptimizer:
    def get_optimal_thresholds(self, symbol, market_state):
        """Use RL to determine optimal thresholds for current state"""
        state_features = self.extract_features(symbol, market_state)
        action = self.rl_agent.choose_action(state_features)
        return self.action_to_thresholds(action)
```

---

### 3. Position Sizing with ML

**Idea**: Use ML to determine optimal position size per trade

**Features**:
- Signal strength (deviation from MA)
- Volatility (ATR)
- Recent win rate
- Account balance
- Correlation with existing positions

**Model**: Regression model predicting optimal position size
- **Input**: Features
- **Output**: Position size multiplier (0.5x to 1.5x of base size)

**Implementation**:
```python
# Extend: mft/src/strategy.py place_order()
def place_order(self, symbol, side, qty_usdt_base):
    # Get ML position size recommendation
    features = self.extract_position_features(symbol, side)
    size_multiplier = self.ml_position_sizer.predict(features)
    
    # Apply ML recommendation
    qty_usdt = qty_usdt_base * size_multiplier
    # ... rest of order logic
```

---

### 4. Exit Timing Optimization

**Idea**: Use ML to predict optimal exit points

**Current**: Fixed TP/SL (2%/1%)
**ML Approach**: Learn when to exit based on market conditions

**Features**:
- Current PnL
- Time in position
- Price action (momentum indicators)
- Volume patterns
- Market regime changes

**Model**: Classification (hold vs exit)
- **Input**: Position features
- **Output**: Exit probability

**Implementation**:
```python
# New module: mft/src/ml_exit_timing.py
class MLExitTiming:
    def should_exit(self, position, market_data):
        """Predict if we should exit position now"""
        features = self.extract_exit_features(position, market_data)
        exit_prob = self.model.predict_proba(features)[0][1]
        return exit_prob > 0.70  # Exit if high probability
```

---

### 5. Symbol Selection with ML

**Idea**: Use ML to prioritize which symbols to trade

**Features**:
- Recent performance per symbol
- Volatility
- Volume
- Correlation with other positions
- Time since last trade

**Model**: Ranking model (prioritize symbols)
- **Input**: Symbol features
- **Output**: Trade priority score

**Implementation**:
```python
# Extend: mft/src/strategy.py run_strategy()
def run_strategy(self):
    # Get ML symbol priorities
    symbol_scores = {}
    for symbol in SYMBOLS:
        features = self.extract_symbol_features(symbol)
        symbol_scores[symbol] = self.ml_symbol_ranker.predict(features)
    
    # Trade highest priority symbols first
    sorted_symbols = sorted(SYMBOLS, key=lambda s: symbol_scores[s], reverse=True)
    for symbol in sorted_symbols:
        # ... trading logic
```

---

## Implementation Priority

### Phase 3A: Feature Engineering (Easiest, High Impact)
1. ✅ Collect features (already have most)
2. ✅ Train simple classifier (Logistic Regression or Random Forest)
3. ✅ Backtest on historical data
4. ✅ Deploy as signal filter

**Timeline**: 1-2 weeks
**Risk**: Low (can disable if underperforms)

### Phase 3B: Dynamic Thresholds (Medium Complexity)
1. ✅ Implement RL agent (Q-learning or bandit)
2. ✅ Train on historical data
3. ✅ Backtest vs fixed thresholds
4. ✅ Deploy if better

**Timeline**: 2-3 weeks
**Risk**: Medium (requires careful tuning)

### Phase 3C: Position Sizing (Medium Complexity)
1. ✅ Collect position sizing features
2. ✅ Train regression model
3. ✅ Backtest vs fixed sizing
4. ✅ Deploy if better

**Timeline**: 2-3 weeks
**Risk**: Medium (can cause larger losses if wrong)

### Phase 3D: Exit Timing (Higher Complexity)
1. ✅ Collect exit timing features
2. ✅ Train classification model
3. ✅ Backtest vs fixed TP/SL
4. ✅ Deploy if better

**Timeline**: 3-4 weeks
**Risk**: High (can miss profits or increase losses)

### Phase 3E: Symbol Selection (Lower Priority)
1. ✅ Implement ranking model
2. ✅ Backtest vs equal priority
3. ✅ Deploy if better

**Timeline**: 1-2 weeks
**Risk**: Low

---

## ML Framework Recommendations

### Option 1: scikit-learn (Simple, Fast)
- **Pros**: Easy to implement, fast training, good for small datasets
- **Cons**: Limited deep learning capabilities
- **Best for**: Phase 3A (Feature Engineering), Phase 3C (Position Sizing)

### Option 2: LightGBM/XGBoost (Gradient Boosting)
- **Pros**: Handles non-linear relationships, feature importance, fast
- **Cons**: Can overfit if not careful
- **Best for**: Phase 3A, Phase 3C, Phase 3D

### Option 3: Reinforcement Learning (Stable-Baselines3)
- **Pros**: Good for sequential decision making
- **Cons**: Requires more data, slower training
- **Best for**: Phase 3B (Dynamic Thresholds)

### Option 4: Simple Neural Network (PyTorch/TensorFlow)
- **Pros**: Can learn complex patterns
- **Cons**: Requires more data, risk of overfitting
- **Best for**: Phase 3D (Exit Timing) if we have enough data

---

## Data Requirements

### Minimum Data Needed
- **Training**: 100+ trades per symbol (we have this)
- **Validation**: 30+ trades per symbol (we have this)
- **Features**: 10-20 features per trade (we can extract these)

### Data Collection
1. ✅ Historical trades (already in DB)
2. ✅ Price data (already in DB)
3. ✅ Signals (already in DB)
4. ⚠️ Need to extract features from historical data

---

## Next Steps

1. **Run Adaptive Backtest** (if not done):
   ```bash
   python3 tools/backtest_adaptive.py --days 60
   ```

2. **Start with Phase 3A** (Feature Engineering):
   - Extract features from historical trades
   - Train simple classifier
   - Backtest on OOS data
   - Deploy if improves performance

3. **Monitor Live Performance**:
   - Compare ML-filtered signals vs all signals
   - Track win rate improvement
   - Iterate based on results

---

## Anti-Overfitting for ML

1. **Train/Test Split**: 70/30 split
2. **Walk-Forward Validation**: Test on rolling windows
3. **Feature Selection**: Use only features with statistical significance
4. **Regularization**: L1/L2 regularization in models
5. **Cross-Validation**: K-fold CV during training
6. **Out-of-Sample Testing**: Always test on unseen data
7. **Live Monitoring**: Compare live vs backtest performance

---

## Quick Start: Phase 3A Implementation

```python
# mft/src/ml_signal_filter.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from database import db_manager, TradeRecord, SignalRecord

class MLSignalFilter:
    def __init__(self):
        self.model = None
        self.feature_names = [
            'price_deviation_pct',
            'trend_strength',
            'volatility',
            'volume_ratio',
            'time_of_day',
            'symbol_encoded'
        ]
    
    def train(self):
        """Train model on historical trades"""
        # Load historical trades and signals
        trades = db_manager.session.query(TradeRecord).all()
        signals = db_manager.session.query(SignalRecord).all()
        
        # Extract features and labels
        X, y = self.extract_features_labels(trades, signals)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Train Accuracy: {train_score:.2%}")
        print(f"Test Accuracy: {test_score:.2%}")
        
        # Check for overfitting
        if train_score - test_score > 0.10:
            print("⚠️ Warning: Possible overfitting detected!")
        
        return self.model
    
    def predict(self, features):
        """Predict if signal will be profitable"""
        if self.model is None:
            return 0.5  # Default: neutral
        return self.model.predict_proba([features])[0][1]
    
    def should_trade(self, signal_value, features, threshold=0.60):
        """Filter signals using ML"""
        if signal_value == 0:  # NEUTRAL
            return False
        prob = self.predict(features)
        return prob >= threshold
```

---

## Summary

**Current**: Phase 1 + Adaptive System (Phase 2B) deployed
**Next**: Start with ML Feature Engineering (Phase 3A) - easiest, highest impact
**Timeline**: 1-2 weeks for Phase 3A
**Risk**: Low (can disable if needed)

