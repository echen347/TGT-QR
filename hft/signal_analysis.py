#!/usr/bin/env python3
"""
Signal Analysis for HFT Alpha
QR-focused analysis: signal quality, distribution, decay, and practical considerations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def load_model_and_features(features_file: Path, model_type: str = 'lasso'):
    """Load features and retrain model for signal generation"""
    print(f"Loading features from {features_file}...")
    df = pd.read_csv(features_file)
    
    # Sort by timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Select features
    exclude_cols = ['symbol', 'snapshot_id', 'timestamp', 'target']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].fillna(0)
    y = df['target'].copy()
    
    # Clean data
    X = X.replace([np.inf, -np.inf], 0)
    valid_mask = ~(y.isna() | np.isinf(y) | (np.abs(y) > 1.0))
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    
    # Time-series split
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    
    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:val_end]
    y_test = y.iloc[val_end:]
    
    # Train model
    if model_type == 'lasso' and HAS_SKLEARN:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Find best alpha (simplified - use validation set)
        best_alpha = 0.001677  # From previous analysis
        model = Lasso(alpha=best_alpha, max_iter=2000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Generate signals on test set
        signals = model.predict(X_test_scaled)
        
        return {
            'signals': signals,
            'targets': y_test.values,
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols
        }
    
    elif model_type == 'xgb' and HAS_XGBOOST:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            early_stopping_rounds=20,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        signals = model.predict(X_test)
        
        return {
            'signals': signals,
            'targets': y_test.values,
            'model': model,
            'scaler': None,
            'feature_cols': feature_cols
        }
    
    else:
        raise ValueError(f"Model type {model_type} not available")


def analyze_signal_quality(signals, targets):
    """Analyze signal quality metrics"""
    print("\n" + "="*60)
    print("Signal Quality Analysis")
    print("="*60)
    
    # Basic statistics
    print(f"\nSignal Statistics:")
    print(f"  Mean: {np.mean(signals):.6f}")
    print(f"  Std: {np.std(signals):.6f}")
    print(f"  Min: {np.min(signals):.6f}")
    print(f"  Max: {np.max(signals):.6f}")
    print(f"  Skewness: {stats.skew(signals):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(signals):.4f}")
    
    # Signal-target correlation
    correlation = np.corrcoef(signals, targets)[0, 1]
    print(f"\nSignal-Target Correlation: {correlation:.4f}")
    
    # Information Coefficient (IC)
    ic = correlation
    print(f"Information Coefficient (IC): {ic:.4f}")
    print(f"  IC > 0.05: Strong signal")
    print(f"  IC 0.02-0.05: Moderate signal")
    print(f"  IC < 0.02: Weak signal")
    
    # Signal distribution by quantile
    print(f"\nSignal Distribution Analysis:")
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    for q in quantiles:
        threshold = np.quantile(signals, q)
        mask = signals >= threshold
        avg_return = np.mean(targets[mask]) if np.sum(mask) > 0 else 0
        print(f"  Top {100*(1-q):.0f}% (signal >= {threshold:.6f}): Avg return = {avg_return:.6f}")
    
    # Win rate analysis
    print(f"\nWin Rate Analysis:")
    for threshold in [0.0, np.quantile(signals, 0.75), np.quantile(signals, 0.9)]:
        long_signals = signals >= threshold
        if np.sum(long_signals) > 0:
            win_rate = np.mean(targets[long_signals] > 0)
            avg_return = np.mean(targets[long_signals])
            print(f"  Signal >= {threshold:.6f}: Win rate = {win_rate:.2%}, Avg return = {avg_return:.6f}")
    
    # Conditional returns by signal strength
    print(f"\nConditional Returns by Signal Strength:")
    signal_bins = np.quantile(signals, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for i in range(len(signal_bins)-1):
        mask = (signals >= signal_bins[i]) & (signals < signal_bins[i+1])
        if np.sum(mask) > 0:
            avg_return = np.mean(targets[mask])
            std_return = np.std(targets[mask])
            sharpe = avg_return / std_return if std_return > 0 else 0
            print(f"  Bin {i+1} [{signal_bins[i]:.6f}, {signal_bins[i+1]:.6f}): "
                  f"Return = {avg_return:.6f}, Sharpe = {sharpe:.4f}, N = {np.sum(mask)}")
    
    return {
        'correlation': correlation,
        'ic': ic,
        'mean': np.mean(signals),
        'std': np.std(signals)
    }


def analyze_signal_decay(signals, targets, max_lag=10):
    """Analyze signal decay over time"""
    print("\n" + "="*60)
    print("Signal Decay Analysis")
    print("="*60)
    
    # Autocorrelation of signals
    signal_acf = []
    for lag in range(1, min(max_lag+1, len(signals))):
        if lag < len(signals):
            corr = np.corrcoef(signals[:-lag], signals[lag:])[0, 1]
            signal_acf.append(corr)
    
    print(f"\nSignal Autocorrelation:")
    for lag, corr in enumerate(signal_acf[:10], 1):
        print(f"  Lag {lag}: {corr:.4f}")
    
    # Predictive power decay
    print(f"\nPredictive Power Decay (IC by lag):")
    for lag in range(min(5, len(signals))):
        if lag < len(targets):
            shifted_targets = targets[lag:]
            shifted_signals = signals[:-lag] if lag > 0 else signals
            if len(shifted_signals) == len(shifted_targets):
                corr = np.corrcoef(shifted_signals, shifted_targets)[0, 1]
                print(f"  Lag {lag}: IC = {corr:.4f}")
    
    return signal_acf


def analyze_turnover(signals, quantile_threshold=0.9):
    """Analyze signal turnover (how often signals change)"""
    print("\n" + "="*60)
    print("Signal Turnover Analysis")
    print("="*60)
    
    # Binary signals (long/short)
    threshold = np.quantile(signals, quantile_threshold)
    binary_signals = (signals >= threshold).astype(int)
    
    # Calculate turnover
    signal_changes = np.sum(np.diff(binary_signals) != 0)
    turnover_rate = signal_changes / len(binary_signals) if len(binary_signals) > 1 else 0
    
    print(f"\nTurnover Statistics:")
    print(f"  Signal threshold (top {100*(1-quantile_threshold):.0f}%): {threshold:.6f}")
    print(f"  Signal changes: {signal_changes} out of {len(binary_signals)-1}")
    print(f"  Turnover rate: {turnover_rate:.2%}")
    print(f"  Average holding period: {1/turnover_rate:.1f} periods" if turnover_rate > 0 else "  Average holding period: N/A")
    
    # Signal persistence
    signal_durations = []
    current_duration = 1
    for i in range(1, len(binary_signals)):
        if binary_signals[i] == binary_signals[i-1]:
            current_duration += 1
        else:
            if binary_signals[i-1] == 1:  # Only count long signal durations
                signal_durations.append(current_duration)
            current_duration = 1
    
    if signal_durations:
        print(f"\nSignal Persistence:")
        print(f"  Mean duration: {np.mean(signal_durations):.1f} periods")
        print(f"  Median duration: {np.median(signal_durations):.1f} periods")
        print(f"  Max duration: {np.max(signal_durations)} periods")
    
    return {
        'turnover_rate': turnover_rate,
        'signal_changes': signal_changes
    }


def analyze_risk_return(signals, targets):
    """Risk-return characteristics"""
    print("\n" + "="*60)
    print("Risk-Return Analysis")
    print("="*60)
    
    # Long-only strategy
    long_mask = signals >= np.quantile(signals, 0.75)
    long_returns = targets[long_mask]
    
    if len(long_returns) > 0:
        mean_return = np.mean(long_returns)
        std_return = np.std(long_returns)
        sharpe = mean_return / std_return if std_return > 0 else 0
        
        # Drawdown analysis
        cumulative = np.cumsum(long_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = np.min(drawdown)
        
        print(f"\nLong-Only Strategy (Top 25% signals):")
        print(f"  Mean return: {mean_return:.6f}")
        print(f"  Std return: {std_return:.6f}")
        print(f"  Sharpe ratio: {sharpe:.4f}")
        print(f"  Max drawdown: {max_drawdown:.6f}")
        print(f"  Win rate: {np.mean(long_returns > 0):.2%}")
        print(f"  Number of trades: {len(long_returns)}")
    
    # Long-short strategy
    short_mask = signals <= np.quantile(signals, 0.25)
    long_returns = targets[long_mask]
    short_returns = -targets[short_mask]  # Invert for short
    
    if len(long_returns) > 0 and len(short_returns) > 0:
        ls_returns = np.concatenate([long_returns, short_returns])
        mean_return = np.mean(ls_returns)
        std_return = np.std(ls_returns)
        sharpe = mean_return / std_return if std_return > 0 else 0
        
        print(f"\nLong-Short Strategy (Top/Bottom 25%):")
        print(f"  Mean return: {mean_return:.6f}")
        print(f"  Std return: {std_return:.6f}")
        print(f"  Sharpe ratio: {sharpe:.4f}")
        print(f"  Win rate: {np.mean(ls_returns > 0):.2%}")
        print(f"  Number of trades: {len(ls_returns)}")


def generate_signal_report(features_file: Path, output_file: Path, model_type: str = 'lasso'):
    """Generate comprehensive signal analysis report"""
    print("="*60)
    print("HFT Alpha Signal Analysis")
    print("="*60)
    
    # Load model and generate signals
    result = load_model_and_features(features_file, model_type)
    signals = result['signals']
    targets = result['targets']
    
    # Run analyses
    quality_metrics = analyze_signal_quality(signals, targets)
    decay_metrics = analyze_signal_decay(signals, targets)
    turnover_metrics = analyze_turnover(signals)
    analyze_risk_return(signals, targets)
    
    # Save summary
    with open(output_file, 'w') as f:
        f.write("HFT Alpha Signal Analysis Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Number of signals: {len(signals)}\n\n")
        f.write("Signal Quality Metrics:\n")
        f.write(f"  Information Coefficient (IC): {quality_metrics['ic']:.4f}\n")
        f.write(f"  Signal-Target Correlation: {quality_metrics['correlation']:.4f}\n")
        f.write(f"  Signal Mean: {quality_metrics['mean']:.6f}\n")
        f.write(f"  Signal Std: {quality_metrics['std']:.6f}\n\n")
        f.write("Turnover Metrics:\n")
        f.write(f"  Turnover Rate: {turnover_metrics['turnover_rate']:.2%}\n")
        f.write(f"  Signal Changes: {turnover_metrics['signal_changes']}\n")
    
    print(f"\n{'='*60}")
    print(f"Report saved to {output_file}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Signal Analysis for HFT Alpha")
    parser.add_argument("features_file", help="Features CSV file")
    parser.add_argument("--model", choices=['lasso', 'xgb'], default='lasso', help="Model type")
    parser.add_argument("--output", default="signal_analysis_report.txt", help="Output report file")
    args = parser.parse_args()
    
    features_file = Path(args.features_file)
    if not features_file.exists():
        print(f"Error: File not found: {features_file}")
        return
    
    generate_signal_report(features_file, Path(args.output), args.model)


if __name__ == "__main__":
    main()

