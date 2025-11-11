#!/usr/bin/env python3
"""
HFT Alpha Analysis Pipeline
Analyzes order book data for high-frequency trading alpha signals.

Features:
- Book pressure (bid/ask imbalance)
- Exponential moving averages
- Quantity weighted average price (VWAP)
- Maker/taker indicators (orders >100 size tend to be makers)
- Order book depth and spread metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Model imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Install with: pip3 install xgboost")

try:
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed. Install with: pip3 install scikit-learn")


class OrderBookReconstructor:
    """Reconstructs order book snapshots from delta updates"""
    
    def __init__(self, maker_size_threshold: float = 100.0):
        """
        Args:
            maker_size_threshold: Orders above this size are likely makers
        """
        self.maker_size_threshold = maker_size_threshold
        self.order_books = {}  # symbol -> {snapshot_id -> {bid: {}, ask: {}}}
    
    def process_snapshot(self, snapshot_df: pd.DataFrame) -> Dict:
        """
        Process a single snapshot (all rows with same id)
        Returns order book state
        """
        symbol = snapshot_df['symbol'].iloc[0]
        snapshot_id = snapshot_df['id'].iloc[0]
        timestamp = snapshot_df['system_ts'].iloc[0]
        
        # Initialize order book if needed
        if symbol not in self.order_books:
            self.order_books[symbol] = {}
        
        # Get current book state (or initialize)
        if snapshot_id not in self.order_books[symbol]:
            self.order_books[symbol][snapshot_id] = {'bid': {}, 'ask': {}}
        
        book = self.order_books[symbol][snapshot_id]
        
        # Update order book from delta
        for _, row in snapshot_df.iterrows():
            price = row['price']
            quantity = row['quantity']
            side = row['side']
            
            if quantity == 0:
                # Remove level
                if price in book[side]:
                    del book[side][price]
            else:
                # Update/add level
                book[side][price] = quantity
        
        return {
            'symbol': symbol,
            'snapshot_id': snapshot_id,
            'timestamp': timestamp,
            'bid': dict(sorted(book['bid'].items(), reverse=True)),  # Descending prices
            'ask': dict(sorted(book['ask'].items())),  # Ascending prices
        }


class HFTFeatureEngineer:
    """Engineers features for HFT alpha prediction"""
    
    def __init__(self, maker_size_threshold: float = 100.0):
        self.maker_size_threshold = maker_size_threshold
    
    def calculate_book_pressure(self, bid_book: Dict, ask_book: Dict, depth_levels: int = 10) -> float:
        """
        Calculate book pressure (bid/ask imbalance)
        Positive = more bid pressure, Negative = more ask pressure
        """
        if not bid_book or not ask_book:
            return 0.0
        
        bid_prices = sorted(bid_book.keys(), reverse=True)[:depth_levels]
        ask_prices = sorted(ask_book.keys())[:depth_levels]
        
        bid_volume = sum(bid_book[p] for p in bid_prices)
        ask_volume = sum(ask_book[p] for p in ask_prices)
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        
        # Normalized pressure: -1 (all ask) to +1 (all bid)
        pressure = (bid_volume - ask_volume) / total_volume
        return pressure
    
    def calculate_vwap(self, book: Dict, side: str, depth_levels: int = 10) -> float:
        """Calculate Volume Weighted Average Price"""
        if not book:
            return 0.0
        
        prices = sorted(book.keys(), reverse=(side == 'bid'))[:depth_levels]
        if not prices:
            return 0.0
        
        total_value = sum(book[p] * p for p in prices)
        total_volume = sum(book[p] for p in prices)
        
        if total_volume == 0:
            return 0.0
        
        return total_value / total_volume
    
    def calculate_spread(self, bid_book: Dict, ask_book: Dict) -> Tuple[float, float]:
        """Calculate bid-ask spread and mid price"""
        if not bid_book or not ask_book:
            return 0.0, 0.0
        
        best_bid = max(bid_book.keys())
        best_ask = min(ask_book.keys())
        
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2.0
        
        return spread, mid_price
    
    def calculate_maker_ratio(self, book: Dict, side: str, depth_levels: int = 10) -> float:
        """
        Calculate ratio of maker orders (size > threshold)
        Makers typically place larger resting orders
        """
        if not book:
            return 0.0
        
        prices = sorted(book.keys(), reverse=(side == 'bid'))[:depth_levels]
        if not prices:
            return 0.0
        
        total_volume = sum(book[p] for p in prices)
        maker_volume = sum(book[p] for p in prices if book[p] >= self.maker_size_threshold)
        
        if total_volume == 0:
            return 0.0
        
        return maker_volume / total_volume
    
    def calculate_depth_imbalance(self, bid_book: Dict, ask_book: Dict, levels: list = [5, 10, 20]) -> Dict[str, float]:
        """Calculate depth imbalance at different levels"""
        imbalances = {}
        
        for level in levels:
            bid_prices = sorted(bid_book.keys(), reverse=True)[:level]
            ask_prices = sorted(ask_book.keys())[:level]
            
            bid_vol = sum(bid_book.get(p, 0) for p in bid_prices)
            ask_vol = sum(ask_book.get(p, 0) for p in ask_prices)
            
            total = bid_vol + ask_vol
            if total > 0:
                imbalances[f'depth_{level}'] = (bid_vol - ask_vol) / total
            else:
                imbalances[f'depth_{level}'] = 0.0
        
        return imbalances
    
    def extract_features(self, book_state: Dict) -> pd.Series:
        """Extract all features from order book state"""
        bid_book = book_state['bid']
        ask_book = book_state['ask']
        
        features = {}
        
        # Book pressure
        features['book_pressure'] = self.calculate_book_pressure(bid_book, ask_book)
        
        # VWAP
        features['bid_vwap'] = self.calculate_vwap(bid_book, 'bid')
        features['ask_vwap'] = self.calculate_vwap(ask_book, 'ask')
        features['mid_vwap'] = (features['bid_vwap'] + features['ask_vwap']) / 2.0
        
        # Spread metrics
        spread, mid_price = self.calculate_spread(bid_book, ask_book)
        features['spread'] = spread
        features['spread_bps'] = (spread / mid_price * 10000) if mid_price > 0 else 0.0
        features['mid_price'] = mid_price
        
        # Maker ratios
        features['bid_maker_ratio'] = self.calculate_maker_ratio(bid_book, 'bid')
        features['ask_maker_ratio'] = self.calculate_maker_ratio(ask_book, 'ask')
        
        # Depth imbalances
        depth_imbalances = self.calculate_depth_imbalance(bid_book, ask_book)
        features.update(depth_imbalances)
        
        # Order book statistics
        if bid_book:
            features['bid_levels'] = len(bid_book)
            features['best_bid'] = max(bid_book.keys())
            features['bid_volume_top10'] = sum(sorted(bid_book.values(), reverse=True)[:10])
        else:
            features['bid_levels'] = 0
            features['best_bid'] = 0.0
            features['bid_volume_top10'] = 0.0
        
        if ask_book:
            features['ask_levels'] = len(ask_book)
            features['best_ask'] = min(ask_book.keys())
            features['ask_volume_top10'] = sum(sorted(ask_book.values(), reverse=True)[:10])
        else:
            features['ask_levels'] = 0
            features['best_ask'] = 0.0
            features['ask_volume_top10'] = 0.0
        
        return pd.Series(features)


def add_ema_features(df: pd.DataFrame, windows: list = [5, 10, 20, 50]) -> pd.DataFrame:
    """Add exponential moving average features"""
    for window in windows:
        df[f'ema_price_{window}'] = df['mid_price'].ewm(span=window, adjust=False).mean()
        df[f'ema_spread_{window}'] = df['spread'].ewm(span=window, adjust=False).mean()
        df[f'ema_pressure_{window}'] = df['book_pressure'].ewm(span=window, adjust=False).mean()
        
        # Price momentum
        df[f'price_momentum_{window}'] = df['mid_price'].pct_change(window)
    
    return df


def create_target(df: pd.DataFrame, forward_periods: int = 5) -> pd.Series:
    """
    Create target variable: future price return
    Positive = price goes up, Negative = price goes down
    """
    # Forward fill mid_price to handle missing values
    df['mid_price'] = df['mid_price'].ffill()
    
    # Calculate future return
    future_price = df.groupby('symbol')['mid_price'].shift(-forward_periods)
    target = (future_price - df['mid_price']) / df['mid_price']
    
    return target


def process_data(file_path: Path, symbol: str = None) -> pd.DataFrame:
    """Process parquet/CSV file and extract features"""
    print(f"Loading data from {file_path}...")
    
    if file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
    
    # Filter by symbol if specified
    if symbol:
        df = df[df['symbol'] == symbol].copy()
        print(f"Filtered to {symbol}: {len(df)} rows")
    
    # Sort by symbol, then by id (snapshot), then by price
    df = df.sort_values(['symbol', 'id', 'system_ts', 'side', 'price'])
    
    print(f"Processing {df['symbol'].nunique()} symbols, {df['id'].nunique()} snapshots...")
    
    # Reconstruct order books
    reconstructor = OrderBookReconstructor(maker_size_threshold=100.0)
    feature_engineer = HFTFeatureEngineer(maker_size_threshold=100.0)
    
    all_features = []
    
    # Process each snapshot
    for (symbol_name, snapshot_id), snapshot_df in df.groupby(['symbol', 'id']):
        try:
            book_state = reconstructor.process_snapshot(snapshot_df)
            features = feature_engineer.extract_features(book_state)
            features['symbol'] = symbol_name
            features['snapshot_id'] = snapshot_id
            features['timestamp'] = book_state['timestamp']
            all_features.append(features)
        except Exception as e:
            print(f"Error processing snapshot {snapshot_id} for {symbol_name}: {e}")
            continue
    
    features_df = pd.DataFrame(all_features)
    
    # Sort by symbol and timestamp
    features_df = features_df.sort_values(['symbol', 'timestamp'])
    
    # Add EMA features and create target per symbol
    processed_dfs = []
    for symbol_name, symbol_df in features_df.groupby('symbol'):
        symbol_df = symbol_df.reset_index(drop=True)
        symbol_df = add_ema_features(symbol_df)
        symbol_df['target'] = create_target(symbol_df)
        processed_dfs.append(symbol_df)
    
    features_df = pd.concat(processed_dfs, ignore_index=True)
    
    # Remove rows with NaN target (last few rows per symbol)
    features_df = features_df.dropna(subset=['target'])
    
    print(f"Extracted {len(features_df)} feature rows")
    print(f"Features: {list(features_df.columns)}")
    
    return features_df


def time_series_split(data, n_splits=5, test_size=0.2):
    """
    Time-series cross-validation: expanding window approach
    Returns list of (train_idx, test_idx) splits
    """
    n = len(data)
    test_start = int(n * (1 - test_size))
    test_size_per_split = (n - test_start) // n_splits
    
    splits = []
    for i in range(n_splits):
        test_end = test_start + (i + 1) * test_size_per_split
        train_end = test_start + i * test_size_per_split
        
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, min(test_end, n))
        
        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    
    return splits


def train_models(features_df: pd.DataFrame, symbol: str = None, use_cv: bool = True):
    """
    Train XGBoost and Lasso models with proper out-of-sample validation
    
    Args:
        features_df: DataFrame with features and target
        symbol: Filter by symbol if specified
        use_cv: If True, use time-series cross-validation
    """
    if symbol:
        features_df = features_df[features_df['symbol'] == symbol].copy()
    
    # Sort by timestamp to ensure chronological order
    if 'timestamp' in features_df.columns:
        features_df = features_df.sort_values('timestamp').reset_index(drop=True)
    
    # Select feature columns (exclude metadata)
    exclude_cols = ['symbol', 'snapshot_id', 'timestamp', 'target']
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    
    X = features_df[feature_cols].fillna(0)
    y = features_df['target'].copy()
    
    # Remove infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    # Remove rows with invalid target values
    valid_mask = ~(y.isna() | np.isinf(y) | (np.abs(y) > 1.0))  # Remove returns >100%
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    
    print(f"After cleaning: {len(X)} samples (removed {len(features_df) - len(X)} invalid rows)")
    print(f"Feature count: {len(feature_cols)}")
    
    results = {}
    
    if use_cv:
        # Time-series cross-validation: Train/Validation/Test split
        # 60% train, 20% validation, 20% test (chronological)
        n = len(X)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        X_train = X.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        
        y_train = y.iloc[:train_end]
        y_val = y.iloc[train_end:val_end]
        y_test = y.iloc[val_end:]
        
        print(f"\nTime-Series Split:")
        print(f"  Train: {len(X_train)} samples ({train_end/n*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({(val_end-train_end)/n*100:.1f}%)")
        print(f"  Test (out-of-sample): {len(X_test)} samples ({(n-val_end)/n*100:.1f}%)")
    else:
        # Simple chronological split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        X_val = X_test
        y_val = y_test
        print(f"\nSimple Split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Test: {len(X_test)} samples")
    
    # Train XGBoost with early stopping and regularization
    if HAS_XGBOOST:
        print("\n" + "="*60)
        print("Training XGBoost Model (with Early Stopping)")
        print("="*60)
        
        # More conservative parameters to reduce overfitting
        model_xgb = xgb.XGBRegressor(
            n_estimators=500,  # More trees but with early stopping
            max_depth=4,  # Reduced depth
            learning_rate=0.05,  # Lower learning rate
            subsample=0.7,  # More aggressive subsampling
            colsample_bytree=0.7,  # More aggressive column sampling
            min_child_weight=5,  # Regularization
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            early_stopping_rounds=20,  # Early stopping
            random_state=42,
            n_jobs=-1
        )
        
        # Fit with early stopping on validation set
        model_xgb.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Predictions
        y_pred_train = model_xgb.predict(X_train)
        y_pred_val = model_xgb.predict(X_val)
        y_pred_test = model_xgb.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"Train R²: {train_r2:.4f}, RMSE: {train_rmse:.6f}")
        print(f"Validation R²: {val_r2:.4f}, RMSE: {val_rmse:.6f}")
        print(f"Test (OOS) R²: {test_r2:.4f}, RMSE: {test_rmse:.6f}")
        print(f"Best iteration: {model_xgb.best_iteration}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model_xgb.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance.head(10).to_string(index=False))
        
        results['xgb'] = {
            'model': model_xgb,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'importance': importance
        }
    
    # Train Lasso with proper validation
    if HAS_SKLEARN:
        print("\n" + "="*60)
        print("Training Lasso Regression")
        print("="*60)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Use validation set to find best alpha (more conservative than CV)
        from sklearn.linear_model import Lasso
        alphas = np.logspace(-4, 0, 50)  # Wider range
        best_alpha = None
        best_val_score = -np.inf
        
        for alpha in alphas:
            model_temp = Lasso(alpha=alpha, max_iter=2000, random_state=42)
            model_temp.fit(X_train_scaled, y_train)
            val_score = r2_score(y_val, model_temp.predict(X_val_scaled))
            if val_score > best_val_score:
                best_val_score = val_score
                best_alpha = alpha
        
        # Train final model with best alpha
        model_lasso = Lasso(alpha=best_alpha, max_iter=2000, random_state=42)
        model_lasso.fit(X_train_scaled, y_train)
        
        y_pred_train = model_lasso.predict(X_train_scaled)
        y_pred_val = model_lasso.predict(X_val_scaled)
        y_pred_test = model_lasso.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"Best alpha: {best_alpha:.6f}")
        print(f"Train R²: {train_r2:.4f}, RMSE: {train_rmse:.6f}")
        print(f"Validation R²: {val_r2:.4f}, RMSE: {val_rmse:.6f}")
        print(f"Test (OOS) R²: {test_r2:.4f}, RMSE: {test_rmse:.6f}")
        
        # Feature importance (absolute coefficients)
        importance = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': model_lasso.coef_,
            'abs_coefficient': np.abs(model_lasso.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        print("\nTop 10 Most Important Features (by |coefficient|):")
        print(importance.head(10).to_string(index=False))
        
        # Count non-zero features
        n_features = np.sum(np.abs(model_lasso.coef_) > 1e-6)
        print(f"\nNon-zero features: {n_features}/{len(feature_cols)}")
        
        results['lasso'] = {
            'model': model_lasso,
            'scaler': scaler,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'importance': importance
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="HFT Alpha Analysis")
    parser.add_argument("input", help="Input parquet or CSV file")
    parser.add_argument("--symbol", help="Filter by symbol (e.g., ETHUSDT)")
    parser.add_argument("--output", default="hft_features.csv", help="Output features CSV")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--no-cv", action="store_true", help="Disable cross-validation (use simple split)")
    args = parser.parse_args()
    
    file_path = Path(args.input)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return
    
    # Check if this is already a features file (has 'target' column)
    if 'target' in pd.read_csv(file_path, nrows=1).columns:
        print(f"Loading pre-processed features from {file_path}...")
        features_df = pd.read_csv(file_path)
        if args.symbol:
            features_df = features_df[features_df['symbol'] == args.symbol].copy()
        print(f"Loaded {len(features_df)} feature rows")
    else:
        # Process raw data
        features_df = process_data(file_path, symbol=args.symbol)
    
    # Save features
    output_path = Path(args.output)
    features_df.to_csv(output_path, index=False)
    print(f"\nSaved features to {output_path}")
    
    # Train models if requested
    if args.train:
        if not HAS_XGBOOST and not HAS_SKLEARN:
            print("\nError: No ML libraries available. Install xgboost and/or scikit-learn")
            return
        
        results = train_models(features_df, symbol=args.symbol, use_cv=not args.no_cv)
        
        # Save model results summary
        summary_path = output_path.parent / f"{output_path.stem}_model_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("HFT Alpha Model Results\n")
            f.write("="*60 + "\n\n")
            
            if 'xgb' in results:
                f.write("XGBoost Results:\n")
                f.write(f"  Train R²: {results['xgb']['train_r2']:.4f}\n")
                f.write(f"  Validation R²: {results['xgb']['val_r2']:.4f}\n")
                f.write(f"  Test (OOS) R²: {results['xgb']['test_r2']:.4f}\n\n")
                f.write("Top Features:\n")
                f.write(results['xgb']['importance'].head(20).to_string() + "\n\n")
            
            if 'lasso' in results:
                f.write("Lasso Results:\n")
                f.write(f"  Best Alpha: {results['lasso']['model'].alpha:.6f}\n")
                f.write(f"  Train R²: {results['lasso']['train_r2']:.4f}\n")
                f.write(f"  Validation R²: {results['lasso']['val_r2']:.4f}\n")
                f.write(f"  Test (OOS) R²: {results['lasso']['test_r2']:.4f}\n\n")
                f.write("Top Features:\n")
                f.write(results['lasso']['importance'].head(20).to_string() + "\n")
        
        print(f"\nSaved model summary to {summary_path}")


if __name__ == "__main__":
    main()

