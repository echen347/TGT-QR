from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import DATABASE_URL
import json
import pandas as pd

Base = declarative_base()

class PriceData(Base):
    """Store historical price data"""
    __tablename__ = 'price_data'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

class TradeRecord(Base):
    """Store trade records"""
    __tablename__ = 'trade_records'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # Buy/Sell
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    value_usdt = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    pnl = Column(Float, default=0)
    fees = Column(Float, default=0)

class PositionRecord(Base):
    """Store position snapshots"""
    __tablename__ = 'position_records'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    position_size = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)

class FundingRecord(Base):
    """Store funding rate payments"""
    __tablename__ = 'funding_records'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    funding_rate = Column(Float, nullable=False)
    funding_fee = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)

class SignalRecord(Base):
    """Store trading signals"""
    __tablename__ = 'signal_records'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    signal = Column(String(20), nullable=False)  # STRONG_BUY, STRONG_SELL, NEUTRAL, etc.
    ma_value = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)

class BalanceSnapshot(Base):
    """Store periodic account balance snapshots for PnL chart evolution"""
    __tablename__ = 'balance_snapshots'

    id = Column(Integer, primary_key=True)
    account_balance = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False, default=0)
    total_pnl = Column(Float, nullable=False, default=0)
    timestamp = Column(DateTime, nullable=False)

class BacktestRun(Base):
    """Store backtest run metadata"""
    __tablename__ = 'backtest_runs'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    symbols = Column(String(200), nullable=False)  # JSON array of symbols
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    parameters = Column(Text, nullable=True)  # JSON of parameters used
    status = Column(String(20), nullable=False, default='completed')  # running, completed, failed
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Pre-calculated summary metrics for the entire run
    total_pnl = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    win_rate = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    avg_return = Column(Float, nullable=True)

class BacktestTrade(Base):
    """Stores individual trades from a backtest run for detailed analysis."""
    __tablename__ = 'backtest_trades'
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, nullable=False)
    symbol = Column(String(20), nullable=False)
    entry_date = Column(DateTime, nullable=False)
    exit_date = Column(DateTime, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    pnl_pct = Column(Float, nullable=False)
    trade_type = Column(String(10), nullable=False) # long/short

class BacktestResult(Base):
    """Store backtest results"""
    __tablename__ = 'backtest_results'

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, nullable=False)  # Link to backtest run
    symbol = Column(String(20), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    total_return_pct = Column(Float, nullable=False)
    total_trades = Column(Integer, nullable=False)
    win_rate_pct = Column(Float, nullable=False)
    avg_win_pct = Column(Float, nullable=False)
    avg_loss_pct = Column(Float, nullable=False)
    max_drawdown_pct = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=False)
    alpha = Column(Float, nullable=False)
    beta = Column(Float, nullable=False)
    volatility = Column(Float, nullable=False)
    calmar_ratio = Column(Float, nullable=False)
    sortino_ratio = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)

# ============================================================================
# SCIENTIFIC METHOD / RESEARCH DASHBOARD MODELS
# ============================================================================

class Experiment(Base):
    """
    Tracks hypothesis-driven experiments using scientific method.
    Each experiment has a clear hypothesis, success criteria, and conclusions.
    """
    __tablename__ = 'experiments'

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    hypothesis = Column(Text, nullable=False)  # "Strategy X works because..."
    strategy = Column(String(50), nullable=False)  # 'ml_rf', 'macd', 'ma', etc.
    symbols = Column(String(200), nullable=False)  # JSON array of symbols
    timeframe = Column(String(10), nullable=True)  # '15', '60', etc.
    parameters = Column(Text, nullable=True)  # JSON of strategy parameters

    # Scientific method fields
    status = Column(String(20), default='pending')  # pending, running, completed, failed
    conclusion = Column(String(20), nullable=True)  # confirmed, rejected, inconclusive
    learnings = Column(Text, nullable=True)  # Key insights from experiment

    # Date ranges for train/test
    train_start = Column(DateTime, nullable=True)
    train_end = Column(DateTime, nullable=True)
    test_start = Column(DateTime, nullable=True)
    test_end = Column(DateTime, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Optional link to backtest run
    backtest_run_id = Column(Integer, nullable=True)


class ExperimentResult(Base):
    """
    Stores train/test split results for OOD validation.
    Each experiment can have multiple results (one per symbol per phase).
    """
    __tablename__ = 'experiment_results'

    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, nullable=False)  # FK to experiments
    symbol = Column(String(20), nullable=False)
    phase = Column(String(10), nullable=False)  # 'train' or 'test'

    # Core metrics
    total_return_pct = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    win_rate_pct = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    max_drawdown_pct = Column(Float, nullable=True)
    avg_win_pct = Column(Float, nullable=True)
    avg_loss_pct = Column(Float, nullable=True)

    # Statistical significance
    trades_sufficient = Column(Integer, nullable=True)  # 1 if trades >= 20, 0 otherwise
    confidence_95_lower = Column(Float, nullable=True)  # Lower bound 95% CI
    confidence_95_upper = Column(Float, nullable=True)  # Upper bound 95% CI

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class ResearchNote(Base):
    """
    Links notes to experiment timeline for tracking research progress.
    """
    __tablename__ = 'research_notes'

    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, nullable=True)  # Optional FK to experiments
    note_type = Column(String(20), nullable=False)  # 'insight', 'bug', 'idea', 'warning'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class DatabaseManager:
    """Database operations manager"""

    def __init__(self):
        self.engine = create_engine(DATABASE_URL, echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def save_price_data(self, symbol, price_data):
        """Save price data to database"""
        try:
            record = PriceData(
                symbol=symbol,
                timestamp=price_data['timestamp'],
                open_price=price_data['open'],
                high_price=price_data['high'],
                low_price=price_data['low'],
                close_price=price_data['close'],
                volume=price_data['volume']
            )
            self.session.merge(record)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Error saving price data: {e}")

    def save_trade_record(self, symbol, side, quantity, price, value_usdt, pnl=0, fees=0):
        """Save trade record to database"""
        try:
            record = TradeRecord(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                value_usdt=value_usdt,
                timestamp=datetime.utcnow(),
                pnl=pnl,
                fees=fees
            )
            self.session.add(record)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Error saving trade record: {e}")

    def save_position_record(self, symbol, position_size, entry_price, current_price, unrealized_pnl):
        """Save position snapshot to database"""
        try:
            record = PositionRecord(
                symbol=symbol,
                position_size=position_size,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                timestamp=datetime.utcnow()
            )
            self.session.add(record)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Error saving position record: {e}")

    def save_funding_record(self, symbol, funding_rate, funding_fee):
        """Save funding rate payment to database"""
        try:
            record = FundingRecord(
                symbol=symbol,
                funding_rate=funding_rate,
                funding_fee=funding_fee,
                timestamp=datetime.utcnow()
            )
            self.session.add(record)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Error saving funding record: {e}")

    def save_signal_record(self, symbol, signal, ma_value, current_price):
        """Save signal to database"""
        try:
            record = SignalRecord(
                symbol=symbol,
                signal=signal,
                ma_value=ma_value,
                current_price=current_price,
                timestamp=datetime.utcnow()
            )
            self.session.add(record)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Error saving signal record: {e}")

    def save_balance_snapshot(self, account_balance, unrealized_pnl=0, total_pnl=0):
        """Save periodic account balance snapshot for PnL chart evolution"""
        try:
            snapshot = BalanceSnapshot(
                account_balance=account_balance,
                unrealized_pnl=unrealized_pnl,
                total_pnl=total_pnl,
                timestamp=datetime.utcnow()
            )
            self.session.add(snapshot)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Error saving balance snapshot: {e}")

    def get_balance_snapshots(self, hours=24):
        """Get balance snapshots for the last N hours"""
        try:
            since_time = datetime.utcnow() - timedelta(hours=hours)
            snapshots = self.session.query(BalanceSnapshot)\
                .filter(BalanceSnapshot.timestamp >= since_time)\
                .order_by(BalanceSnapshot.timestamp.asc())\
                .all()
            return [{
                'timestamp': s.timestamp,
                'account_balance': s.account_balance,
                'unrealized_pnl': s.unrealized_pnl,
                'total_pnl': s.total_pnl
            } for s in snapshots]
        except Exception as e:
            print(f"Error getting balance snapshots: {e}")
            return []

    def create_backtest_run(self, name, description, symbols, start_date, end_date, parameters=None):
        """Create a new backtest run record"""
        try:
            run = BacktestRun(
                name=name,
                description=description,
                symbols=json.dumps(symbols),
                start_date=start_date,
                end_date=end_date,
                parameters=json.dumps(parameters) if parameters else None,
                status='running'
            )
            self.session.add(run)
            self.session.commit()
            return run.id
        except Exception as e:
            self.session.rollback()
            print(f"Error creating backtest run: {e}")
            return None

    def update_backtest_run_status(self, run_id, status):
        """Update backtest run status"""
        try:
            run = self.session.query(BacktestRun).filter(BacktestRun.id == run_id).first()
            if run:
                run.status = status
                self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Error updating backtest run status: {e}")

    def update_backtest_run_summary(self, run_id, summary_data):
        """Updates a backtest run with summary metrics."""
        try:
            run = self.session.query(BacktestRun).filter_by(id=run_id).one_or_none()
            if run:
                run.total_pnl = summary_data.get('total_pnl')
                run.total_trades = summary_data.get('total_trades')
                run.win_rate = summary_data.get('win_rate')
                run.sharpe_ratio = summary_data.get('sharpe_ratio')
                run.max_drawdown = summary_data.get('max_drawdown')
                run.avg_return = summary_data.get('avg_return')
                self.session.commit()
                # Use app.logger if available, otherwise print
                try:
                    from flask import current_app
                    current_app.logger.info(f"Updated summary for run {run_id}.")
                except (ImportError, RuntimeError):
                    print(f"Updated summary for run {run_id}.")
            else:
                # Use app.logger if available, otherwise print
                try:
                    from flask import current_app
                    current_app.logger.warning(f"Could not find run {run_id} to update summary.")
                except (ImportError, RuntimeError):
                    print(f"Could not find run {run_id} to update summary.")
        except Exception as e:
            self.session.rollback()
            # Use app.logger if available, otherwise print
            try:
                from flask import current_app
                current_app.logger.error(f"Error updating backtest run summary for run {run_id}: {e}")
            except (ImportError, RuntimeError):
                print(f"Error updating backtest run summary for run {run_id}: {e}")
            raise

    def save_backtest_trades(self, run_id, symbol, trades_df):
        """Saves the detailed trades from a backtest dataframe to the database."""
        if trades_df.empty:
            return
        
        try:
            records = []
            for _, row in trades_df.iterrows():
                # Ensure we have the necessary columns before proceeding
                if 'exit_date' in row and pd.notna(row['exit_date']):
                    record = BacktestTrade(
                        run_id=run_id,
                        symbol=symbol,
                        entry_date=row['entry_date'],
                        exit_date=row['exit_date'],
                        entry_price=row['entry_price'],
                        exit_price=row['exit_price'],
                        pnl_pct=row['pnl'],
                        trade_type=row['type']
                    )
                    records.append(record)
            
            if records:
                self.session.bulk_save_objects(records)
                self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Error saving backtest trades for run_id {run_id}: {e}")

    def save_backtest_result(self, run_id, symbol, start_date, end_date, metrics):
        """Save backtest results to database with run_id"""
        try:
            record = BacktestResult(
                run_id=run_id,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                total_return_pct=metrics.get('total_return_pct', 0),
                total_trades=metrics.get('total_trades', 0),
                win_rate_pct=metrics.get('win_rate_pct', 0),
                avg_win_pct=metrics.get('avg_win_pct', 0),
                avg_loss_pct=metrics.get('avg_loss_pct', 0),
                max_drawdown_pct=metrics.get('max_drawdown_pct', 0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0),
                alpha=metrics.get('alpha', 0),
                beta=metrics.get('beta', 0),
                volatility=metrics.get('volatility', 0),
                calmar_ratio=metrics.get('calmar_ratio', 0),
                sortino_ratio=metrics.get('sortino_ratio', 0),
                timestamp=datetime.utcnow()
            )
            self.session.add(record)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Error saving backtest result: {e}")

    def get_backtest_runs(self, limit=20):
        """Get recent backtest runs"""
        try:
            runs = self.session.query(BacktestRun)\
                .order_by(BacktestRun.created_at.desc())\
                .limit(limit)\
                .all()

            return [{
                'id': r.id,
                'name': r.name,
                'description': r.description,
                'symbols': json.loads(r.symbols),
                'start_date': r.start_date.strftime('%Y-%m-%d'),
                'end_date': r.end_date.strftime('%Y-%m-%d'),
                'status': r.status,
                'parameters': json.loads(r.parameters) if r.parameters else {},
                'created_at': r.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                # Include the pre-calculated summary data
                'summary': {
                    'total_pnl': r.total_pnl,
                    'total_trades': r.total_trades,
                    'win_rate': r.win_rate,
                    'sharpe_ratio': r.sharpe_ratio,
                    'max_drawdown': r.max_drawdown,
                    'avg_return': r.avg_return
                }
            } for r in runs]
        except Exception as e:
            print(f"Error getting backtest runs: {e}")
            return []

    def get_backtest_results(self, run_id=None, limit=50):
        """Get backtest results, optionally filtered by run_id"""
        try:
            query = self.session.query(BacktestResult)

            if run_id:
                query = query.filter(BacktestResult.run_id == run_id)

            records = query.order_by(BacktestResult.timestamp.desc()).limit(limit).all()

            results = []
            for r in records:
                results.append({
                    'id': int(r.id),
                    'run_id': int(r.run_id),
                    'symbol': r.symbol,
                    'start_date': r.start_date.strftime('%Y-%m-%d'),
                    'end_date': r.end_date.strftime('%Y-%m-%d'),
                    'total_return_pct': float(r.total_return_pct),
                    'total_trades': int(r.total_trades),
                    'win_rate_pct': float(r.win_rate_pct),
                    'sharpe_ratio': float(r.sharpe_ratio),
                    'max_drawdown_pct': float(r.max_drawdown_pct),
                    'pnl': float(r.total_pnl) if hasattr(r, 'total_pnl') else 0,  # Use actual PnL data
                    'timestamp': r.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })

            return results
        except Exception as e:
            print(f"Error getting backtest results: {e}")
            return []

    def get_chart_data_for_run(self, run_id):
        """Optimized query to get only necessary data for backtest charts."""
        try:
            # Note: The BacktestResult model does not store individual PnL values.
            # This is a limitation of the current schema.
            # For this optimization to work, we would need to store trade history
            # from backtests. As a workaround, we return placeholder data.
            # This highlights a schema design issue to be addressed later.
            
            # Use actual backtest data instead of placeholders:
            results = self.session.query(
                BacktestResult.symbol, 
                BacktestResult.total_return_pct, 
                BacktestResult.max_drawdown_pct,
                BacktestResult.timestamp
            ).filter(BacktestResult.run_id == run_id).all()

            return [{
                'symbol': r.symbol,
                'pnl': r.total_return_pct,  # Use actual backtest return data
                'timestamp': r.timestamp.strftime('%Y-%m-%d %H:%M:%S') if r.timestamp else datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            } for r in results]

        except Exception as e:
            print(f"Error getting chart data for run {run_id}: {e}")
            return []

    def get_trades_for_run(self, run_id):
        """Fetches all individual trades for a specific backtest run."""
        try:
            trades = self.session.query(BacktestTrade).filter(BacktestTrade.run_id == run_id).all()
            return [{
                'symbol': t.symbol,
                'entry_date': t.entry_date.isoformat(),
                'exit_date': t.exit_date.isoformat(),
                'pnl_pct': t.pnl_pct,
                'trade_type': t.trade_type
            } for t in trades]
        except Exception as e:
            print(f"Error getting trades for run {run_id}: {e}")
            return []

    def clear_all_backtests(self):
        """Deletes all backtest runs, results, and trades from the database."""
        try:
            self.session.query(BacktestTrade).delete()
            self.session.query(BacktestResult).delete()
            self.session.query(BacktestRun).delete()
            self.session.commit()
            print("All backtest data has been cleared.")
        except Exception as e:
            self.session.rollback()
            print(f"Error clearing backtest data: {e}")

    def get_recent_prices(self, symbol, limit=100):
        """Get recent price data for a symbol"""
        try:
            records = self.session.query(PriceData)\
                .filter(PriceData.symbol == symbol)\
                .order_by(PriceData.timestamp.desc())\
                .limit(limit)\
                .all()

            prices = []
            for record in reversed(records):  # Chronological order
                prices.append({
                    'timestamp': record.timestamp,
                    'open': record.open_price,
                    'high': record.high_price,
                    'low': record.low_price,
                    'close': record.close_price,
                    'volume': record.volume
                })

            return prices
        except Exception as e:
            print(f"Error getting recent prices: {e}")
            return []

    def get_historical_prices(self, symbol, start_date=None, end_date=None):
        """Get historical price data for a symbol within a date range"""
        try:
            query = self.session.query(PriceData).filter(PriceData.symbol == symbol)
            
            if start_date:
                query = query.filter(PriceData.timestamp >= start_date)
            if end_date:
                query = query.filter(PriceData.timestamp <= end_date)
            
            records = query.order_by(PriceData.timestamp.asc()).all()

            prices = []
            for record in records:
                prices.append({
                    'timestamp': record.timestamp,
                    'open': record.open_price,
                    'high': record.high_price,
                    'low': record.low_price,
                    'close': record.close_price,
                    'volume': record.volume
                })

            return prices
        except Exception as e:
            print(f"Error getting historical prices: {e}")
            return []

    def save_price_data(self, symbol, price, volume):
        """Save current price data for a symbol"""
        try:
            from datetime import datetime
            price_record = PriceData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                open_price=price,
                high_price=price,
                low_price=price,
                close_price=price,
                volume=volume
            )
            self.session.add(price_record)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Error saving price data for {symbol}: {e}")

    def get_pnl_summary(self, hours=24):
        """Get PnL summary for the last N hours"""
        try:
            since_time = datetime.utcnow() - timedelta(hours=hours)

            # Get trade PnL
            trades = self.session.query(TradeRecord)\
                .filter(TradeRecord.timestamp >= since_time)\
                .all()

            total_trade_pnl = sum(trade.pnl for trade in trades)

            # Get funding fees
            fundings = self.session.query(FundingRecord)\
                .filter(FundingRecord.timestamp >= since_time)\
                .all()

            total_funding = sum(funding.funding_fee for funding in fundings)

            # Get current unrealized PnL (from latest position records)
            positions = self.session.query(PositionRecord)\
                .filter(PositionRecord.timestamp >= since_time)\
                .all()

            current_unrealized_pnl = 0
            if positions:
                # Get the most recent position record for each symbol
                latest_positions = {}
                for pos in positions:
                    if pos.symbol not in latest_positions:
                        latest_positions[pos.symbol] = pos

                current_unrealized_pnl = sum(pos.unrealized_pnl for pos in latest_positions.values())

            return {
                'total_trade_pnl': total_trade_pnl,
                'total_funding_fees': total_funding,
                'current_unrealized_pnl': current_unrealized_pnl,
                'net_pnl': total_trade_pnl + total_funding + current_unrealized_pnl
            }

        except Exception as e:
            print(f"Error getting PnL summary: {e}")
            return {'total_trade_pnl': 0, 'total_funding_fees': 0, 'current_unrealized_pnl': 0, 'net_pnl': 0}

    # =========================================================================
    # EXPERIMENT / RESEARCH DASHBOARD METHODS
    # =========================================================================

    def create_experiment(self, name, hypothesis, strategy, symbols, timeframe=None,
                          parameters=None, train_start=None, train_end=None,
                          test_start=None, test_end=None):
        """Create a new experiment"""
        try:
            experiment = Experiment(
                name=name,
                hypothesis=hypothesis,
                strategy=strategy,
                symbols=json.dumps(symbols) if isinstance(symbols, list) else symbols,
                timeframe=timeframe,
                parameters=json.dumps(parameters) if parameters else None,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                status='pending'
            )
            self.session.add(experiment)
            self.session.commit()
            return experiment.id
        except Exception as e:
            self.session.rollback()
            print(f"Error creating experiment: {e}")
            return None

    def get_experiment(self, experiment_id):
        """Get a single experiment by ID"""
        try:
            exp = self.session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if exp:
                return {
                    'id': exp.id,
                    'name': exp.name,
                    'hypothesis': exp.hypothesis,
                    'strategy': exp.strategy,
                    'symbols': json.loads(exp.symbols) if exp.symbols else [],
                    'timeframe': exp.timeframe,
                    'parameters': json.loads(exp.parameters) if exp.parameters else {},
                    'status': exp.status,
                    'conclusion': exp.conclusion,
                    'learnings': exp.learnings,
                    'train_start': exp.train_start,
                    'train_end': exp.train_end,
                    'test_start': exp.test_start,
                    'test_end': exp.test_end,
                    'created_at': exp.created_at,
                    'completed_at': exp.completed_at
                }
            return None
        except Exception as e:
            print(f"Error getting experiment: {e}")
            return None

    def get_all_experiments(self, limit=50):
        """Get all experiments, most recent first"""
        try:
            exps = self.session.query(Experiment).order_by(Experiment.created_at.desc()).limit(limit).all()
            return [{
                'id': exp.id,
                'name': exp.name,
                'hypothesis': exp.hypothesis,
                'strategy': exp.strategy,
                'symbols': json.loads(exp.symbols) if exp.symbols else [],
                'status': exp.status,
                'conclusion': exp.conclusion,
                'created_at': exp.created_at
            } for exp in exps]
        except Exception as e:
            print(f"Error getting experiments: {e}")
            return []

    def update_experiment(self, experiment_id, **kwargs):
        """Update an experiment"""
        try:
            exp = self.session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if exp:
                for key, value in kwargs.items():
                    if hasattr(exp, key):
                        if key in ['symbols', 'parameters'] and isinstance(value, (list, dict)):
                            value = json.dumps(value)
                        setattr(exp, key, value)
                self.session.commit()
                return True
            return False
        except Exception as e:
            self.session.rollback()
            print(f"Error updating experiment: {e}")
            return False

    def save_experiment_result(self, experiment_id, symbol, phase, metrics):
        """Save experiment result (train or test metrics)"""
        try:
            result = ExperimentResult(
                experiment_id=experiment_id,
                symbol=symbol,
                phase=phase,
                total_return_pct=metrics.get('total_return_pct'),
                total_trades=metrics.get('total_trades'),
                win_rate_pct=metrics.get('win_rate_pct'),
                sharpe_ratio=metrics.get('sharpe_ratio'),
                sortino_ratio=metrics.get('sortino_ratio'),
                max_drawdown_pct=metrics.get('max_drawdown_pct'),
                avg_win_pct=metrics.get('avg_win_pct'),
                avg_loss_pct=metrics.get('avg_loss_pct'),
                trades_sufficient=1 if metrics.get('total_trades', 0) >= 20 else 0,
                confidence_95_lower=metrics.get('confidence_95_lower'),
                confidence_95_upper=metrics.get('confidence_95_upper')
            )
            self.session.add(result)
            self.session.commit()
            return result.id
        except Exception as e:
            self.session.rollback()
            print(f"Error saving experiment result: {e}")
            return None

    def get_experiment_results(self, experiment_id):
        """Get all results for an experiment"""
        try:
            results = self.session.query(ExperimentResult)\
                .filter(ExperimentResult.experiment_id == experiment_id)\
                .order_by(ExperimentResult.phase, ExperimentResult.symbol).all()

            return [{
                'id': r.id,
                'symbol': r.symbol,
                'phase': r.phase,
                'total_return_pct': r.total_return_pct,
                'total_trades': r.total_trades,
                'win_rate_pct': r.win_rate_pct,
                'sharpe_ratio': r.sharpe_ratio,
                'sortino_ratio': r.sortino_ratio,
                'max_drawdown_pct': r.max_drawdown_pct,
                'avg_win_pct': r.avg_win_pct,
                'avg_loss_pct': r.avg_loss_pct,
                'trades_sufficient': r.trades_sufficient,
                'confidence_95_lower': r.confidence_95_lower,
                'confidence_95_upper': r.confidence_95_upper
            } for r in results]
        except Exception as e:
            print(f"Error getting experiment results: {e}")
            return []

    def add_research_note(self, content, note_type='insight', experiment_id=None):
        """Add a research note"""
        try:
            note = ResearchNote(
                experiment_id=experiment_id,
                note_type=note_type,
                content=content
            )
            self.session.add(note)
            self.session.commit()
            return note.id
        except Exception as e:
            self.session.rollback()
            print(f"Error adding research note: {e}")
            return None

    def get_research_notes(self, experiment_id=None, limit=50):
        """Get research notes"""
        try:
            query = self.session.query(ResearchNote)
            if experiment_id:
                query = query.filter(ResearchNote.experiment_id == experiment_id)
            notes = query.order_by(ResearchNote.created_at.desc()).limit(limit).all()

            return [{
                'id': n.id,
                'experiment_id': n.experiment_id,
                'note_type': n.note_type,
                'content': n.content,
                'created_at': n.created_at
            } for n in notes]
        except Exception as e:
            print(f"Error getting research notes: {e}")
            return []

    def close(self):
        """Close database connection"""
        self.session.close()

# Global database instance
db_manager = DatabaseManager()
