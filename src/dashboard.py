from flask import Flask, render_template, jsonify, request
import plotly.graph_objs as go
import plotly.utils
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import *
from database import db_manager
# The global instances will be passed in, not imported
# from risk_manager import risk_manager 
# from strategy import strategy
import json
from flask_moment import Moment
import random
import logging

app = Flask(__name__)
# Rename to avoid conflicts
flask_moment = Moment(app)

# These will hold the shared instances
strategy_instance = None
risk_manager_instance = None

class TradingDashboard:
    """Handles data fetching for the Flask dashboard"""

    def __init__(self, strategy, risk_manager):
        self.db = db_manager
        self.strategy = strategy
        self.risk_manager = risk_manager

    def get_system_status(self):
        """Get current system status from the risk manager"""
        bot_is_active = not self.risk_manager.is_stopped
        return {
            'dashboard_status': 'Active',
            'bot_status': 'Active' if bot_is_active else 'Inactive',
            'api_status': 'OK' if self.strategy.session else 'Error',  # Check actual API connection
        }

    def get_market_data(self):
        """Get live market data from the strategy instance"""
        return self.strategy.market_state

    def get_alerts(self):
        """Get current alerts from the risk manager"""
        return self.risk_manager.get_alerts()

    def get_pnl_chart_data(self):
        """Get PnL data for charting - tracking account balance over time"""
        try:
            # Get all trade records to calculate realized PnL
            from database import TradeRecord
            trades = self.db.session.query(TradeRecord)\
                .order_by(TradeRecord.timestamp)\
                .all()

            if not trades:
                # No trades yet, just show current account balance
                try:
                    balance = self.strategy.session.get_wallet_balance(
                        accountType="UNIFIED",
                        coin="USDT"
                    )
                    if balance['retCode'] == 0:
                        wallet_balance = float(balance['result']['list'][0]['coin'][0]['walletBalance'])
                        from pytz import timezone
                        est = timezone('US/Eastern')
                        now_est = datetime.utcnow().replace(tzinfo=timezone('UTC')).astimezone(est)
                        return {
                            'timestamps': [now_est.strftime('%m/%d %I:%M%p EST')],
                            'pnl_values': [0],
                            'cumulative_pnl': [0],
                            'current_pnl': 0,
                            'account_balance': wallet_balance
                        }
                except:
                    pass
                return {'error': 'No trading activity yet. Waiting for first trade...'}

            # Calculate cumulative PnL from trades
            timestamps = []
            cumulative_pnl = []
            running_pnl = 0
            
            for trade in trades:
                timestamps.append(trade.timestamp)
                # Use actual PnL from database, fallback to 0 if not available
                trade_pnl = trade.pnl if hasattr(trade, 'pnl') and trade.pnl is not None else 0
                running_pnl += trade_pnl
                cumulative_pnl.append(running_pnl)

            # Add current unrealized PnL from open positions
            current_positions = self.strategy.get_current_positions()
            unrealized_pnl = 0
            for symbol, pos in current_positions.items():
                if pos.get('position_size', 0) != 0:
                    # Calculate unrealized PnL
                    unrealized_pnl += pos.get('unrealized_pnl', 0)
            
            total_pnl = running_pnl + unrealized_pnl

            # Get current account balance
            account_balance = 0
            try:
                balance = self.strategy.session.get_wallet_balance(
                    accountType="UNIFIED",
                    coin="USDT"
                )
                if balance['retCode'] == 0:
                    account_balance = float(balance['result']['list'][0]['coin'][0]['walletBalance'])
            except Exception as e:
                logging.error(f"Failed to fetch wallet balance: {e}")

            # Convert timestamps to EST
            from pytz import timezone
            est = timezone('US/Eastern')
            timestamps_est = [ts.replace(tzinfo=timezone('UTC')).astimezone(est) for ts in timestamps[-50:]]
            
            return {
                'timestamps': [ts.strftime('%m/%d %I:%M%p EST') for ts in timestamps_est],  # Last 50 points in EST
                'pnl_values': [cumulative_pnl[i] - cumulative_pnl[i-1] if i > 0 else cumulative_pnl[0] 
                              for i in range(max(0, len(cumulative_pnl)-50), len(cumulative_pnl))],
                'cumulative_pnl': cumulative_pnl[-50:],
                'current_pnl': total_pnl,
                'account_balance': account_balance
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            logging.error(f"PnL chart error: {e}")
            return {'error': 'Unable to load PnL data'}

    def get_positions_data(self):
        """Get current positions data from Bybit API"""
        try:
            # Get positions directly from the strategy (which queries Bybit)
            current_positions = self.strategy.get_current_positions()
            
            positions_list = []
            for symbol, pos in current_positions.items():
                position_size = pos.get('position_size', 0)
                if position_size != 0:  # Only show open positions
                    entry_price = pos.get('entry_price', 0)
                    current_price = pos.get('current_price', 0)
                    side = pos.get('side', 'N/A')
                    
                    # Calculate PnL
                    if entry_price > 0 and current_price > 0:
                        direction = 1 if side == 'Buy' else -1
                        pnl = (current_price - entry_price) * abs(position_size) * direction
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100 * direction
                    else:
                        pnl = 0
                        pnl_pct = 0
                    
                    positions_list.append({
                        'symbol': symbol,
                        'size': position_size,
                        'side': side,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct
                    })
            
            return positions_list
        except Exception as e:
            logging.error(f"Error fetching positions: {e}")
            return []

    def get_signals_data(self):
        """Get recent signals data"""
        try:
            signals = self.db.session.query(SignalRecord)\
                .filter(SignalRecord.timestamp >= datetime.utcnow() - timedelta(hours=6))\
                .all()

            return [{
                'symbol': s.symbol,
                'signal': s.signal,
                'ma_value': s.ma_value,
                'current_price': s.current_price,
                'timestamp': s.timestamp.strftime('%H:%M:%S')
            } for s in signals]
        except Exception as e:
            return []

    def get_risk_status(self):
        """Get current risk status"""
        return self.risk_manager.get_risk_status()

    def create_pnl_chart(self):
        """Create PnL chart using Plotly"""
        pnl_data = self.get_pnl_chart_data()

        if 'error' in pnl_data:
            return "<p>No chart data available</p>"

        fig = go.Figure()

        # Add cumulative PnL line
        fig.add_trace(go.Scatter(
            x=pnl_data['timestamps'],
            y=pnl_data['cumulative_pnl'],
            mode='lines+markers',
            name='Cumulative PnL',
            line=dict(color='blue', width=2)
        ))

        # Add individual trade PnL as bars
        colors = ['green' if x >= 0 else 'red' for x in pnl_data['pnl_values']]
        fig.add_trace(go.Bar(
            x=pnl_data['timestamps'],
            y=pnl_data['pnl_values'],
            name='Trade PnL',
            marker_color=colors,
            opacity=0.6
        ))

        fig.update_layout(
            title='24h PnL Performance',
            xaxis_title='Time',
            yaxis_title='PnL (USDT)',
            template='plotly_white',
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return fig.to_html(full_html=False, include_plotlyjs=True)

@app.route('/')
def dashboard():
    """Render the main dashboard"""
    # Use the shared instances
    trading_dashboard = TradingDashboard(strategy_instance, risk_manager_instance)
    pnl_chart_data = trading_dashboard.get_pnl_chart_data()
    positions_data = trading_dashboard.get_positions_data()
    signals_data = trading_dashboard.get_signals_data()
    risk = trading_dashboard.get_risk_status()
    system_status = trading_dashboard.get_system_status()
    alerts = trading_dashboard.get_alerts()
    market_context = trading_dashboard.get_market_data() # Renamed for clarity
    last_updated_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    return render_template('dashboard.html',
                           pnl_chart=pnl_chart_data,
                           positions=positions_data,
                           signals=signals_data,
                           risk=risk,
                           system_status=system_status,
                           alerts=alerts,
                           market_context=market_context, # Pass with the new name
                           last_updated=last_updated_str,
                           utc_now=datetime.utcnow(),
                           moment=flask_moment) # Pass the object itself

@app.route('/api/status')
def api_status():
    """API endpoint for real-time status"""
    dashboard = TradingDashboard(strategy_instance, risk_manager_instance)

    return jsonify({
        'risk_status': dashboard.get_risk_status(),
        'positions': dashboard.get_positions_data(),
        'recent_signals': dashboard.get_signals_data()[:5],  # Last 5 signals
        'pnl_summary': dashboard.get_pnl_chart_data()
    })

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Endpoint to trigger the emergency stop and close all positions."""
    data = request.get_json()
    if not data or data.get('password') != 'chaewon':
        return jsonify({'status': 'error', 'message': 'Incorrect password.'}), 401

    try:
        logging.warning("EMERGENCY STOP ACTIVATED VIA DASHBOARD - CLOSING ALL POSITIONS")
        
        # First, get all current positions
        positions = strategy_instance.get_current_positions()
        closed_positions = []
        errors = []
        
        # Close each open position at market price
        for symbol, position in positions.items():
            if position.get('position_size', 0) != 0:
                try:
                    # Determine the closing side (opposite of current position)
                    close_side = "Sell" if position['side'] == "Buy" else "Buy"
                    position_value = abs(position.get('position_value', 0))
                    
                    logging.warning(f"EMERGENCY: Closing {symbol} position - {close_side} {position_value} USDT")
                    
                    # Place market order to close position
                    success = strategy_instance.place_order(symbol, close_side, position_value)
                    
                    if success:
                        closed_positions.append(f"{symbol}: {close_side} {position_value} USDT")
                    else:
                        errors.append(f"{symbol}: Failed to close")
                        
                except Exception as e:
                    error_msg = f"{symbol}: {str(e)}"
                    errors.append(error_msg)
                    logging.error(f"Error closing {symbol}: {e}")
        
        # Now activate emergency stop to prevent new trades
        risk_manager_instance.emergency_stop()
        
        message = f"Emergency stop activated. Closed {len(closed_positions)} position(s)."
        if closed_positions:
            message += f" Closed: {', '.join(closed_positions)}."
        if errors:
            message += f" Errors: {', '.join(errors)}."
            
        return jsonify({
            'status': 'success' if not errors else 'partial',
            'message': message,
            'closed': closed_positions,
            'errors': errors
        })
    except Exception as e:
        logging.error(f"Emergency stop error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/set_parameter', methods=['POST'])
def api_set_parameter():
    """Set strategy parameters"""
    try:
        data = request.get_json()
        param = data.get('parameter')
        value = data.get('value')

        # In a real system, this would update strategy parameters
        # For now, just log the request
        print(f"Parameter update requested: {param} = {value}")

        return jsonify({
            'success': True,
            'message': f'Parameter {param} set to {value}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/backtest')
def backtest_dashboard():
    return ("Backtest dashboard disabled. Use CLI backtesting.", 404)

@app.route('/api/backtest_results')
def api_backtest_results():
    return jsonify({'error': 'Backtest API disabled. Use CLI backtesting.'}), 404

@app.route('/api/backtest_chart_data')
def api_backtest_chart_data():
    return jsonify({'error': 'Backtest API disabled. Use CLI backtesting.'}), 404

@app.route('/api/clear_backtests', methods=['POST'])
def api_clear_backtests():
    """API endpoint to delete all backtest data."""
    try:
        db_manager.clear_all_backtests()
        return jsonify({'success': True, 'message': 'All backtest data has been cleared.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/delete_backtest/<int:run_id>', methods=['DELETE'])
def api_delete_backtest(run_id):
    """API endpoint to delete a specific backtest run."""
    try:
        # Delete trades first (foreign key constraint)
        db_manager.session.query(db_manager.BacktestTrade).filter(db_manager.BacktestTrade.run_id == run_id).delete()
        # Delete results
        db_manager.session.query(db_manager.BacktestResult).filter(db_manager.BacktestResult.run_id == run_id).delete()
        # Delete the run
        db_manager.session.query(db_manager.BacktestRun).filter(db_manager.BacktestRun.id == run_id).delete()
        db_manager.session.commit()
        return jsonify({'success': True, 'message': f'Backtest run {run_id} has been deleted.'})
    except Exception as e:
        db_manager.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/backtest_runs')
def api_backtest_runs():
    return jsonify({'error': 'Backtest API disabled. Use CLI backtesting.'}), 404

@app.route('/api/run_backtest', methods=['POST'])
def api_run_backtest():
    return jsonify({'error': 'Backtest API disabled. Use CLI backtesting.'}), 404

@app.route('/api/market_data')
def api_market_data():
    """Provides aggregated market data for the advanced charts."""
    try:
        # Fetch recent data for all symbols in SYMBOLS
        symbols = SYMBOLS
        all_data = {}
        
        # Use a shorter period for dashboard visuals to keep it fast
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24) # 24 hours of data
        
        for symbol in symbols:
            # Fetch only last 50 candles for dashboard speed
            klines = db_manager.get_recent_prices(symbol, limit=50)
            if klines:
                df = pd.DataFrame(klines)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                all_data[symbol] = df
        
        if not all_data:
            return jsonify({'error': 'No market data available'}), 500

        # --- Prepare data for charts ---
        # Labels (timestamps) - use first available symbol as reference
        reference_symbol = next(iter(all_data.keys()))
        labels = all_data[reference_symbol].index.strftime('%H:%M').tolist()
        
        # 1. Price Data
        prices = {s.replace('USDT', '').lower(): df['close'].tolist() for s, df in all_data.items()}

        # 2. Volume Data
        volumes = {s.replace('USDT', '').lower(): df['volume'].tolist() for s, df in all_data.items()}

        # 3. Volatility Data (ATR)
        volatility = {}
        for symbol, df in all_data.items():
            if len(df) < 2:
                volatility[symbol.replace('USDT', '').lower()] = [0] * len(df)
                continue

            # Calculate True Range
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))

            # Handle NaN values in tr2 and tr3 for first row
            tr2 = tr2.fillna(df['high'] - df['low'])
            tr3 = tr3.fillna(df['high'] - df['low'])

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14, min_periods=1).mean()
            volatility[symbol.replace('USDT', '').lower()] = atr.fillna(0).tolist()

        # 4. MACD Data (calculate for each symbol)
        macd_data = {}
        for symbol, df in all_data.items():
            if len(df) >= 26:  # Need enough data for MACD
                exp1 = df['close'].ewm(span=12, adjust=False).mean()
                exp2 = df['close'].ewm(span=26, adjust=False).mean()
                macd_line = exp1 - exp2
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                macd_data[symbol] = {
                    'macd': macd_line.fillna(0).tolist(),
                    'signal': signal_line.fillna(0).tolist()
                }
            else:
                # Default values for insufficient data
                macd_data[symbol] = {
                    'macd': [0] * len(df),
                    'signal': [0] * len(df)
                }

        macd = {
            'btc_macd': macd_data.get('BTCUSDT', macd_data.get(reference_symbol, {})).get('macd', [0]),
            'btc_signal': macd_data.get('BTCUSDT', macd_data.get(reference_symbol, {})).get('signal', [0])
        }

        # 5. Technical Indicators
        indicators = {}
        for symbol, df in all_data.items():
            if len(df) < 14:  # Minimum data for basic calculations
                indicators[symbol.replace('USDT', '').lower()] = {
                    'rsi': 50,  # Neutral RSI
                    'macd': '0.00/0.00/0.00',
                    'bb': f"{df['close'].iloc[-1] * 0.98:.0f}-{df['close'].iloc[-1] * 1.02:.0f}",
                    'stoch': '50/50'
                }
                continue

            # RSI Calculation (use available window size)
            window_size = min(14, len(df) - 1)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window_size).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window_size).mean()

            # Avoid division by zero
            rs = gain / loss.replace(0, 1e-10)  # Replace 0 with small value
            rsi = 100 - (100 / (1 + rs.iloc[-1]))

            # Bollinger Bands (use available data)
            bb_window = min(20, len(df))
            sma = df['close'].rolling(window=bb_window).mean().iloc[-1]
            std = df['close'].rolling(window=bb_window).std().iloc[-1]
            bb_upper = sma + (std * 2)
            bb_lower = sma - (std * 2)

            # Stochastic Oscillator (use available data)
            stoch_window = min(14, len(df))
            lowest_low = df['low'].rolling(window=stoch_window).min().iloc[-1]
            highest_high = df['high'].rolling(window=stoch_window).max().iloc[-1]
            stoch_k = 100 * ((df['close'].iloc[-1] - lowest_low) / max(highest_high - lowest_low, 1e-10))

            # Get MACD for current symbol
            symbol_macd = macd_data.get(symbol, {}).get('macd', [0])
            symbol_signal = macd_data.get(symbol, {}).get('signal', [0])
            macd_value = f"{symbol_macd[-1]:.2f}/{symbol_signal[-1]:.2f}/{symbol_macd[-1] - symbol_signal[-1]:.2f}" if symbol_macd else '0.00/0.00/0.00'

            indicators[symbol.replace('USDT', '').lower()] = {
                'rsi': round(rsi, 1),
                'macd': macd_value,
                'bb': f"{bb_lower:.0f}-{bb_upper:.0f}",
                'stoch': f"{stoch_k:.0f}/50"
            }

        # Construct the final JSON response
        chart_data = {
            'labels': labels,
            'prices': prices,
            'volumes': volumes,
            'volatility': volatility,
            'macd': macd,
            'indicators': indicators,
            # Correlation and Depth are complex and will be left as placeholders for now
        }

        return jsonify(chart_data)

    except Exception as e:
        # Log the error for debugging
        print(f"Error in /api/market_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to fetch market data'}), 500


@app.route('/logs')
def logs_page():
    return render_template('logs.html')

@app.route('/api/logs')
def api_logs():
    try:
        log_file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'trading.log')
        if not os.path.exists(log_file_path):
            return jsonify({'error': 'Log file not found.'}), 404

        with open(log_file_path, 'r') as f:
            # Read the last 200 lines for performance
            lines = f.readlines()[-200:]
            return jsonify({'logs': ''.join(lines)})
    except Exception as e:
        logging.error(f"Error reading log file: {e}")
        return jsonify({'error': 'An error occurred while reading logs.'}), 500

@app.route('/reset_stop', methods=['POST'])
def reset_stop():
    """Resets the emergency stop."""
    data = request.get_json()
    if not data or data.get('password') != 'chaewon':
        return jsonify({'status': 'error', 'message': 'Incorrect password.'}), 401
    
    try:
        risk_manager_instance.reset_stop()
        logging.warning("EMERGENCY STOP RESET VIA DASHBOARD")
        return jsonify({'status': 'success', 'message': 'Emergency stop reset. Trading is re-enabled.'})
    except Exception as e:
        logging.error(f"Error resetting emergency stop: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Function to run the dashboard with shared instances
def run_dashboard(strategy, risk_manager):
    global strategy_instance, risk_manager_instance
    strategy_instance = strategy
    risk_manager_instance = risk_manager
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False)

if __name__ == '__main__':
    # This part is for standalone testing, which needs to be updated
    # from src.strategy import MovingAverageStrategy
    # from src.risk_manager import RiskManager
    # strategy = MovingAverageStrategy()
    # risk_manager = RiskManager()
    # run_dashboard(strategy, risk_manager)
    pass
