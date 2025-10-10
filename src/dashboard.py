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
            'api_status': 'OK',  # Placeholder - can be improved
        }

    def get_market_data(self):
        """Get live market data from the strategy instance"""
        return self.strategy.market_state

    def get_alerts(self):
        """Get current alerts from the risk manager"""
        return self.risk_manager.get_alerts()

    def get_pnl_chart_data(self):
        """Get PnL data for charting"""
        try:
            # Get last 24 hours of PnL data
            pnl_records = self.db.session.query(TradeRecord)\
                .filter(TradeRecord.timestamp >= datetime.utcnow() - timedelta(hours=24))\
                .all()

            if not pnl_records:
                return {'error': 'No trade data available'}

            # Create time series data
            timestamps = [r.timestamp for r in pnl_records]
            pnl_values = [r.pnl for r in pnl_records]
            cumulative_pnl = pd.Series(pnl_values).cumsum().tolist()

            return {
                'timestamps': [ts.strftime('%H:%M') for ts in timestamps],
                'pnl_values': pnl_values,
                'cumulative_pnl': cumulative_pnl,
                'current_pnl': sum(pnl_values)
            }
        except Exception as e:
            return {'error': str(e)}

    def get_positions_data(self):
        """Get current positions data"""
        try:
            positions = self.db.session.query(PositionRecord)\
                .filter(PositionRecord.timestamp >= datetime.utcnow() - timedelta(minutes=30))\
                .all()

            current_positions = {}
            for pos in positions:
                if pos.symbol not in current_positions:
                    current_positions[pos.symbol] = pos

            return [{
                'symbol': pos.symbol,
                'size': pos.position_size,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'pnl': pos.unrealized_pnl,
                'pnl_pct': (pos.unrealized_pnl / (pos.position_size * pos.entry_price)) * 100 if pos.position_size > 0 else 0
            } for pos in current_positions.values()]
        except Exception as e:
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
    """Endpoint to trigger the emergency stop."""
    data = request.get_json()
    if not data or data.get('password') != 'chaewon':
        return jsonify({'status': 'error', 'message': 'Incorrect password.'}), 401

    try:
        risk_manager_instance.emergency_stop()
        logging.warning("EMERGENCY STOP ACTIVATED VIA DASHBOARD")
        return jsonify({'status': 'success', 'message': 'Emergency stop triggered. Bot will stop trading.'})
    except Exception as e:
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
    """Backtest results and visualization dashboard"""
    trading_dashboard = TradingDashboard(strategy_instance, risk_manager_instance)

    # Get recent backtest results
    backtest_results = db_manager.get_backtest_results(20)

    # Get market data for charts
    market_data = trading_dashboard.get_market_data()

    return render_template('backtest_dashboard.html',
                         backtest_results=backtest_results,
                         market_data=market_data,
                         last_updated=datetime.now())

@app.route('/api/backtest_results')
def api_backtest_results():
    """
    API endpoint for backtest results.
    Now includes a calculated summary for the overview panel.
    """
    run_id = request.args.get('run_id', type=int)
    if not run_id:
        return jsonify({'error': 'run_id is required'}), 400

    results = db_manager.get_backtest_results(run_id=run_id, limit=500)

    # The summary is now pre-calculated and fetched with the run list.
    # This endpoint now only needs to return the detailed results.

    return jsonify({
        'results': results,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/backtest_chart_data')
def api_backtest_chart_data():
    """API endpoint for optimized backtest chart data"""
    run_id = request.args.get('run_id', type=int)
    if not run_id:
        return jsonify({'error': 'run_id is required'}), 400

    # This query will be much faster as it only pulls the necessary columns
    trade_data = db_manager.get_trades_for_run(run_id)
    
    return jsonify({
        'chart_data': trade_data,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/clear_backtests', methods=['POST'])
def api_clear_backtests():
    """API endpoint to delete all backtest data."""
    try:
        db_manager.clear_all_backtests()
        return jsonify({'success': True, 'message': 'All backtest data has been cleared.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/backtest_runs')
def api_backtest_runs():
    """API endpoint for backtest runs"""
    dashboard = TradingDashboard(strategy_instance, risk_manager_instance)
    runs = db_manager.get_backtest_runs(20)

    return jsonify({
        'runs': runs,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/run_backtest', methods=['POST'])
def api_run_backtest():
    """API endpoint to run backtests from web interface"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', ['BTCUSDT', 'ETHUSDT'])
        days = data.get('days', 30)

        # Import and run backtester
        from tools.backtester import Backtester
        from datetime import datetime, timedelta

        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()

        run_name = f"Web Backtest - {', '.join(symbols)} - {days} days"
        run_description = f"Backtest run from web dashboard: {len(symbols)} symbols, {days} days of data"

        backtester = Backtester(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            run_name=run_name,
            run_description=run_description
        )
        if not backtester.run_id:
            # If the backtester failed to create a run record, it's an error
            return jsonify({'success': False, 'error': 'Failed to create a backtest run in the database.'}), 500

        try:
            # The backtester's run method will handle fetching, simulation, and saving results.
            backtester.run()
            return jsonify({'success': True, 'run_id': backtester.run_id, 'message': f'Backtest completed for run ID {backtester.run_id}.'})
        except Exception as e:
            # If any part of the backtest run fails, update the status to 'failed'
            if backtester.run_id:
                db_manager.update_backtest_run_status(backtester.run_id, 'failed')
            print(f"Error during backtest run: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)}), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/market_data')
def api_market_data():
    """Provides aggregated market data for the advanced charts."""
    try:
        # Fetch recent data for key symbols (e.g., BTC, ETH, BNB)
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        all_data = {}
        
        # Use a shorter period for dashboard visuals to keep it fast
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24) # 24 hours of data
        
        for symbol in symbols:
            # This is a simplified fetch, ideally this would be cached or more robust
            klines = db_manager.get_recent_prices(symbol, limit=200) # Fetch last 200 candles
            if klines:
                df = pd.DataFrame(klines)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                all_data[symbol] = df
        
        if not all_data:
            return jsonify({'error': 'No market data available'}), 500

        # --- Prepare data for charts ---
        # Labels (timestamps) - using BTC as the reference
        labels = all_data['BTCUSDT'].index.strftime('%H:%M').tolist()
        
        # 1. Price Data
        prices = {s.replace('USDT', '').lower(): df['close'].tolist() for s, df in all_data.items()}

        # 2. Volume Data
        volumes = {s.replace('USDT', '').lower(): df['volume'].tolist() for s, df in all_data.items()}

        # 3. Volatility Data (ATR)
        volatility = {}
        for symbol, df in all_data.items():
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            volatility[symbol.replace('USDT', '').lower()] = atr.fillna(0).tolist()

        # 4. MACD Data (using BTC as an example)
        btc_df = all_data['BTCUSDT']
        exp1 = btc_df['close'].ewm(span=12, adjust=False).mean()
        exp2 = btc_df['close'].ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd = {
            'btc_macd': macd_line.fillna(0).tolist(),
            'btc_signal': signal_line.fillna(0).tolist()
        }

        # Construct the final JSON response
        chart_data = {
            'labels': labels,
            'prices': prices,
            'volumes': volumes,
            'volatility': volatility,
            'macd': macd,
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
        log_file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'trading_system.log')
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
