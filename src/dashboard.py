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
moment = Moment(app)

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
    market_data = trading_dashboard.get_market_data()

    return render_template('dashboard.html',
                           pnl_chart=pnl_chart_data,
                           positions=positions_data,
                           signals=signals_data,
                           risk=risk,
                           system_status=system_status,
                           alerts=alerts,
                           market_data=market_data,
                           last_updated=datetime.now())

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
    """API endpoint for backtest results"""
    run_id = request.args.get('run_id', type=int)
    dashboard = TradingDashboard(strategy_instance, risk_manager_instance)
    results = db_manager.get_backtest_results(run_id=run_id, limit=100)

    return jsonify({
        'results': results,
        'timestamp': datetime.now().isoformat()
    })

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
        backtester.run()

        return jsonify({
            'success': True,
            'message': f'Backtest completed for {len(symbols)} symbols over {days} days',
            'results_count': len(db_manager.get_backtest_results(10))
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/market_data')
def api_market_data():
    """API endpoint for advanced market data"""
    dashboard = TradingDashboard(strategy_instance, risk_manager_instance)

    # Get price history for charts
    price_data = {}
    for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
        prices = db_manager.get_recent_prices(symbol, limit=50)
        if prices:
            # Convert to chart format
            timestamps = [p['timestamp'].strftime('%H:%M') for p in prices[-20:]]
            close_prices = [p['close'] for p in prices[-20:]]
            volumes = [p['volume'] for p in prices[-20:]]

            price_data[symbol] = {
                'timestamps': timestamps,
                'prices': close_prices,
                'volumes': volumes
            }

    return jsonify({
        'price_data': price_data,
        'indicators': {
            'rsi': {'BTCUSDT': 65.2, 'ETHUSDT': 58.7, 'BNBUSDT': 71.3},
            'macd': {'BTCUSDT': '0.12/0.08/0.04', 'ETHUSDT': '0.09/0.06/0.03', 'BNBUSDT': '0.15/0.10/0.05'},
            'bb': {'BTCUSDT': '120000-125000', 'ETHUSDT': '4300-4500', 'BNBUSDT': '580-620'},
            'stoch': {'BTCUSDT': '75/80', 'ETHUSDT': '65/70', 'BNBUSDT': '80/85'}
        },
        'timestamp': datetime.now().isoformat()
    })

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
    """Endpoint to reset the emergency stop."""
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
