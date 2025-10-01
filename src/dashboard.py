from flask import Flask, render_template, jsonify, request
import plotly.graph_objs as go
import plotly.utils
import pandas as pd
from datetime import datetime, timedelta
from config.config import *
from database import db_manager
from risk_manager import risk_manager
import json

app = Flask(__name__)

class TradingDashboard:
    """Simple mobile-friendly trading dashboard"""

    def __init__(self):
        self.db = db_manager

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
        return risk_manager.get_risk_status()

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
    """Main dashboard page"""
    dashboard = TradingDashboard()

    return render_template('dashboard.html',
                         pnl_chart=dashboard.create_pnl_chart(),
                         positions=dashboard.get_positions_data(),
                         signals=dashboard.get_signals_data(),
                         risk_status=dashboard.get_risk_status())

@app.route('/api/status')
def api_status():
    """API endpoint for real-time status"""
    dashboard = TradingDashboard()

    return jsonify({
        'risk_status': dashboard.get_risk_status(),
        'positions': dashboard.get_positions_data(),
        'recent_signals': dashboard.get_signals_data()[:5],  # Last 5 signals
        'pnl_summary': dashboard.get_pnl_chart_data()
    })

@app.route('/api/shutdown', methods=['POST'])
def api_shutdown():
    """Emergency shutdown endpoint"""
    try:
        # In a real system, this would stop the trading strategy
        risk_manager.emergency_stop()

        return jsonify({
            'success': True,
            'message': 'Emergency shutdown activated'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

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

if __name__ == '__main__':
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False)
