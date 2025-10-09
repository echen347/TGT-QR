from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import DATABASE_URL
import json

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
    signal = Column(Integer, nullable=False)  # -1, 0, 1
    ma_value = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)

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

    def close(self):
        """Close database connection"""
        self.session.close()

# Global database instance
db_manager = DatabaseManager()
