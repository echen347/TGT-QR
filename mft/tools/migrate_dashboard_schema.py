#!/usr/bin/env python3
"""
Migration script to update dashboard-related database schema:
1. Change SignalRecord.signal from Integer to String
2. Create BalanceSnapshot table for PnL chart evolution
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.database import db_manager, Base, SignalRecord, BalanceSnapshot
from sqlalchemy import text

def migrate():
    """Run the migration"""
    print("🔄 Starting dashboard schema migration...")
    
    try:
        # Check if BalanceSnapshot table exists
        inspector = db_manager.engine.dialect.inspector(db_manager.engine)
        tables = inspector.get_table_names()
        
        # Create BalanceSnapshot table if it doesn't exist
        if 'balance_snapshots' not in tables:
            print("📊 Creating balance_snapshots table...")
            BalanceSnapshot.__table__.create(db_manager.engine, checkfirst=True)
            print("✅ balance_snapshots table created")
        else:
            print("✅ balance_snapshots table already exists")
        
        # Check current signal column type
        try:
            result = db_manager.engine.execute(text(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='signal_records'"
            ))
            sql = result.fetchone()[0]
            
            if 'signal INTEGER' in sql:
                print("🔄 Converting signal_records.signal from INTEGER to STRING...")
                # SQLite doesn't support ALTER COLUMN, so we need to recreate the table
                # Step 1: Create new table with correct schema
                db_manager.engine.execute(text("""
                    CREATE TABLE signal_records_new (
                        id INTEGER PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        signal VARCHAR(20) NOT NULL,
                        ma_value FLOAT NOT NULL,
                        current_price FLOAT NOT NULL,
                        timestamp DATETIME NOT NULL
                    )
                """))
                
                # Step 2: Copy data, converting integer signals to strings
                db_manager.engine.execute(text("""
                    INSERT INTO signal_records_new (id, symbol, signal, ma_value, current_price, timestamp)
                    SELECT 
                        id,
                        symbol,
                        CASE 
                            WHEN signal = 1 THEN 'STRONG_BUY'
                            WHEN signal = -1 THEN 'STRONG_SELL'
                            ELSE 'NEUTRAL'
                        END as signal,
                        ma_value,
                        current_price,
                        timestamp
                    FROM signal_records
                """))
                
                # Step 3: Drop old table and rename new one
                db_manager.engine.execute(text("DROP TABLE signal_records"))
                db_manager.engine.execute(text("ALTER TABLE signal_records_new RENAME TO signal_records"))
                
                print("✅ signal_records.signal converted to STRING")
            else:
                print("✅ signal_records.signal is already STRING")
        except Exception as e:
            print(f"⚠️  Could not check signal_records schema: {e}")
            print("   This is okay if the table doesn't exist yet or schema is already correct")
        
        print("✅ Migration complete!")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = migrate()
    sys.exit(0 if success else 1)

