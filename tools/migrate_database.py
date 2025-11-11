#!/usr/bin/env python3
"""
Database Migration Script
Adds missing columns to existing tables if they don't exist
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.database import db_manager
from sqlalchemy import text, inspect

def migrate_backtest_runs():
    """Add missing summary columns to backtest_runs table"""
    inspector = inspect(db_manager.engine)
    
    if 'backtest_runs' not in inspector.get_table_names():
        print("❌ backtest_runs table doesn't exist")
        return False
    
    # Get current columns
    current_cols = [c['name'] for c in inspector.get_columns('backtest_runs')]
    print(f"Current columns: {current_cols}")
    
    # Columns to add
    columns_to_add = {
        'total_pnl': 'FLOAT',
        'total_trades': 'INTEGER',
        'win_rate': 'FLOAT',
        'sharpe_ratio': 'FLOAT',
        'max_drawdown': 'FLOAT',
        'avg_return': 'FLOAT'
    }
    
    added = []
    for col_name, col_type in columns_to_add.items():
        if col_name not in current_cols:
            try:
                db_manager.session.execute(text(f"ALTER TABLE backtest_runs ADD COLUMN {col_name} {col_type}"))
                db_manager.session.commit()
                added.append(col_name)
                print(f"✅ Added column: {col_name}")
            except Exception as e:
                db_manager.session.rollback()
                print(f"❌ Error adding column {col_name}: {e}")
                return False
    
    if added:
        print(f"\n✅ Successfully added {len(added)} columns: {', '.join(added)}")
    else:
        print("\n✅ All columns already exist - schema is up to date!")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 DATABASE MIGRATION")
    print("=" * 60)
    
    success = migrate_backtest_runs()
    
    if success:
        print("\n✅ Migration completed successfully!")
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)

