#!/usr/bin/env python3
"""
Migrate database to add condition and furnishing columns
"""

from src.database.connection import engine
from sqlalchemy import text

conn = engine.connect()
trans = conn.begin()

try:
    print("Adding condition column...")
    try:
        conn.execute(text("ALTER TABLE listings ADD COLUMN condition VARCHAR(50)"))
        print("  Added condition column")
    except Exception as e:
        if "already exists" in str(e):
            print("  condition column already exists")
        else:
            raise
    
    print("Adding furnishing column...")
    try:
        conn.execute(text("ALTER TABLE listings ADD COLUMN furnishing VARCHAR(50)"))
        print("  Added furnishing column")
    except Exception as e:
        if "already exists" in str(e):
            print("  furnishing column already exists")
        else:
            raise
    
    print("Adding indexes...")
    try:
        conn.execute(text("CREATE INDEX idx_listing_condition ON listings(condition)"))
        print("  Added condition index")
    except Exception as e:
        if "already exists" in str(e):
            print("  condition index already exists")
        else:
            raise
    
    try:
        conn.execute(text("CREATE INDEX idx_listing_furnishing ON listings(furnishing)"))
        print("  Added furnishing index")
    except Exception as e:
        if "already exists" in str(e):
            print("  furnishing index already exists")
        else:
            raise
    
    trans.commit()
    print("\nDatabase migration completed successfully!")
    
except Exception as e:
    trans.rollback()
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    conn.close()
