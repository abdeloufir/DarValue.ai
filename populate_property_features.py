#!/usr/bin/env python3
"""
Extract and populate property_type, condition, and furnishing from titles/descriptions
"""

from src.database.connection import engine
from sqlalchemy import text
import pandas as pd
import re

def extract_property_type(title, description=''):
    """Extract property type from title and description"""
    text = (str(title) + ' ' + str(description)).lower()
    
    # Keywords for each property type
    villa_keywords = ['villa', 'villas']
    house_keywords = ['maison', 'house', 'townhouse', 'riad']
    apartment_keywords = ['appartement', 'apartment', 'apt', 'studio', 'appart']
    land_keywords = ['terrain', 'land', 'plot']
    commercial_keywords = ['commercial', 'bureau', 'office', 'local', 'shop', 'boutique']
    
    if any(kw in text for kw in villa_keywords):
        return 'villa'
    elif any(kw in text for kw in house_keywords):
        return 'house'
    elif any(kw in text for kw in apartment_keywords):
        return 'apartment'
    elif any(kw in text for kw in land_keywords):
        return 'land'
    elif any(kw in text for kw in commercial_keywords):
        return 'commercial'
    else:
        return 'apartment'  # Default to apartment

def extract_condition(title, description=''):
    """Extract property condition from title and description"""
    text = (str(title) + ' ' + str(description)).lower()
    
    if any(word in text for word in ['neuf', 'new', 'newly built', 'nouveau', 'brand new']):
        return 'New'
    elif any(word in text for word in ['rénov', 'renovated', 'restoration', 'refurb']):
        return 'Renovated'
    elif any(word in text for word in ['ancien', 'old', 'historic', 'vintage']):
        return 'Old'
    else:
        return 'Standard'

def extract_furnishing(title, description=''):
    """Extract furnishing status from title and description"""
    text = (str(title) + ' ' + str(description)).lower()
    
    if 'meublé' in text or 'furnished' in text or 'meuble' in text:
        return 'Furnished'
    elif 'non meublé' in text or 'unfurnished' in text or 'non meuble' in text:
        return 'Unfurnished'
    else:
        return 'Unknown'

# Connect to database
conn = engine.connect()
trans = conn.begin()

try:
    # Get all listings
    query = "SELECT id, title, description FROM listings WHERE price_mad > 0 ORDER BY id"
    df = pd.read_sql_query(text(query), conn)
    
    print(f"Processing {len(df)} listings...")
    
    # Extract features
    df['property_type'] = df.apply(lambda row: extract_property_type(row['title'], row.get('description', '')), axis=1)
    df['condition'] = df.apply(lambda row: extract_condition(row['title'], row.get('description', '')), axis=1)
    df['furnishing'] = df.apply(lambda row: extract_furnishing(row['title'], row.get('description', '')), axis=1)
    
    # Statistics
    print("\nExtracted Features Distribution:")
    print("Property Type:")
    print(df['property_type'].value_counts())
    print("\nCondition:")
    print(df['condition'].value_counts())
    print("\nFurnishing:")
    print(df['furnishing'].value_counts())
    
    # Update database
    print("\nUpdating database...")
    updated = 0
    for idx, row in df.iterrows():
        update_query = text("""
            UPDATE listings 
            SET property_type = :property_type,
                condition = :condition,
                furnishing = :furnishing
            WHERE id = :id
        """)
        conn.execute(update_query, {
            'property_type': row['property_type'],
            'condition': row['condition'],
            'furnishing': row['furnishing'],
            'id': row['id']
        })
        updated += 1
        if updated % 100 == 0:
            print(f"  Updated {updated}/{len(df)} listings...")
    
    trans.commit()
    print(f"\nSuccessfully updated {updated} listings!")
    
    # Verify
    conn_verify = engine.connect()
    verify_query = text("""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN property_type IS NOT NULL THEN 1 END) as with_property_type,
            COUNT(CASE WHEN condition IS NOT NULL THEN 1 END) as with_condition,
            COUNT(CASE WHEN furnishing IS NOT NULL THEN 1 END) as with_furnishing
        FROM listings
    """)
    result = conn_verify.execute(verify_query).fetchone()
    print("\nDatabase Verification:")
    print(f"  Total listings: {result[0]}")
    print(f"  With property_type: {result[1]} ({result[1]/result[0]*100:.1f}%)")
    print(f"  With condition: {result[2]} ({result[2]/result[0]*100:.1f}%)")
    print(f"  With furnishing: {result[3]} ({result[3]/result[0]*100:.1f}%)")
    conn_verify.close()
    
except Exception as e:
    trans.rollback()
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    conn.close()

print("\nFeature population complete!")
