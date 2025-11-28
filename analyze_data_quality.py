"""
Analyze and report data quality issues in the property listings
"""
import pandas as pd
from sqlalchemy import text
from src.database.connection import engine
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load data from database"""
    query = """
    SELECT 
        id, title, description, city, neighborhood, price_mad, surface_m2, rooms, bathrooms, property_type
    FROM listings
    ORDER BY price_mad ASC
    LIMIT 2000
    """
    df = pd.read_sql(text(query), engine)
    return df

def check_data_quality(df):
    """Identify data quality issues"""
    
    df['price_per_m2'] = df['price_mad'] / df['surface_m2']
    
    # Define issue categories
    issues = {
        'suspiciously_low_price': [],
        'suspiciously_low_price_per_m2': [],
        'suspiciously_high_price_per_m2': [],
        'impossibly_small_surface': [],
        'impossibly_large_surface': [],
        'suspiciously_high_price': [],
        'missing_critical_data': []
    }
    
    # Check each property
    for idx, row in df.iterrows():
        price = row['price_mad']
        surface = row['surface_m2']
        price_per_m2 = row['price_per_m2']
        
        issue_list = []
        
        # Missing critical data
        if pd.isna(price) or pd.isna(surface):
            issue_list.append('missing_critical_data')
        
        # Price per m² checks
        if price_per_m2 < 500:
            issue_list.append('suspiciously_low_price_per_m2')
        elif price_per_m2 > 500000:
            issue_list.append('suspiciously_high_price_per_m2')
        
        # Absolute price checks
        if price < 50000:
            issue_list.append('suspiciously_low_price')
        elif price > 1500000000:
            issue_list.append('suspiciously_high_price')
        
        # Surface checks
        if surface < 20:
            issue_list.append('impossibly_small_surface')
        elif surface > 50000:
            issue_list.append('impossibly_large_surface')
        
        # Record issues
        for issue in issue_list:
            issues[issue].append({
                'id': row['id'],
                'title': row['title'][:60] if pd.notna(row['title']) else 'N/A',
                'city': row['city'],
                'neighborhood': row['neighborhood'],
                'price': price,
                'surface': surface,
                'price_per_m2': price_per_m2,
                'rooms': row['rooms'],
                'property_type': row['property_type']
            })
    
    return issues

def print_issues(issues):
    """Pretty print data quality issues"""
    
    print("\n" + "="*100)
    print("DATA QUALITY ISSUES REPORT")
    print("="*100)
    
    # Suspiciously low prices
    if issues['suspiciously_low_price']:
        print(f"\n📉 SUSPICIOUSLY LOW PRICES (< 50,000 MAD) - {len(issues['suspiciously_low_price'])} issues")
        print("-" * 100)
        for item in issues['suspiciously_low_price'][:10]:
            print(f"  ID: {item['id']} | {item['title']}")
            print(f"    Price: {item['price']:,.0f} MAD | Surface: {item['surface']} m² | Price/m²: {item['price_per_m2']:,.0f} MAD/m²")
            print(f"    Location: {item['city']}, {item['neighborhood']} | Rooms: {item['rooms']}")
            print()
    
    # Suspiciously low price per m²
    if issues['suspiciously_low_price_per_m2']:
        print(f"\n💔 SUSPICIOUSLY LOW PRICE/M² (< 500 MAD/m²) - {len(issues['suspiciously_low_price_per_m2'])} issues")
        print("-" * 100)
        for item in issues['suspiciously_low_price_per_m2'][:10]:
            print(f"  ID: {item['id']} | {item['title']}")
            print(f"    Price: {item['price']:,.0f} MAD | Surface: {item['surface']} m² | Price/m²: {item['price_per_m2']:,.0f} MAD/m²")
            print(f"    Location: {item['city']}, {item['neighborhood']} | Type: {item['property_type']}")
            print()
    
    # Suspiciously high price per m²
    if issues['suspiciously_high_price_per_m2']:
        print(f"\n💎 SUSPICIOUSLY HIGH PRICE/M² (> 500,000 MAD/m²) - {len(issues['suspiciously_high_price_per_m2'])} issues")
        print("-" * 100)
        for item in issues['suspiciously_high_price_per_m2'][:10]:
            print(f"  ID: {item['id']} | {item['title']}")
            print(f"    Price: {item['price']:,.0f} MAD | Surface: {item['surface']} m² | Price/m²: {item['price_per_m2']:,.0f} MAD/m²")
            print(f"    Location: {item['city']}, {item['neighborhood']}")
            print()
    
    # Impossibly small surfaces
    if issues['impossibly_small_surface']:
        print(f"\n📦 IMPOSSIBLY SMALL SURFACE (< 20 m²) - {len(issues['impossibly_small_surface'])} issues")
        print("-" * 100)
        for item in issues['impossibly_small_surface'][:10]:
            print(f"  ID: {item['id']} | {item['title']}")
            print(f"    Price: {item['price']:,.0f} MAD | Surface: {item['surface']} m² | Price/m²: {item['price_per_m2']:,.0f} MAD/m²")
            print(f"    Location: {item['city']}, {item['neighborhood']}")
            print()
    
    # Impossibly large surfaces
    if issues['impossibly_large_surface']:
        print(f"\n🏞️ IMPOSSIBLY LARGE SURFACE (> 50,000 m²) - {len(issues['impossibly_large_surface'])} issues")
        print("-" * 100)
        for item in issues['impossibly_large_surface'][:10]:
            print(f"  ID: {item['id']} | {item['title']}")
            print(f"    Price: {item['price']:,.0f} MAD | Surface: {item['surface']} m² | Price/m²: {item['price_per_m2']:,.0f} MAD/m²")
            print(f"    Location: {item['city']}, {item['neighborhood']}")
            print()
    
    # Suspiciously high prices
    if issues['suspiciously_high_price']:
        print(f"\n🚀 SUSPICIOUSLY HIGH PRICES (> 1.5B MAD) - {len(issues['suspiciously_high_price'])} issues")
        print("-" * 100)
        for item in issues['suspiciously_high_price'][:10]:
            print(f"  ID: {item['id']} | {item['title']}")
            print(f"    Price: {item['price']:,.0f} MAD | Surface: {item['surface']} m² | Price/m²: {item['price_per_m2']:,.0f} MAD/m²")
            print(f"    Location: {item['city']}, {item['neighborhood']}")
            print()
    
    # Missing critical data
    if issues['missing_critical_data']:
        print(f"\n❌ MISSING CRITICAL DATA - {len(issues['missing_critical_data'])} issues")
        print("-" * 100)
        for item in issues['missing_critical_data'][:10]:
            print(f"  ID: {item['id']} | {item['title']}")
            print(f"    Price: {item['price']} | Surface: {item['surface']}")
            print(f"    Location: {item['city']}, {item['neighborhood']}")
            print()
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    total_issues = sum(len(v) for v in issues.values())
    print(f"Total data quality issues found: {total_issues}")
    for issue_type, items in issues.items():
        if items:
            print(f"  • {issue_type}: {len(items)} issues")

def main():
    logger.info("Loading property data from database...")
    df = load_data()
    logger.info(f"Loaded {len(df)} listings")
    
    logger.info("Analyzing data quality...")
    issues = check_data_quality(df)
    
    print_issues(issues)

if __name__ == '__main__':
    main()
