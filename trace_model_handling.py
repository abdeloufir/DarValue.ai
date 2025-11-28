"""
Trace how the model handles bad data quality records through the pipeline
"""
import pandas as pd
import numpy as np
from sqlalchemy import text
from src.database.connection import engine
import logging
import sys
import io

# Fix encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load all data including bad records"""
    query = """
    SELECT 
        id, title, city, price_mad, surface_m2, rooms, bathrooms
    FROM listings
    ORDER BY price_mad ASC
    LIMIT 2000
    """
    df = pd.read_sql(text(query), engine)
    return df

def trace_outlier_removal(df):
    """Trace step-by-step outlier removal"""
    
    print("\n" + "="*120)
    print("STEP-BY-STEP MODEL DATA QUALITY HANDLING")
    print("="*120)
    
    print(f"\n[STEP 1] Raw Data")
    print(f"   Total records: {len(df)}")
    print(f"   Price range: {df['price_mad'].min():,.0f} - {df['price_mad'].max():,.0f} MAD")
    print(f"   Surface range: {df['surface_m2'].min():,.0f} - {df['surface_m2'].max():,.0f} m²")
    
    # Show sample bad records
    bad_samples = df[df['price_mad'] < 100].head(3)
    print(f"\n   Sample bad records:")
    for idx, row in bad_samples.iterrows():
        print(f"      ID {row['id']}: Price={row['price_mad']:,.0f} MAD, Surface={row['surface_m2']} m², Price/m²={row['price_mad']/row['surface_m2'] if row['surface_m2'] > 0 else 'N/A'}")
    
    df_step1 = df.copy()
    
    # Step 2: Remove NaN values in critical columns
    print(f"\n[STEP 2] Remove Missing Critical Data")
    critical_cols = ['price_mad', 'surface_m2', 'city']
    before_missing = len(df_step1)
    df_step2 = df_step1.dropna(subset=critical_cols)
    removed_missing = before_missing - len(df_step2)
    print(f"   Records with missing price/surface/city: {removed_missing}")
    print(f"   Remaining: {len(df_step2)}")
    
    # Step 3: Data quality checks - Price per m²
    print(f"\n[STEP 3a] Check Price per m² Range (500-500,000 MAD/m²)")
    df_step2['price_per_m2'] = df_step2['price_mad'] / df_step2['surface_m2']
    invalid_price_per_m2 = (df_step2['price_per_m2'] < 500) | (df_step2['price_per_m2'] > 500000)
    print(f"   Invalid price/m²: {invalid_price_per_m2.sum()}")
    
    # Show examples
    bad_price_m2 = df_step2[invalid_price_per_m2 & (df_step2['price_per_m2'] < 500)].head(3)
    print(f"\n   Examples of suspiciously low price/m²:")
    for idx, row in bad_price_m2.iterrows():
        print(f"      ID {row['id']}: Price={row['price_mad']:,.0f} MAD, Surface={row['surface_m2']} m², Price/m²={row['price_per_m2']:,.1f} MAD/m² [REMOVED]")
    
    # Step 4: Data quality checks - Surface bounds
    print(f"\n[STEP 3b] Check Surface Range (20-50,000 m²)")
    invalid_surface = (df_step2['surface_m2'] < 20) | (df_step2['surface_m2'] > 50000)
    print(f"   Invalid surface: {invalid_surface.sum()}")
    
    # Step 5: Data quality checks - Price bounds
    print(f"\n[STEP 3c] Check Absolute Price Range (50k-1.5B MAD)")
    invalid_price = (df_step2['price_mad'] < 50000) | (df_step2['price_mad'] > 1500000000)
    print(f"   Invalid price: {invalid_price.sum()}")
    
    # Show examples
    bad_price = df_step2[invalid_price & (df_step2['price_mad'] < 50000)].head(3)
    print(f"\n   Examples of suspiciously low price:")
    for idx, row in bad_price.iterrows():
        print(f"      ID {row['id']}: Price={row['price_mad']:,.0f} MAD, Surface={row['surface_m2']} m² [REMOVED]")
    
    # Combine data quality issues
    invalid_rows = invalid_price_per_m2 | invalid_surface | invalid_price
    before_invalid = len(df_step2)
    df_step3 = df_step2[~invalid_rows].copy()
    removed_invalid = before_invalid - len(df_step3)
    
    print(f"\n[STEP 3 SUMMARY] Data Quality Validation")
    print(f"   Total invalid records found: {invalid_rows.sum()}")
    print(f"   - Invalid price/m²: {invalid_price_per_m2.sum()}")
    print(f"   - Invalid surface: {invalid_surface.sum()}")
    print(f"   - Invalid price: {invalid_price.sum()}")
    print(f"   Remaining: {len(df_step3)}")
    
    # Step 6: Statistical outliers
    print(f"\n[STEP 4] Statistical Outlier Detection (3 std dev)")
    price_mean = df_step3['price_mad'].mean()
    price_std = df_step3['price_mad'].std()
    price_range = (price_mean - 3*price_std, price_mean + 3*price_std)
    
    surface_mean = df_step3['surface_m2'].mean()
    surface_std = df_step3['surface_m2'].std()
    surface_range = (surface_mean - 3*surface_std, surface_mean + 3*surface_std)
    
    statistical_outliers = (
        (df_step3['price_mad'] < price_range[0]) | (df_step3['price_mad'] > price_range[1]) |
        (df_step3['surface_m2'] < surface_range[0]) | (df_step3['surface_m2'] > surface_range[1])
    )
    
    print(f"   Price range: {price_range[0]:,.0f} - {price_range[1]:,.0f} MAD")
    print(f"   Surface range: {surface_range[0]:,.0f} - {surface_range[1]:,.0f} m²")
    print(f"   Statistical outliers found: {statistical_outliers.sum()}")
    
    df_final = df_step3[~statistical_outliers].copy()
    
    # Final summary
    print(f"\n" + "="*120)
    print("FINAL PIPELINE SUMMARY")
    print("="*120)
    print(f"Initial records:           {len(df):,}")
    print(f"After removing missing:    {len(df_step2):,} (-{before_missing - len(df_step2)})")
    print(f"After data quality checks: {len(df_step3):,} (-{before_invalid - len(df_step3)})")
    print(f"After statistical filter:  {len(df_final):,} (-{len(df_step3) - len(df_final)})")
    print(f"\nTotal removed: {len(df) - len(df_final):,} ({(len(df) - len(df_final))/len(df)*100:.1f}%)")
    print(f"Clean data retained: {len(df_final):,} ({len(df_final)/len(df)*100:.1f}%)")
    
    # Statistics on final dataset
    print(f"\n[FINAL CLEAN DATASET] Statistics")
    print(f"   Price: {df_final['price_mad'].min():,.0f} - {df_final['price_mad'].max():,.0f} MAD")
    print(f"   Average price: {df_final['price_mad'].mean():,.0f} MAD")
    print(f"   Median price: {df_final['price_mad'].median():,.0f} MAD")
    print(f"\n   Surface: {df_final['surface_m2'].min():.0f} - {df_final['surface_m2'].max():.0f} m²")
    print(f"   Average surface: {df_final['surface_m2'].mean():.0f} m²")
    print(f"\n   Price/m²: {df_final['price_per_m2'].min():,.0f} - {df_final['price_per_m2'].max():,.0f} MAD/m²")
    print(f"   Average price/m²: {df_final['price_per_m2'].mean():,.0f} MAD/m²")
    print(f"   Median price/m²: {df_final['price_per_m2'].median():,.0f} MAD/m²")
    
    # Show example of a good record
    print(f"\n[EXAMPLE] Clean Record Ready for Training:")
    good_record = df_final.iloc[0]
    print(f"   ID {good_record['id']}: {good_record['title'][:50] if isinstance(good_record['title'], str) else 'N/A'}...")
    print(f"   Price: {good_record['price_mad']:,.0f} MAD")
    print(f"   Surface: {good_record['surface_m2']} m²")
    print(f"   Price/m²: {good_record['price_per_m2']:,.0f} MAD/m²")
    print(f"   City: {good_record['city']}")
    print(f"   Status: [READY FOR MODEL TRAINING]")

def main():
    logger.info("Loading property data from database...")
    df = load_data()
    logger.info(f"Loaded {len(df)} listings\n")
    
    trace_outlier_removal(df)

if __name__ == '__main__':
    main()
