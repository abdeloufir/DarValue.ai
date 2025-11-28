"""
Training and evaluation script for the property analyzer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sqlalchemy import text
from src.database.connection import DatabaseManager
from src.models.property_analyzer import PropertyAnalyzer
import json
from loguru import logger

# Configure logging - use simple print for Windows compatibility
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def safe_print(msg):
    """Safe print that handles unicode"""
    try:
        print(msg.rstrip())
    except (UnicodeEncodeError, AttributeError):
        print(str(msg).encode('utf-8', errors='replace').decode('utf-8').rstrip())

logger.remove()
logger.add(safe_print, level="INFO", format="<level>{level: <8}</level> | {message}")


def load_data_from_database() -> pd.DataFrame:
    """Load listing data from database"""
    db = DatabaseManager()
    
    query = """
    SELECT 
        id,
        source_id,
        title,
        city,
        neighborhood,
        price_mad,
        surface_m2,
        rooms,
        bathrooms,
        property_type,
        latitude,
        longitude
    FROM listings 
    WHERE price_mad IS NOT NULL 
        AND surface_m2 IS NOT NULL
        AND surface_m2 > 0
        AND price_mad > 0
    ORDER BY city, price_mad
    """
    
    try:
        with db.get_session() as session:
            df = pd.read_sql(text(query), session.connection())
        
        logger.info(f"✅ Loaded {len(df)} listings from database")
        return df
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return None


def analyze_dataset(df: pd.DataFrame):
    """Analyze the dataset"""
    logger.info("\n" + "="*60)
    logger.info("📊 DATASET ANALYSIS")
    logger.info("="*60)
    
    logger.info(f"\nTotal listings: {len(df):,}")
    logger.info(f"\n🏙️ Cities: {df['city'].nunique()}")
    for city in sorted(df['city'].unique()):
        count = len(df[df['city'] == city])
        logger.info(f"   - {city}: {count:,} listings")
    
    logger.info(f"\n💰 Price Statistics (MAD):")
    logger.info(f"   - Min: {df['price_mad'].min():,.0f}")
    logger.info(f"   - Max: {df['price_mad'].max():,.0f}")
    logger.info(f"   - Mean: {df['price_mad'].mean():,.0f}")
    logger.info(f"   - Median: {df['price_mad'].median():,.0f}")
    logger.info(f"   - Std Dev: {df['price_mad'].std():,.0f}")
    
    logger.info(f"\n📐 Surface Area Statistics (m²):")
    logger.info(f"   - Min: {df['surface_m2'].min():.1f}")
    logger.info(f"   - Max: {df['surface_m2'].max():.1f}")
    logger.info(f"   - Mean: {df['surface_m2'].mean():.1f}")
    logger.info(f"   - Median: {df['surface_m2'].median():.1f}")
    
    logger.info(f"\n🔑 Rooms Statistics:")
    logger.info(f"   - Min: {df['rooms'].min():.0f}")
    logger.info(f"   - Max: {df['rooms'].max():.0f}")
    logger.info(f"   - Mean: {df['rooms'].mean():.1f}")
    
    # Price per m²
    df_temp = df.copy()
    df_temp['price_per_m2'] = df_temp['price_mad'] / df_temp['surface_m2']
    
    logger.info(f"\n💲 Price per m² Statistics:")
    logger.info(f"   - Min: {df_temp['price_per_m2'].min():,.0f} MAD/m²")
    logger.info(f"   - Max: {df_temp['price_per_m2'].max():,.0f} MAD/m²")
    logger.info(f"   - Mean: {df_temp['price_per_m2'].mean():,.0f} MAD/m²")
    logger.info(f"   - Median: {df_temp['price_per_m2'].median():,.0f} MAD/m²")


def train_analyzer(df: pd.DataFrame) -> PropertyAnalyzer:
    """Train the property analyzer"""
    logger.info("\n" + "="*60)
    logger.info("🚀 TRAINING PROPERTY ANALYZER")
    logger.info("="*60)
    
    # Initialize analyzer
    analyzer = PropertyAnalyzer()
    
    # Prepare data
    df_prepared = analyzer.prepare_data(df)
    logger.info(f"\n✅ Data prepared: {len(df_prepared)} listings ready for modeling")
    
    # Calculate market statistics
    market_stats = analyzer.calculate_market_stats(df_prepared)
    logger.info(f"✅ Market statistics calculated for {len(market_stats)} cities")
    
    # Build price prediction model
    logger.info("\n🏗️ Building price prediction model...")
    metrics = analyzer.build_price_model(df_prepared)
    
    # Analyze appreciation
    logger.info("\n📈 Analyzing appreciation patterns...")
    appreciation = analyzer.analyze_appreciation(df_prepared, forecast_years=3)
    
    logger.info("\n3-Year Appreciation Forecast by City:")
    for city, data in sorted(appreciation.items()):
        logger.info(f"\n   {city.upper()}")
        logger.info(f"   - Current avg price: {data['current_avg_price']:,.0f} MAD")
        logger.info(f"   - Forecast price: {data['forecasted_price']:,.0f} MAD")
        logger.info(f"   - Total appreciation: {data['total_appreciation_percentage']:.1f}%")
        logger.info(f"   - Annual rate: {data['annual_appreciation_rate']*100:.1f}%")
    
    # Estimate rental yields
    logger.info("\n💰 Estimating rental yields...")
    rental_yields = analyzer.estimate_rental_yield(df_prepared)
    
    logger.info("\nRental Yield Estimates by City:")
    for city, data in sorted(rental_yields.items()):
        logger.info(f"\n   {city.upper()}")
        logger.info(f"   - Avg property price: {data['avg_property_price']:,.0f} MAD")
        logger.info(f"   - Est. monthly rent: {data['estimated_monthly_rent']:,.0f} MAD")
        logger.info(f"   - Gross yield: {data['gross_rental_yield']:.2f}%")
        logger.info(f"   - Net yield (after costs): {data['net_rental_yield']:.2f}%")
    
    # Save models
    analyzer.save_models()
    logger.info("\n✅ Models saved successfully")
    
    return analyzer


def generate_sample_recommendations(analyzer: PropertyAnalyzer, df: pd.DataFrame):
    """Generate sample Buy/Hold/Sell recommendations"""
    logger.info("\n" + "="*60)
    logger.info("🎯 SAMPLE BUY/HOLD/SELL RECOMMENDATIONS")
    logger.info("="*60)
    
    # Sample properties from each city
    for city in df['city'].unique():
        city_data = df[df['city'] == city].sample(min(2, len(df[df['city'] == city])))
        
        logger.info(f"\n\n📍 {city.upper()}")
        logger.info("-" * 60)
        
        for _, prop in city_data.iterrows():
            recommendation = analyzer.generate_recommendations(
                property_price=prop['price_mad'],
                property_city=prop['city'],
                property_surface=prop['surface_m2'],
                property_rooms=prop['rooms']
            )
            
            logger.info(f"\nProperty: {prop['title'][:60]}")
            logger.info(f"Price: {prop['price_mad']:,.0f} MAD | Surface: {prop['surface_m2']:.0f} m²")
            logger.info(f"\n🔮 RECOMMENDATION: {recommendation['recommendation']}")
            logger.info(f"   Confidence: {recommendation['confidence']:.0f}%")
            logger.info(f"   Valuation: {recommendation['price_valuation']} ({recommendation['price_deviation_percent']:.1f}%)")
            logger.info(f"   Expected yield: {recommendation['expected_annual_yield']:.2f}%")
            logger.info(f"   3-year forecast: {recommendation['appreciation_forecast']:,.0f} MAD")
            logger.info(f"   Reason: {recommendation['reasoning']}")


def main():
    """Main training and analysis pipeline"""
    logger.info("\n" + "="*80)
    logger.info("🏠 DARVALUE.AI - PROPERTY ANALYSIS & PREDICTION SYSTEM")
    logger.info("="*80)
    
    # Load data
    df = load_data_from_database()
    if df is None or len(df) == 0:
        logger.error("No data available for analysis")
        return
    
    # Analyze dataset
    analyze_dataset(df)
    
    # Train analyzer
    analyzer = train_analyzer(df)
    
    # Generate sample recommendations
    generate_sample_recommendations(analyzer, df)
    
    logger.info("\n" + "="*80)
    logger.info("✅ ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
