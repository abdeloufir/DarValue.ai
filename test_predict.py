#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/S580381/Documents/GitHub/DarValue.ai')

from src.models.property_analyzer import PropertyAnalyzer
from src.database.connection import engine
import pandas as pd
from sqlalchemy import text

try:
    # Initialize analyzer
    analyzer = PropertyAnalyzer()
    
    # Load and prepare data
    query = "SELECT id, title, description, price_mad, surface_m2, rooms, bathrooms, city, neighborhood, property_type FROM listings LIMIT 2000"
    df = pd.read_sql(text(query), engine)
    
    print(f"Loaded {len(df)} listings")
    
    # Prepare data
    df_prepared = analyzer.prepare_data(df)
    print(f"Prepared {len(df_prepared)} listings")
    
    # Calculate stats
    analyzer.calculate_market_stats(df_prepared)
    print("Market stats calculated")
    
    analyzer.analyze_appreciation(df_prepared)
    print("Appreciation analyzed")
    
    analyzer.estimate_rental_yield(df_prepared)
    print("Rental yield estimated")
    
    # Build model
    analyzer.build_price_model(df_prepared)
    print("Model built successfully")
    
    # Test prediction
    test_data = {
        'surface_m2': 120,
        'rooms': 3,
        'bathrooms': 2,
        'city': 'Casablanca',
        'neighborhood': 'Anfa',
        'property_type': 'apartment',
        'condition': 'Standard',
        'furnishing': 'Unknown'
    }
    
    print(f"\nTesting prediction with: {test_data}")
    
    result = analyzer.predict_property(test_data)
    print(f"Prediction result: {result}")
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
