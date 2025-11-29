import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

from src.models.property_analyzer import PropertyAnalyzer
import pandas as pd
from sqlalchemy import text
from src.database.connection import engine

query = 'SELECT price_mad, surface_m2, rooms, bathrooms, city, neighborhood, property_type, condition, furnishing FROM listings LIMIT 2000'
df = pd.read_sql(text(query), engine)

analyzer = PropertyAnalyzer()
df_prepared = analyzer.prepare_data(df)
analyzer.calculate_market_stats(df_prepared)
analyzer.analyze_appreciation(df_prepared)
analyzer.estimate_rental_yield(df_prepared)
analyzer.build_price_model(df_prepared)

print('\nTest predictions:')
for prop_type in ['Apartment', 'Villa', 'House']:
    test_data = {
        'surface_m2': 200,
        'rooms': 3,
        'bathrooms': 2,
        'city': 'Casablanca',
        'neighborhood': 'Californie',
        'property_type': prop_type,
        'condition': 'Standard',
        'furnishing': 'Unknown'
    }
    result = analyzer.predict_property(test_data)
    price = result['predicted_price']
    print(f'{prop_type:12}: {price:>12,.0f} MAD')
