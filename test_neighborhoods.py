from src.database.connection import engine
from sqlalchemy import text
import pandas as pd

result = pd.read_sql(text('SELECT DISTINCT city FROM listings'), engine)
print('Cities:', result['city'].tolist())

print('\nNeighborhoods for Casablanca:')
result2 = pd.read_sql(text('SELECT DISTINCT neighborhood FROM listings WHERE city = :city LIMIT 20'), engine, params={'city': 'Casablanca'})
print(result2['neighborhood'].tolist())
