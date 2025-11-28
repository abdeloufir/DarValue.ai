from src.database.connection import engine
from sqlalchemy import text
import pandas as pd

result = pd.read_sql(text('SELECT neighborhood, COUNT(*) as count, AVG(price_mad) as avg_price, AVG(surface_m2) as avg_surface, AVG(price_mad/surface_m2) as avg_price_m2 FROM listings WHERE city = :city AND neighborhood IS NOT NULL GROUP BY neighborhood ORDER BY avg_price DESC LIMIT 15'), engine, params={"city": "Casablanca"})
print("\nCasablanca Neighborhoods:")
print(result)

result2 = pd.read_sql(text('SELECT COUNT(*) as total, AVG(price_mad) as avg_price FROM listings WHERE city = :city'), engine, params={"city": "Casablanca"})
print("\nCasablanca Overall:")
print(result2)
