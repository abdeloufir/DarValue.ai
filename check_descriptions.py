from src.database.connection import engine
from sqlalchemy import text
import pandas as pd

# Check sample descriptions
result = pd.read_sql(text('SELECT city, title, description FROM listings LIMIT 10'), engine)
for idx, row in result.iterrows():
    print(f"\nCity: {row['city']}")
    print(f"Title: {row['title'][:100] if row['title'] else 'None'}")
    print(f"Description: {str(row['description'])[:200] if row['description'] else 'None'}")
