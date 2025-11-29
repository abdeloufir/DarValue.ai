"""
Revert neighborhood-city mismatch fixes.
This script undoes the fixes that were just applied.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.database.connection import engine
from sqlalchemy import text
import pandas as pd

# Mappings to revert (reverse of what was fixed)
REVERTS = [
    ('Gueliz', 'Marrakech', 'Casablanca'),
    ('Guéliz', 'Marrakech', 'Casablanca'),
    ('Hivernage', 'Marrakech', 'Casablanca'),
    ('Palmeraie', 'Marrakech', 'Casablanca'),
    ('Agdal', 'Casablanca', 'Fes'),
    ('Californie', 'Casablanca', 'Fes'),
    ('Route de Fez', 'Casablanca', 'Fes'),
    ('Kasbah', 'Casablanca', 'Marrakech'),
    ('Kech', 'Casablanca', 'Marrakech'),
    ('Agdal', 'Casablanca', 'Rabat'),
    ('Aviation - Mabella', 'Casablanca', 'Rabat'),
    ('Hassan - Centre Ville', 'Casablanca', 'Rabat'),
    ('Hay Al Kora', 'Casablanca', 'Rabat'),
    ('Hay El Menzah', 'Casablanca', 'Rabat'),
    ('Hay Nahda', 'Casablanca', 'Rabat'),
    ('Hay Riad', 'Casablanca', 'Rabat'),
    ('Kébibat', 'Casablanca', 'Rabat'),
    ("L'Ocean", 'Casablanca', 'Rabat'),
    ('Marjane', 'Casablanca', 'Rabat'),
    ('Médina', 'Casablanca', 'Rabat'),
    ('Riyad', 'Casablanca', 'Rabat'),
    ('Souissi', 'Casablanca', 'Rabat'),
    ('Administratif', 'Casablanca', 'Tangier'),
    ('Boukhalef', 'Casablanca', 'Tangier'),
    ('Californie', 'Casablanca', 'Tangier'),
    ('De La Plage', 'Casablanca', 'Tangier'),
    ('Hay El Boughaz', 'Casablanca', 'Tangier'),
    ('Hay Hassani', 'Casablanca', 'Tangier'),
    ('Jbel Kbir', 'Casablanca', 'Tangier'),
    ('Malabata', 'Casablanca', 'Tangier'),
    ('Médina', 'Casablanca', 'Tangier'),
    ('Mesnana', 'Casablanca', 'Tangier'),
    ('Mina', 'Casablanca', 'Tangier'),
    ('Moulay Youssef', 'Casablanca', 'Tangier'),
    ('Route Nationale Assilah (N1)', 'Casablanca', 'Tangier'),
    ('Sania', 'Casablanca', 'Tangier'),
    ('Tanja Balia', 'Casablanca', 'Tangier'),
    ('Ziaten', 'Casablanca', 'Tangier'),
]

def revert_changes():
    """Revert all neighborhood-city changes"""
    with engine.connect() as conn:
        reverted_count = 0
        
        for neighborhood, current_city, original_city in REVERTS:
            # Check if there are listings to revert
            count_query = """
            SELECT COUNT(*) as cnt FROM listings 
            WHERE neighborhood = :neighborhood AND city = :city
            """
            count_result = pd.read_sql(
                text(count_query), 
                conn, 
                params={'neighborhood': neighborhood, 'city': current_city}
            )
            count = count_result['cnt'].iloc[0] if len(count_result) > 0 else 0
            
            if count > 0:
                # Revert listings
                update_query = """
                UPDATE listings 
                SET city = :original_city 
                WHERE neighborhood = :neighborhood AND city = :current_city
                """
                conn.execute(
                    text(update_query),
                    {
                        'original_city': original_city,
                        'neighborhood': neighborhood,
                        'current_city': current_city
                    }
                )
                conn.commit()
                print(f"✅ Reverted {count} listings: '{neighborhood}' from {current_city} → {original_city}")
                reverted_count += count
        
        print(f"\n✅ All changes reverted! ({reverted_count} listings updated)")

if __name__ == '__main__':
    revert_changes()
