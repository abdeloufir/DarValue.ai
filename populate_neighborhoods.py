from src.database.connection import engine
from sqlalchemy import text
import pandas as pd
import re

# Moroccan neighborhoods/areas by city
NEIGHBORHOODS = {
    'Casablanca': ['Anfa', 'Beaulieu', 'Californie', 'Gauthier', 'Oasis', 'Wilaya', 'Ain Sebaa', 'Oulfa', 'Maarif', 'Zenata', 'Racine', 'Hay Mohammadi', 'Bd Zerktouni'],
    'Rabat': ['Agdal', 'Ryad', 'Souissi', 'Takaddoum', 'Medina', 'Hay Riad', 'Hassan', 'Océan', 'Salé Medina'],
    'Marrakech': ['Medina', 'Gueliz', 'Palmeraie', 'Hivernage', 'Safi', 'Amesfouel', 'Kasbah', 'Menara'],
    'Fes': ['Medina', 'Ville Nouvelle', 'Saada', 'Zaouia', 'Andalus', 'Ziat'],
    'Tangier': ['Medina', 'Ville Nouvelle', 'Malabata', 'Hazaz', 'Kasbah', 'Sidi Kacem'],
    'Agadir': ['Medina', 'Talborjt', 'Drarga', 'Oued Souss', 'Kasbah', 'Centre Ville']
}

# Extract neighborhood from description and title
def extract_neighborhood(title, description, city):
    text_to_search = ""
    if title and isinstance(title, str):
        text_to_search = title.lower()
    if description and isinstance(description, str):
        text_to_search += " " + description.lower()
    
    if not text_to_search or city not in NEIGHBORHOODS:
        return None
    
    for neighborhood in NEIGHBORHOODS[city]:
        if neighborhood.lower() in text_to_search:
            return neighborhood
    
    return None

# Get all listings and update neighborhoods
print("Loading listings...")
with engine.connect() as conn:
    # Get listings without neighborhoods
    result = conn.execute(text('SELECT id, city, title, description FROM listings WHERE neighborhood IS NULL LIMIT 2000'))
    rows = result.fetchall()
    
    print(f"Found {len(rows)} listings to update")
    
    updated = 0
    for i, (listing_id, city, title, description) in enumerate(rows):
        neighborhood = extract_neighborhood(title, description, city)
        if neighborhood:
            conn.execute(
                text('UPDATE listings SET neighborhood = :neighborhood WHERE id = :id'),
                {"neighborhood": neighborhood, "id": listing_id}
            )
            updated += 1
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} listings, updated {updated}...")
    
    conn.commit()

print(f"Done! Updated {updated} neighborhoods from descriptions.")

# Show summary
result = pd.read_sql(text('SELECT city, neighborhood, COUNT(*) as count FROM listings WHERE neighborhood IS NOT NULL GROUP BY city, neighborhood ORDER BY city, count DESC'), engine)
print("\nUpdated neighborhoods:")
print(result)
