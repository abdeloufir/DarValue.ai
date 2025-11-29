"""
Fix neighborhood-city mismatches in the database.
Some listings have neighborhoods from one city but are labeled as belonging to another.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.database.connection import engine
from sqlalchemy import text
import pandas as pd

# Known neighborhoods by city (comprehensive list)
NEIGHBORHOODS_BY_CITY = {
    'Casablanca': [
        'Abattoirs', 'Achakar', 'Administratif', 'agdal', 'Ain Borja', 'Ain Chock', 
        'Ain Diab', 'Ain Diab Extension', 'Aïn Sebaâ', 'Al Azhar', 'Al Irfane', 
        'Al Wifaq', 'Alsace Lorraine', 'Amicales', 'Amsernate', 'Anfa', 'Anza', 
        'Aouama Gharbia', 'Aourir', 'Arset Ben Chebli', 'Arset Ihiri', 
        'Arset Moulay Moussa', 'Assaka', 'Aviation - Mabella', 'Beauséjour', 
        'Bella Vista', 'Belvedere', 'Belvédère', 'Ben Serguaou', 'Bouaakkaz', 
        'Bouargane', 'Boukhalef', 'Boulevard Med 6', 'Bourgogne Ouest', 'Branes 2', 
        'Branes Kdima', 'CIL (Hay Salam)', 'Californie', 'Camp Al Ghoul', 'Castilla', 
        'Charaf', 'Cité Adrar', 'De La Plage', 'Derb Ghalef', 'Diour Jamaa', 
        'Du Golf', 'El Haj El Mokhtar', 'El Manar - El Hank', 'El Mrabet', 
        'Ferme Bretonne (Hay Arraha)', 'Founti', 'Franceville', 'Gauthier', 'Guich Oudaya', 
        'Hassan - Centre Ville', 'Haut Agdal', 'Haut Anza', 'Haut Founty', 'Hay Al Farah', 
        'Hay Al Kora', 'Hay Al Wafaa', 'Hay Dakhla', 'Hay El Boughaz', 'Hay El Menzah', 
        'Hay Hassani', 'Hay Houda', 'Hay Mabrouka', 'Hay Mohammadi', 'Hay Nahda', 
        'Hay Qods', 'Hay Riad', 'Hay Salam', 'Hay Targa', 'Hay Zaytoun', 'Hermitage', 
        'Iberie', 'Illigh', 'Izdihar', 'Jbel Kbir', 'Kasbah', 'Kech', 'Kébibat', 
        "L'Ocean", 'Laymoune', 'Lekhiam', 'Lekrimat', 'Les Hôpitaux', 'Les Orangers', 
        'Les princesses', 'Longchamps (Hay Al Hanâa)', 'Maarif', 'Majorelle', 'Malabata', 
        'Manar', 'Marchan', 'Marjane', 'Maârif', 'Maârif Extension', 'Medina', 
        'Mers Sultan', 'Mesnana', 'Mhamid', 'Mina', 'Moujahidine', 'Moulay Youssef', 
        'Mozart', 'Msala', 'Médina', 'Nassim 2', 'Oasis', 'Ouest', 'Oulfa', 'Palmier', 
        'Park', 'Plateau (Al Batha)', 'Polo', 'Quartier Administratif', 
        'Quartier Des Ambassades', 'Racine', 'Rais', 'Riad Salam', 'Rif', 'Riviera', 
        'Riyad', 'Riyad Extension', 'Rmila', 'Rmilat', 'Roches', 'Roches Noires', 
        'Route Amizmiz', 'Route Casablanca', 'Route Nationale Assilah (N1)', 'Route de Fez', 
        'Route de Ouarzazate', 'Route de Tahanaout', "Route de l'Ourika", 'Samlalia', 
        'Sania', 'Secteur Touristique', 'Sidi Ben Slimane El Jazouli', 'Sidi Bou Amar', 
        'Sidi Maarouf', 'Sidi Maârouf', 'Siusse', 'Skhirat', 'Souissi', 'Star Hill', 
        'Taddart', 'Taddart Anza', 'Talborjt', 'Tanger City Center', 'Tanja Balia', 
        'Tilila', 'Val Fleury', 'Ville Nouvelle', 'Ziaten', 'Zone Industrielle Agadir', 
        'Zone Industrielle Mghogha', 'iberia', 'marguerites'
    ],
    'Marrakech': [
        'Gueliz', 'Hivernage', 'Kasbah', 'Kech', 'Palmeraie'
    ],
    'Rabat': [
        'Agdal', 'Aviation - Mabella', 'Hassan - Centre Ville', 'Hay Al Kora', 
        'Hay El Menzah', 'Hay Nahda', 'Hay Riad', 'Kébibat', "L'Ocean", 'Marjane', 
        'Médina', 'Riyad', 'Souissi'
    ],
    'Tangier': [
        'Medina', 'Ville Nouvelle', 'Malabata', 'Hazaz', 'Kasbah'
    ],
    'Fes': [
        'Medina', 'Ville Nouvelle', 'Saada', 'Zaouia', 'Andalus'
    ],
    'Agadir': [
        'Medina', 'Talborjt', 'Drarga', 'Oued Souss', 'Kasbah'
    ]
}

def normalize_text(text):
    """Normalize text for comparison"""
    import unicodedata
    if not text:
        return ''
    nfd = unicodedata.normalize('NFD', text)
    cleaned = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    return cleaned.lower().strip()

def find_correct_city_for_neighborhood(neighborhood):
    """Find which city a neighborhood belongs to"""
    normalized_input = normalize_text(neighborhood)
    
    for city, neighborhoods in NEIGHBORHOODS_BY_CITY.items():
        for nb in neighborhoods:
            if normalize_text(nb) == normalized_input:
                return city
    
    return None

def fix_mismatches():
    """Find and fix neighborhood-city mismatches"""
    with engine.connect() as conn:
        # Get all unique (city, neighborhood) combinations
        query = """
        SELECT DISTINCT city, neighborhood FROM listings 
        WHERE neighborhood IS NOT NULL AND neighborhood != '' AND city IS NOT NULL
        ORDER BY city, neighborhood
        """
        df = pd.read_sql(text(query), conn)
        
        print(f"Found {len(df)} unique city-neighborhood combinations\n")
        
        mismatches = []
        updates = []
        
        for idx, row in df.iterrows():
            city = row['city']
            neighborhood = row['neighborhood']
            correct_city = find_correct_city_for_neighborhood(neighborhood)
            
            if correct_city and correct_city != city:
                # This is a mismatch
                mismatches.append({
                    'neighborhood': neighborhood,
                    'wrong_city': city,
                    'correct_city': correct_city
                })
                updates.append((neighborhood, city, correct_city))
                print(f"MISMATCH: '{neighborhood}' is labeled '{city}' but belongs to '{correct_city}'")
        
        if not mismatches:
            print("✅ No mismatches found!")
            return
        
        print(f"\n⚠️  Found {len(mismatches)} mismatches. Fixing...\n")
        
        # Apply fixes
        for neighborhood, wrong_city, correct_city in updates:
            # Count listings to update
            count_query = """
            SELECT COUNT(*) as cnt FROM listings 
            WHERE neighborhood = :neighborhood AND city = :city
            """
            count_result = pd.read_sql(
                text(count_query), 
                conn, 
                params={'neighborhood': neighborhood, 'city': wrong_city}
            )
            count = count_result['cnt'].iloc[0] if len(count_result) > 0 else 0
            
            # Update listings
            update_query = """
            UPDATE listings 
            SET city = :correct_city 
            WHERE neighborhood = :neighborhood AND city = :wrong_city
            """
            conn.execute(
                text(update_query),
                {
                    'correct_city': correct_city,
                    'neighborhood': neighborhood,
                    'wrong_city': wrong_city
                }
            )
            conn.commit()
            print(f"✅ Updated {count} listings: '{neighborhood}' from {wrong_city} → {correct_city}")
        
        print(f"\n✅ All {len(updates)} mismatches fixed!")

if __name__ == '__main__':
    fix_mismatches()
