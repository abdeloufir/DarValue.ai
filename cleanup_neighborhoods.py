"""
Clean up bad neighborhood data from the database
Remove entries that are clearly not neighborhood names (descriptions, adjectives, etc.)
"""

import pandas as pd
from sqlalchemy import text
from src.database.connection import engine

# List of words/patterns that are clearly not neighborhood names
GARBAGE_NEIGHBORHOODS = {
    # Descriptive words
    '102m2', '1ere', '2ch', '69m2', 'agréable', 'ancienne', 'apartement', 'app', 
    'apparemment', 'appart', 'appartemen', 'appartement', 'apparterment', 'apparts', 
    'appatement', 'apptement', 'architecture', 'auncien', 'beau', 'bel', 'belle', 
    'bonne', 'bureau', 'chambre', 'chambres', 'charmant', 'charmante', 'clés', 
    'complexe', 'coquette', 'derniers', 'des', 'domaine', 'duplex', 'duplexe', 
    'école', 'élégant', 'élégante', 'emplacement', 'entre', 'étage', 'évasion',
    'exceptionnel', 'exceptionnelle', 'exclusivite', 'fantastique', 'golf', 'grand',
    'grande', 'haut', 'hotel', 'immeuble', 'incroyable', 'joli', 'jolie', 'location',
    'lumineux', 'luxueuse', 'luxueux', 'luxury', 'magnifique', 'maison', 'manifique',
    'moderne', 'modern', 'nouveau', 'nouvelle', 'nozha', 'occasion', 'opportunité',
    'orangerie', 'palace', 'palais', 'particulier', 'penthouse', 'pièces', 'Plage',
    'prestigia', 'prestigieux', 'programme', 'projet', 'propriétaire', 'pure', 'rare',
    'rdc', 'residence', 'résidence', 'rez', 'riad', 'route', 'sofia', 'somptueuse',
    'spacieux', 'studio', 'sublime', 'superbe', 'tanger', 'terrain', 'top', 'tres',
    'très', 'trouvez', 'une', 'vend', 'vente', 'vie', 'viila', 'villa', 'ville',
    'votre', 'très', 'centre', 'Centre', 'Centre Ville', 'Centre ville',
    
    # Generic city names that shouldn't be neighborhoods
    'Casablanca', 'Casablanca Finance City', 'Rabat', 'Agadir', 'Marrakech', 'Fes', 'Tangier', 'Tanger',
    
    # Generic descriptive phrases
    'City', 'Del', 'Des', 'Du', 'Et', 'Hassan', 'L', 'La', 'Le', 'Les', 'L\'', 
    'Port', 'Ras', 'Route', 'Rue', 'Saint', 'Sidi',
}

# Make a case-insensitive set
GARBAGE_LOWER = {g.lower() for g in GARBAGE_NEIGHBORHOODS}

# Get current neighborhoods
query = '''
SELECT DISTINCT neighborhood, COUNT(*) as count
FROM listings
WHERE neighborhood IS NOT NULL AND neighborhood != ''
GROUP BY neighborhood
ORDER BY neighborhood
'''
result = pd.read_sql(text(query), engine)

print(f"Total neighborhoods before cleanup: {len(result)}")
print("\nNeighborhoods to remove:")
to_remove = []
for idx, row in result.iterrows():
    neighborhood = row['neighborhood']
    if neighborhood.lower() in GARBAGE_LOWER:
        to_remove.append(neighborhood)
        print(f"  - '{neighborhood}' ({row['count']} listings)")

print(f"\nTotal to remove: {len(to_remove)}")

if to_remove:
    # Update database to set these to 'Unknown'
    with engine.connect() as conn:
        for neighborhood in to_remove:
            update_query = '''
            UPDATE listings
            SET neighborhood = 'Unknown'
            WHERE neighborhood = :neighborhood
            '''
            conn.execute(text(update_query), {'neighborhood': neighborhood})
        conn.commit()
    
    print(f"✓ Updated {len(to_remove)} neighborhoods to 'Unknown'")

# Verify
result_after = pd.read_sql(text(query), engine)
print(f"\nTotal neighborhoods after cleanup: {len(result_after)}")
print(f"Real neighborhoods remaining:")
for idx, row in result_after.iterrows():
    print(f"  - {row['neighborhood']}: {row['count']} listings")
