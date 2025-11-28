"""
Extract neighborhoods from Mubawab URLs and listing data
The URL slug contains the neighborhood name
"""
import re
import urllib.parse
import psycopg2
from psycopg2.extras import execute_batch

# Connect to PostgreSQL
conn = psycopg2.connect(
    host='127.0.0.1',
    user='darvalue_user',
    password='darvalue_password_123',
    database='darvalue_db'
)
cursor = conn.cursor()

def extract_neighborhood_from_url(source_url):
    """Extract neighborhood from Mubawab URL slug"""
    if not source_url:
        return None
    
    try:
        # URL format: https://www.mubawab.ma/fr/pa/ID/neighborhood-and-title-slug
        # Extract the slug part after the last /
        slug = source_url.split('/')[-1]
        
        # URL decode to get the actual text
        slug = urllib.parse.unquote(slug)
        
        # Remove the listing ID if present (usually at the start)
        slug = re.sub(r'^[0-9]+-', '', slug)
        
        # Split by % or - or special characters and extract potential neighborhood names
        parts = re.split(r'[-–%_]', slug)
        
        if parts:
            # Return the first significant part (usually the neighborhood)
            for part in parts:
                part = part.strip()
                if len(part) > 2 and part.lower() not in ['à', 'vendre', 'de', 'un', 'et', 'le', 'la', 'les', 'en', 'fr', 'pa']:
                    return part
    except:
        pass
    
    return None

def extract_neighborhood_from_title(title):
    """Extract neighborhood from title text"""
    if not title:
        return None
    
    # Common patterns: "à Neighborhood", "Neighborhood à", "in Neighborhood"
    patterns = [
        r'à\s+([A-Z][a-z\s]+?)(?:\s*[––-]|\s+à\s+|$)',  # à Neighborhood
        r'([A-Z][a-z\s]+?)\s+à\s+',  # Neighborhood à
        r'Résidence\s+([A-Za-z\s]+?)(?:\s*[––-]|$)',  # Résidence Name
        r'Hay\s+([A-Za-z]+)',  # Hay Neighborhood
        r'Les\s+([A-Za-z]+)',  # Les Neighborhood
    ]
    
    for pattern in patterns:
        match = re.search(pattern, title)
        if match:
            neighborhood = match.group(1).strip()
            if len(neighborhood) > 2:
                return neighborhood
    
    return None

# Get all listings with NULL/empty neighborhoods
cursor.execute("""
    SELECT id, title, source_url 
    FROM listings 
    WHERE (neighborhood IS NULL OR neighborhood = '')
    ORDER BY id
""")

listings = cursor.fetchall()
print(f"Processing {len(listings)} listings without neighborhoods...")

updates = []
for listing_id, title, source_url in listings:
    # Try to extract from URL first
    neighborhood = extract_neighborhood_from_url(source_url)
    
    # If not found, try title
    if not neighborhood:
        neighborhood = extract_neighborhood_from_title(title)
    
    if neighborhood:
        updates.append((neighborhood, listing_id))

print(f"Found neighborhoods for {len(updates)} listings")

# Show some examples
if updates:
    print("\nSample extractions:")
    for i, (hood, lid) in enumerate(updates[:10]):
        cursor.execute("SELECT title FROM listings WHERE id = %s", (lid,))
        title = cursor.fetchone()[0]
        print(f"  ID {lid}: '{title[:60]}' -> {hood}")

# Update the database
if updates:
    query = "UPDATE listings SET neighborhood = %s WHERE id = %s"
    execute_batch(cursor, query, updates, page_size=500)
    conn.commit()
    print(f"\nUpdated {len(updates)} listings with neighborhoods")
else:
    print("No neighborhoods found to update")

cursor.close()
conn.close()
