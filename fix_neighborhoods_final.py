"""
Fix remaining bad neighborhoods from the second pass
"""
import re
import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2.extras import execute_batch
from time import sleep

# Connect to PostgreSQL
conn = psycopg2.connect(
    host='127.0.0.1',
    user='darvalue_user',
    password='darvalue_password_123',
    database='darvalue_db'
)
cursor = conn.cursor()

def extract_neighborhood_from_page(url):
    """Scrape the actual Mubawab page to extract neighborhood from greyTit element"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the h3 with class greyTit - contains "Neighborhood à City" format
        grey_tit = soup.select_one('h3.greyTit')
        
        if grey_tit:
            location_text = grey_tit.get_text()
            # Clean up whitespace and newlines
            location_text = ' '.join(location_text.split())
            
            # Format is typically: "Neighborhood à City" or "Neighborhood\nCity"
            # Split by 'à' or by newline/multiple spaces
            parts = re.split(r'\s*à\s*|\s{2,}|\n', location_text)
            
            if parts:
                # First part is the neighborhood
                neighborhood = parts[0].strip()
                if neighborhood and len(neighborhood) > 2:
                    return neighborhood
        
        return None
        
    except Exception as e:
        return None

# List of remaining bad neighborhood values
bad_neighborhoods = [
    'appartement', 'villa', 'splendide', 'superbe', 'magnifique', 'vente', 'bel', 'très',
    'belle', 'vend', 'duplex', 'studio', 'maison', 'somptueuse', 'trés', 'somptueuse',
    'villas', 'appartements', 'studios', 'duplexes', 'maisons'
]

# Get all listings with bad neighborhoods
query = "SELECT id, source_url, neighborhood FROM listings WHERE source_platform = 'mubawab' AND neighborhood IN (" + ",".join([f"'{nh}'" for nh in bad_neighborhoods]) + ")"
cursor.execute(query)
bad_listings = cursor.fetchall()

print(f"Found {len(bad_listings)} listings with remaining bad neighborhoods")

updates = []

for i, (listing_id, url, current_neighborhood) in enumerate(bad_listings):
    if i % 20 == 0:
        print(f"Progress: {i}/{len(bad_listings)}")
    
    neighborhood = extract_neighborhood_from_page(url)
    
    if neighborhood and neighborhood not in bad_neighborhoods:
        updates.append((neighborhood, listing_id))
    
    # Be nice to the server
    sleep(0.3)

print(f"\nFixed neighborhoods for {len(updates)} listings")

# Update the database
if updates:
    query = "UPDATE listings SET neighborhood = %s WHERE id = %s"
    execute_batch(cursor, query, updates, page_size=100)
    conn.commit()
    print(f"Updated {len(updates)} listings with correct neighborhoods")

cursor.close()
conn.close()
