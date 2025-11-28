"""
Extract neighborhoods from Mubawab listing pages using the greyTit selector
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
        print(f"Error scraping {url}: {e}")
        return None

# Get all listings without neighborhoods
cursor.execute("""
    SELECT id, title, source_url 
    FROM listings 
    WHERE (neighborhood IS NULL OR neighborhood = '')
    AND source_platform = 'mubawab'
    ORDER BY id
""")

listings = cursor.fetchall()
print(f"Processing {len(listings)} listings without neighborhoods...")

updates = []
errors = []

for i, (listing_id, title, url) in enumerate(listings):
    if i % 20 == 0:
        print(f"Progress: {i}/{len(listings)}")
    
    neighborhood = extract_neighborhood_from_page(url)
    
    if neighborhood:
        updates.append((neighborhood, listing_id))
    else:
        errors.append(listing_id)
    
    # Be nice to the server - don't hammer it
    sleep(0.3)

print(f"\nFound neighborhoods for {len(updates)} listings")
print(f"Failed to extract from {len(errors)} listings")

# Show some samples
if updates:
    print("\nSample extractions:")
    for hood, lid in updates[:15]:
        print(f"  ID {lid}: {hood}")

# Update the database
if updates:
    query = "UPDATE listings SET neighborhood = %s WHERE id = %s"
    execute_batch(cursor, query, updates, page_size=100)
    conn.commit()
    print(f"\nUpdated {len(updates)} listings with neighborhoods")

cursor.close()
conn.close()
