"""
Try to find actual property listings on Mubawab using different approaches
"""

import requests
from bs4 import BeautifulSoup
import re
import time

def find_mubawab_listings():
    print("=== Finding Actual Mubawab Property Listings ===")
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'fr,en-US;q=0.7,en;q=0.3',
    })
    
    # Try different URL patterns that might work
    test_patterns = [
        "https://www.mubawab.ma/fr/ct/casablanca/immobilier-a-vendre",
        "https://www.mubawab.ma/fr/ct/casablanca/immobilier-a-vendre?page=1",
        "https://www.mubawab.ma/fr/st/casablanca/appartements-a-vendre",
        "https://www.mubawab.ma/fr/st/casablanca/villas-a-vendre",
        "https://www.mubawab.ma/fr/ct/casablanca",
        "https://www.mubawab.ma/en/ct/casablanca/real-estate-for-sale"
    ]
    
    for pattern in test_patterns:
        print(f"\n--- Testing: {pattern} ---")
        try:
            response = session.get(pattern, timeout=15)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for all links and analyze their patterns
                all_links = soup.find_all('a', href=True)
                
                # Filter for potential property listing links
                property_patterns = [
                    r'/property/',
                    r'/annonce/',
                    r'/ad/',
                    r'/prop/',
                    r'/listing/',
                    r'/bien/',
                    r'/\d+/',  # Links with numbers (property IDs)
                ]
                
                property_links = []
                for link in all_links:
                    href = link.get('href', '')
                    for pattern_regex in property_patterns:
                        if re.search(pattern_regex, href) and 'mubawab.ma' in href:
                            property_links.append(href)
                            break
                
                unique_property_links = list(set(property_links))
                print(f"Found {len(unique_property_links)} unique property links:")
                
                for i, link in enumerate(unique_property_links[:10]):
                    print(f"  {i+1}. {link}")
                    
                if unique_property_links:
                    # Found some property links, this pattern works
                    print(f"✅ This pattern works! Found {len(unique_property_links)} listings")
                    return pattern, unique_property_links
                    
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(1)
    
    print("\n❌ No working patterns found with property listings")
    return None, []

def test_direct_search():
    """Try the search functionality directly"""
    print("\n=== Testing Direct Search Functionality ===")
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Content-Type': 'application/x-www-form-urlencoded',
    })
    
    # Try a search with form data
    search_url = "https://www.mubawab.ma/search"
    search_data = {
        'city': 'casablanca',
        'type': 'apartment',
        'transaction': 'sale'
    }
    
    try:
        response = session.post(search_url, data=search_data, timeout=15)
        print(f"Search POST status: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            links = soup.find_all('a', href=True)
            listing_links = [link.get('href') for link in links if '/property/' in link.get('href', '')]
            print(f"Found {len(listing_links)} property links via search")
            
    except Exception as e:
        print(f"Search error: {e}")

if __name__ == "__main__":
    working_pattern, listings = find_mubawab_listings()
    if not working_pattern:
        test_direct_search()