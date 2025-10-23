"""
Research Mubawab website structure to fix URL patterns
"""

import requests
from bs4 import BeautifulSoup
import time

def analyze_mubawab_structure():
    print("=== Analyzing Mubawab Website Structure ===")
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'fr,en-US;q=0.7,en;q=0.3',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    })
    
    # Test main page first
    print("1. Testing main page...")
    try:
        response = session.get("https://www.mubawab.ma/", timeout=10)
        print(f"Main page status: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for navigation links that might show the correct structure
            nav_links = soup.find_all('a', href=True)
            search_links = [link for link in nav_links if any(word in link.get('href', '').lower() for word in ['acheter', 'buy', 'vente', 'sale', 'search'])]
            
            print(f"Found {len(search_links)} potential search links:")
            for i, link in enumerate(search_links[:10]):
                href = link.get('href')
                text = link.get_text().strip()[:30]
                print(f"  {i+1}. {href} - '{text}'")
                
    except Exception as e:
        print(f"Error accessing main page: {e}")
    
    # Test different URL patterns for real estate search
    print("\n2. Testing various search URL patterns...")
    test_urls = [
        "https://www.mubawab.ma/fr/cc/immobilier-casablanca",
        "https://www.mubawab.ma/fr/sc/appartements-casablanca",
        "https://www.mubawab.ma/fr/ct/casablanca/immobilier-a-vendre",
        "https://www.mubawab.ma/fr/ct/casablanca/appartements-a-vendre", 
        "https://www.mubawab.ma/fr/casablanca/immobilier-a-vendre",
        "https://www.mubawab.ma/en/casablanca/real-estate-for-sale",
        "https://www.mubawab.ma/fr/acheter/appartements/casablanca",
        "https://www.mubawab.ma/search?city=casablanca",
        "https://www.mubawab.ma/fr/recherche/casablanca"
    ]
    
    working_urls = []
    for url in test_urls:
        try:
            print(f"Testing: {url}")
            response = session.get(url, timeout=10)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                working_urls.append(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for listing cards or links
                links = soup.find_all('a', href=True)
                listing_links = [link for link in links if any(word in link.get('href', '') for word in ['/property/', '/annonce/', '/ad/', 'immobilier'])]
                print(f"  Found {len(listing_links)} potential listing links")
                
                if listing_links:
                    for i, link in enumerate(listing_links[:3]):
                        print(f"    {i+1}. {link.get('href')}")
                break
                
        except Exception as e:
            print(f"  Error: {e}")
        
        time.sleep(1)
    
    print(f"\n3. Working URLs found: {len(working_urls)}")
    for url in working_urls:
        print(f"  ✅ {url}")

if __name__ == "__main__":
    analyze_mubawab_structure()