"""
Analyze individual Mubawab listing page structure
"""

import requests
from bs4 import BeautifulSoup

def analyze_mubawab_listing():
    url = "https://www.mubawab.ma/fr/pa/8152102/appartement-haut-standing-%C3%A0-vendre-les-princesses"
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    try:
        response = session.get(url, timeout=15)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for title
            print("=== Looking for title ===")
            title_selectors = [
                'h1',
                '.title',
                '[data-testid="ad-title"]',
                '.property-title',
                '.listing-title'
            ]
            for selector in title_selectors:
                elem = soup.select_one(selector)
                if elem:
                    print(f"Found title with '{selector}': {elem.get_text().strip()}")
                    break
            
            # Look for price
            print("\n=== Looking for price ===")
            price_selectors = [
                '.price',
                '.cost',
                '[class*="price"]',
                '[class*="cost"]',
                'span[class*="price"]',
                'div[class*="price"]'
            ]
            for selector in price_selectors:
                elem = soup.select_one(selector)
                if elem:
                    print(f"Found price with '{selector}': {elem.get_text().strip()}")
                    break
            
            # Look for location
            print("\n=== Looking for location ===")
            location_selectors = [
                '.location',
                '.address',
                '[class*="location"]',
                '[class*="address"]',
                '.city'
            ]
            for selector in location_selectors:
                elem = soup.select_one(selector)
                if elem:
                    print(f"Found location with '{selector}': {elem.get_text().strip()}")
                    break
            
            # Look for common class patterns
            print("\n=== Common class patterns ===")
            all_classes = set()
            for elem in soup.find_all(class_=True):
                if isinstance(elem.get('class'), list):
                    all_classes.update(elem.get('class'))
                    
            relevant_classes = [cls for cls in all_classes if any(word in cls.lower() for word in ['price', 'title', 'location', 'surface', 'room', 'detail'])]
            print(f"Relevant classes: {relevant_classes[:15]}")
            
            # Look for any structured data
            print("\n=== Looking for structured data ===")
            scripts = soup.find_all('script', type='application/ld+json')
            for script in scripts:
                print(f"Found JSON-LD: {script.get_text()[:100]}...")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_mubawab_listing()