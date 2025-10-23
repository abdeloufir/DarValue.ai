"""
Deep dive into Mubawab listing structure to find all data elements
"""

import requests
from bs4 import BeautifulSoup

def analyze_mubawab_detailed():
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
            
            # Look for elements with specific classes we found
            print("=== Analyzing specific classes ===")
            
            # Check fullPicturesPrice
            price_elem = soup.select_one('.fullPicturesPrice')
            if price_elem:
                print(f"fullPicturesPrice: {price_elem.get_text().strip()}")
            
            # Check adDetails
            details_elem = soup.select_one('.adDetails')
            if details_elem:
                print(f"adDetails: {details_elem.get_text().strip()[:200]}...")
            
            # Look for all text that might contain price (numbers with DH/MAD)
            print("\n=== Looking for price patterns in all text ===")
            all_text = soup.get_text()
            import re
            price_matches = re.findall(r'(\d[\d\s,\.]*)\s*(?:DH|MAD|dh|mad|dirham)', all_text, re.IGNORECASE)
            print(f"Found price patterns: {price_matches[:5]}")
            
            # Look for surface patterns
            surface_matches = re.findall(r'(\d+(?:[,\.]\d+)?)\s*m[²2]', all_text, re.IGNORECASE)
            print(f"Found surface patterns: {surface_matches[:5]}")
            
            # Look for room patterns
            room_matches = re.findall(r'(\d+)\s*(?:chambre|bedroom|room)', all_text, re.IGNORECASE)
            print(f"Found room patterns: {room_matches[:5]}")
            
            # Check all elements with data attributes
            print("\n=== Elements with data attributes ===")
            data_elements = soup.find_all(attrs=lambda x: x and any(key.startswith('data-') for key in x.keys()))
            for elem in data_elements[:10]:
                attrs = {k: v for k, v in elem.attrs.items() if k.startswith('data-')}
                if attrs:
                    print(f"{elem.name}: {attrs}")
            
            # Look for meta tags that might contain structured data
            print("\n=== Meta tags ===")
            meta_tags = soup.find_all('meta')
            relevant_meta = [tag for tag in meta_tags if tag.get('property') or tag.get('name')]
            for meta in relevant_meta[:10]:
                prop = meta.get('property') or meta.get('name')
                content = meta.get('content', '')
                if any(word in prop.lower() for word in ['price', 'location', 'title', 'description']):
                    print(f"{prop}: {content}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_mubawab_detailed()