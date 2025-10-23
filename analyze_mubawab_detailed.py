"""
Deep dive into working Mubawab URL pattern
"""

import requests
from bs4 import BeautifulSoup
import re

def analyze_working_mubawab_url():
    print("=== Deep Analysis of Working Mubawab URL ===")
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'fr,en-US;q=0.7,en;q=0.3',
    })
    
    base_url = "https://www.mubawab.ma/fr/ct/casablanca/immobilier-a-vendre"
    
    try:
        print(f"Analyzing: {base_url}")
        response = session.get(base_url, timeout=15)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for listing elements with various selectors
            print("\n=== Looking for listing cards ===")
            
            # Try different selectors that might contain listings
            selectors_to_try = [
                'article',
                '.listing',
                '.property',
                '.ad-item',
                '.item',
                '[class*="listing"]',
                '[class*="property"]',
                '[class*="card"]',
                'a[href*="/property/"]',
                'a[href*="/annonce/"]',
                'a[href*="/ad/"]'
            ]
            
            for selector in selectors_to_try:
                elements = soup.select(selector)
                if elements:
                    print(f"Found {len(elements)} elements with selector '{selector}'")
                    
                    # Check if these elements contain links
                    for i, elem in enumerate(elements[:3]):
                        links = elem.find_all('a', href=True) if elem.name != 'a' else [elem]
                        for link in links:
                            href = link.get('href', '')
                            if href and any(word in href for word in ['/property/', '/annonce/', '/ad/', 'immobilier']):
                                print(f"  {i+1}. {href}")
                                
            # Look for pagination to understand URL structure
            print("\n=== Looking for pagination ===")
            pagination_links = soup.find_all('a', href=True)
            page_links = [link for link in pagination_links if re.search(r'page=\d+', link.get('href', ''))]
            
            if page_links:
                print(f"Found {len(page_links)} pagination links:")
                for link in page_links[:5]:
                    print(f"  {link.get('href')}")
            
            # Check what city patterns work
            print("\n=== Testing other cities ===")
            cities = ['rabat', 'marrakech', 'fes', 'agadir', 'tanger']
            
            for city in cities:
                test_url = f"https://www.mubawab.ma/fr/ct/{city}/immobilier-a-vendre"
                try:
                    resp = session.get(test_url, timeout=10)
                    print(f"  {city}: {resp.status_code}")
                except:
                    print(f"  {city}: Error")
            
            # Look for the actual HTML structure
            print("\n=== HTML Structure Analysis ===")
            # Check if it's a SPA (Single Page Application) that loads content via JavaScript
            scripts = soup.find_all('script')
            js_content = ' '.join([script.get_text() for script in scripts])
            
            if 'react' in js_content.lower() or 'angular' in js_content.lower() or 'vue' in js_content.lower():
                print("⚠️  This appears to be a Single Page Application (SPA)")
                print("   Content may be loaded dynamically via JavaScript")
                
            # Look for any API endpoints in the JavaScript
            api_patterns = re.findall(r'["\'](?:https?://[^"\']*api[^"\']*|/api/[^"\']*)["\']', js_content)
            if api_patterns:
                print(f"Found {len(api_patterns)} potential API endpoints:")
                for pattern in api_patterns[:5]:
                    print(f"  {pattern}")
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_working_mubawab_url()