"""
Test Mubawab scraper with multiple pages
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_collection.scrapers.mubawab_scraper import MubawabScraper

def test_mubawab_multiple_pages():
    print("=== Testing Mubawab Scraper with Multiple Pages ===")
    scraper = MubawabScraper()
    
    # Test URL extraction for casablanca with 2 pages
    urls = scraper.get_listing_urls('casablanca', max_pages=2)
    print(f"Found {len(urls)} listing URLs across 2 pages")
    
    if urls:
        print(f"\n=== Testing listing scraping on {min(3, len(urls))} listings ===")
        successful = 0
        for i, url in enumerate(urls[:3]):
            print(f"\nTesting listing {i+1}: {url}")
            listing = scraper.scrape_listing(url)
            if listing:
                print(f"  ✅ Title: {listing.title}")
                print(f"  💰 Price: {listing.price_mad} MAD")
                print(f"  📍 City: {listing.city}")
                print(f"  📐 Surface: {listing.surface_m2} m²")
                print(f"  🏠 Rooms: {listing.rooms}")
                successful += 1
            else:
                print(f"  ❌ Failed to scrape")
        
        print(f"\n📊 Success rate: {successful}/{min(3, len(urls))} ({100*successful/min(3, len(urls)):.1f}%)")

if __name__ == "__main__":
    test_mubawab_multiple_pages()