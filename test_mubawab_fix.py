"""
Test the updated Mubawab scraper
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_collection.scrapers.mubawab_scraper import MubawabScraper

def test_mubawab():
    print("=== Testing Updated Mubawab Scraper ===")
    scraper = MubawabScraper()
    
    # Test URL extraction for casablanca (multiple pages)
    urls = scraper.get_listing_urls('casablanca', max_pages=5)
    print(f"Found {len(urls)} listing URLs")
    
    for i, url in enumerate(urls[:5]):
        print(f"  {i+1}. {url}")
    
    if urls:
        print(f"\n=== Testing listing scraping ===")
        # Test scraping first listing
        listing = scraper.scrape_listing(urls[0])
        if listing:
            print(f"Successfully scraped: {listing.title}")
            print(f"Price: {listing.price_mad}")
            print(f"City: {listing.city}")
            print(f"Surface: {listing.surface_m2}")
        else:
            print("Failed to scrape listing content")

if __name__ == "__main__":
    test_mubawab()