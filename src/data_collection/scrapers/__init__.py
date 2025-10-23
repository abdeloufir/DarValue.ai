"""
Scrapers package initialization
"""

from .base_scraper import BaseScraper, PropertyListing
from .mubawab_scraper import MubawabScraper

__all__ = [
    'BaseScraper',
    'PropertyListing', 
    'MubawabScraper'
]


def get_scraper(platform: str) -> BaseScraper:
    """Factory function to get scraper instance by platform name"""
    scrapers = {
        'mubawab': MubawabScraper
    }
    
    if platform.lower() not in scrapers:
        raise ValueError(f"Unknown platform: {platform}. Available: {list(scrapers.keys())}")
    
    return scrapers[platform.lower()]()