"""
Scrapers package initialization
"""

from .base_scraper import BaseScraper, PropertyListing
from .avito_scraper import AvitoScraper
from .mubawab_scraper import MubawabScraper
from .sarouty_scraper import SaroutyScraper

__all__ = [
    'BaseScraper',
    'PropertyListing', 
    'AvitoScraper',
    'MubawabScraper',
    'SaroutyScraper'
]


def get_scraper(platform: str) -> BaseScraper:
    """Factory function to get scraper instance by platform name"""
    scrapers = {
        'avito': AvitoScraper,
        'mubawab': MubawabScraper,
        'sarouty': SaroutyScraper
    }
    
    if platform.lower() not in scrapers:
        raise ValueError(f"Unknown platform: {platform}. Available: {list(scrapers.keys())}")
    
    return scrapers[platform.lower()]()