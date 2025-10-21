"""
Base scraper class for real estate websites
"""

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import random
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from loguru import logger
import re
from urllib.parse import urljoin, urlparse


@dataclass
class PropertyListing:
    """Data class for property listing information"""
    title: str
    description: Optional[str]
    city: str
    neighborhood: Optional[str]
    price_mad: Optional[int]
    surface_m2: Optional[float]
    rooms: Optional[int]
    bathrooms: Optional[int]
    property_type: Optional[str]
    amenities: List[str]
    image_urls: List[str]
    agent_name: Optional[str]
    agent_phone: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    source_url: str
    source_id: str


class BaseScraper(ABC):
    """Base class for real estate website scrapers"""
    
    def __init__(self, base_url: str, platform_name: str):
        self.base_url = base_url
        self.platform_name = platform_name
        self.session = requests.Session()
        self.driver = None
        self.setup_session()
    
    def setup_session(self):
        """Configure requests session with headers"""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def get_selenium_driver(self) -> webdriver.Chrome:
        """Create and configure Selenium Chrome driver"""
        if self.driver is None:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
        
        return self.driver
    
    def close_driver(self):
        """Close Selenium driver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def random_delay(self, min_seconds: float = 1, max_seconds: float = 3):
        """Add random delay to avoid detection"""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)
    
    def extract_price(self, price_text: str) -> Optional[int]:
        """Extract price from text and convert to MAD"""
        if not price_text:
            return None
        
        # Remove common price prefixes/suffixes
        price_text = re.sub(r'[^\d\s]', '', price_text)
        price_text = re.sub(r'\s+', '', price_text)
        
        # Extract number
        price_match = re.search(r'\d+', price_text)
        if price_match:
            return int(price_match.group())
        
        return None
    
    def extract_surface(self, surface_text: str) -> Optional[float]:
        """Extract surface area from text"""
        if not surface_text:
            return None
        
        # Look for number followed by m2 or similar
        surface_match = re.search(r'(\d+(?:\.\d+)?)\s*m²?', surface_text.lower())
        if surface_match:
            return float(surface_match.group(1))
        
        return None
    
    def extract_rooms(self, rooms_text: str) -> Optional[int]:
        """Extract number of rooms from text"""
        if not rooms_text:
            return None
        
        # Look for digits
        rooms_match = re.search(r'(\d+)', rooms_text)
        if rooms_match:
            return int(rooms_match.group(1))
        
        return None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def extract_coordinates_from_url(self, url: str) -> tuple[Optional[float], Optional[float]]:
        """Extract latitude and longitude from URL if present"""
        # Look for common coordinate patterns in URLs
        coord_pattern = r'(?:lat|latitude)[=:]?([-]?\d+\.?\d*)[,&]?.*?(?:lng|lon|longitude)[=:]?([-]?\d+\.?\d*)'
        match = re.search(coord_pattern, url, re.IGNORECASE)
        
        if match:
            try:
                lat = float(match.group(1))
                lng = float(match.group(2))
                return lat, lng
            except ValueError:
                pass
        
        return None, None
    
    @abstractmethod
    def get_listing_urls(self, city: str, max_pages: int = 10) -> List[str]:
        """Get list of individual listing URLs for a city"""
        pass
    
    @abstractmethod
    def scrape_listing(self, url: str) -> Optional[PropertyListing]:
        """Scrape individual listing details"""
        pass
    
    def scrape_city(self, city: str, max_pages: int = 10) -> List[PropertyListing]:
        """Scrape all listings for a city"""
        logger.info(f"Starting to scrape {city} on {self.platform_name}")
        
        listing_urls = self.get_listing_urls(city, max_pages)
        logger.info(f"Found {len(listing_urls)} listing URLs for {city}")
        
        listings = []
        for i, url in enumerate(listing_urls):
            try:
                logger.debug(f"Scraping listing {i+1}/{len(listing_urls)}: {url}")
                listing = self.scrape_listing(url)
                
                if listing:
                    listings.append(listing)
                    logger.debug(f"Successfully scraped listing: {listing.title[:50]}...")
                else:
                    logger.warning(f"Failed to scrape listing: {url}")
                
                # Add delay between requests
                self.random_delay()
                
            except Exception as e:
                logger.error(f"Error scraping listing {url}: {e}")
                continue
        
        logger.info(f"Successfully scraped {len(listings)} listings for {city}")
        return listings
    
    def __del__(self):
        """Cleanup when scraper is destroyed"""
        self.close_driver()