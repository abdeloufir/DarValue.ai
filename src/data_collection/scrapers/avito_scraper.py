"""
Avito.ma scraper for real estate listings
"""

import re
import json
from typing import List, Optional
from urllib.parse import urljoin, quote
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from loguru import logger

from .base_scraper import BaseScraper, PropertyListing


class AvitoScraper(BaseScraper):
    """Scraper for Avito.ma real estate listings"""
    
    def __init__(self):
        super().__init__("https://www.avito.ma", "Avito")
        self.city_mappings = {
            'casablanca': 'casablanca',
            'rabat': 'rabat',
            'marrakech': 'marrakech',
            'tangier': 'tanger',
            'fes': 'fes',
            'agadir': 'agadir'
        }
    
    def get_listing_urls(self, city: str, max_pages: int = 10) -> List[str]:
        """Get listing URLs for a specific city"""
        city_slug = self.city_mappings.get(city.lower(), city.lower())
        listing_urls = []
        
        for page in range(1, max_pages + 1):
            try:
                # Construct search URL for real estate in city
                search_url = f"{self.base_url}/fr/immobilier/appartements_et_maisons-{city_slug}?o={page}"
                
                logger.debug(f"Fetching page {page}: {search_url}")
                response = self.session.get(search_url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find listing links
                listing_elements = soup.find_all('a', {'data-testid': 'ad-card-link'})
                
                if not listing_elements:
                    # Try alternative selectors
                    listing_elements = soup.find_all('a', href=re.compile(r'/immobilier/'))
                
                page_urls = []
                for element in listing_elements:
                    href = element.get('href')
                    if href:
                        full_url = urljoin(self.base_url, href)
                        if full_url not in listing_urls:
                            listing_urls.append(full_url)
                            page_urls.append(full_url)
                
                logger.debug(f"Found {len(page_urls)} listings on page {page}")
                
                # If no listings found, we've reached the end
                if not page_urls:
                    logger.info(f"No more listings found on page {page}, stopping")
                    break
                
                self.random_delay()
                
            except Exception as e:
                logger.error(f"Error fetching page {page}: {e}")
                break
        
        return listing_urls
    
    def scrape_listing(self, url: str) -> Optional[PropertyListing]:
        """Scrape individual listing from Avito"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic information
            title = self._extract_title(soup)
            description = self._extract_description(soup)
            price_mad = self._extract_price(soup)
            
            # Extract property details
            details = self._extract_property_details(soup)
            surface_m2 = details.get('surface_m2')
            rooms = details.get('rooms')
            bathrooms = details.get('bathrooms')
            property_type = details.get('property_type')
            
            # Extract location
            location_info = self._extract_location(soup)
            city = location_info.get('city')
            neighborhood = location_info.get('neighborhood')
            
            # Extract coordinates if available
            latitude, longitude = self._extract_coordinates(soup, url)
            
            # Extract images
            image_urls = self._extract_images(soup)
            
            # Extract agent info
            agent_info = self._extract_agent_info(soup)
            agent_name = agent_info.get('name')
            agent_phone = agent_info.get('phone')
            
            # Extract amenities
            amenities = self._extract_amenities(soup)
            
            # Extract source ID from URL
            source_id = self._extract_source_id(url)
            
            if not title or not city:
                logger.warning(f"Missing essential data for listing: {url}")
                return None
            
            return PropertyListing(
                title=title,
                description=description,
                city=city,
                neighborhood=neighborhood,
                price_mad=price_mad,
                surface_m2=surface_m2,
                rooms=rooms,
                bathrooms=bathrooms,
                property_type=property_type,
                amenities=amenities,
                image_urls=image_urls,
                agent_name=agent_name,
                agent_phone=agent_phone,
                latitude=latitude,
                longitude=longitude,
                source_url=url,
                source_id=source_id
            )
            
        except Exception as e:
            logger.error(f"Error scraping Avito listing {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract listing title"""
        title_selectors = [
            'h1[data-testid="ad-title"]',
            'h1.font-bold',
            'h1',
            '.adview-title'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                return self.clean_text(title_elem.get_text())
        
        return None
    
    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract listing description"""
        desc_selectors = [
            '[data-testid="ad-description"]',
            '.description',
            '.adview-description'
        ]
        
        for selector in desc_selectors:
            desc_elem = soup.select_one(selector)
            if desc_elem:
                return self.clean_text(desc_elem.get_text())
        
        return None
    
    def _extract_price(self, soup: BeautifulSoup) -> Optional[int]:
        """Extract price in MAD"""
        price_selectors = [
            '[data-testid="ad-price"]',
            '.price',
            '.adview-price'
        ]
        
        for selector in price_selectors:
            price_elem = soup.select_one(selector)
            if price_elem:
                price_text = price_elem.get_text()
                return self.extract_price(price_text)
        
        return None
    
    def _extract_property_details(self, soup: BeautifulSoup) -> dict:
        """Extract property details like surface, rooms, etc."""
        details = {}
        
        # Look for property details section
        details_section = soup.find('div', class_=re.compile(r'details|specifications'))
        if not details_section:
            details_section = soup
        
        # Extract surface area
        surface_patterns = [
            r'(\d+(?:\.\d+)?)\s*m²',
            r'Surface[:\s]*(\d+(?:\.\d+)?)',
            r'Superficie[:\s]*(\d+(?:\.\d+)?)'
        ]
        
        text = details_section.get_text()
        for pattern in surface_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                details['surface_m2'] = float(match.group(1))
                break
        
        # Extract rooms
        rooms_patterns = [
            r'(\d+)\s*pièces?',
            r'(\d+)\s*chambres?',
            r'Pièces[:\s]*(\d+)',
            r'Chambres[:\s]*(\d+)'
        ]
        
        for pattern in rooms_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                details['rooms'] = int(match.group(1))
                break
        
        # Extract bathrooms
        bath_patterns = [
            r'(\d+)\s*salle[s]?\s*de\s*bain',
            r'(\d+)\s*SDB',
            r'Salles de bain[:\s]*(\d+)'
        ]
        
        for pattern in bath_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                details['bathrooms'] = int(match.group(1))
                break
        
        # Extract property type
        if 'appartement' in text.lower():
            details['property_type'] = 'apartment'
        elif 'maison' in text.lower():
            details['property_type'] = 'house'
        elif 'villa' in text.lower():
            details['property_type'] = 'villa'
        
        return details
    
    def _extract_location(self, soup: BeautifulSoup) -> dict:
        """Extract location information"""
        location = {}
        
        # Look for location elements
        location_selectors = [
            '[data-testid="ad-location"]',
            '.location',
            '.adview-location'
        ]
        
        location_text = ""
        for selector in location_selectors:
            location_elem = soup.select_one(selector)
            if location_elem:
                location_text = location_elem.get_text()
                break
        
        if location_text:
            # Parse location text (usually format: "Neighborhood, City")
            parts = [part.strip() for part in location_text.split(',')]
            if len(parts) >= 2:
                location['neighborhood'] = parts[0]
                location['city'] = parts[1]
            elif len(parts) == 1:
                location['city'] = parts[0]
        
        return location
    
    def _extract_coordinates(self, soup: BeautifulSoup, url: str) -> tuple[Optional[float], Optional[float]]:
        """Extract coordinates from map or URL"""
        # First try to extract from URL
        lat, lng = self.extract_coordinates_from_url(url)
        if lat and lng:
            return lat, lng
        
        # Look for map data in page
        script_tags = soup.find_all('script')
        for script in script_tags:
            if script.string:
                # Look for coordinate patterns in JavaScript
                coord_match = re.search(r'lat["\s]*[:=]\s*([-]?\d+\.?\d*)[,\s].*?lng["\s]*[:=]\s*([-]?\d+\.?\d*)', script.string)
                if coord_match:
                    try:
                        return float(coord_match.group(1)), float(coord_match.group(2))
                    except ValueError:
                        continue
        
        return None, None
    
    def _extract_images(self, soup: BeautifulSoup) -> List[str]:
        """Extract image URLs"""
        image_urls = []
        
        # Look for image gallery
        img_selectors = [
            'img[data-testid="ad-image"]',
            '.gallery img',
            '.carousel img',
            '.slider img'
        ]
        
        for selector in img_selectors:
            images = soup.select(selector)
            for img in images:
                src = img.get('src') or img.get('data-src') or img.get('data-original')
                if src and src.startswith('http'):
                    image_urls.append(src)
        
        return list(set(image_urls))  # Remove duplicates
    
    def _extract_agent_info(self, soup: BeautifulSoup) -> dict:
        """Extract agent contact information"""
        agent_info = {}
        
        # Look for agent/seller information
        agent_section = soup.find('div', class_=re.compile(r'seller|agent|contact'))
        if not agent_section:
            agent_section = soup
        
        # Extract agent name
        name_selectors = [
            '.seller-name',
            '.agent-name',
            '[data-testid="seller-name"]'
        ]
        
        for selector in name_selectors:
            name_elem = agent_section.select_one(selector)
            if name_elem:
                agent_info['name'] = self.clean_text(name_elem.get_text())
                break
        
        # Extract phone (usually requires JavaScript or hidden)
        phone_pattern = r'(\+212|0)[0-9\s\-]{8,}'
        phone_match = re.search(phone_pattern, agent_section.get_text())
        if phone_match:
            agent_info['phone'] = phone_match.group()
        
        return agent_info
    
    def _extract_amenities(self, soup: BeautifulSoup) -> List[str]:
        """Extract amenities and features"""
        amenities = []
        
        # Common amenity keywords in French/Arabic
        amenity_keywords = [
            'parking', 'garage', 'jardin', 'terrasse', 'balcon', 'piscine',
            'ascenseur', 'climatisation', 'chauffage', 'meublé', 'cuisine équipée'
        ]
        
        text = soup.get_text().lower()
        for keyword in amenity_keywords:
            if keyword in text:
                amenities.append(keyword)
        
        return amenities
    
    def _extract_source_id(self, url: str) -> str:
        """Extract listing ID from URL"""
        # Avito URLs typically contain the ID
        id_match = re.search(r'/(\d+)$', url)
        if id_match:
            return id_match.group(1)
        
        # Fallback: use last part of URL
        return url.split('/')[-1]