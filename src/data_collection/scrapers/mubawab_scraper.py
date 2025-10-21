"""
Mubawab.ma scraper for real estate listings
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


class MubawabScraper(BaseScraper):
    """Scraper for Mubawab.ma real estate listings"""
    
    def __init__(self):
        super().__init__("https://www.mubawab.ma", "Mubawab")
        self.city_mappings = {
            'casablanca': 'casablanca',
            'rabat': 'rabat',
            'marrakech': 'marrakech',
            'tangier': 'tangier',
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
                search_url = f"{self.base_url}/en/buy/morocco/{city_slug}/real-estate?page={page}"
                
                logger.debug(f"Fetching page {page}: {search_url}")
                response = self.session.get(search_url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find listing links
                listing_elements = soup.find_all('a', class_=re.compile(r'listing-card|property-card'))
                
                if not listing_elements:
                    # Try alternative selectors
                    listing_elements = soup.find_all('a', href=re.compile(r'/property/'))
                
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
        """Scrape individual listing from Mubawab"""
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
            logger.error(f"Error scraping Mubawab listing {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract listing title"""
        title_selectors = [
            'h1.property-title',
            'h1.listing-title',
            '.title h1',
            'h1'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                return self.clean_text(title_elem.get_text())
        
        return None
    
    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract listing description"""
        desc_selectors = [
            '.property-description',
            '.description-content',
            '.listing-description',
            '.description'
        ]
        
        for selector in desc_selectors:
            desc_elem = soup.select_one(selector)
            if desc_elem:
                return self.clean_text(desc_elem.get_text())
        
        return None
    
    def _extract_price(self, soup: BeautifulSoup) -> Optional[int]:
        """Extract price in MAD"""
        price_selectors = [
            '.property-price',
            '.listing-price',
            '.price-value',
            '.price'
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
        
        # Look for property details in structured format
        detail_items = soup.find_all('div', class_=re.compile(r'detail-item|property-detail|spec'))
        
        for item in detail_items:
            label = item.find('span', class_=re.compile(r'label|key'))
            value = item.find('span', class_=re.compile(r'value'))
            
            if label and value:
                label_text = label.get_text().lower()
                value_text = value.get_text()
                
                if 'surface' in label_text or 'area' in label_text:
                    surface = self.extract_surface(value_text)
                    if surface:
                        details['surface_m2'] = surface
                
                elif 'room' in label_text or 'pièce' in label_text:
                    rooms = self.extract_rooms(value_text)
                    if rooms:
                        details['rooms'] = rooms
                
                elif 'bath' in label_text or 'sdb' in label_text:
                    bathrooms = self.extract_rooms(value_text)
                    if bathrooms:
                        details['bathrooms'] = bathrooms
                
                elif 'type' in label_text:
                    prop_type = value_text.lower()
                    if 'apartment' in prop_type or 'appartement' in prop_type:
                        details['property_type'] = 'apartment'
                    elif 'house' in prop_type or 'maison' in prop_type:
                        details['property_type'] = 'house'
                    elif 'villa' in prop_type:
                        details['property_type'] = 'villa'
        
        # Fallback: extract from general text
        if not details:
            text = soup.get_text()
            
            # Surface area
            surface_match = re.search(r'(\d+(?:\.\d+)?)\s*m²', text)
            if surface_match:
                details['surface_m2'] = float(surface_match.group(1))
            
            # Rooms
            rooms_match = re.search(r'(\d+)\s*(?:rooms?|pièces?|chambres?)', text, re.IGNORECASE)
            if rooms_match:
                details['rooms'] = int(rooms_match.group(1))
        
        return details
    
    def _extract_location(self, soup: BeautifulSoup) -> dict:
        """Extract location information"""
        location = {}
        
        # Look for location elements
        location_selectors = [
            '.property-location',
            '.listing-location',
            '.address',
            '.location'
        ]
        
        location_text = ""
        for selector in location_selectors:
            location_elem = soup.select_one(selector)
            if location_elem:
                location_text = location_elem.get_text()
                break
        
        if location_text:
            # Parse location text
            parts = [part.strip() for part in location_text.split(',')]
            if len(parts) >= 2:
                location['neighborhood'] = parts[0]
                location['city'] = parts[-1]  # City is usually last
            elif len(parts) == 1:
                location['city'] = parts[0]
        
        # Also check breadcrumbs
        breadcrumbs = soup.find('nav', class_=re.compile(r'breadcrumb'))
        if breadcrumbs:
            links = breadcrumbs.find_all('a')
            for link in links:
                text = link.get_text().strip()
                if text and text not in ['Home', 'Buy', 'Morocco']:
                    if not location.get('city'):
                        location['city'] = text
        
        return location
    
    def _extract_coordinates(self, soup: BeautifulSoup, url: str) -> tuple[Optional[float], Optional[float]]:
        """Extract coordinates from map or scripts"""
        # First try to extract from URL
        lat, lng = self.extract_coordinates_from_url(url)
        if lat and lng:
            return lat, lng
        
        # Look for Google Maps embed or scripts
        script_tags = soup.find_all('script')
        for script in script_tags:
            if script.string:
                # Look for coordinate patterns
                patterns = [
                    r'lat["\s]*[:=]\s*([-]?\d+\.?\d*)[,\s].*?lng["\s]*[:=]\s*([-]?\d+\.?\d*)',
                    r'latitude["\s]*[:=]\s*([-]?\d+\.?\d*)[,\s].*?longitude["\s]*[:=]\s*([-]?\d+\.?\d*)',
                    r'center["\s]*[:=]\s*\{\s*lat[:\s]*([-]?\d+\.?\d*)[,\s]*lng[:\s]*([-]?\d+\.?\d*)'
                ]
                
                for pattern in patterns:
                    coord_match = re.search(pattern, script.string, re.IGNORECASE)
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
            '.property-gallery img',
            '.listing-images img',
            '.image-gallery img',
            '.photos img'
        ]
        
        for selector in img_selectors:
            images = soup.select(selector)
            for img in images:
                src = img.get('src') or img.get('data-src') or img.get('data-original')
                if src:
                    if src.startswith('//'):
                        src = 'https:' + src
                    elif src.startswith('/'):
                        src = urljoin(self.base_url, src)
                    
                    if src.startswith('http'):
                        image_urls.append(src)
        
        return list(set(image_urls))  # Remove duplicates
    
    def _extract_agent_info(self, soup: BeautifulSoup) -> dict:
        """Extract agent contact information"""
        agent_info = {}
        
        # Look for agent/agency information
        agent_section = soup.find('div', class_=re.compile(r'agent|agency|contact|seller'))
        if not agent_section:
            agent_section = soup
        
        # Extract agent name
        name_selectors = [
            '.agent-name',
            '.agency-name',
            '.contact-name',
            '.seller-name'
        ]
        
        for selector in name_selectors:
            name_elem = agent_section.select_one(selector)
            if name_elem:
                agent_info['name'] = self.clean_text(name_elem.get_text())
                break
        
        # Extract phone
        phone_pattern = r'(\+212|0)[0-9\s\-]{8,}'
        phone_match = re.search(phone_pattern, agent_section.get_text())
        if phone_match:
            agent_info['phone'] = phone_match.group().strip()
        
        return agent_info
    
    def _extract_amenities(self, soup: BeautifulSoup) -> List[str]:
        """Extract amenities and features"""
        amenities = []
        
        # Look for amenities section
        amenities_section = soup.find('div', class_=re.compile(r'amenities|features|facilities'))
        if amenities_section:
            # Extract from structured list
            amenity_items = amenities_section.find_all(['li', 'span', 'div'])
            for item in amenity_items:
                text = item.get_text().strip().lower()
                if text:
                    amenities.append(text)
        
        # Fallback: look for common amenity keywords
        if not amenities:
            amenity_keywords = [
                'parking', 'garage', 'garden', 'terrace', 'balcony', 'pool',
                'elevator', 'air conditioning', 'heating', 'furnished'
            ]
            
            text = soup.get_text().lower()
            for keyword in amenity_keywords:
                if keyword in text:
                    amenities.append(keyword)
        
        return amenities
    
    def _extract_source_id(self, url: str) -> str:
        """Extract listing ID from URL"""
        # Mubawab URLs typically contain the ID
        id_match = re.search(r'/property/(\w+)', url)
        if id_match:
            return id_match.group(1)
        
        # Fallback: use last part of URL
        return url.split('/')[-1]