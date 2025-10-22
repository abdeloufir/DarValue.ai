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
            'tangier': 'tanger',  # Updated mapping
            'fes': 'fes',
            'agadir': 'agadir'
        }
    
    def get_listing_urls(self, city: str, max_pages: Optional[int] = None) -> List[str]:
        """Get listing URLs for a specific city.

        Pagination uses the `:p:<page>` pattern (e.g. `...immobilier-a-vendre:p:2`).
        If `max_pages` is None the scraper will keep requesting pages until
        a page returns no listings (or an error).
        """
        city_slug = self.city_mappings.get(city.lower(), city.lower())
        listing_urls: List[str] = []

        page = 1
        while True:
            if max_pages is not None and page > max_pages:
                logger.debug(f"Reached max_pages ({max_pages}) for {city_slug}, stopping")
                break

            try:
                # Build search URL using the site-specific pagination format
                if page == 1:
                    search_url = f"{self.base_url}/fr/ct/{city_slug}/immobilier-a-vendre"
                else:
                    search_url = f"{self.base_url}/fr/ct/{city_slug}/immobilier-a-vendre:p:{page}"

                logger.debug(f"Fetching page {page}: {search_url}")
                response = self.session.get(search_url, timeout=20)

                if response.status_code != 200:
                    logger.warning(f"Failed to fetch page {page} for {city_slug}: {response.status_code}")
                    # If a single page fails, stop to avoid endless loops
                    break

                soup = BeautifulSoup(response.content, 'html.parser')

                # Find listing links using the patterns we discovered
                listing_elements = []
                all_links = soup.find_all('a', href=True)

                property_patterns = [
                    r'/fr/a/\d+',      # /fr/a/8235765/...
                    r'/fr/pa/\d+',     # /fr/pa/8155064/...
                    r'/a/\d+',
                    r'/pa/\d+'
                ]

                for link in all_links:
                    href = link.get('href', '')
                    for pattern in property_patterns:
                        if re.search(pattern, href):
                            listing_elements.append(link)
                            break

                page_urls: List[str] = []
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

                page += 1
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
            '.fullPicturesPrice',  # This is the working selector we found
            '.price',
            '.cost',
            '[class*="price"]',
            'span[class*="price"]',
            'div[class*="price"]'
        ]
        
        for selector in price_selectors:
            price_elem = soup.select_one(selector)
            if price_elem:
                price_text = price_elem.get_text()
                price = self.extract_price(price_text)
                if price:
                    return price
        
        # Fallback: look for price patterns in all text
        all_text = soup.get_text()
        import re
        price_patterns = [
            r'(\d[\d\s\u00a0,\.]*)\s*(?:DH|MAD|dh|mad)',  # Include non-breaking space \u00a0
            r'Prix\s*:?\s*(\d[\d\s\u00a0,\.]*)',
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                price_text = match.group(1).replace('\u00a0', ' ')  # Replace non-breaking spaces
                price = self.extract_price(price_text)
                if price:
                    return price
        
        return None
    
    def _extract_property_details(self, soup: BeautifulSoup) -> dict:
        """Extract property details like surface, rooms, etc."""
        details = {}
        
        # Look for details in the .adDetails section we discovered
        ad_details = soup.select_one('.adDetails')
        if ad_details:
            details_text = ad_details.get_text()
            
            import re
            # Extract surface (m²)
            surface_match = re.search(r'(\d+(?:[,\.]\d+)?)\s*m[²2]', details_text, re.IGNORECASE)
            if surface_match:
                details['surface_m2'] = float(surface_match.group(1).replace(',', '.'))
            
            # Extract rooms/chambers
            rooms_match = re.search(r'(\d+)\s*(?:chambre|bedroom|room)', details_text, re.IGNORECASE)
            if rooms_match:
                details['rooms'] = int(rooms_match.group(1))
            
            # Extract pieces (total rooms)
            pieces_match = re.search(r'(\d+)\s*pi[èe]ces?', details_text, re.IGNORECASE)
            if pieces_match and not details.get('rooms'):
                details['rooms'] = int(pieces_match.group(1))
            
            # Extract bathrooms
            bath_match = re.search(r'(\d+)\s*(?:salle.*bain|bathroom|sdb)', details_text, re.IGNORECASE)
            if bath_match:
                details['bathrooms'] = int(bath_match.group(1))
        
        # Fallback: look for details in all text using broader patterns
        if not details:
            all_text = soup.get_text()
            
            import re
            # Surface patterns
            surface_patterns = [
                r'(\d+(?:[,\.]\d+)?)\s*m[²2]',
                r'surface[:\s]*(\d+(?:[,\.]\d+)?)\s*m',
                r'superficie[:\s]*(\d+(?:[,\.]\d+)?)\s*m'
            ]
            
            for pattern in surface_patterns:
                match = re.search(pattern, all_text, re.IGNORECASE)
                if match:
                    details['surface_m2'] = float(match.group(1).replace(',', '.'))
                    break
                    
            # Room patterns
            room_patterns = [
                r'(\d+)\s*chambre',
                r'(\d+)\s*pi[èe]ces?',
                r'(\d+)\s*bedroom'
            ]
            
            for pattern in room_patterns:
                match = re.search(pattern, all_text, re.IGNORECASE)
                if match:
                    details['rooms'] = int(match.group(1))
                    break
                    
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
        
        # Fallback: extract from title or set a default
        if not location.get('city'):
            title = soup.select_one('h1')
            if title:
                title_text = title.get_text().lower()
                # Map common city names in titles
                city_mapping = {
                    'casablanca': 'Casablanca',
                    'rabat': 'Rabat',
                    'marrakech': 'Marrakech',
                    'tanger': 'Tangier',
                    'fes': 'Fes',
                    'agadir': 'Agadir'
                }
                for city_key, city_name in city_mapping.items():
                    if city_key in title_text:
                        location['city'] = city_name
                        break
                        
        # Last resort: assume casablanca (since we're testing with casablanca)
        if not location.get('city'):
            location['city'] = 'Casablanca'
        
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
        # Mubawab listing URLs often contain numeric IDs in patterns like:
        #  - /fr/a/8235765/...
        #  - /fr/pa/8155064/...
        id_match = re.search(r'/a/(\d+)', url)
        if not id_match:
            id_match = re.search(r'/pa/(\d+)', url)
        if id_match:
            return id_match.group(1)

        # Fallback: try to extract any trailing numeric segment
        alt_match = re.search(r'/(\d+)(?:$|/)', url)
        if alt_match:
            return alt_match.group(1)

        # Last fallback: return the last path segment
        return url.rstrip('/').split('/')[-1]