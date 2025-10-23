"""
Main data collection pipeline for DarValue.ai
Orchestrates scraping, enrichment, and storage
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.orm import Session
from loguru import logger
import time

from src.database import get_db_session, Listing, ListingEnrichment, ListingImage, ScrapingLog
from .scrapers import get_scraper, PropertyListing
from .enrichment import GeospatialEnricher, MoroccanDataEnricher
from .enrichment.image_collector import ImageDownloader, CloudImageStorage, ImageProcessor


@dataclass
class PipelineConfig:
    """Configuration for the data collection pipeline"""
    cities: List[str]
    platforms: List[str]
    max_pages_per_city: int = 10
    enable_geospatial_enrichment: bool = True
    enable_image_download: bool = True
    enable_cloud_storage: bool = False
    cloud_provider: str = "aws"
    google_maps_api_key: Optional[str] = None
    aws_config: Optional[Dict] = None
    gcp_config: Optional[Dict] = None
    max_workers: int = 4
    delay_between_platforms: int = 5


class DataCollectionPipeline:
    """Main pipeline for collecting and processing real estate data"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.session_id = str(uuid.uuid4())
        
        # Initialize components
        self.geospatial_enricher = GeospatialEnricher(
            google_maps_api_key=config.google_maps_api_key
        ) if config.enable_geospatial_enrichment else None
        
        self.moroccan_enricher = MoroccanDataEnricher()
        
        self.image_downloader = ImageDownloader() if config.enable_image_download else None
        
        self.cloud_storage = None
        if config.enable_cloud_storage:
            if config.cloud_provider == "aws" and config.aws_config:
                self.cloud_storage = CloudImageStorage("aws", **config.aws_config)
            elif config.cloud_provider == "gcp" and config.gcp_config:
                self.cloud_storage = CloudImageStorage("gcp", **config.gcp_config)
        
        self.image_processor = ImageProcessor()
        
        # Statistics
        self.stats = {
            'total_listings_found': 0,
            'total_listings_new': 0,
            'total_listings_updated': 0,
            'total_errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete data collection pipeline"""
        logger.info(f"Starting data collection pipeline (Session: {self.session_id})")
        self.stats['start_time'] = datetime.now()
        
        try:
            # Process each platform
            for platform in self.config.platforms:
                try:
                    logger.info(f"Processing platform: {platform}")
                    self._process_platform(platform)
                    
                    # Delay between platforms to be respectful
                    if len(self.config.platforms) > 1:
                        logger.info(f"Waiting {self.config.delay_between_platforms}s before next platform...")
                        time.sleep(self.config.delay_between_platforms)
                        
                except Exception as e:
                    logger.error(f"Error processing platform {platform}: {e}")
                    self.stats['total_errors'] += 1
                    continue
            
            self.stats['end_time'] = datetime.now()
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
            logger.info(f"Pipeline completed in {duration:.1f}s")
            logger.info(f"Statistics: {self.stats}")
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.stats['end_time'] = datetime.now()
            raise
    
    def _process_platform(self, platform: str):
        """Process a single platform across all cities"""
        platform_start = datetime.now()
        platform_stats = {
            'listings_found': 0,
            'listings_new': 0,
            'listings_updated': 0,
            'errors': 0
        }
        
        # Log scraping session start
        db_session = get_db_session()
        try:
            scraping_log = ScrapingLog(
                platform=platform,
                session_id=self.session_id,
                started_at=platform_start,
                status='running',
                config_snapshot={'cities': self.config.cities, 'max_pages': self.config.max_pages_per_city}
            )
            db_session.add(scraping_log)
            db_session.commit()
            log_id = scraping_log.id
        except Exception as e:
            logger.error(f"Error creating scraping log: {e}")
            db_session.rollback()
            log_id = None
        finally:
            db_session.close()
        
        try:
            # Get scraper instance
            scraper = get_scraper(platform)
            
            # Process each city
            for city in self.config.cities:
                try:
                    logger.info(f"Scraping {city} on {platform}")
                    city_stats = self._process_city(scraper, city, platform)
                    
                    # Update platform stats
                    platform_stats['listings_found'] += city_stats['found']
                    platform_stats['listings_new'] += city_stats['new']
                    platform_stats['listings_updated'] += city_stats['updated']
                    platform_stats['errors'] += city_stats['errors']
                    
                except Exception as e:
                    logger.error(f"Error processing {city} on {platform}: {e}")
                    platform_stats['errors'] += 1
                    continue
            
            # Update global stats
            self.stats['total_listings_found'] += platform_stats['listings_found']
            self.stats['total_listings_new'] += platform_stats['listings_new']
            self.stats['total_listings_updated'] += platform_stats['listings_updated']
            self.stats['total_errors'] += platform_stats['errors']
            
            # Update scraping log
            if log_id:
                db_session = get_db_session()
                try:
                    scraping_log = db_session.query(ScrapingLog).get(log_id)
                    if scraping_log:
                        scraping_log.finished_at = datetime.now()
                        scraping_log.duration_seconds = int((scraping_log.finished_at - scraping_log.started_at).total_seconds())
                        scraping_log.status = 'completed'
                        scraping_log.listings_found = platform_stats['listings_found']
                        scraping_log.listings_new = platform_stats['listings_new']
                        scraping_log.listings_updated = platform_stats['listings_updated']
                        scraping_log.errors_count = platform_stats['errors']
                        db_session.commit()
                except Exception as e:
                    logger.error(f"Error updating scraping log: {e}")
                    db_session.rollback()
                finally:
                    db_session.close()
            
        except Exception as e:
            logger.error(f"Error with {platform} scraper: {e}")
            # Update log as failed
            if log_id:
                db_session = get_db_session()
                try:
                    scraping_log = db_session.query(ScrapingLog).get(log_id)
                    if scraping_log:
                        scraping_log.finished_at = datetime.now()
                        scraping_log.status = 'failed'
                        scraping_log.error_message = str(e)
                        db_session.commit()
                except Exception:
                    db_session.rollback()
                finally:
                    db_session.close()
            raise
        
        finally:
            # Clean up scraper
            if hasattr(scraper, 'close_driver'):
                scraper.close_driver()
    
    def _process_city(self, scraper, city: str, platform: str) -> Dict[str, int]:
        """Process a single city with a scraper"""
        city_stats = {'found': 0, 'new': 0, 'updated': 0, 'errors': 0}
        
        try:
            # Scrape listings
            listings = scraper.scrape_city(city, self.config.max_pages_per_city)
            city_stats['found'] = len(listings)
            
            logger.info(f"Found {len(listings)} listings for {city} on {platform}")
            
            # Process listings in parallel
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                
                for listing in listings:
                    future = executor.submit(self._process_single_listing, listing, platform)
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result == 'new':
                            city_stats['new'] += 1
                        elif result == 'updated':
                            city_stats['updated'] += 1
                    except Exception as e:
                        logger.error(f"Error processing listing: {e}")
                        city_stats['errors'] += 1
            
        except Exception as e:
            logger.error(f"Error scraping {city} on {platform}: {e}")
            city_stats['errors'] += 1
        
        return city_stats
    
    def _process_single_listing(self, property_listing: PropertyListing, platform: str) -> str:
        """Process a single listing: store, enrich, and download images"""
        db_session = get_db_session()
        try:
            # Check if listing already exists
            existing_listing = db_session.query(Listing).filter_by(
                source_platform=platform,
                source_id=property_listing.source_id
            ).first()
            
            if existing_listing:
                # Update existing listing
                self._update_listing(db_session, existing_listing, property_listing)
                result = 'updated'
            else:
                # Create new listing
                existing_listing = self._create_listing(db_session, property_listing, platform)
                result = 'new'
            
            db_session.commit()
            listing_id = existing_listing.id
            
            # Geospatial enrichment (if enabled and coordinates available)
            if (self.config.enable_geospatial_enrichment and 
                property_listing.latitude and property_listing.longitude):
                
                self._enrich_listing_geospatial(
                    db_session, listing_id, 
                    property_listing.latitude, property_listing.longitude, 
                    property_listing.city
                )
            
            # Image processing (if enabled and images available)
            if (self.config.enable_image_download and 
                property_listing.image_urls):
                
                self._process_listing_images(
                    db_session, listing_id, property_listing.image_urls
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing listing {property_listing.source_id}: {e}")
            db_session.rollback()
            raise
        finally:
            db_session.close()
    
    def _create_listing(self, db_session: Session, property_listing: PropertyListing, platform: str) -> Listing:
        """Create a new listing in the database"""
        listing = Listing(
            source_platform=platform,
            source_id=property_listing.source_id,
            source_url=property_listing.source_url,
            title=property_listing.title,
            description=property_listing.description,
            city=property_listing.city,
            neighborhood=property_listing.neighborhood,
            latitude=property_listing.latitude,
            longitude=property_listing.longitude,
            price_mad=property_listing.price_mad,
            surface_m2=property_listing.surface_m2,
            rooms=property_listing.rooms,
            bathrooms=property_listing.bathrooms,
            property_type=property_listing.property_type,
            amenities=property_listing.amenities,
            image_urls=property_listing.image_urls,
            agent_name=property_listing.agent_name,
            agent_phone=property_listing.agent_phone,
            has_geolocation=bool(property_listing.latitude and property_listing.longitude),
            has_images=bool(property_listing.image_urls),
            scraped_at=datetime.now()
        )
        
        # Calculate price per m2 if both price and surface are available
        if listing.price_mad and listing.surface_m2 and listing.surface_m2 > 0:
            listing.price_per_m2 = listing.price_mad / listing.surface_m2
        
        db_session.add(listing)
        db_session.flush()  # Get the ID
        
        logger.debug(f"Created new listing: {listing.title[:50]}...")
        return listing
    
    def _update_listing(self, db_session: Session, existing_listing: Listing, 
                       property_listing: PropertyListing):
        """Update an existing listing"""
        # Update fields that might have changed
        existing_listing.title = property_listing.title
        existing_listing.description = property_listing.description
        existing_listing.price_mad = property_listing.price_mad
        existing_listing.surface_m2 = property_listing.surface_m2
        existing_listing.rooms = property_listing.rooms
        existing_listing.bathrooms = property_listing.bathrooms
        existing_listing.amenities = property_listing.amenities
        existing_listing.image_urls = property_listing.image_urls
        existing_listing.agent_name = property_listing.agent_name
        existing_listing.agent_phone = property_listing.agent_phone
        existing_listing.updated_at = datetime.now()
        
        # Update geolocation if not previously available
        if not existing_listing.latitude and property_listing.latitude:
            existing_listing.latitude = property_listing.latitude
            existing_listing.longitude = property_listing.longitude
            existing_listing.has_geolocation = True
        
        # Update price per m2
        if existing_listing.price_mad and existing_listing.surface_m2 and existing_listing.surface_m2 > 0:
            existing_listing.price_per_m2 = existing_listing.price_mad / existing_listing.surface_m2
        
        logger.debug(f"Updated listing: {existing_listing.title[:50]}...")
    
    def _enrich_listing_geospatial(self, db_session: Session, listing_id: int, 
                                  latitude: float, longitude: float, city: str):
        """Add geospatial enrichment to a listing"""
        try:
            logger.debug(f"Enriching listing {listing_id} with geospatial data")
            
            # Get enrichment data
            enrichment_data = self.geospatial_enricher.enrich_listing(latitude, longitude, city)
            
            # Check if enrichment already exists
            existing_enrichment = db_session.query(ListingEnrichment).filter_by(
                listing_id=listing_id
            ).first()
            
            if existing_enrichment:
                # Update existing
                for attr_name in dir(enrichment_data):
                    if not attr_name.startswith('_'):
                        value = getattr(enrichment_data, attr_name)
                        if value is not None:
                            setattr(existing_enrichment, attr_name, value)
            else:
                # Create new enrichment
                enrichment = ListingEnrichment(
                    listing_id=listing_id,
                    distance_to_city_center=enrichment_data.distance_to_city_center,
                    distance_to_nearest_school=enrichment_data.distance_to_nearest_school,
                    distance_to_nearest_hospital=enrichment_data.distance_to_nearest_hospital,
                    distance_to_beach=enrichment_data.distance_to_beach,
                    distance_to_airport=enrichment_data.distance_to_airport,
                    walkability_score=enrichment_data.walkability_score,
                    schools_count_1km=enrichment_data.schools_count_1km,
                    restaurants_count_1km=enrichment_data.restaurants_count_1km,
                    shops_count_1km=enrichment_data.shops_count_1km,
                    parks_count_1km=enrichment_data.parks_count_1km,
                    neighborhood_income_level=enrichment_data.neighborhood_income_level,
                    street_quality_score=enrichment_data.street_quality_score,
                    osm_data=enrichment_data.osm_data,
                    enriched_at=datetime.now(),
                    data_sources=['osm', 'nominatim']
                )
                db_session.add(enrichment)
            
            db_session.commit()
            logger.debug(f"Successfully enriched listing {listing_id}")
            
        except Exception as e:
            logger.error(f"Error enriching listing {listing_id}: {e}")
            db_session.rollback()
    
    def _process_listing_images(self, db_session: Session, listing_id: int, 
                               image_urls: List[str]):
        """Download and process images for a listing"""
        try:
            logger.debug(f"Processing {len(image_urls)} images for listing {listing_id}")
            
            # Download images
            image_metadata_list = self.image_downloader.download_images(image_urls, str(listing_id))
            
            # Upload to cloud storage if enabled
            if self.cloud_storage:
                image_metadata_list = self.cloud_storage.batch_upload_images(
                    image_metadata_list, str(listing_id)
                )
            
            # Store image records in database
            for metadata in image_metadata_list:
                # Check if image already exists
                existing_image = db_session.query(ListingImage).filter_by(
                    listing_id=listing_id,
                    original_url=metadata.url
                ).first()
                
                if existing_image:
                    # Update existing
                    existing_image.storage_path = metadata.storage_path
                    existing_image.width = metadata.width
                    existing_image.height = metadata.height
                    existing_image.file_size_bytes = metadata.file_size
                    existing_image.format = metadata.format
                    existing_image.room_type = metadata.room_type
                    existing_image.is_exterior = metadata.is_exterior
                    existing_image.quality_score = metadata.quality_score
                    existing_image.downloaded = not bool(metadata.download_error)
                    existing_image.download_error = metadata.download_error
                    existing_image.downloaded_at = datetime.now()
                else:
                    # Create new
                    image_record = ListingImage(
                        listing_id=listing_id,
                        original_url=metadata.url,
                        storage_path=metadata.storage_path,
                        local_path=f"data/images/{metadata.filename}" if metadata.filename else None,
                        width=metadata.width,
                        height=metadata.height,
                        file_size_bytes=metadata.file_size,
                        format=metadata.format,
                        room_type=metadata.room_type,
                        is_exterior=metadata.is_exterior,
                        quality_score=metadata.quality_score,
                        downloaded=not bool(metadata.download_error),
                        download_error=metadata.download_error,
                        downloaded_at=datetime.now() if not metadata.download_error else None
                    )
                    db_session.add(image_record)
            
            # Update listing to mark images as downloaded
            listing = db_session.query(Listing).get(listing_id)
            if listing:
                listing.images_downloaded = True
            
            db_session.commit()
            logger.debug(f"Successfully processed images for listing {listing_id}")
            
        except Exception as e:
            logger.error(f"Error processing images for listing {listing_id}: {e}")
            db_session.rollback()


def create_pipeline_config(
    cities: List[str] = ['casablanca', 'rabat', 'marrakech', 'tangier'],
    platforms: List[str] = ['mubawab'],
    **kwargs
) -> PipelineConfig:
    """Helper function to create pipeline configuration"""
    return PipelineConfig(
        cities=cities,
        platforms=platforms,
        **kwargs
    )