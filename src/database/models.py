"""
Database models for DarValue.ai real estate platform
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, Boolean, 
    JSON, ForeignKey, Index, DECIMAL
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class Listing(Base):
    """Main listings table for real estate properties"""
    __tablename__ = 'listings'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Source information
    source_platform = Column(String(50), nullable=False)  # avito, mubawab, sarouty
    source_id = Column(String(100), nullable=False)  # Original listing ID
    source_url = Column(Text, nullable=True)
    
    # Basic property information
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    
    # Location
    city = Column(String(100), nullable=False, index=True)
    neighborhood = Column(String(200), nullable=True, index=True)
    address = Column(Text, nullable=True)
    latitude = Column(DECIMAL(10, 8), nullable=True, index=True)
    longitude = Column(DECIMAL(11, 8), nullable=True, index=True)
    
    # Price and property details
    price_mad = Column(Integer, nullable=True, index=True)  # Price in Moroccan Dirhams
    price_per_m2 = Column(Float, nullable=True)
    surface_m2 = Column(Float, nullable=True, index=True)
    
    # Property characteristics
    rooms = Column(Integer, nullable=True)
    bedrooms = Column(Integer, nullable=True)
    bathrooms = Column(Integer, nullable=True)
    property_type = Column(String(50), nullable=True, index=True)  # apartment, house, villa, etc.
    
    # Features and amenities
    amenities = Column(JSON, nullable=True)  # Parking, pool, garden, etc.
    features = Column(JSON, nullable=True)  # Furnished, new construction, etc.
    
    # Images
    image_urls = Column(JSON, nullable=True)  # List of image URLs
    main_image_url = Column(Text, nullable=True)
    images_downloaded = Column(Boolean, default=False)
    
    # Agent/contact information
    agent_name = Column(String(200), nullable=True)
    agent_phone = Column(String(50), nullable=True)
    
    # Status and metadata
    is_active = Column(Boolean, default=True, index=True)
    scraped_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Data quality flags
    has_geolocation = Column(Boolean, default=False, index=True)
    has_images = Column(Boolean, default=False, index=True)
    data_quality_score = Column(Float, nullable=True)
    
    # Relationships
    enriched_data = relationship("ListingEnrichment", back_populates="listing", uselist=False)
    images = relationship("ListingImage", back_populates="listing")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_listing_location', 'city', 'neighborhood'),
        Index('idx_listing_price_surface', 'price_mad', 'surface_m2'),
        Index('idx_listing_coords', 'latitude', 'longitude'),
        Index('idx_listing_source', 'source_platform', 'source_id'),
    )
    
    def __repr__(self):
        return f"<Listing(id={self.id}, title='{self.title[:50]}...', city='{self.city}', price={self.price_mad})>"


class ListingEnrichment(Base):
    """Enriched geospatial and neighborhood data for listings"""
    __tablename__ = 'listing_enrichments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    listing_id = Column(Integer, ForeignKey('listings.id'), unique=True, nullable=False)
    
    # Distance-based features (in meters)
    distance_to_city_center = Column(Float, nullable=True)
    distance_to_nearest_school = Column(Float, nullable=True)
    distance_to_nearest_hospital = Column(Float, nullable=True)
    distance_to_nearest_metro = Column(Float, nullable=True)
    distance_to_beach = Column(Float, nullable=True)
    distance_to_airport = Column(Float, nullable=True)
    
    # Neighborhood statistics
    neighborhood_avg_price_m2 = Column(Float, nullable=True)
    neighborhood_property_density = Column(Float, nullable=True)
    walkability_score = Column(Float, nullable=True)
    
    # Points of interest counts (within 1km radius)
    schools_count_1km = Column(Integer, nullable=True)
    restaurants_count_1km = Column(Integer, nullable=True)
    shops_count_1km = Column(Integer, nullable=True)
    parks_count_1km = Column(Integer, nullable=True)
    
    # Infrastructure quality indicators
    street_quality_score = Column(Float, nullable=True)
    noise_level_estimate = Column(Float, nullable=True)
    air_quality_index = Column(Float, nullable=True)
    
    # Demographic and economic data
    neighborhood_income_level = Column(String(20), nullable=True)  # low, medium, high
    population_density = Column(Float, nullable=True)
    
    # OSM and government data
    osm_data = Column(JSON, nullable=True)
    census_data = Column(JSON, nullable=True)
    
    # Enrichment metadata
    enriched_at = Column(DateTime, default=func.now())
    data_sources = Column(JSON, nullable=True)  # Track which APIs were used
    
    # Relationship
    listing = relationship("Listing", back_populates="enriched_data")
    
    def __repr__(self):
        return f"<ListingEnrichment(listing_id={self.listing_id}, distance_to_center={self.distance_to_city_center})>"


class ListingImage(Base):
    """Individual images for listings with metadata"""
    __tablename__ = 'listing_images'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    listing_id = Column(Integer, ForeignKey('listings.id'), nullable=False)
    
    # Image information
    original_url = Column(Text, nullable=False)
    storage_path = Column(Text, nullable=True)  # S3/GCS path
    local_path = Column(Text, nullable=True)  # Local file path if stored locally
    
    # Image metadata
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    format = Column(String(10), nullable=True)  # jpg, png, webp
    
    # AI-powered image classification
    room_type = Column(String(50), nullable=True)  # bedroom, kitchen, living_room, exterior
    is_exterior = Column(Boolean, nullable=True)
    is_interior = Column(Boolean, nullable=True)
    quality_score = Column(Float, nullable=True)  # Image quality assessment
    
    # Computer vision features
    cv_features = Column(JSON, nullable=True)  # Extracted visual features
    objects_detected = Column(JSON, nullable=True)  # Objects found in image
    
    # Status
    downloaded = Column(Boolean, default=False)
    processed = Column(Boolean, default=False)
    download_error = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    downloaded_at = Column(DateTime, nullable=True)
    processed_at = Column(DateTime, nullable=True)
    
    # Relationship
    listing = relationship("Listing", back_populates="images")
    
    __table_args__ = (
        Index('idx_image_listing', 'listing_id'),
        Index('idx_image_room_type', 'room_type'),
    )
    
    def __repr__(self):
        return f"<ListingImage(id={self.id}, listing_id={self.listing_id}, room_type='{self.room_type}')>"


class ScrapingLog(Base):
    """Log table for tracking scraping activities"""
    __tablename__ = 'scraping_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Scraping session info
    platform = Column(String(50), nullable=False, index=True)
    session_id = Column(String(100), nullable=False, index=True)
    
    # Results
    listings_found = Column(Integer, nullable=True)
    listings_new = Column(Integer, nullable=True)
    listings_updated = Column(Integer, nullable=True)
    errors_count = Column(Integer, nullable=True)
    
    # Timing
    started_at = Column(DateTime, nullable=False)
    finished_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    
    # Status and errors
    status = Column(String(20), nullable=False)  # running, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Configuration used
    config_snapshot = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<ScrapingLog(id={self.id}, platform='{self.platform}', status='{self.status}')>"