"""
Test suite for DarValue.ai data collection system
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.data_collection.scrapers.base_scraper import BaseScraper, PropertyListing
from src.data_collection.scrapers.avito_scraper import AvitoScraper
from src.data_collection.enrichment.geospatial_enricher import GeospatialEnricher
from src.data_collection.enrichment.image_collector import ImageDownloader, ImageMetadata
from src.database.models import Listing, ListingEnrichment
from config.settings import ConfigManager


class TestBaseScraper:
    """Test base scraper functionality"""
    
    def test_property_listing_creation(self):
        """Test PropertyListing dataclass"""
        listing = PropertyListing(
            title="Test Apartment",
            description="Beautiful apartment",
            city="Casablanca",
            neighborhood="Maarif",
            price_mad=1500000,
            surface_m2=80.0,
            rooms=3,
            bathrooms=2,
            property_type="apartment",
            amenities=["parking", "elevator"],
            image_urls=["http://example.com/image1.jpg"],
            agent_name="John Doe",
            agent_phone="+212600000000",
            latitude=33.5731,
            longitude=-7.5898,
            source_url="http://example.com/listing",
            source_id="123456"
        )
        
        assert listing.title == "Test Apartment"
        assert listing.city == "Casablanca"
        assert listing.price_mad == 1500000
        assert listing.surface_m2 == 80.0
        assert len(listing.amenities) == 2
    
    def test_price_extraction(self):
        """Test price extraction from text"""
        scraper = AvitoScraper()
        
        assert scraper.extract_price("1 500 000 DH") == 1500000
        assert scraper.extract_price("Price: 850000") == 850000
        assert scraper.extract_price("2,300,000 MAD") == 2300000
        assert scraper.extract_price("No price") is None
    
    def test_surface_extraction(self):
        """Test surface area extraction"""
        scraper = AvitoScraper()
        
        assert scraper.extract_surface("80 m²") == 80.0
        assert scraper.extract_surface("120.5 m2") == 120.5
        assert scraper.extract_surface("Surface: 95 m²") == 95.0
        assert scraper.extract_surface("No surface") is None
    
    def test_coordinate_extraction(self):
        """Test coordinate extraction from URLs"""
        scraper = AvitoScraper()
        
        url1 = "http://example.com/listing?lat=33.5731&lng=-7.5898"
        lat, lng = scraper.extract_coordinates_from_url(url1)
        assert lat == 33.5731
        assert lng == -7.5898
        
        url2 = "http://example.com/listing"
        lat, lng = scraper.extract_coordinates_from_url(url2)
        assert lat is None
        assert lng is None


class TestGeospatialEnricher:
    """Test geospatial enrichment functionality"""
    
    def test_distance_calculation(self):
        """Test distance calculation to city center"""
        enricher = GeospatialEnricher()
        
        # Test Casablanca coordinates
        distance = enricher._calculate_distance_to_city_center(
            33.5731, -7.5898, "casablanca"
        )
        assert distance is not None
        assert distance < 1000  # Should be very close to city center
    
    def test_airport_distance(self):
        """Test distance calculation to airport"""
        enricher = GeospatialEnricher()
        
        distance = enricher._calculate_distance_to_airport(
            33.5731, -7.5898, "casablanca"
        )
        assert distance is not None
        assert distance > 10000  # Should be some distance from city center to airport
    
    @patch('src.data_collection.enrichment.geospatial_enricher.ox.geometries_from_bbox')
    def test_osm_data_retrieval(self, mock_osm):
        """Test OSM data retrieval"""
        # Mock OSM response
        mock_osm.return_value = MagicMock()
        mock_osm.return_value.empty = True
        
        enricher = GeospatialEnricher()
        osm_data = enricher._get_osm_data(33.5731, -7.5898)
        
        assert osm_data is not None
        assert 'walkability_score' in osm_data
    
    def test_geocoding(self):
        """Test address geocoding"""
        enricher = GeospatialEnricher()
        
        with patch.object(enricher.geocoder, 'geocode') as mock_geocode:
            mock_location = Mock()
            mock_location.latitude = 33.5731
            mock_location.longitude = -7.5898
            mock_geocode.return_value = mock_location
            
            lat, lng = enricher.geocode_address("Maarif", "Casablanca")
            assert lat == 33.5731
            assert lng == -7.5898


class TestImageCollector:
    """Test image collection and processing"""
    
    def test_image_metadata_creation(self):
        """Test ImageMetadata dataclass"""
        metadata = ImageMetadata(
            url="http://example.com/image.jpg",
            filename="listing_01_abc123.jpg",
            width=800,
            height=600,
            file_size=150000,
            format="JPEG",
            room_type="bedroom",
            is_exterior=False,
            quality_score=0.8
        )
        
        assert metadata.width == 800
        assert metadata.room_type == "bedroom"
        assert metadata.quality_score == 0.8
    
    @patch('requests.Session.get')
    @patch('PIL.Image.open')
    def test_image_download(self, mock_image_open, mock_requests_get):
        """Test image download functionality"""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content.return_value = [b'fake_image_data']
        mock_requests_get.return_value = mock_response
        
        # Mock PIL Image
        mock_img = Mock()
        mock_img.size = (800, 600)
        mock_img.format = 'JPEG'
        mock_image_open.return_value.__enter__.return_value = mock_img
        
        downloader = ImageDownloader("test_images")
        
        with patch('os.path.getsize', return_value=150000):
            with patch.object(downloader, '_assess_image_quality', return_value=0.8):
                with patch.object(downloader, '_classify_room_type', return_value='bedroom'):
                    with patch.object(downloader, '_is_exterior_image', return_value=False):
                        metadata = downloader.download_single_image(
                            "http://example.com/image.jpg", "listing123", 0
                        )
        
        assert metadata.width == 800
        assert metadata.height == 600
        assert metadata.format == 'JPEG'
        assert metadata.room_type == 'bedroom'


class TestConfigManager:
    """Test configuration management"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        assert config is not None
        assert hasattr(config, 'database')
        assert hasattr(config, 'scraping')
        assert hasattr(config, 'target_cities')
        assert len(config.target_cities) > 0
    
    def test_database_url_generation(self):
        """Test database URL generation"""
        config_manager = ConfigManager()
        url = config_manager.get_database_url()
        
        assert url is not None
        assert url.startswith('postgresql://')


class TestDatabaseModels:
    """Test database models"""
    
    def test_listing_model(self):
        """Test Listing model creation"""
        listing = Listing(
            source_platform="avito",
            source_id="123456",
            source_url="http://example.com/listing",
            title="Test Apartment",
            city="Casablanca",
            price_mad=1500000,
            surface_m2=80.0,
            rooms=3,
            property_type="apartment"
        )
        
        assert listing.source_platform == "avito"
        assert listing.title == "Test Apartment"
        assert listing.price_mad == 1500000
    
    def test_listing_enrichment_model(self):
        """Test ListingEnrichment model"""
        enrichment = ListingEnrichment(
            listing_id=1,
            distance_to_city_center=5000.0,
            walkability_score=75.0,
            schools_count_1km=3
        )
        
        assert enrichment.listing_id == 1
        assert enrichment.distance_to_city_center == 5000.0
        assert enrichment.walkability_score == 75.0


@pytest.fixture
def mock_database():
    """Mock database session for testing"""
    with patch('src.database.get_db_session') as mock_session:
        mock_db = Mock()
        mock_session.return_value = mock_db
        yield mock_db


class TestDataPipeline:
    """Test the main data collection pipeline"""
    
    @patch('src.data_collection.pipeline.get_scraper')
    def test_pipeline_config_creation(self, mock_get_scraper):
        """Test pipeline configuration"""
        from src.data_collection import create_pipeline_config
        
        config = create_pipeline_config(
            cities=['casablanca', 'rabat'],
            platforms=['avito'],
            max_pages_per_city=5
        )
        
        assert len(config.cities) == 2
        assert len(config.platforms) == 1
        assert config.max_pages_per_city == 5


def test_run_basic_scraper():
    """Integration test for basic scraping functionality"""
    scraper = AvitoScraper()
    
    # Test URL generation (without actually making requests)
    city_slug = scraper.city_mappings.get('casablanca')
    assert city_slug == 'casablanca'
    
    # Test scraper initialization
    assert scraper.platform_name == "Avito"
    assert scraper.base_url == "https://www.avito.ma"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])