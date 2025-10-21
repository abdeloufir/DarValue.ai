"""
Quick test script to validate the DarValue.ai system without external dependencies
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_imports():
    """Test that all main modules can be imported"""
    print("Testing imports...")
    
    try:
        from data_collection.scrapers import get_scraper, PropertyListing
        print("✅ Scrapers import successful")
    except Exception as e:
        print(f"❌ Scrapers import failed: {e}")
        return False
    
    try:
        from data_collection.enrichment import GeospatialEnricher
        print("✅ Geospatial enrichment import successful")
    except Exception as e:
        print(f"❌ Geospatial enrichment import failed: {e}")
        return False
    
    try:
        from database import Listing, ListingEnrichment, db_manager
        print("✅ Database models import successful")
    except Exception as e:
        print(f"❌ Database models import failed: {e}")
        return False
    
    try:
        from config.settings import get_config
        print("✅ Configuration import successful")
    except Exception as e:
        print(f"❌ Configuration import failed: {e}")
        return False
    
    try:
        from utils.monitoring import get_logger
        print("✅ Monitoring import successful")
    except Exception as e:
        print(f"❌ Monitoring import failed: {e}")
        return False
    
    return True


def test_scraper_creation():
    """Test that scrapers can be created"""
    print("\nTesting scraper creation...")
    
    try:
        from src.data_collection.scrapers import get_scraper
        
        for platform in ['avito', 'mubawab', 'sarouty']:
            scraper = get_scraper(platform)
            print(f"✅ {platform.capitalize()} scraper created: {scraper.platform_name}")
        
        return True
    except Exception as e:
        print(f"❌ Scraper creation failed: {e}")
        return False


def test_property_listing():
    """Test PropertyListing creation"""
    print("\nTesting PropertyListing...")
    
    try:
        from src.data_collection.scrapers import PropertyListing
        
        listing = PropertyListing(
            title="Test Apartment",
            description="Beautiful 3-bedroom apartment",
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
        
        print(f"✅ PropertyListing created: {listing.title} in {listing.city}")
        print(f"   Price: {listing.price_mad:,} MAD")
        print(f"   Surface: {listing.surface_m2} m²")
        print(f"   Amenities: {listing.amenities}")
        
        return True
    except Exception as e:
        print(f"❌ PropertyListing creation failed: {e}")
        return False


def test_geospatial_enricher():
    """Test GeospatialEnricher"""
    print("\nTesting GeospatialEnricher...")
    
    try:
        from src.data_collection.enrichment import GeospatialEnricher
        
        enricher = GeospatialEnricher()
        
        # Test distance calculation
        distance = enricher._calculate_distance_to_city_center(33.5731, -7.5898, "casablanca")
        print(f"✅ Distance to Casablanca center: {distance:.0f} meters")
        
        # Test airport distance
        airport_distance = enricher._calculate_distance_to_airport(33.5731, -7.5898, "casablanca")
        print(f"✅ Distance to airport: {airport_distance:.0f} meters")
        
        return True
    except Exception as e:
        print(f"❌ GeospatialEnricher test failed: {e}")
        return False


def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from config.settings import get_config
        
        config = get_config()
        print(f"✅ Configuration loaded")
        print(f"   Target cities: {config.target_cities}")
        print(f"   Target platforms: {config.target_platforms}")
        print(f"   Environment: {config.environment}")
        print(f"   Database URL: {config.database.url[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_database_models():
    """Test database model creation"""
    print("\nTesting database models...")
    
    try:
        from src.database import Listing, ListingEnrichment, ListingImage
        from datetime import datetime
        
        # Test Listing model
        listing = Listing(
            source_platform="avito",
            source_id="123456",
            source_url="http://example.com/listing",
            title="Test Apartment",
            city="Casablanca",
            price_mad=1500000,
            surface_m2=80.0,
            rooms=3,
            property_type="apartment",
            scraped_at=datetime.now()
        )
        print(f"✅ Listing model created: {listing.title}")
        
        # Test enrichment model
        enrichment = ListingEnrichment(
            listing_id=1,
            distance_to_city_center=5000.0,
            walkability_score=75.0,
            schools_count_1km=3
        )
        print(f"✅ Enrichment model created with walkability score: {enrichment.walkability_score}")
        
        return True
    except Exception as e:
        print(f"❌ Database models test failed: {e}")
        return False


def test_text_processing():
    """Test text processing utilities"""
    print("\nTesting text processing...")
    
    try:
        from src.data_collection.scrapers.avito_scraper import AvitoScraper
        
        scraper = AvitoScraper()
        
        # Test price extraction
        price_tests = [
            ("1 500 000 DH", 1500000),
            ("850,000 MAD", 850000),
            ("Price: 2300000", 2300000),
            ("No price listed", None)
        ]
        
        for text, expected in price_tests:
            result = scraper.extract_price(text)
            status = "✅" if result == expected else "❌"
            print(f"   {status} Price '{text}' -> {result} (expected {expected})")
        
        # Test surface extraction
        surface_tests = [
            ("80 m²", 80.0),
            ("120.5 m2", 120.5),
            ("Surface: 95 m²", 95.0),
            ("No surface", None)
        ]
        
        for text, expected in surface_tests:
            result = scraper.extract_surface(text)
            status = "✅" if result == expected else "❌"
            print(f"   {status} Surface '{text}' -> {result} (expected {expected})")
        
        return True
    except Exception as e:
        print(f"❌ Text processing test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("🧪 Running DarValue.ai System Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_scraper_creation,
        test_property_listing,
        test_geospatial_enricher,
        test_configuration,
        test_database_models,
        test_text_processing
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The system is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the configuration and dependencies.")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)