"""
Main entry point for running the DarValue.ai data collection pipeline
"""

import sys
from pathlib import Path
import argparse
from typing import List, Optional

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config.settings import get_config
from src.utils.monitoring import initialize_monitoring
from src.data_collection import DataCollectionPipeline, create_pipeline_config
from src.database import create_database, db_manager


def setup_database():
    """Initialize database tables"""
    print("Setting up database...")
    try:
        # Test connection
        if not db_manager.test_connection():
            print("❌ Database connection failed!")
            return False
        
        # Create tables
        create_database()
        print("✅ Database setup completed")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False


def run_data_collection(
    cities: Optional[List[str]] = None,
    platforms: Optional[List[str]] = None,
    max_pages: int = 5
):
    """Run the data collection pipeline"""
    
    # Load configuration
    config = get_config()
    
    # Initialize monitoring
    initialize_monitoring({
        'logging': {
            'level': config.logging.level,
            'format': config.logging.format,
            'log_file': config.logging.log_file,
            'rotation': config.logging.rotation,
            'retention': config.logging.retention
        },
        'enable_monitoring': config.monitoring.enable_monitoring,
        'prometheus_port': config.monitoring.prometheus_port
    })
    
    from src.utils import get_logger
    logger = get_logger('main')
    
    logger.info("🚀 Starting DarValue.ai data collection pipeline")
    
    # Use config defaults if not specified
    cities = cities or config.target_cities
    platforms = platforms or config.target_platforms
    
    logger.info(f"Cities: {cities}")
    logger.info(f"Platforms: {platforms}")
    logger.info(f"Max pages per city: {max_pages}")
    
    # Create pipeline configuration
    pipeline_config = create_pipeline_config(
        cities=cities,
        platforms=platforms,
        max_pages_per_city=max_pages,
        enable_geospatial_enrichment=config.geospatial.enable_enrichment,
        enable_image_download=config.images.enable_download,
        enable_cloud_storage=config.images.enable_cloud_storage,
        cloud_provider=config.images.cloud_provider,
        google_maps_api_key=config.geospatial.google_maps_api_key,
        aws_config=config.aws.__dict__ if config.images.cloud_provider == 'aws' else None,
        gcp_config=config.gcp.__dict__ if config.images.cloud_provider == 'gcp' else None,
        max_workers=config.scraping.max_workers,
        delay_between_platforms=config.scraping.delay_between_platforms
    )
    
    # Run pipeline
    try:
        pipeline = DataCollectionPipeline(pipeline_config)
        results = pipeline.run_pipeline()
        
        logger.info("📊 Pipeline Results:")
        logger.info(f"  📋 Total listings found: {results['total_listings_found']}")
        logger.info(f"  ✨ New listings: {results['total_listings_new']}")
        logger.info(f"  🔄 Updated listings: {results['total_listings_updated']}")
        logger.info(f"  ❌ Errors: {results['total_errors']}")
        
        duration = (results['end_time'] - results['start_time']).total_seconds()
        logger.info(f"  ⏱️  Duration: {duration:.1f} seconds")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='DarValue.ai Data Collection Pipeline')
    
    parser.add_argument(
        '--cities',
        nargs='+',
        help='Cities to scrape (default: all configured cities)',
        choices=['casablanca', 'rabat', 'marrakech', 'tangier', 'fes', 'agadir']
    )
    
    parser.add_argument(
        '--platforms',
        nargs='+',
        help='Platforms to scrape (default: all configured platforms)',
        choices=['mubawab']
    )
    
    parser.add_argument(
        '--max-pages',
        type=int,
        default=5,
        help='Maximum pages to scrape per city (default: 5)'
    )
    
    parser.add_argument(
        '--setup-db',
        action='store_true',
        help='Setup database tables'
    )
    
    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='Test database connection'
    )
    
    args = parser.parse_args()
    
    # Handle database setup
    if args.setup_db:
        setup_database()
        return
    
    # Handle connection test
    if args.test_connection:
        if db_manager.test_connection():
            print("✅ Database connection successful!")
        else:
            print("❌ Database connection failed!")
        return
    
    # Setup database if needed
    try:
        if not setup_database():
            print("Database setup failed. Exiting.")
            sys.exit(1)
    except Exception as e:
        print(f"Database setup error: {e}")
        sys.exit(1)
    
    # Run data collection
    try:
        run_data_collection(
            cities=args.cities,
            platforms=args.platforms,
            max_pages=args.max_pages
        )
        print("\n🎉 Data collection completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Data collection interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Data collection failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()