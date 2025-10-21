"""
Configuration management for DarValue.ai
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from decouple import config
import yaml
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str
    echo: bool = False
    pool_size: int = 20
    max_overflow: int = 30
    pool_recycle: int = 3600


@dataclass
class ScrapingConfig:
    """Web scraping configuration"""
    max_pages_per_city: int = 10
    max_workers: int = 4
    delay_between_platforms: int = 5
    enable_headless: bool = True
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    request_timeout: int = 30
    retry_attempts: int = 3


@dataclass
class GeospatialConfig:
    """Geospatial enrichment configuration"""
    enable_enrichment: bool = True
    google_maps_api_key: Optional[str] = None
    osm_cache_duration: int = 3600  # seconds
    max_poi_radius: int = 1000  # meters


@dataclass
class ImageConfig:
    """Image processing configuration"""
    enable_download: bool = True
    enable_cloud_storage: bool = False
    cloud_provider: str = "aws"  # aws or gcp
    local_storage_path: str = "data/images"
    max_image_size: int = 5 * 1024 * 1024  # 5MB
    quality_threshold: float = 0.3
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['JPEG', 'PNG', 'WEBP']


@dataclass
class AWSConfig:
    """AWS configuration"""
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    region: str = "us-east-1"
    s3_bucket: str = "darvalue-images"


@dataclass
class GCPConfig:
    """Google Cloud Platform configuration"""
    project_id: Optional[str] = None
    bucket_name: str = "darvalue-images"
    credentials_path: Optional[str] = None


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
    rotation: str = "1 week"
    retention: str = "30 days"
    log_file: str = "logs/darvalue.log"


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    enable_monitoring: bool = False
    prometheus_port: int = 8000
    metrics_prefix: str = "darvalue"


@dataclass
class AppConfig:
    """Main application configuration"""
    # Component configurations (required)
    database: DatabaseConfig
    scraping: ScrapingConfig
    geospatial: GeospatialConfig
    images: ImageConfig
    aws: AWSConfig
    gcp: GCPConfig
    logging: LoggingConfig
    monitoring: MonitoringConfig
    
    # Optional configurations with defaults
    debug: bool = False
    environment: str = "development"
    app_name: str = "DarValue.ai"
    version: str = "1.0.0"
    
    # Cities to scrape
    target_cities: List[str] = None
    # Platforms to use
    target_platforms: List[str] = None
    
    def __post_init__(self):
        if self.target_cities is None:
            self.target_cities = ['casablanca', 'rabat', 'marrakech', 'tangier', 'fes', 'agadir']
        if self.target_platforms is None:
            self.target_platforms = ['avito', 'mubawab', 'sarouty']


class ConfigManager:
    """Configuration manager that loads settings from multiple sources"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._config: Optional[AppConfig] = None
    
    def load_config(self) -> AppConfig:
        """Load configuration from environment variables and config files"""
        if self._config is not None:
            return self._config
        
        # Load from environment variables (with .env support)
        self._config = AppConfig(
            debug=config('DEBUG', default=False, cast=bool),
            environment=config('ENVIRONMENT', default='development'),
            
            database=DatabaseConfig(
                url=config('DATABASE_URL', default='postgresql://postgres:password@localhost:5432/darvalue_db'),
                echo=config('DATABASE_ECHO', default=False, cast=bool),
                pool_size=config('DATABASE_POOL_SIZE', default=20, cast=int),
                max_overflow=config('DATABASE_MAX_OVERFLOW', default=30, cast=int),
                pool_recycle=config('DATABASE_POOL_RECYCLE', default=3600, cast=int)
            ),
            
            scraping=ScrapingConfig(
                max_pages_per_city=config('SCRAPING_MAX_PAGES_PER_CITY', default=10, cast=int),
                max_workers=config('SCRAPING_MAX_WORKERS', default=4, cast=int),
                delay_between_platforms=config('SCRAPING_DELAY_BETWEEN_PLATFORMS', default=5, cast=int),
                enable_headless=config('SCRAPING_ENABLE_HEADLESS', default=True, cast=bool),
                request_timeout=config('SCRAPING_REQUEST_TIMEOUT', default=30, cast=int),
                retry_attempts=config('SCRAPING_RETRY_ATTEMPTS', default=3, cast=int)
            ),
            
            geospatial=GeospatialConfig(
                enable_enrichment=config('ENABLE_GEOSPATIAL_ENRICHMENT', default=True, cast=bool),
                google_maps_api_key=config('GOOGLE_MAPS_API_KEY', default=None),
                osm_cache_duration=config('OSM_CACHE_DURATION', default=3600, cast=int),
                max_poi_radius=config('MAX_POI_RADIUS', default=1000, cast=int)
            ),
            
            images=ImageConfig(
                enable_download=config('ENABLE_IMAGE_DOWNLOAD', default=True, cast=bool),
                enable_cloud_storage=config('ENABLE_CLOUD_STORAGE', default=False, cast=bool),
                cloud_provider=config('CLOUD_PROVIDER', default='aws'),
                local_storage_path=config('LOCAL_STORAGE_PATH', default='data/images'),
                max_image_size=config('MAX_IMAGE_SIZE', default=5*1024*1024, cast=int),
                quality_threshold=config('IMAGE_QUALITY_THRESHOLD', default=0.3, cast=float)
            ),
            
            aws=AWSConfig(
                access_key_id=config('AWS_ACCESS_KEY_ID', default=None),
                secret_access_key=config('AWS_SECRET_ACCESS_KEY', default=None),
                region=config('AWS_REGION', default='us-east-1'),
                s3_bucket=config('AWS_S3_BUCKET', default='darvalue-images')
            ),
            
            gcp=GCPConfig(
                project_id=config('GCP_PROJECT_ID', default=None),
                bucket_name=config('GCP_BUCKET_NAME', default='darvalue-images'),
                credentials_path=config('GOOGLE_APPLICATION_CREDENTIALS', default=None)
            ),
            
            logging=LoggingConfig(
                level=config('LOG_LEVEL', default='INFO'),
                format=config('LOG_FORMAT', default='{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}'),
                rotation=config('LOG_ROTATION', default='1 week'),
                retention=config('LOG_RETENTION', default='30 days'),
                log_file=config('LOG_FILE', default='logs/darvalue.log')
            ),
            
            monitoring=MonitoringConfig(
                enable_monitoring=config('ENABLE_MONITORING', default=False, cast=bool),
                prometheus_port=config('PROMETHEUS_PORT', default=8000, cast=int),
                metrics_prefix=config('METRICS_PREFIX', default='darvalue')
            )
        )
        
        # Load additional configuration from YAML if exists
        yaml_config_path = self.config_dir / "config.yaml"
        if yaml_config_path.exists():
            self._load_yaml_config(yaml_config_path)
        
        return self._config
    
    def _load_yaml_config(self, config_path: Path):
        """Load additional configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            
            # Override with YAML values
            if 'cities' in yaml_config:
                self._config.target_cities = yaml_config['cities']
            
            if 'platforms' in yaml_config:
                self._config.target_platforms = yaml_config['platforms']
            
            # Add any custom scraper configurations
            if 'scrapers' in yaml_config:
                # Store custom scraper configs for later use
                self._custom_scraper_configs = yaml_config['scrapers']
                
        except Exception as e:
            print(f"Warning: Could not load YAML config from {config_path}: {e}")
    
    def get_scraper_config(self, platform: str) -> Dict[str, Any]:
        """Get platform-specific scraper configuration"""
        default_config = {
            'max_pages': self._config.scraping.max_pages_per_city,
            'delay': 1.0,
            'timeout': self._config.scraping.request_timeout,
            'retries': self._config.scraping.retry_attempts
        }
        
        # Return custom config if available
        if hasattr(self, '_custom_scraper_configs'):
            return self._custom_scraper_configs.get(platform, default_config)
        
        return default_config
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        return self._config.database.url
    
    def get_aws_config(self) -> Dict[str, Any]:
        """Get AWS configuration for cloud storage"""
        return {
            'aws_access_key_id': self._config.aws.access_key_id,
            'aws_secret_access_key': self._config.aws.secret_access_key,
            'region': self._config.aws.region,
            'bucket_name': self._config.aws.s3_bucket
        }
    
    def get_gcp_config(self) -> Dict[str, Any]:
        """Get GCP configuration for cloud storage"""
        config_dict = {
            'project_id': self._config.gcp.project_id,
            'bucket_name': self._config.gcp.bucket_name
        }
        
        # Add credentials if available
        if self._config.gcp.credentials_path:
            # Set environment variable for GCP client
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self._config.gcp.credentials_path
        
        return config_dict
    
    def save_config_template(self, output_path: str = "config/config.yaml"):
        """Save a configuration template file"""
        template = {
            'cities': ['casablanca', 'rabat', 'marrakech', 'tangier', 'fes', 'agadir'],
            'platforms': ['avito', 'mubawab', 'sarouty'],
            'scrapers': {
                'avito': {
                    'max_pages': 10,
                    'delay': 1.5,
                    'timeout': 30
                },
                'mubawab': {
                    'max_pages': 10,
                    'delay': 1.0,
                    'timeout': 25
                },
                'sarouty': {
                    'max_pages': 8,
                    'delay': 2.0,
                    'timeout': 35,
                    'use_selenium': True
                }
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(template, f, default_flow_style=False, allow_unicode=True)


# Global config manager instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get the application configuration"""
    return config_manager.load_config()


def get_database_url() -> str:
    """Get database URL"""
    return config_manager.get_database_url()


def get_aws_config() -> Dict[str, Any]:
    """Get AWS configuration"""
    return config_manager.get_aws_config()


def get_gcp_config() -> Dict[str, Any]:
    """Get GCP configuration"""
    return config_manager.get_gcp_config()