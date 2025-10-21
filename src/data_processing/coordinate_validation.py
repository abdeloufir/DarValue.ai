"""
Coordinate validation and geocoding module for real estate data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import requests
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import geopandas as gpd
from shapely.geometry import Point
from loguru import logger

from ..utils.monitoring import get_logger


@dataclass
class GeocodingResult:
    """Result of geocoding operation"""
    address: str
    latitude: Optional[float]
    longitude: Optional[float]
    confidence: float
    source: str
    formatted_address: Optional[str] = None


@dataclass
class CoordinateValidationResult:
    """Result of coordinate validation"""
    is_valid: bool
    latitude: Optional[float]
    longitude: Optional[float]
    validation_method: str
    confidence_score: float
    errors: List[str]


class CoordinateValidator:
    """Validates and fixes coordinates for real estate data"""
    
    def __init__(self):
        self.logger = get_logger('coordinate_validator')
        
        # Morocco bounding box (approximate)
        self.morocco_bounds = {
            'min_lat': 21.0,    # Southern border
            'max_lat': 36.0,    # Northern border  
            'min_lon': -17.5,   # Western border
            'max_lon': -1.0     # Eastern border
        }
        
        # Major Moroccan cities with approximate coordinates
        self.city_coordinates = {
            'casablanca': {'lat': 33.5731, 'lon': -7.5898},
            'rabat': {'lat': 34.0209, 'lon': -6.8416},
            'marrakech': {'lat': 31.6295, 'lon': -7.9811},
            'marrakesh': {'lat': 31.6295, 'lon': -7.9811},  # Alternative spelling
            'tangier': {'lat': 35.7595, 'lon': -5.8340},
            'tanger': {'lat': 35.7595, 'lon': -5.8340},     # Alternative spelling
            'fes': {'lat': 34.0331, 'lon': -5.0003},
            'fez': {'lat': 34.0331, 'lon': -5.0003},        # Alternative spelling
            'agadir': {'lat': 30.4278, 'lon': -9.5981},
            'meknes': {'lat': 33.8935, 'lon': -5.5473},
            'oujda': {'lat': 34.6814, 'lon': -1.9086},
            'kenitra': {'lat': 34.2610, 'lon': -6.5802},
            'tetouan': {'lat': 35.5889, 'lon': -5.3626},
            'safi': {'lat': 32.2994, 'lon': -9.2372},
            'mohammedia': {'lat': 33.6862, 'lon': -7.3830},
            'khouribga': {'lat': 32.8811, 'lon': -6.9061},
            'beni mellal': {'lat': 32.3373, 'lon': -6.3498},
            'el jadida': {'lat': 33.2316, 'lon': -8.5007},
            'nador': {'lat': 35.1740, 'lon': -2.9287}
        }
        
        # Initialize geocoder
        self.geocoder = Nominatim(user_agent="darvalue_real_estate_geocoder")
        
        # Rate limiting for geocoding APIs
        self.rate_limit_delay = 1.0  # seconds between requests
        
    def validate_coordinates(self, df: pd.DataFrame) -> Dict[int, CoordinateValidationResult]:
        """Validate coordinates in the DataFrame"""
        self.logger.info(f"Validating coordinates for {len(df)} records")
        
        validation_results = {}
        
        for idx, row in df.iterrows():
            lat = row.get('latitude')
            lon = row.get('longitude')
            city = row.get('city', '').lower() if row.get('city') else ''
            
            result = self._validate_single_coordinate(lat, lon, city)
            validation_results[idx] = result
            
            if idx % 1000 == 0:
                self.logger.info(f"Validated {idx} coordinates")
        
        valid_count = sum(1 for r in validation_results.values() if r.is_valid)
        self.logger.info(f"Validation complete: {valid_count}/{len(df)} coordinates are valid")
        
        return validation_results
    
    def _validate_single_coordinate(self, lat: Any, lon: Any, city: str) -> CoordinateValidationResult:
        """Validate a single coordinate pair"""
        errors = []
        confidence_score = 0.0
        
        # Check if coordinates exist
        if pd.isna(lat) or pd.isna(lon):
            errors.append("Missing coordinates")
            return CoordinateValidationResult(
                is_valid=False,
                latitude=None,
                longitude=None,
                validation_method='existence_check',
                confidence_score=0.0,
                errors=errors
            )
        
        try:
            lat_float = float(lat)
            lon_float = float(lon)
        except (ValueError, TypeError):
            errors.append("Invalid coordinate format")
            return CoordinateValidationResult(
                is_valid=False,
                latitude=None,
                longitude=None,
                validation_method='format_check',
                confidence_score=0.0,
                errors=errors
            )
        
        # Check if coordinates are within Morocco bounds
        if not self._is_within_morocco_bounds(lat_float, lon_float):
            errors.append("Coordinates outside Morocco")
            confidence_score -= 0.5
        else:
            confidence_score += 0.4
        
        # Check if coordinates match expected city location
        if city and city in self.city_coordinates:
            city_coords = self.city_coordinates[city]
            distance_km = self._calculate_distance(
                lat_float, lon_float,
                city_coords['lat'], city_coords['lon']
            )
            
            # Allow up to 50km from city center (reasonable for metropolitan areas)
            if distance_km <= 50:
                confidence_score += 0.4
            elif distance_km <= 100:
                confidence_score += 0.2
                errors.append(f"Coordinates far from {city} center ({distance_km:.1f}km)")
            else:
                confidence_score -= 0.3
                errors.append(f"Coordinates very far from {city} center ({distance_km:.1f}km)")
        
        # Check for common invalid coordinates (0,0), (1,1), etc.
        if self._is_likely_invalid_coordinate(lat_float, lon_float):
            errors.append("Suspicious coordinate values")
            confidence_score -= 0.4
        
        # Final validation
        is_valid = confidence_score >= 0.3 and len([e for e in errors if 'outside Morocco' in e]) == 0
        
        return CoordinateValidationResult(
            is_valid=is_valid,
            latitude=lat_float,
            longitude=lon_float,
            validation_method='comprehensive',
            confidence_score=max(0.0, confidence_score),
            errors=errors
        )
    
    def _is_within_morocco_bounds(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within Morocco's bounding box"""
        return (
            self.morocco_bounds['min_lat'] <= lat <= self.morocco_bounds['max_lat'] and
            self.morocco_bounds['min_lon'] <= lon <= self.morocco_bounds['max_lon']
        )
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in kilometers using Haversine formula"""
        from math import radians, sin, cos, sqrt, atan2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        # Earth's radius in kilometers
        R = 6371
        distance = R * c
        
        return distance
    
    def _is_likely_invalid_coordinate(self, lat: float, lon: float) -> bool:
        """Check for obviously invalid coordinates"""
        invalid_patterns = [
            # Exact zeros or ones
            (lat == 0.0 and lon == 0.0),
            (lat == 1.0 and lon == 1.0),
            # Repeated digits
            (abs(lat - round(lat)) < 1e-6 and abs(lon - round(lon)) < 1e-6),
            # Very high precision (suspicious)
            (len(str(lat).split('.')[-1]) > 10 or len(str(lon).split('.')[-1]) > 10)
        ]
        
        return any(invalid_patterns)


class GeocodingService:
    """Geocoding service for addresses and missing coordinates"""
    
    def __init__(self):
        self.logger = get_logger('geocoding_service')
        self.geocoder = Nominatim(user_agent="darvalue_real_estate_geocoder")
        self.rate_limit_delay = 1.1  # Nominatim requires 1 request per second
        
        # Cache for geocoding results
        self.geocoding_cache = {}
        
    def geocode_missing_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Geocode missing coordinates using available address information"""
        self.logger.info(f"Starting geocoding for records with missing coordinates")
        
        df_geocoded = df.copy()
        
        # Identify records needing geocoding
        needs_geocoding = df_geocoded[
            (df_geocoded['latitude'].isna() | df_geocoded['longitude'].isna()) |
            (df_geocoded['latitude'] == 0) | (df_geocoded['longitude'] == 0)
        ]
        
        self.logger.info(f"Found {len(needs_geocoding)} records needing geocoding")
        
        successful_geocodes = 0
        
        for idx, row in needs_geocoding.iterrows():
            # Build address string
            address = self._build_address_string(row)
            
            if not address:
                continue
            
            # Check cache first
            if address in self.geocoding_cache:
                result = self.geocoding_cache[address]
            else:
                result = self._geocode_address(address)
                self.geocoding_cache[address] = result
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
            
            if result.latitude and result.longitude:
                df_geocoded.at[idx, 'latitude'] = result.latitude
                df_geocoded.at[idx, 'longitude'] = result.longitude
                df_geocoded.at[idx, 'geocoding_confidence'] = result.confidence
                df_geocoded.at[idx, 'geocoding_source'] = result.source
                successful_geocodes += 1
            
            if idx % 100 == 0:
                self.logger.info(f"Geocoded {idx} addresses, {successful_geocodes} successful")
        
        self.logger.info(f"Geocoding complete: {successful_geocodes} successful geocodes")
        return df_geocoded
    
    def _build_address_string(self, row: pd.Series) -> str:
        """Build address string from available fields"""
        address_parts = []
        
        # Add neighborhood if available
        if pd.notna(row.get('neighborhood')):
            address_parts.append(str(row['neighborhood']))
        
        # Add city (required)
        if pd.notna(row.get('city')):
            address_parts.append(str(row['city']))
        else:
            return ""  # Can't geocode without city
        
        # Add country
        address_parts.append("Morocco")
        
        return ", ".join(address_parts)
    
    def _geocode_address(self, address: str) -> GeocodingResult:
        """Geocode a single address"""
        try:
            location = self.geocoder.geocode(
                address,
                country_codes=['ma'],  # Restrict to Morocco
                timeout=10
            )
            
            if location:
                return GeocodingResult(
                    address=address,
                    latitude=location.latitude,
                    longitude=location.longitude,
                    confidence=0.8,  # Nominatim doesn't provide confidence scores
                    source='nominatim',
                    formatted_address=location.address
                )
            else:
                return GeocodingResult(
                    address=address,
                    latitude=None,
                    longitude=None,
                    confidence=0.0,
                    source='nominatim'
                )
                
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            self.logger.warning(f"Geocoding failed for {address}: {e}")
            return GeocodingResult(
                address=address,
                latitude=None,
                longitude=None,
                confidence=0.0,
                source='error'
            )
    
    def fallback_city_geocoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback: assign city center coordinates for records that couldn't be geocoded"""
        self.logger.info("Applying fallback city center coordinates")
        
        df_fallback = df.copy()
        
        # City coordinates from CoordinateValidator
        city_coordinates = {
            'casablanca': {'lat': 33.5731, 'lon': -7.5898},
            'rabat': {'lat': 34.0209, 'lon': -6.8416},
            'marrakech': {'lat': 31.6295, 'lon': -7.9811},
            'marrakesh': {'lat': 31.6295, 'lon': -7.9811},
            'tangier': {'lat': 35.7595, 'lon': -5.8340},
            'tanger': {'lat': 35.7595, 'lon': -5.8340},
            'fes': {'lat': 34.0331, 'lon': -5.0003},
            'fez': {'lat': 34.0331, 'lon': -5.0003},
            'agadir': {'lat': 30.4278, 'lon': -9.5981}
        }
        
        fallback_count = 0
        
        for idx, row in df_fallback.iterrows():
            if pd.isna(row['latitude']) or pd.isna(row['longitude']) or \
               row['latitude'] == 0 or row['longitude'] == 0:
                
                city = str(row.get('city', '')).lower().strip()
                
                if city in city_coordinates:
                    coords = city_coordinates[city]
                    # Add small random offset to avoid all properties having exact same coordinates
                    lat_offset = np.random.uniform(-0.05, 0.05)  # ~5km offset
                    lon_offset = np.random.uniform(-0.05, 0.05)
                    
                    df_fallback.at[idx, 'latitude'] = coords['lat'] + lat_offset
                    df_fallback.at[idx, 'longitude'] = coords['lon'] + lon_offset
                    df_fallback.at[idx, 'geocoding_confidence'] = 0.3  # Low confidence
                    df_fallback.at[idx, 'geocoding_source'] = 'city_center_fallback'
                    fallback_count += 1
        
        self.logger.info(f"Applied fallback coordinates to {fallback_count} records")
        return df_fallback


class CoordinateEnrichment:
    """Enriches data with additional geographic information"""
    
    def __init__(self):
        self.logger = get_logger('coordinate_enrichment')
    
    def enrich_with_reverse_geocoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich coordinates with reverse geocoded address information"""
        self.logger.info("Starting reverse geocoding enrichment")
        
        df_enriched = df.copy()
        geocoder = Nominatim(user_agent="darvalue_real_estate_reverse_geocoder")
        
        enriched_count = 0
        
        for idx, row in df_enriched.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                try:
                    location = geocoder.reverse(
                        f"{row['latitude']}, {row['longitude']}",
                        timeout=10
                    )
                    
                    if location and location.raw.get('address'):
                        address = location.raw['address']
                        
                        # Extract useful information
                        df_enriched.at[idx, 'reverse_geocoded_neighborhood'] = address.get('neighbourhood', address.get('suburb'))
                        df_enriched.at[idx, 'reverse_geocoded_city'] = address.get('city', address.get('town', address.get('village')))
                        df_enriched.at[idx, 'reverse_geocoded_postal_code'] = address.get('postcode')
                        df_enriched.at[idx, 'reverse_geocoded_full_address'] = location.address
                        
                        enriched_count += 1
                    
                    # Rate limiting
                    time.sleep(1.1)
                    
                except Exception as e:
                    self.logger.warning(f"Reverse geocoding failed for {idx}: {e}")
                    continue
            
            if idx % 100 == 0:
                self.logger.info(f"Reverse geocoded {idx} coordinates, {enriched_count} successful")
        
        self.logger.info(f"Reverse geocoding complete: {enriched_count} coordinates enriched")
        return df_enriched
    
    def calculate_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate distance-based features"""
        self.logger.info("Calculating distance-based features")
        
        df_distance = df.copy()
        
        # City centers for distance calculation
        city_centers = {
            'casablanca': (33.5731, -7.5898),
            'rabat': (34.0209, -6.8416),
            'marrakech': (31.6295, -7.9811),
            'tangier': (35.7595, -5.8340),
            'fes': (34.0331, -5.0003),
            'agadir': (30.4278, -9.5981)
        }
        
        # Calculate distance to city center
        for idx, row in df_distance.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                city = str(row.get('city', '')).lower().strip()
                
                if city in city_centers:
                    city_lat, city_lon = city_centers[city]
                    distance = self._calculate_distance(
                        row['latitude'], row['longitude'],
                        city_lat, city_lon
                    )
                    df_distance.at[idx, 'distance_to_city_center_km'] = distance
        
        # Calculate distance to coast (approximate)
        # Morocco's Atlantic coast runs roughly north-south
        for idx, row in df_distance.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                # Simplified distance to Atlantic coast
                # Using longitude -7.5 as approximate coast line
                coast_distance = abs(row['longitude'] + 7.5) * 111  # Rough conversion to km
                df_distance.at[idx, 'distance_to_coast_km'] = coast_distance
        
        self.logger.info("Distance features calculated")
        return df_distance
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in kilometers"""
        from math import radians, sin, cos, sqrt, atan2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        # Earth's radius in kilometers
        R = 6371
        distance = R * c
        
        return distance