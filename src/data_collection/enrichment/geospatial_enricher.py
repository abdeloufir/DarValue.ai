"""
Geospatial enrichment module for adding location-based features to listings
"""

import requests
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import osmnx as ox
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from loguru import logger
import time
import re


@dataclass
class LocationEnrichment:
    """Data class for enriched location information"""
    distance_to_city_center: Optional[float] = None
    distance_to_nearest_school: Optional[float] = None
    distance_to_nearest_hospital: Optional[float] = None
    distance_to_nearest_metro: Optional[float] = None
    distance_to_beach: Optional[float] = None
    distance_to_airport: Optional[float] = None
    neighborhood_avg_price_m2: Optional[float] = None
    walkability_score: Optional[float] = None
    schools_count_1km: Optional[int] = None
    restaurants_count_1km: Optional[int] = None
    shops_count_1km: Optional[int] = None
    parks_count_1km: Optional[int] = None
    street_quality_score: Optional[float] = None
    neighborhood_income_level: Optional[str] = None
    osm_data: Optional[Dict] = None


class GeospatialEnricher:
    """Main class for enriching property listings with geospatial data"""
    
    def __init__(self, google_maps_api_key: Optional[str] = None):
        self.google_maps_api_key = google_maps_api_key
        self.geocoder = Nominatim(user_agent="darvalue_ai")
        
        # Moroccan city centers (approximate coordinates)
        self.city_centers = {
            'casablanca': (33.5731, -7.5898),
            'rabat': (34.0209, -6.8416),
            'marrakech': (31.6295, -7.9811),
            'tangier': (35.7595, -5.8340),
            'fes': (34.0181, -5.0078),
            'agadir': (30.4278, -9.5981)
        }
        
        # Major airports
        self.airports = {
            'casablanca': (33.3676, -7.5897),  # Mohammed V International
            'rabat': (34.0515, -6.7515),      # Rabat-Salé
            'marrakech': (31.6069, -8.0363),  # Marrakech Menara
            'tangier': (35.7269, -5.9169),   # Ibn Battouta
            'fes': (33.9273, -4.9676),       # Fès–Saïs
            'agadir': (30.3814, -9.5463)     # Agadir–Al Massira
        }
    
    def enrich_listing(self, latitude: float, longitude: float, city: str) -> LocationEnrichment:
        """Enrich a single listing with geospatial data"""
        logger.debug(f"Enriching location: {latitude}, {longitude} in {city}")
        
        enrichment = LocationEnrichment()
        
        try:
            # Basic distance calculations
            enrichment.distance_to_city_center = self._calculate_distance_to_city_center(
                latitude, longitude, city
            )
            enrichment.distance_to_airport = self._calculate_distance_to_airport(
                latitude, longitude, city
            )
            
            # OSM-based enrichment
            osm_data = self._get_osm_data(latitude, longitude)
            enrichment.osm_data = osm_data
            
            if osm_data:
                enrichment.distance_to_nearest_school = osm_data.get('nearest_school_distance')
                enrichment.distance_to_nearest_hospital = osm_data.get('nearest_hospital_distance')
                enrichment.distance_to_beach = osm_data.get('nearest_beach_distance')
                enrichment.schools_count_1km = osm_data.get('schools_count_1km', 0)
                enrichment.restaurants_count_1km = osm_data.get('restaurants_count_1km', 0)
                enrichment.shops_count_1km = osm_data.get('shops_count_1km', 0)
                enrichment.parks_count_1km = osm_data.get('parks_count_1km', 0)
                enrichment.walkability_score = osm_data.get('walkability_score')
            
            # Google Maps enrichment (if API key available)
            if self.google_maps_api_key:
                gmaps_data = self._get_google_maps_data(latitude, longitude)
                if gmaps_data:
                    # Update with more accurate Google data
                    enrichment.distance_to_nearest_metro = gmaps_data.get('nearest_metro_distance')
            
            # Neighborhood analysis
            neighborhood_data = self._analyze_neighborhood(latitude, longitude, city)
            enrichment.neighborhood_income_level = neighborhood_data.get('income_level')
            enrichment.street_quality_score = neighborhood_data.get('street_quality')
            
        except Exception as e:
            logger.error(f"Error enriching location {latitude}, {longitude}: {e}")
        
        return enrichment
    
    def _calculate_distance_to_city_center(self, lat: float, lng: float, city: str) -> Optional[float]:
        """Calculate distance to city center in meters"""
        city_center = self.city_centers.get(city.lower())
        if not city_center:
            return None
        
        try:
            distance = geodesic((lat, lng), city_center).meters
            return distance
        except Exception as e:
            logger.error(f"Error calculating distance to city center: {e}")
            return None
    
    def _calculate_distance_to_airport(self, lat: float, lng: float, city: str) -> Optional[float]:
        """Calculate distance to nearest airport in meters"""
        airport = self.airports.get(city.lower())
        if not airport:
            return None
        
        try:
            distance = geodesic((lat, lng), airport).meters
            return distance
        except Exception as e:
            logger.error(f"Error calculating distance to airport: {e}")
            return None
    
    def _get_osm_data(self, lat: float, lng: float, radius: int = 1000) -> Optional[Dict]:
        """Get OpenStreetMap data around the location"""
        try:
            # Create a point and buffer for analysis
            point = Point(lng, lat)
            
            # Get OSM data for the area
            # Note: This is a simplified version. In production, you'd want to cache this data
            bbox = self._create_bbox(lat, lng, radius)
            
            osm_data = {}
            
            # Get different types of POIs
            poi_types = {
                'schools': ['school', 'university', 'kindergarten'],
                'hospitals': ['hospital', 'clinic', 'doctors'],
                'restaurants': ['restaurant', 'cafe', 'fast_food'],
                'shops': ['shop', 'mall', 'supermarket'],
                'parks': ['park', 'garden', 'playground']
            }
            
            for category, tags in poi_types.items():
                try:
                    count = 0
                    nearest_distance = None
                    
                    for tag in tags:
                        try:
                            # Use OSMnx to get POIs
                            pois = ox.geometries_from_bbox(
                                bbox['north'], bbox['south'], bbox['east'], bbox['west'],
                                tags={tag: True}
                            )
                            
                            if not pois.empty:
                                # Count POIs within 1km
                                pois_gdf = gpd.GeoDataFrame(pois)
                                pois_gdf = pois_gdf.to_crs('EPSG:4326')
                                
                                # Calculate distances
                                distances = []
                                for _, poi in pois_gdf.iterrows():
                                    if hasattr(poi.geometry, 'centroid'):
                                        poi_point = poi.geometry.centroid
                                    else:
                                        poi_point = poi.geometry
                                    
                                    distance = geodesic(
                                        (lat, lng), 
                                        (poi_point.y, poi_point.x)
                                    ).meters
                                    
                                    if distance <= radius:
                                        count += 1
                                    
                                    distances.append(distance)
                                
                                if distances and (nearest_distance is None or min(distances) < nearest_distance):
                                    nearest_distance = min(distances)
                        
                        except Exception as tag_error:
                            logger.debug(f"Error getting {tag} data: {tag_error}")
                            continue
                    
                    osm_data[f'{category}_count_1km'] = count
                    if nearest_distance is not None:
                        osm_data[f'nearest_{category[:-1]}_distance'] = nearest_distance
                
                except Exception as category_error:
                    logger.debug(f"Error processing {category}: {category_error}")
                    continue
            
            # Calculate walkability score based on POI density
            total_pois = sum([
                osm_data.get('schools_count_1km', 0),
                osm_data.get('restaurants_count_1km', 0),
                osm_data.get('shops_count_1km', 0),
                osm_data.get('parks_count_1km', 0)
            ])
            
            # Simple walkability score (0-100)
            osm_data['walkability_score'] = min(100, total_pois * 2)
            
            # Add coastal proximity
            try:
                # Simple check for beach proximity using OSM
                water_bodies = ox.geometries_from_bbox(
                    bbox['north'], bbox['south'], bbox['east'], bbox['west'],
                    tags={'natural': 'coastline'}
                )
                
                if not water_bodies.empty:
                    # Calculate distance to nearest coastline
                    min_distance = float('inf')
                    for _, water in water_bodies.iterrows():
                        distance = geodesic(
                            (lat, lng),
                            (water.geometry.centroid.y, water.geometry.centroid.x)
                        ).meters
                        min_distance = min(min_distance, distance)
                    
                    osm_data['nearest_beach_distance'] = min_distance
            
            except Exception as e:
                logger.debug(f"Error calculating beach distance: {e}")
            
            return osm_data
            
        except Exception as e:
            logger.error(f"Error getting OSM data: {e}")
            return None
    
    def _create_bbox(self, lat: float, lng: float, radius_meters: int) -> Dict:
        """Create bounding box around a point"""
        # Approximate conversion (1 degree ≈ 111km at equator)
        lat_offset = radius_meters / 111000
        lng_offset = radius_meters / (111000 * abs(lat))
        
        return {
            'north': lat + lat_offset,
            'south': lat - lat_offset,
            'east': lng + lng_offset,
            'west': lng - lng_offset
        }
    
    def _get_google_maps_data(self, lat: float, lng: float) -> Optional[Dict]:
        """Get additional data from Google Maps API"""
        if not self.google_maps_api_key:
            return None
        
        try:
            # Example: Get nearby transit stations
            places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            
            params = {
                'location': f"{lat},{lng}",
                'radius': 1000,
                'type': 'transit_station',
                'key': self.google_maps_api_key
            }
            
            response = requests.get(places_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('results'):
                    # Find nearest transit station
                    min_distance = float('inf')
                    for result in data['results']:
                        place_lat = result['geometry']['location']['lat']
                        place_lng = result['geometry']['location']['lng']
                        
                        distance = geodesic((lat, lng), (place_lat, place_lng)).meters
                        min_distance = min(min_distance, distance)
                    
                    return {'nearest_metro_distance': min_distance}
            
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Error getting Google Maps data: {e}")
        
        return None
    
    def _analyze_neighborhood(self, lat: float, lng: float, city: str) -> Dict:
        """Analyze neighborhood characteristics"""
        neighborhood_data = {}
        
        try:
            # Simple neighborhood income estimation based on location
            # In a real implementation, this would use census data or other sources
            
            # Distance from city center as proxy for neighborhood quality
            center_distance = self._calculate_distance_to_city_center(lat, lng, city)
            
            if center_distance is not None:
                # Simple heuristic: closer to center = higher income (not always true)
                if center_distance < 5000:  # Within 5km
                    neighborhood_data['income_level'] = 'high'
                    neighborhood_data['street_quality'] = 8.5
                elif center_distance < 15000:  # 5-15km
                    neighborhood_data['income_level'] = 'medium'
                    neighborhood_data['street_quality'] = 7.0
                else:  # >15km
                    neighborhood_data['income_level'] = 'low'
                    neighborhood_data['street_quality'] = 5.5
        
        except Exception as e:
            logger.error(f"Error analyzing neighborhood: {e}")
        
        return neighborhood_data
    
    def geocode_address(self, address: str, city: str) -> Optional[Tuple[float, float]]:
        """Geocode an address to get coordinates"""
        try:
            full_address = f"{address}, {city}, Morocco"
            location = self.geocoder.geocode(full_address, timeout=10)
            
            if location:
                return location.latitude, location.longitude
            
        except Exception as e:
            logger.error(f"Error geocoding address '{address}': {e}")
        
        return None
    
    def reverse_geocode(self, lat: float, lng: float) -> Optional[Dict]:
        """Reverse geocode coordinates to get address information"""
        try:
            location = self.geocoder.reverse((lat, lng), timeout=10)
            
            if location:
                address = location.raw.get('address', {})
                return {
                    'neighborhood': address.get('suburb') or address.get('neighbourhood'),
                    'city': address.get('city') or address.get('town'),
                    'postcode': address.get('postcode'),
                    'formatted_address': location.address
                }
        
        except Exception as e:
            logger.error(f"Error reverse geocoding {lat}, {lng}: {e}")
        
        return None


class MoroccanDataEnricher:
    """Enricher for Moroccan government and regional data"""
    
    def __init__(self):
        self.data_gov_base_url = "https://data.gov.ma"
        self.cached_census_data = {}
    
    def get_census_data(self, city: str) -> Optional[Dict]:
        """Get census data for a city from Moroccan government sources"""
        try:
            # This is a placeholder - in practice, you'd integrate with actual APIs
            # or download and process shapefiles/CSV files from data.gov.ma
            
            census_data = {
                'casablanca': {
                    'population': 3359818,
                    'density_per_km2': 9030,
                    'unemployment_rate': 0.12,
                    'literacy_rate': 0.87
                },
                'rabat': {
                    'population': 1777186,
                    'density_per_km2': 4950,
                    'unemployment_rate': 0.10,
                    'literacy_rate': 0.91
                },
                'marrakech': {
                    'population': 928850,
                    'density_per_km2': 4365,
                    'unemployment_rate': 0.08,
                    'literacy_rate': 0.68
                }
            }
            
            return census_data.get(city.lower())
            
        except Exception as e:
            logger.error(f"Error getting census data for {city}: {e}")
            return None
    
    def get_infrastructure_data(self, lat: float, lng: float) -> Optional[Dict]:
        """Get infrastructure quality data"""
        # Placeholder for infrastructure data
        # Would integrate with actual Moroccan infrastructure databases
        return {
            'road_quality_index': 7.2,
            'water_access': True,
            'electricity_reliability': 0.95,
            'internet_coverage': 0.78
        }