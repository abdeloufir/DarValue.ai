"""
Data deduplication module for removing duplicate real estate listings
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from loguru import logger
from difflib import SequenceMatcher
import hashlib

from ..database import get_db_session, Listing
from ..utils.monitoring import get_logger


@dataclass
class DuplicateResult:
    """Result of duplicate detection"""
    original_id: int
    duplicate_ids: List[int]
    similarity_score: float
    match_type: str  # exact, fuzzy, coordinate, etc.


class ListingDeduplicator:
    """Handles deduplication of real estate listings"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.logger = get_logger('deduplicator')
        
        # Thresholds for different matching criteria
        self.coordinate_threshold = 50  # meters
        self.price_similarity_threshold = 0.9  # 10% difference allowed
        self.surface_similarity_threshold = 0.9  # 10% difference allowed
        
    def find_duplicates(self, data) -> List[DuplicateResult]:
        """Find all duplicate listings - accepts either SQLAlchemy session or pandas DataFrame"""
        from sqlalchemy.orm import Session
        import pandas as pd
        
        self.logger.info("Starting duplicate detection process")
        
        if isinstance(data, Session):
            # Database session - load listings from database
            listings = data.query(Listing).filter(
                Listing.is_active == True
            ).all()
            
            self.logger.info(f"Analyzing {len(listings)} listings for duplicates")
            df = self._listings_to_dataframe(listings)
            
        elif isinstance(data, pd.DataFrame):
            # DataFrame directly passed - validate required columns
            required_columns = ['id', 'source_platform', 'source_id', 'title', 'description', 
                              'city', 'neighborhood', 'price_mad', 'surface_m2', 'rooms', 
                              'bathrooms', 'property_type', 'latitude', 'longitude', 'amenities']
            
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                # Add missing columns with default values
                for col in missing_columns:
                    if col in ['amenities']:
                        data[col] = data[col] if col in data.columns else [[] for _ in range(len(data))]
                    elif col in ['scraped_at']:
                        data[col] = pd.Timestamp.now()
                    else:
                        data[col] = None
            
            df = data.copy()
            self.logger.info(f"Analyzing {len(df)} listings for duplicates")
            
        else:
            raise ValueError("Data must be either SQLAlchemy Session or pandas DataFrame")
        
        # Find duplicates using multiple strategies
        duplicates = []
        processed_ids = set()
        
        for i, listing in df.iterrows():
            if listing['id'] in processed_ids:
                continue
                
            # Find duplicates for this listing
            listing_duplicates = self._find_listing_duplicates(listing, df, processed_ids)
            
            if listing_duplicates:
                duplicates.append(listing_duplicates)
                processed_ids.add(listing['id'])
                processed_ids.update(listing_duplicates.duplicate_ids)
        
        self.logger.info(f"Found {len(duplicates)} duplicate groups")
        return duplicates
    
    def _listings_to_dataframe(self, listings: List[Listing]) -> pd.DataFrame:
        """Convert listing objects to pandas DataFrame"""
        data = []
        
        for listing in listings:
            data.append({
                'id': listing.id,
                'source_platform': listing.source_platform,
                'source_id': listing.source_id,
                'title': listing.title or '',
                'description': listing.description or '',
                'city': listing.city or '',
                'neighborhood': listing.neighborhood or '',
                'price_mad': listing.price_mad,
                'surface_m2': listing.surface_m2,
                'rooms': listing.rooms,
                'bathrooms': listing.bathrooms,
                'property_type': listing.property_type or '',
                'latitude': float(listing.latitude) if listing.latitude else None,
                'longitude': float(listing.longitude) if listing.longitude else None,
                'amenities': listing.amenities or [],
                'scraped_at': listing.scraped_at
            })
        
        return pd.DataFrame(data)
    
    def _find_listing_duplicates(self, listing: pd.Series, df: pd.DataFrame, 
                               processed_ids: Set[int]) -> Optional[DuplicateResult]:
        """Find duplicates for a specific listing"""
        
        duplicates = []
        best_similarity = 0.0
        match_type = 'none'
        
        # Filter candidates (same city, similar price range)
        candidates = df[
            (df['id'] != listing['id']) &
            (~df['id'].isin(processed_ids)) &
            (df['city'] == listing['city'])
        ].copy()
        
        if candidates.empty:
            return None
        
        # 1. Exact coordinate match
        if listing['latitude'] is not None and listing['longitude'] is not None:
            coord_matches = self._find_coordinate_duplicates(listing, candidates)
            if coord_matches:
                duplicates.extend(coord_matches)
                match_type = 'coordinate'
                best_similarity = 1.0
        
        # 2. Title similarity match
        if not duplicates:
            title_matches = self._find_title_duplicates(listing, candidates)
            if title_matches:
                similarity = self._calculate_title_similarity(listing['title'], 
                                                            candidates.loc[title_matches[0], 'title'])
                if similarity > best_similarity:
                    duplicates = title_matches
                    match_type = 'title'
                    best_similarity = similarity
        
        # 3. Property feature match (price + surface + location)
        if not duplicates:
            feature_matches = self._find_feature_duplicates(listing, candidates)
            if feature_matches:
                similarity = self._calculate_feature_similarity(listing, candidates.loc[feature_matches[0]])
                if similarity > best_similarity:
                    duplicates = feature_matches
                    match_type = 'features'
                    best_similarity = similarity
        
        # 4. Source ID match across platforms
        source_matches = self._find_source_id_duplicates(listing, candidates)
        if source_matches:
            duplicates.extend(source_matches)
            match_type = 'source_id'
            best_similarity = max(best_similarity, 0.95)
        
        if duplicates and best_similarity >= self.similarity_threshold:
            return DuplicateResult(
                original_id=listing['id'],
                duplicate_ids=list(set(duplicates)),  # Remove duplicates from duplicates list
                similarity_score=best_similarity,
                match_type=match_type
            )
        
        return None
    
    def _find_coordinate_duplicates(self, listing: pd.Series, candidates: pd.DataFrame) -> List[int]:
        """Find duplicates based on coordinates"""
        import pandas as pd
        import numpy as np
        
        # Validate listing coordinates
        if (pd.isna(listing['latitude']) or pd.isna(listing['longitude']) or 
            listing['latitude'] is None or listing['longitude'] is None or
            not np.isfinite(listing['latitude']) or not np.isfinite(listing['longitude'])):
            return []
        
        matches = []
        
        for idx, candidate in candidates.iterrows():
            # Validate candidate coordinates
            if (pd.isna(candidate['latitude']) or pd.isna(candidate['longitude']) or 
                candidate['latitude'] is None or candidate['longitude'] is None or
                not np.isfinite(candidate['latitude']) or not np.isfinite(candidate['longitude'])):
                continue
            
            try:
                # Calculate distance between coordinates
                distance = self._calculate_distance(
                    listing['latitude'], listing['longitude'],
                    candidate['latitude'], candidate['longitude']
                )
                
                if distance <= self.coordinate_threshold:
                    # Additional checks for coordinate matches
                    if (self._price_similarity(listing['price_mad'], candidate['price_mad']) > 0.8 or
                        self._surface_similarity(listing['surface_m2'], candidate['surface_m2']) > 0.8):
                        matches.append(candidate['id'])
                        
            except Exception as e:
                # Log and skip invalid coordinate pairs
                self.logger.warning(f"Error calculating distance between coordinates: {e}")
                continue
        
        return matches
    
    def _find_title_duplicates(self, listing: pd.Series, candidates: pd.DataFrame) -> List[int]:
        """Find duplicates based on title similarity"""
        if not listing['title'] or len(listing['title'].strip()) < 10:
            return []
        
        matches = []
        listing_title_clean = self._clean_title(listing['title'])
        
        for idx, candidate in candidates.iterrows():
            if not candidate['title']:
                continue
            
            candidate_title_clean = self._clean_title(candidate['title'])
            similarity = self._calculate_title_similarity(listing_title_clean, candidate_title_clean)
            
            if similarity >= self.similarity_threshold:
                # Additional verification for title matches
                if (self._location_similarity(listing, candidate) > 0.7 and
                    (self._price_similarity(listing['price_mad'], candidate['price_mad']) > 0.7 or
                     self._surface_similarity(listing['surface_m2'], candidate['surface_m2']) > 0.7)):
                    matches.append(candidate['id'])
        
        return matches
    
    def _find_feature_duplicates(self, listing: pd.Series, candidates: pd.DataFrame) -> List[int]:
        """Find duplicates based on property features"""
        matches = []
        
        for idx, candidate in candidates.iterrows():
            # Calculate overall feature similarity
            similarity = self._calculate_feature_similarity(listing, candidate)
            
            if similarity >= self.similarity_threshold:
                matches.append(candidate['id'])
        
        return matches
    
    def _find_source_id_duplicates(self, listing: pd.Series, candidates: pd.DataFrame) -> List[int]:
        """Find duplicates with same source ID from different platforms"""
        if not listing['source_id']:
            return []
        
        matches = []
        
        # Look for same source_id from different platforms
        for idx, candidate in candidates.iterrows():
            if (candidate['source_id'] == listing['source_id'] and 
                candidate['source_platform'] != listing['source_platform']):
                matches.append(candidate['id'])
        
        return matches
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in meters"""
        from geopy.distance import geodesic
        return geodesic((lat1, lon1), (lat2, lon2)).meters
    
    def _clean_title(self, title: str) -> str:
        """Clean title text for comparison"""
        if not title:
            return ""
        
        # Convert to lowercase
        title = title.lower()
        
        # Remove common prefixes/suffixes
        patterns_to_remove = [
            r'^(vente|à vendre|for sale|sale)\s*',
            r'\s*(urgent|opportunity|occasion).*$',
            r'\s*-\s*(avito|mubawab|sarouty).*$',
            r'\s*\([^)]*\)\s*',  # Remove parentheses content
        ]
        
        for pattern in patterns_to_remove:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # Normalize spaces and special characters
        title = re.sub(r'[^\w\s]', ' ', title)
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles"""
        if not title1 or not title2:
            return 0.0
        
        # Use sequence matcher for basic similarity
        seq_similarity = SequenceMatcher(None, title1, title2).ratio()
        
        # Use TF-IDF for semantic similarity
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=None)
            tfidf_matrix = vectorizer.fit_transform([title1, title2])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            cosine_sim = 0.0
        
        # Combine both similarities
        return max(seq_similarity, cosine_sim)
    
    def _calculate_feature_similarity(self, listing1: pd.Series, listing2: pd.Series) -> float:
        """Calculate overall feature similarity between two listings"""
        similarities = []
        weights = []
        
        # Price similarity
        price_sim = self._price_similarity(listing1['price_mad'], listing2['price_mad'])
        similarities.append(price_sim)
        weights.append(0.3)
        
        # Surface similarity
        surface_sim = self._surface_similarity(listing1['surface_m2'], listing2['surface_m2'])
        similarities.append(surface_sim)
        weights.append(0.25)
        
        # Location similarity
        location_sim = self._location_similarity(listing1, listing2)
        similarities.append(location_sim)
        weights.append(0.25)
        
        # Rooms similarity
        rooms_sim = self._categorical_similarity(listing1['rooms'], listing2['rooms'])
        similarities.append(rooms_sim)
        weights.append(0.1)
        
        # Property type similarity
        type_sim = self._categorical_similarity(listing1['property_type'], listing2['property_type'])
        similarities.append(type_sim)
        weights.append(0.1)
        
        # Weighted average
        total_weight = sum(weights)
        weighted_similarity = sum(s * w for s, w in zip(similarities, weights)) / total_weight
        
        return weighted_similarity
    
    def _price_similarity(self, price1: Optional[float], price2: Optional[float]) -> float:
        """Calculate price similarity"""
        if price1 is None or price2 is None or price1 <= 0 or price2 <= 0:
            return 0.0
        
        ratio = min(price1, price2) / max(price1, price2)
        return ratio
    
    def _surface_similarity(self, surface1: Optional[float], surface2: Optional[float]) -> float:
        """Calculate surface area similarity"""
        if surface1 is None or surface2 is None or surface1 <= 0 or surface2 <= 0:
            return 0.0
        
        ratio = min(surface1, surface2) / max(surface1, surface2)
        return ratio
    
    def _location_similarity(self, listing1: pd.Series, listing2: pd.Series) -> float:
        """Calculate location similarity"""
        similarities = []
        
        # City similarity (exact match required)
        if listing1['city'] == listing2['city']:
            similarities.append(1.0)
        else:
            return 0.0  # Different cities = no similarity
        
        # Neighborhood similarity
        if listing1['neighborhood'] and listing2['neighborhood']:
            if listing1['neighborhood'] == listing2['neighborhood']:
                similarities.append(1.0)
            else:
                # Calculate text similarity for neighborhood names
                neighborhood_sim = self._calculate_title_similarity(
                    listing1['neighborhood'], listing2['neighborhood']
                )
                similarities.append(neighborhood_sim)
        else:
            similarities.append(0.5)  # Partial credit for missing neighborhood
        
        # Coordinate similarity
        if (listing1['latitude'] is not None and listing1['longitude'] is not None and
            listing2['latitude'] is not None and listing2['longitude'] is not None):
            
            import pandas as pd
            import numpy as np
            
            # Validate coordinates are finite
            if (not pd.isna(listing1['latitude']) and not pd.isna(listing1['longitude']) and
                not pd.isna(listing2['latitude']) and not pd.isna(listing2['longitude']) and
                np.isfinite(listing1['latitude']) and np.isfinite(listing1['longitude']) and
                np.isfinite(listing2['latitude']) and np.isfinite(listing2['longitude'])):
                
                try:
                    distance = self._calculate_distance(
                        listing1['latitude'], listing1['longitude'],
                        listing2['latitude'], listing2['longitude']
                    )
                    
                    # Convert distance to similarity (0-1 scale)
                    if distance <= 100:  # Very close
                        coord_sim = 1.0
                    elif distance <= 500:  # Close
                        coord_sim = 0.8
                    elif distance <= 1000:  # Same area
                        coord_sim = 0.6
                    else:
                        coord_sim = 0.0
                    
                    similarities.append(coord_sim)
                except Exception as e:
                    self.logger.warning(f"Error calculating coordinate similarity: {e}")
                    # Skip coordinate similarity if calculation fails
        
        return np.mean(similarities) if similarities else 0.0
    
    def _categorical_similarity(self, val1, val2) -> float:
        """Calculate similarity for categorical values"""
        if val1 is None and val2 is None:
            return 1.0
        elif val1 is None or val2 is None:
            return 0.5
        elif val1 == val2:
            return 1.0
        else:
            return 0.0
    
    def remove_duplicates(self, data, duplicates: List[DuplicateResult], 
                         strategy: str = 'keep_newest'):
        """Remove duplicate listings - accepts either SQLAlchemy session or pandas DataFrame"""
        from sqlalchemy.orm import Session
        import pandas as pd
        
        if isinstance(data, Session):
            # Database session - remove from database
            return self._remove_duplicates_from_db(data, duplicates, strategy)
        elif isinstance(data, pd.DataFrame):
            # DataFrame - return filtered DataFrame
            return self._remove_duplicates_from_df(data, duplicates, strategy)
        else:
            raise ValueError("Data must be either SQLAlchemy Session or pandas DataFrame")
    
    def _remove_duplicates_from_db(self, session: Session, duplicates: List[DuplicateResult], 
                                  strategy: str = 'keep_newest') -> int:
        """Remove duplicate listings from database"""
        removed_count = 0
        
        for duplicate_group in duplicates:
            try:
                # Get all listings in the duplicate group
                all_ids = [duplicate_group.original_id] + duplicate_group.duplicate_ids
                listings = session.query(Listing).filter(Listing.id.in_(all_ids)).all()
                
                if not listings:
                    continue
                
                # Determine which listing to keep
                keep_listing = self._select_best_listing_db(listings, strategy)
                
                # Mark others as inactive
                for listing in listings:
                    if listing.id != keep_listing.id:
                        listing.is_active = False
                        removed_count += 1
                        self.logger.debug(f"Marked listing {listing.id} as duplicate of {keep_listing.id}")
                
                session.commit()
                
            except Exception as e:
                self.logger.error(f"Error removing duplicates for group {duplicate_group.original_id}: {e}")
                session.rollback()
        
        self.logger.info(f"Removed {removed_count} duplicate listings")
        return removed_count
    
    def _remove_duplicates_from_df(self, df: pd.DataFrame, duplicates: List[DuplicateResult], 
                                  strategy: str = 'keep_newest') -> pd.DataFrame:
        """Remove duplicate listings from DataFrame"""
        import pandas as pd
        
        ids_to_remove = set()
        
        for duplicate_group in duplicates:
            try:
                # Get all IDs in the duplicate group
                all_ids = [duplicate_group.original_id] + duplicate_group.duplicate_ids
                group_df = df[df['id'].isin(all_ids)].copy()
                
                if group_df.empty:
                    continue
                
                # Determine which listing to keep
                keep_id = self._select_best_listing_df(group_df, strategy)
                
                # Mark others for removal
                for listing_id in all_ids:
                    if listing_id != keep_id:
                        ids_to_remove.add(listing_id)
                        self.logger.debug(f"Marking listing {listing_id} as duplicate of {keep_id}")
                        
            except Exception as e:
                self.logger.error(f"Error processing duplicates for group {duplicate_group.original_id}: {e}")
        
        # Remove duplicates from DataFrame
        result_df = df[~df['id'].isin(ids_to_remove)].copy()
        removed_count = len(df) - len(result_df)
        
        self.logger.info(f"Removed {removed_count} duplicate listings from DataFrame")
        return result_df
    
    def _select_best_listing_db(self, listings: List[Listing], strategy: str) -> Listing:
        """Select the best listing to keep from a group of duplicates (database version)"""
        
        if strategy == 'keep_newest':
            return max(listings, key=lambda x: x.scraped_at)
        
        elif strategy == 'keep_most_complete':
            def completeness_score(listing):
                score = 0
                if listing.price_mad: score += 1
                if listing.surface_m2: score += 1
                if listing.latitude and listing.longitude: score += 1
                if listing.rooms: score += 1
                if listing.description: score += 1
                if listing.image_urls: score += len(listing.image_urls)
                if listing.amenities: score += len(listing.amenities)
                return score
            
            return max(listings, key=completeness_score)
        
        elif strategy == 'keep_best_platform':
            # Preference order: sarouty > mubawab > avito
            platform_priority = {'sarouty': 3, 'mubawab': 2, 'avito': 1}
            return max(listings, key=lambda x: platform_priority.get(x.source_platform, 0))
        
        else:
            # Default: keep first one
            return listings[0]
    
    def _select_best_listing_df(self, df: pd.DataFrame, strategy: str) -> int:
        """Select the best listing ID to keep from a group of duplicates (DataFrame version)"""
        import pandas as pd
        
        if strategy == 'keep_newest':
            if 'scraped_at' in df.columns:
                return df.loc[df['scraped_at'].idxmax(), 'id']
            else:
                return df.iloc[0]['id']
        
        elif strategy == 'keep_most_complete':
            def completeness_score(row):
                score = 0
                if pd.notna(row.get('price_mad')): score += 1
                if pd.notna(row.get('surface_m2')): score += 1
                if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')): score += 1
                if pd.notna(row.get('rooms')): score += 1
                if pd.notna(row.get('description')): score += 1
                if row.get('amenities'): score += len(row.get('amenities', []))
                return score
            
            df['completeness'] = df.apply(completeness_score, axis=1)
            return df.loc[df['completeness'].idxmax(), 'id']
        
        elif strategy == 'keep_best_platform':
            # Preference order: sarouty > mubawab > avito
            platform_priority = {'sarouty': 3, 'mubawab': 2, 'avito': 1}
            df['platform_score'] = df['source_platform'].map(lambda x: platform_priority.get(x, 0))
            return df.loc[df['platform_score'].idxmax(), 'id']
        
        else:
            # Default: keep first one
            return df.iloc[0]['id']
    
    def _select_best_listing(self, listings: List[Listing], strategy: str) -> Listing:
        """Select the best listing to keep from a group of duplicates"""
        
        if strategy == 'keep_newest':
            return max(listings, key=lambda x: x.scraped_at)
        
        elif strategy == 'keep_most_complete':
            def completeness_score(listing):
                score = 0
                if listing.price_mad: score += 1
                if listing.surface_m2: score += 1
                if listing.latitude and listing.longitude: score += 1
                if listing.rooms: score += 1
                if listing.description: score += 1
                if listing.image_urls: score += len(listing.image_urls)
                if listing.amenities: score += len(listing.amenities)
                return score
            
            return max(listings, key=completeness_score)
        
        elif strategy == 'keep_best_platform':
            # Preference order: sarouty > mubawab > avito
            platform_priority = {'sarouty': 3, 'mubawab': 2, 'avito': 1}
            return max(listings, key=lambda x: platform_priority.get(x.source_platform, 0))
        
        else:
            # Default: keep first one
            return listings[0]
    
    def generate_deduplication_report(self, duplicates: List[DuplicateResult]) -> Dict:
        """Generate a report about found duplicates"""
        if not duplicates:
            return {'total_groups': 0, 'total_duplicates': 0, 'match_types': {}}
        
        total_duplicates = sum(len(dup.duplicate_ids) for dup in duplicates)
        match_type_counts = {}
        
        for dup in duplicates:
            match_type = dup.match_type
            match_type_counts[match_type] = match_type_counts.get(match_type, 0) + 1
        
        avg_similarity = np.mean([dup.similarity_score for dup in duplicates])
        
        return {
            'total_groups': len(duplicates),
            'total_duplicates': total_duplicates,
            'average_similarity': avg_similarity,
            'match_types': match_type_counts,
            'highest_similarity': max(dup.similarity_score for dup in duplicates),
            'lowest_similarity': min(dup.similarity_score for dup in duplicates)
        }