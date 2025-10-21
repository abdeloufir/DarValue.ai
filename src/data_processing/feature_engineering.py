"""
Feature engineering pipeline for real estate data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
from loguru import logger

from src.utils.monitoring import get_logger


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    feature_name: str
    feature_type: str  # 'derived', 'interaction', 'transformation', 'aggregation'
    dependencies: List[str]
    transformation_func: Optional[Callable] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class FeatureEngineeredData:
    """Result of feature engineering"""
    data: pd.DataFrame
    feature_metadata: Dict[str, Dict[str, Any]]
    original_features: List[str]
    engineered_features: List[str]
    feature_importance: Optional[Dict[str, float]] = None


class FeatureEngineer:
    """Advanced feature engineering for real estate data"""
    
    def __init__(self):
        self.logger = get_logger('feature_engineer')
        self.scalers = {}
        self.feature_metadata = {}
        
        # Feature engineering configurations
        self.feature_configs = [
            # Price-based features
            FeatureConfig('price_per_room', 'derived', ['price_mad', 'rooms']),
            FeatureConfig('price_per_bathroom', 'derived', ['price_mad', 'bathrooms']),
            FeatureConfig('surface_per_room', 'derived', ['surface_m2', 'rooms']),
            FeatureConfig('bathroom_ratio', 'derived', ['bathrooms', 'rooms']),
            
            # Location features
            FeatureConfig('location_premium_score', 'derived', ['distance_to_city_center_km']),
            FeatureConfig('coastal_proximity_score', 'derived', ['distance_to_coast_km']),
            FeatureConfig('coordinate_precision', 'derived', ['latitude', 'longitude']),
            
            # Property characteristics
            FeatureConfig('luxury_score', 'aggregation', ['pool', 'garden', 'elevator', 'parking']),
            FeatureConfig('amenity_count', 'aggregation', ['parking', 'elevator', 'pool', 'garden', 'furnished']),
            FeatureConfig('space_efficiency', 'derived', ['surface_m2', 'rooms']),
            
            # Market position features
            FeatureConfig('price_percentile_city', 'aggregation', ['price_mad', 'city_cleaned']),
            FeatureConfig('surface_percentile_city', 'aggregation', ['surface_m2', 'city_cleaned']),
            FeatureConfig('neighborhood_density', 'aggregation', ['neighborhood_cleaned', 'city_cleaned']),
            
            # Interaction features
            FeatureConfig('city_property_premium', 'interaction', ['city_cleaned', 'property_type_cleaned']),
            FeatureConfig('size_location_interaction', 'interaction', ['surface_m2', 'distance_to_city_center_km']),
            FeatureConfig('luxury_location_interaction', 'interaction', ['luxury_score', 'location_premium_score']),
        ]
    
    def engineer_features(self, df: pd.DataFrame, target_column: str = 'price_mad') -> FeatureEngineeredData:
        """Engineer comprehensive features for real estate data"""
        
        self.logger.info(f"Starting feature engineering for {len(df)} records")
        
        # Create a copy of the data
        df_engineered = df.copy()
        original_features = list(df.columns)
        
        # 1. Basic derived features
        df_engineered = self._create_basic_derived_features(df_engineered)
        
        # 2. Location-based features
        df_engineered = self._create_location_features(df_engineered)
        
        # 3. Property characteristic features
        df_engineered = self._create_property_features(df_engineered)
        
        # 4. Market position features
        df_engineered = self._create_market_features(df_engineered, target_column)
        
        # 5. Interaction features
        df_engineered = self._create_interaction_features(df_engineered)
        
        # 6. Aggregation features
        df_engineered = self._create_aggregation_features(df_engineered)
        
        # 7. Transformation features
        df_engineered = self._create_transformation_features(df_engineered)
        
        # 8. Time-based features (if available)
        df_engineered = self._create_time_features(df_engineered)
        
        # 9. Text-based features
        df_engineered = self._create_text_features(df_engineered)
        
        # 10. Statistical features
        df_engineered = self._create_statistical_features(df_engineered)
        
        # Get list of engineered features
        engineered_features = [col for col in df_engineered.columns if col not in original_features]
        
        # Calculate feature importance if target is available
        feature_importance = None
        if target_column in df_engineered.columns:
            feature_importance = self._calculate_feature_importance(df_engineered, target_column)
        
        self.logger.info(f"Feature engineering complete: {len(engineered_features)} new features created")
        
        return FeatureEngineeredData(
            data=df_engineered,
            feature_metadata=self.feature_metadata,
            original_features=original_features,
            engineered_features=engineered_features,
            feature_importance=feature_importance
        )
    
    def _create_basic_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic derived features"""
        
        # Price ratios
        if 'price_mad' in df.columns and 'rooms' in df.columns:
            df['price_per_room'] = df['price_mad'] / df['rooms'].replace(0, np.nan)
            self.feature_metadata['price_per_room'] = {
                'type': 'derived', 'dependencies': ['price_mad', 'rooms']
            }
        
        if 'price_mad' in df.columns and 'bathrooms' in df.columns:
            df['price_per_bathroom'] = df['price_mad'] / df['bathrooms'].replace(0, np.nan)
            self.feature_metadata['price_per_bathroom'] = {
                'type': 'derived', 'dependencies': ['price_mad', 'bathrooms']
            }
        
        # Surface ratios
        if 'surface_m2' in df.columns and 'rooms' in df.columns:
            df['surface_per_room'] = df['surface_m2'] / df['rooms'].replace(0, np.nan)
            df['room_density'] = df['rooms'] / df['surface_m2'].replace(0, np.nan)
            
            self.feature_metadata['surface_per_room'] = {
                'type': 'derived', 'dependencies': ['surface_m2', 'rooms']
            }
            self.feature_metadata['room_density'] = {
                'type': 'derived', 'dependencies': ['rooms', 'surface_m2']
            }
        
        # Bathroom ratios
        if 'bathrooms' in df.columns and 'rooms' in df.columns:
            df['bathroom_to_room_ratio'] = df['bathrooms'] / df['rooms'].replace(0, np.nan)
            self.feature_metadata['bathroom_to_room_ratio'] = {
                'type': 'derived', 'dependencies': ['bathrooms', 'rooms']
            }
        
        # Floor features
        if 'floor' in df.columns:
            df['is_ground_floor'] = (df['floor'] == 0).astype(int)
            df['is_top_floor'] = df['floor'] > 5  # Assume buildings > 5 floors
            df['floor_category'] = pd.cut(df['floor'], bins=[-1, 0, 2, 5, 20], 
                                        labels=['ground', 'low', 'mid', 'high'])
            
            self.feature_metadata['is_ground_floor'] = {'type': 'derived', 'dependencies': ['floor']}
            self.feature_metadata['is_top_floor'] = {'type': 'derived', 'dependencies': ['floor']}
        
        return df
    
    def _create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create location-based features"""
        
        # Distance-based scores
        if 'distance_to_city_center_km' in df.columns:
            # Location premium (closer to center = higher score)
            max_distance = df['distance_to_city_center_km'].max()
            df['location_premium_score'] = 1 - (df['distance_to_city_center_km'] / max_distance)
            
            # Location categories
            df['location_category'] = pd.cut(
                df['distance_to_city_center_km'],
                bins=[0, 2, 5, 10, 50],
                labels=['city_center', 'inner', 'suburban', 'outer']
            )
            
            self.feature_metadata['location_premium_score'] = {
                'type': 'derived', 'dependencies': ['distance_to_city_center_km']
            }
        
        if 'distance_to_coast_km' in df.columns:
            # Coastal proximity score
            df['coastal_proximity_score'] = 1 / (1 + df['distance_to_coast_km'])
            df['is_coastal'] = (df['distance_to_coast_km'] < 5).astype(int)
            
            self.feature_metadata['coastal_proximity_score'] = {
                'type': 'derived', 'dependencies': ['distance_to_coast_km']
            }
        
        # Coordinate precision (indicator of data quality)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            lat_precision = df['latitude'].astype(str).str.split('.').str[1].str.len()
            lon_precision = df['longitude'].astype(str).str.split('.').str[1].str.len()
            df['coordinate_precision'] = (lat_precision + lon_precision) / 2
            
            # Coordinate clustering features
            df['lat_rounded'] = df['latitude'].round(2)
            df['lon_rounded'] = df['longitude'].round(2)
            df['coordinate_cluster'] = df['lat_rounded'].astype(str) + '_' + df['lon_rounded'].astype(str)
            
            self.feature_metadata['coordinate_precision'] = {
                'type': 'derived', 'dependencies': ['latitude', 'longitude']
            }
        
        return df
    
    def _create_property_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create property characteristic features"""
        
        # Amenity features
        amenity_columns = ['parking', 'elevator', 'pool', 'garden', 'furnished', 
                          'air_conditioning', 'heating', 'balcony', 'terrace']
        available_amenities = [col for col in amenity_columns if col in df.columns]
        
        if available_amenities:
            # Total amenity count
            df['amenity_count'] = df[available_amenities].sum(axis=1)
            
            # Luxury score (weighted amenities)
            luxury_weights = {'pool': 3, 'garden': 2, 'elevator': 1.5, 'parking': 1, 
                            'air_conditioning': 1, 'furnished': 0.5}
            
            df['luxury_score'] = 0
            for amenity in available_amenities:
                weight = luxury_weights.get(amenity, 1)
                df['luxury_score'] += df[amenity] * weight
            
            # Amenity density (amenities per room)
            if 'rooms' in df.columns:
                df['amenity_density'] = df['amenity_count'] / df['rooms'].replace(0, np.nan)
            
            self.feature_metadata['amenity_count'] = {
                'type': 'aggregation', 'dependencies': available_amenities
            }
            self.feature_metadata['luxury_score'] = {
                'type': 'derived', 'dependencies': available_amenities
            }
        
        # Property size categories
        if 'surface_m2' in df.columns:
            df['size_category'] = pd.cut(
                df['surface_m2'],
                bins=[0, 50, 100, 150, 250, 1000],
                labels=['small', 'medium', 'large', 'very_large', 'mansion']
            )
            
            # Space efficiency score
            if 'rooms' in df.columns:
                df['space_efficiency'] = df['surface_m2'] / df['rooms'].replace(0, np.nan)
        
        # Property age features (if construction year available)
        if 'construction_year' in df.columns:
            current_year = pd.Timestamp.now().year
            df['property_age'] = current_year - df['construction_year']
            df['age_category'] = pd.cut(
                df['property_age'],
                bins=[0, 5, 15, 30, 100],
                labels=['new', 'recent', 'mature', 'old']
            )
            df['is_new_construction'] = (df['property_age'] <= 2).astype(int)
        
        return df
    
    def _create_market_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Create market-based features"""
        
        # City-level market features
        if 'city_cleaned' in df.columns and target_column in df.columns:
            city_stats = df.groupby('city_cleaned')[target_column].agg(['mean', 'median', 'std', 'count'])
            
            df['city_price_mean'] = df['city_cleaned'].map(city_stats['mean'])
            df['city_price_median'] = df['city_cleaned'].map(city_stats['median'])
            df['city_price_std'] = df['city_cleaned'].map(city_stats['std'])
            df['city_listing_count'] = df['city_cleaned'].map(city_stats['count'])
            
            # Price deviation from city mean
            df['price_deviation_from_city'] = df[target_column] - df['city_price_mean']
            df['price_zscore_in_city'] = (df[target_column] - df['city_price_mean']) / df['city_price_std']
            
            # Percentile ranking within city
            df['price_percentile_in_city'] = df.groupby('city_cleaned')[target_column].rank(pct=True)
        
        # Neighborhood-level features
        if 'neighborhood_cleaned' in df.columns:
            neighborhood_counts = df['neighborhood_cleaned'].value_counts()
            df['neighborhood_popularity'] = df['neighborhood_cleaned'].map(neighborhood_counts)
            
            # Neighborhood price statistics (if enough data)
            if target_column in df.columns:
                neighborhood_stats = df.groupby('neighborhood_cleaned')[target_column].agg(['mean', 'count'])
                
                # Only use neighborhoods with sufficient data
                valid_neighborhoods = neighborhood_stats[neighborhood_stats['count'] >= 5]
                df['neighborhood_price_mean'] = df['neighborhood_cleaned'].map(valid_neighborhoods['mean'])
        
        # Property type market features
        if 'property_type_cleaned' in df.columns and target_column in df.columns:
            property_stats = df.groupby('property_type_cleaned')[target_column].agg(['mean', 'median'])
            df['property_type_price_mean'] = df['property_type_cleaned'].map(property_stats['mean'])
            
            # Price premium over property type average
            df['price_premium_over_type'] = df[target_column] - df['property_type_price_mean']
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        
        # Size × Location interactions
        if 'surface_m2' in df.columns and 'location_premium_score' in df.columns:
            df['size_location_interaction'] = df['surface_m2'] * df['location_premium_score']
        
        # Luxury × Location interactions
        if 'luxury_score' in df.columns and 'location_premium_score' in df.columns:
            df['luxury_location_interaction'] = df['luxury_score'] * df['location_premium_score']
        
        # Price per m² × City interactions
        if 'price_per_m2' in df.columns and 'city_cleaned' in df.columns:
            city_dummies = pd.get_dummies(df['city_cleaned'], prefix='city')
            for col in city_dummies.columns:
                df[f'price_m2_{col}'] = df['price_per_m2'] * city_dummies[col]
        
        # Room count × Property type interactions
        if 'rooms' in df.columns and 'property_type_cleaned' in df.columns:
            type_dummies = pd.get_dummies(df['property_type_cleaned'], prefix='type')
            for col in type_dummies.columns:
                df[f'rooms_{col}'] = df['rooms'] * type_dummies[col]
        
        return df
    
    def _create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregation features"""
        
        # Group-based aggregations
        groupby_features = [
            ('city_cleaned', ['surface_m2', 'rooms']),
            ('property_type_cleaned', ['price_per_m2']),
            ('neighborhood_cleaned', ['luxury_score'])
        ]
        
        for group_col, agg_cols in groupby_features:
            if group_col in df.columns:
                available_agg_cols = [col for col in agg_cols if col in df.columns]
                
                for agg_col in available_agg_cols:
                    group_stats = df.groupby(group_col)[agg_col].agg(['mean', 'std', 'count'])
                    
                    df[f'{agg_col}_mean_by_{group_col}'] = df[group_col].map(group_stats['mean'])
                    df[f'{agg_col}_std_by_{group_col}'] = df[group_col].map(group_stats['std'])
                    
                    # Z-score within group
                    df[f'{agg_col}_zscore_by_{group_col}'] = (
                        (df[agg_col] - df[f'{agg_col}_mean_by_{group_col}']) / 
                        df[f'{agg_col}_std_by_{group_col}']
                    )
        
        return df
    
    def _create_transformation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create transformed features"""
        
        # Log transformations for skewed numerical features
        numerical_cols = ['price_mad', 'surface_m2', 'price_per_m2']
        
        for col in numerical_cols:
            if col in df.columns:
                # Fill NaN/None values with 0 for log transformations
                col_data = df[col].fillna(0)
                
                # Ensure positive values for log transformation
                col_data = col_data.clip(lower=0)
                
                # Log transformation (add 1 to handle zeros)
                df[f'{col}_log'] = np.log1p(col_data)
                
                # Square root transformation
                df[f'{col}_sqrt'] = np.sqrt(col_data)
                
                # Box-Cox inspired transformation
                df[f'{col}_boxcox'] = np.sign(col_data) * np.log1p(np.abs(col_data))
        
        # Binning continuous variables
        if 'price_per_m2' in df.columns:
            price_per_m2_clean = df['price_per_m2'].fillna(0)
            try:
                df['price_per_m2_bin'] = pd.qcut(price_per_m2_clean, q=10, labels=False, duplicates='drop')
            except ValueError:
                # Fallback if qcut fails
                df['price_per_m2_bin'] = pd.cut(price_per_m2_clean, bins=10, labels=False)
        
        if 'surface_m2' in df.columns:
            surface_m2_clean = df['surface_m2'].fillna(0)
            try:
                df['surface_m2_bin'] = pd.qcut(surface_m2_clean, q=10, labels=False, duplicates='drop')
            except ValueError:
                # Fallback if qcut fails
                df['surface_m2_bin'] = pd.cut(surface_m2_clean, bins=10, labels=False)
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        
        # If scraping date is available
        if 'scraped_at' in df.columns:
            df['scraped_at'] = pd.to_datetime(df['scraped_at'], errors='coerce')
            
            df['scraping_year'] = df['scraped_at'].dt.year
            df['scraping_month'] = df['scraped_at'].dt.month
            df['scraping_day_of_week'] = df['scraped_at'].dt.dayofweek
            df['scraping_quarter'] = df['scraped_at'].dt.quarter
            
            # Season features
            df['scraping_season'] = df['scraping_month'].map({
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'autumn', 10: 'autumn', 11: 'autumn'
            })
        
        # Time since listing (if listing date available)
        if 'listed_at' in df.columns and 'scraped_at' in df.columns:
            df['listed_at'] = pd.to_datetime(df['listed_at'], errors='coerce')
            df['days_on_market'] = (df['scraped_at'] - df['listed_at']).dt.days
            
            df['is_new_listing'] = (df['days_on_market'] <= 7).astype(int)
            df['is_stale_listing'] = (df['days_on_market'] >= 180).astype(int)
        
        return df
    
    def _create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create text-based features"""
        
        # Title features
        if 'title_cleaned' in df.columns:
            df['title_length'] = df['title_cleaned'].fillna('').str.len()
            df['title_word_count'] = df['title_cleaned'].fillna('').str.split().str.len()
            
            # Keyword features
            keywords = ['luxury', 'modern', 'new', 'renovated', 'spacious', 'bright']
            for keyword in keywords:
                df[f'title_has_{keyword}'] = df['title_cleaned'].fillna('').str.lower().str.contains(keyword).astype(int)
        
        # Description features
        if 'description_cleaned' in df.columns:
            df['description_length'] = df['description_cleaned'].fillna('').str.len()
            df['description_word_count'] = df['description_cleaned'].fillna('').str.split().str.len()
            
            # Sentiment indicators
            positive_words = ['excellent', 'perfect', 'beautiful', 'amazing', 'stunning']
            negative_words = ['needs', 'repair', 'old', 'small', 'dark']
            
            df['description_positive_sentiment'] = df['description_cleaned'].fillna('').str.lower().str.count('|'.join(positive_words))
            df['description_negative_sentiment'] = df['description_cleaned'].fillna('').str.lower().str.count('|'.join(negative_words))
            
            df['description_sentiment_score'] = (
                df['description_positive_sentiment'] - df['description_negative_sentiment']
            )
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Rolling statistics by city
        if 'city_cleaned' in df.columns:
            for col in numerical_cols[:5]:  # Limit to avoid too many features
                if col in ['price_mad', 'surface_m2', 'price_per_m2']:
                    city_rolling = df.groupby('city_cleaned')[col].rolling(window=10, min_periods=3)
                    
                    df[f'{col}_rolling_mean'] = city_rolling.mean().reset_index(0, drop=True)
                    df[f'{col}_rolling_std'] = city_rolling.std().reset_index(0, drop=True)
        
        # Rank features
        rank_cols = ['price_mad', 'surface_m2', 'luxury_score']
        for col in rank_cols:
            if col in df.columns:
                df[f'{col}_rank'] = df[col].rank(pct=True)
                df[f'{col}_rank_in_city'] = df.groupby('city_cleaned')[col].rank(pct=True)
        
        return df
    
    def _calculate_feature_importance(self, df: pd.DataFrame, target_column: str) -> Dict[str, float]:
        """Calculate feature importance using mutual information"""
        
        # Select numerical features for importance calculation
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and ID columns
        feature_cols = [col for col in numerical_features 
                       if col != target_column and 
                       not col.lower().endswith('_id') and
                       col != 'id']
        
        if len(feature_cols) == 0:
            return {}
        
        # Prepare data
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target_column].fillna(df[target_column].median())
        
        # Calculate mutual information
        try:
            mi_scores = mutual_info_regression(X, y, random_state=42)
            feature_importance = dict(zip(feature_cols, mi_scores))
            
            # Normalize scores
            max_score = max(feature_importance.values()) if feature_importance.values() else 1
            feature_importance = {k: v/max_score for k, v in feature_importance.items()}
            
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            self.logger.warning(f"Could not calculate feature importance: {e}")
            return {}
    
    def select_top_features(self, df: pd.DataFrame, target_column: str, 
                           k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """Select top k features using univariate selection"""
        
        # Get numerical features
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numerical_features 
                       if col != target_column and 
                       not col.lower().endswith('_id')]
        
        if len(feature_cols) <= k:
            return df, feature_cols
        
        # Prepare data
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target_column].fillna(df[target_column].median())
        
        # Select top k features
        selector = SelectKBest(score_func=f_regression, k=min(k, len(feature_cols)))
        
        try:
            selector.fit(X, y)
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
            
            # Include target and essential columns
            essential_cols = [target_column, 'city_cleaned', 'property_type_cleaned']
            all_selected = selected_features + [col for col in essential_cols if col in df.columns]
            
            self.logger.info(f"Selected {len(selected_features)} top features from {len(feature_cols)} total")
            
            return df[all_selected], selected_features
            
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}")
            return df, feature_cols
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Scale numerical features"""
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Choose scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        df_scaled = df.copy()
        
        # Scale numerical columns
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols].fillna(0))
        
        # Store scaler for future use
        self.scalers[method] = scaler
        
        self.logger.info(f"Scaled {len(numerical_cols)} numerical features using {method} scaling")
        
        return df_scaled
    
    def get_feature_summary(self, engineered_data: FeatureEngineeredData) -> Dict[str, Any]:
        """Get summary of engineered features"""
        
        summary = {
            'total_original_features': len(engineered_data.original_features),
            'total_engineered_features': len(engineered_data.engineered_features),
            'feature_types': {},
            'top_important_features': {},
            'data_shape': engineered_data.data.shape,
            'memory_usage_mb': engineered_data.data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Count features by type
        for feature, metadata in engineered_data.feature_metadata.items():
            feature_type = metadata.get('type', 'unknown')
            summary['feature_types'][feature_type] = summary['feature_types'].get(feature_type, 0) + 1
        
        # Top important features
        if engineered_data.feature_importance:
            summary['top_important_features'] = dict(
                list(engineered_data.feature_importance.items())[:10]
            )
        
        return summary