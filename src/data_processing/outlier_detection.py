"""
Outlier detection and data cleaning module for real estate data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from ..utils.monitoring import get_logger


@dataclass
class OutlierDetectionResult:
    """Result of outlier detection"""
    feature: str
    outlier_indices: List[int]
    outlier_values: List[float]
    detection_method: str
    threshold_used: float


@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    total_records: int
    missing_values: Dict[str, int]
    outliers_detected: Dict[str, int]
    data_types: Dict[str, str]
    unique_values: Dict[str, int]
    quality_score: float


class OutlierDetector:
    """Detects and handles outliers in real estate data"""
    
    def __init__(self):
        self.logger = get_logger('outlier_detector')
        
        # Domain-specific thresholds for Moroccan real estate
        self.price_thresholds = {
            'casablanca': {'min': 200000, 'max': 50000000},  # 200K to 50M MAD
            'rabat': {'min': 150000, 'max': 30000000},
            'marrakech': {'min': 300000, 'max': 80000000},  # Higher due to luxury market
            'tangier': {'min': 150000, 'max': 25000000},
            'fes': {'min': 100000, 'max': 15000000},
            'agadir': {'min': 200000, 'max': 35000000}
        }
        
        self.surface_thresholds = {
            'min': 15,    # 15 m² minimum (studio)
            'max': 2000   # 2000 m² maximum (very large house/villa)
        }
        
        self.rooms_thresholds = {
            'min': 1,     # At least 1 room
            'max': 15     # Maximum 15 rooms
        }
        
        self.price_per_m2_thresholds = {
            'casablanca': {'min': 8000, 'max': 80000},   # 8K to 80K MAD/m²
            'rabat': {'min': 6000, 'max': 60000},
            'marrakech': {'min': 10000, 'max': 100000},  # Higher luxury market
            'tangier': {'min': 5000, 'max': 50000},
            'fes': {'min': 4000, 'max': 40000},
            'agadir': {'min': 7000, 'max': 70000}
        }
    
    def detect_outliers(self, df: pd.DataFrame) -> List[OutlierDetectionResult]:
        """Detect outliers using multiple methods"""
        self.logger.info(f"Starting outlier detection on {len(df)} records")
        
        outlier_results = []
        
        # 1. Domain-specific outliers
        domain_outliers = self._detect_domain_outliers(df)
        outlier_results.extend(domain_outliers)
        
        # 2. Statistical outliers (IQR method)
        statistical_outliers = self._detect_statistical_outliers(df)
        outlier_results.extend(statistical_outliers)
        
        # 3. Isolation Forest for multivariate outliers
        isolation_outliers = self._detect_isolation_forest_outliers(df)
        outlier_results.extend(isolation_outliers)
        
        # 4. Z-score outliers
        zscore_outliers = self._detect_zscore_outliers(df)
        outlier_results.extend(zscore_outliers)
        
        self.logger.info(f"Detected outliers in {len(outlier_results)} features")
        return outlier_results
    
    def _detect_domain_outliers(self, df: pd.DataFrame) -> List[OutlierDetectionResult]:
        """Detect outliers based on domain knowledge"""
        outliers = []
        
        # Price outliers by city
        if 'price_mad' in df.columns and 'city' in df.columns:
            price_outliers = []
            
            for city in df['city'].unique():
                if pd.isna(city):
                    continue
                    
                city_data = df[df['city'] == city]
                thresholds = self.price_thresholds.get(city.lower(), 
                                                     self.price_thresholds['casablanca'])
                
                outlier_mask = (
                    (city_data['price_mad'] < thresholds['min']) |
                    (city_data['price_mad'] > thresholds['max'])
                )
                
                city_outliers = city_data[outlier_mask].index.tolist()
                price_outliers.extend(city_outliers)
            
            if price_outliers:
                outliers.append(OutlierDetectionResult(
                    feature='price_mad',
                    outlier_indices=price_outliers,
                    outlier_values=df.loc[price_outliers, 'price_mad'].tolist(),
                    detection_method='domain_knowledge',
                    threshold_used=0.0
                ))
        
        # Surface area outliers
        if 'surface_m2' in df.columns:
            surface_outliers = df[
                (df['surface_m2'] < self.surface_thresholds['min']) |
                (df['surface_m2'] > self.surface_thresholds['max'])
            ].index.tolist()
            
            if surface_outliers:
                outliers.append(OutlierDetectionResult(
                    feature='surface_m2',
                    outlier_indices=surface_outliers,
                    outlier_values=df.loc[surface_outliers, 'surface_m2'].tolist(),
                    detection_method='domain_knowledge',
                    threshold_used=0.0
                ))
        
        # Rooms outliers
        if 'rooms' in df.columns:
            rooms_outliers = df[
                (df['rooms'] < self.rooms_thresholds['min']) |
                (df['rooms'] > self.rooms_thresholds['max'])
            ].index.tolist()
            
            if rooms_outliers:
                outliers.append(OutlierDetectionResult(
                    feature='rooms',
                    outlier_indices=rooms_outliers,
                    outlier_values=df.loc[rooms_outliers, 'rooms'].tolist(),
                    detection_method='domain_knowledge',
                    threshold_used=0.0
                ))
        
        # Price per m² outliers
        if 'price_per_m2' in df.columns and 'city' in df.columns:
            price_per_m2_outliers = []
            
            for city in df['city'].unique():
                if pd.isna(city):
                    continue
                    
                city_data = df[df['city'] == city]
                thresholds = self.price_per_m2_thresholds.get(city.lower(),
                                                            self.price_per_m2_thresholds['casablanca'])
                
                outlier_mask = (
                    (city_data['price_per_m2'] < thresholds['min']) |
                    (city_data['price_per_m2'] > thresholds['max'])
                )
                
                city_outliers = city_data[outlier_mask].index.tolist()
                price_per_m2_outliers.extend(city_outliers)
            
            if price_per_m2_outliers:
                outliers.append(OutlierDetectionResult(
                    feature='price_per_m2',
                    outlier_indices=price_per_m2_outliers,
                    outlier_values=df.loc[price_per_m2_outliers, 'price_per_m2'].tolist(),
                    detection_method='domain_knowledge',
                    threshold_used=0.0
                ))
        
        return outliers
    
    def _detect_statistical_outliers(self, df: pd.DataFrame) -> List[OutlierDetectionResult]:
        """Detect outliers using IQR method"""
        outliers = []
        numerical_columns = ['price_mad', 'surface_m2', 'rooms', 'bathrooms', 'price_per_m2']
        
        for column in numerical_columns:
            if column not in df.columns:
                continue
                
            series = df[column].dropna()
            if len(series) < 10:  # Need sufficient data
                continue
            
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            outlier_indices = df[outlier_mask].index.tolist()
            
            if outlier_indices:
                outliers.append(OutlierDetectionResult(
                    feature=column,
                    outlier_indices=outlier_indices,
                    outlier_values=df.loc[outlier_indices, column].tolist(),
                    detection_method='iqr',
                    threshold_used=1.5
                ))
        
        return outliers
    
    def _detect_isolation_forest_outliers(self, df: pd.DataFrame) -> List[OutlierDetectionResult]:
        """Detect multivariate outliers using Isolation Forest"""
        outliers = []
        
        # Select numerical features for multivariate analysis
        numerical_features = ['price_mad', 'surface_m2', 'rooms', 'bathrooms']
        available_features = [col for col in numerical_features if col in df.columns]
        
        if len(available_features) < 2:
            return outliers
        
        # Prepare data
        feature_data = df[available_features].dropna()
        if len(feature_data) < 50:  # Need sufficient data for Isolation Forest
            return outliers
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)
        
        # Apply Isolation Forest
        isolation_forest = IsolationForest(
            contamination=0.1,  # Expect 10% outliers
            random_state=42,
            n_estimators=100
        )
        
        outlier_labels = isolation_forest.fit_predict(scaled_data)
        outlier_indices = feature_data[outlier_labels == -1].index.tolist()
        
        if outlier_indices:
            outliers.append(OutlierDetectionResult(
                feature='multivariate',
                outlier_indices=outlier_indices,
                outlier_values=[],  # Multivariate outlier, no single value
                detection_method='isolation_forest',
                threshold_used=0.1
            ))
        
        return outliers
    
    def _detect_zscore_outliers(self, df: pd.DataFrame) -> List[OutlierDetectionResult]:
        """Detect outliers using Z-score method"""
        outliers = []
        numerical_columns = ['price_mad', 'surface_m2', 'rooms', 'bathrooms', 'price_per_m2']
        z_threshold = 3.0  # Standard threshold
        
        for column in numerical_columns:
            if column not in df.columns:
                continue
                
            series = df[column].dropna()
            if len(series) < 10:
                continue
            
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(series))
            outlier_mask = z_scores > z_threshold
            
            # Map back to original DataFrame indices
            outlier_indices = series[outlier_mask].index.tolist()
            
            if outlier_indices:
                outliers.append(OutlierDetectionResult(
                    feature=column,
                    outlier_indices=outlier_indices,
                    outlier_values=df.loc[outlier_indices, column].tolist(),
                    detection_method='zscore',
                    threshold_used=z_threshold
                ))
        
        return outliers
    
    def remove_outliers(self, df: pd.DataFrame, outlier_results: List[OutlierDetectionResult],
                       strategy: str = 'remove') -> pd.DataFrame:
        """Remove or handle outliers based on strategy"""
        self.logger.info(f"Handling outliers with strategy: {strategy}")
        
        if strategy == 'remove':
            return self._remove_outliers(df, outlier_results)
        elif strategy == 'cap':
            return self._cap_outliers(df, outlier_results)
        elif strategy == 'transform':
            return self._transform_outliers(df, outlier_results)
        else:
            self.logger.warning(f"Unknown strategy: {strategy}, returning original data")
            return df
    
    def _remove_outliers(self, df: pd.DataFrame, outlier_results: List[OutlierDetectionResult]) -> pd.DataFrame:
        """Remove outlier records from DataFrame"""
        all_outlier_indices = set()
        
        for result in outlier_results:
            # Only remove based on domain knowledge and severe statistical outliers
            if result.detection_method in ['domain_knowledge', 'iqr']:
                all_outlier_indices.update(result.outlier_indices)
        
        # Remove outliers
        clean_df = df.drop(index=list(all_outlier_indices))
        
        removed_count = len(df) - len(clean_df)
        self.logger.info(f"Removed {removed_count} outlier records ({removed_count/len(df)*100:.1f}%)")
        
        return clean_df
    
    def _cap_outliers(self, df: pd.DataFrame, outlier_results: List[OutlierDetectionResult]) -> pd.DataFrame:
        """Cap outliers at reasonable thresholds"""
        df_capped = df.copy()
        
        for result in outlier_results:
            if result.feature == 'multivariate':
                continue
                
            feature = result.feature
            series = df_capped[feature]
            
            # Calculate reasonable bounds (5th and 95th percentiles)
            lower_bound = series.quantile(0.05)
            upper_bound = series.quantile(0.95)
            
            # Cap outliers
            df_capped[feature] = np.clip(series, lower_bound, upper_bound)
        
        self.logger.info("Capped outliers to 5th-95th percentile range")
        return df_capped
    
    def _transform_outliers(self, df: pd.DataFrame, outlier_results: List[OutlierDetectionResult]) -> pd.DataFrame:
        """Transform outliers using log transformation"""
        df_transformed = df.copy()
        
        # Apply log transformation to price-related features
        price_features = ['price_mad', 'price_per_m2']
        
        for feature in price_features:
            if feature in df_transformed.columns:
                # Add small constant to handle zeros
                df_transformed[f'{feature}_log'] = np.log1p(df_transformed[feature])
        
        self.logger.info("Applied log transformation to price features")
        return df_transformed


class DataQualityAssessment:
    """Assesses overall data quality of real estate dataset"""
    
    def __init__(self):
        self.logger = get_logger('data_quality')
    
    def assess_quality(self, df: pd.DataFrame) -> DataQualityReport:
        """Generate comprehensive data quality report"""
        self.logger.info(f"Assessing data quality for {len(df)} records")
        
        # Missing values analysis
        missing_values = {}
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                missing_values[column] = missing_count
        
        # Data types
        data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Unique values
        unique_values = {}
        categorical_columns = ['city', 'neighborhood', 'property_type', 'source_platform']
        for column in categorical_columns:
            if column in df.columns:
                unique_values[column] = df[column].nunique()
        
        # Detect outliers for quality scoring
        outlier_detector = OutlierDetector()
        outlier_results = outlier_detector.detect_outliers(df)
        
        outliers_detected = {}
        for result in outlier_results:
            outliers_detected[result.feature] = len(result.outlier_indices)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(df, missing_values, outliers_detected)
        
        return DataQualityReport(
            total_records=len(df),
            missing_values=missing_values,
            outliers_detected=outliers_detected,
            data_types=data_types,
            unique_values=unique_values,
            quality_score=quality_score
        )
    
    def _calculate_quality_score(self, df: pd.DataFrame, missing_values: Dict[str, int], 
                                outliers_detected: Dict[str, int]) -> float:
        """Calculate overall data quality score (0-1)"""
        scores = []
        
        # Completeness score (based on missing values)
        important_fields = ['price_mad', 'surface_m2', 'city', 'title']
        completeness_scores = []
        
        for field in important_fields:
            if field in df.columns:
                missing_pct = missing_values.get(field, 0) / len(df)
                completeness_scores.append(1 - missing_pct)
        
        if completeness_scores:
            scores.append(np.mean(completeness_scores))
        
        # Validity score (based on outliers)
        total_outliers = sum(outliers_detected.values())
        outlier_rate = total_outliers / len(df) if len(df) > 0 else 0
        validity_score = max(0, 1 - outlier_rate)
        scores.append(validity_score)
        
        # Consistency score (basic check for price/surface consistency)
        if 'price_mad' in df.columns and 'surface_m2' in df.columns:
            valid_price_surface = df[
                (df['price_mad'] > 0) & 
                (df['surface_m2'] > 0) & 
                (df['price_mad'] / df['surface_m2'] > 1000) &  # At least 1000 MAD/m²
                (df['price_mad'] / df['surface_m2'] < 200000)  # At most 200K MAD/m²
            ]
            consistency_score = len(valid_price_surface) / len(df[df['price_mad'].notna() & df['surface_m2'].notna()])
            scores.append(consistency_score)
        
        return np.mean(scores) if scores else 0.0
    
    def generate_quality_visualizations(self, df: pd.DataFrame, save_path: str = None):
        """Generate data quality visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Real Estate Data Quality Assessment', fontsize=16)
        
        # 1. Missing values heatmap
        missing_data = df.isnull().sum()
        axes[0, 0].bar(range(len(missing_data)), missing_data.values)
        axes[0, 0].set_title('Missing Values by Column')
        axes[0, 0].set_xticks(range(len(missing_data)))
        axes[0, 0].set_xticklabels(missing_data.index, rotation=45)
        
        # 2. Price distribution
        if 'price_mad' in df.columns:
            df['price_mad'].dropna().hist(bins=50, ax=axes[0, 1], alpha=0.7)
            axes[0, 1].set_title('Price Distribution')
            axes[0, 1].set_xlabel('Price (MAD)')
        
        # 3. Surface area distribution
        if 'surface_m2' in df.columns:
            df['surface_m2'].dropna().hist(bins=50, ax=axes[0, 2], alpha=0.7)
            axes[0, 2].set_title('Surface Area Distribution')
            axes[0, 2].set_xlabel('Surface (m²)')
        
        # 4. City distribution
        if 'city' in df.columns:
            city_counts = df['city'].value_counts()
            city_counts.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Properties by City')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Property type distribution
        if 'property_type' in df.columns:
            type_counts = df['property_type'].value_counts()
            type_counts.plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%')
            axes[1, 1].set_title('Property Types')
        
        # 6. Price vs Surface scatter
        if 'price_mad' in df.columns and 'surface_m2' in df.columns:
            valid_data = df[(df['price_mad'] > 0) & (df['surface_m2'] > 0)]
            axes[1, 2].scatter(valid_data['surface_m2'], valid_data['price_mad'], alpha=0.5)
            axes[1, 2].set_title('Price vs Surface Area')
            axes[1, 2].set_xlabel('Surface (m²)')
            axes[1, 2].set_ylabel('Price (MAD)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Quality visualizations saved to {save_path}")
        
        return fig