"""
City-specific dataset splitting module for real estate data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from loguru import logger

from ..utils.monitoring import get_logger


@dataclass
class DatasetSplit:
    """Dataset split configuration"""
    city: str
    train_data: pd.DataFrame
    val_data: pd.DataFrame
    test_data: pd.DataFrame
    feature_columns: List[str]
    target_column: str
    split_stats: Dict[str, Any]


@dataclass
class SplitConfiguration:
    """Configuration for dataset splitting"""
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    stratify_column: Optional[str] = None
    min_samples_per_city: int = 100
    balance_datasets: bool = True


class CityDatasetSplitter:
    """Splits real estate data into city-specific datasets"""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.logger = get_logger('city_dataset_splitter')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Target Moroccan cities
        self.target_cities = [
            'casablanca', 'rabat', 'marrakech', 
            'tangier', 'fes', 'agadir'
        ]
        
        # Feature groups for different model types
        self.feature_groups = {
            'basic': [
                'surface_m2', 'rooms', 'bathrooms', 'floor',
                'property_type_encoded', 'condition_encoded'
            ],
            'location': [
                'city_encoded', 'neighborhood_encoded', 'latitude', 'longitude',
                'distance_to_city_center_km', 'distance_to_coast_km'
            ],
            'amenities': [
                'parking', 'elevator', 'pool', 'garden', 'furnished',
                'air_conditioning', 'heating', 'balcony', 'terrace'
            ],
            'market': [
                'price_category_encoded', 'price_per_m2_category_encoded',
                'price_percentile_in_city', 'neighborhood_frequency_in_city'
            ],
            'derived': [
                'price_per_room', 'city_property_interaction',
                'property_frequency_in_city', 'age_category'
            ]
        }
    
    def split_datasets(self, df: pd.DataFrame, 
                      config: SplitConfiguration = SplitConfiguration(),
                      target_column: str = 'price_mad') -> Dict[str, DatasetSplit]:
        """Split data into city-specific datasets"""
        
        self.logger.info(f"Splitting {len(df)} records into city-specific datasets")
        
        # Prepare data
        df_clean = self._prepare_data_for_splitting(df, target_column)
        
        # Get feature columns
        feature_columns = self._select_feature_columns(df_clean)
        
        city_splits = {}
        
        # Create splits for each target city
        for city in self.target_cities:
            city_data = df_clean[df_clean['city_cleaned'] == city].copy()
            
            if len(city_data) < config.min_samples_per_city:
                self.logger.warning(f"Insufficient data for {city}: {len(city_data)} samples")
                continue
            
            # Create train/val/test splits
            split_result = self._create_city_split(
                city_data, city, feature_columns, target_column, config
            )
            
            if split_result:
                city_splits[city] = split_result
                self.logger.info(f"Created splits for {city}: "
                               f"train={len(split_result.train_data)}, "
                               f"val={len(split_result.val_data)}, "
                               f"test={len(split_result.test_data)}")
        
        # Create combined dataset
        combined_split = self._create_combined_split(
            df_clean, feature_columns, target_column, config
        )
        city_splits['combined'] = combined_split
        
        # Save splits to disk
        self._save_splits_to_disk(city_splits)
        
        # Generate split report
        self._generate_split_report(city_splits)
        
        self.logger.info(f"Dataset splitting complete: {len(city_splits)} datasets created")
        return city_splits
    
    def _prepare_data_for_splitting(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Prepare data for splitting"""
        
        # Create a copy
        df_prep = df.copy()
        
        # Remove records with missing target values
        df_prep = df_prep[df_prep[target_column].notna()]
        
        # Ensure city_cleaned column exists
        if 'city_cleaned' not in df_prep.columns:
            if 'city' in df_prep.columns:
                # Create city_cleaned from city column
                df_prep['city_cleaned'] = df_prep['city'].str.lower().str.strip()
                self.logger.info("Created city_cleaned column from city column")
            else:
                raise ValueError("Neither 'city_cleaned' nor 'city' column found in data")
        
        # Remove records with missing city information
        df_prep = df_prep[df_prep['city_cleaned'].notna()]
        
        # Standardize city names
        df_prep['city_cleaned'] = df_prep['city_cleaned'].str.lower().str.strip()
        
        # Remove extreme outliers in target variable
        Q1 = df_prep[target_column].quantile(0.01)
        Q99 = df_prep[target_column].quantile(0.99)
        df_prep = df_prep[
            (df_prep[target_column] >= Q1) & 
            (df_prep[target_column] <= Q99)
        ]
        
        # Add derived features if not present
        df_prep = self._add_derived_features(df_prep)
        
        self.logger.info(f"Prepared {len(df_prep)} records for splitting")
        return df_prep
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for modeling"""
        
        # Age category (if construction year available)
        if 'construction_year' in df.columns:
            current_year = pd.Timestamp.now().year
            df['property_age'] = current_year - df['construction_year']
            df['age_category'] = pd.cut(
                df['property_age'], 
                bins=[0, 5, 15, 30, 100], 
                labels=['new', 'recent', 'mature', 'old']
            )
        
        # Room density (rooms per surface area)
        if 'rooms' in df.columns and 'surface_m2' in df.columns:
            df['room_density'] = df['rooms'] / df['surface_m2']
        
        # Luxury score (combination of amenities)
        luxury_features = ['pool', 'garden', 'elevator', 'parking']
        available_luxury = [f for f in luxury_features if f in df.columns]
        if available_luxury:
            df['luxury_score'] = df[available_luxury].sum(axis=1)
        
        # Location score (proximity to city center)
        if 'distance_to_city_center_km' in df.columns:
            df['location_score'] = 1 / (1 + df['distance_to_city_center_km'])
        
        return df
    
    def _select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Select feature columns for modeling"""
        
        all_features = []
        
        # Add features from each group that exist in the data
        for group_name, features in self.feature_groups.items():
            available_features = [f for f in features if f in df.columns]
            all_features.extend(available_features)
            self.logger.info(f"{group_name} features: {len(available_features)}/{len(features)} available")
        
        # Add any additional numerical features
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        additional_features = [
            col for col in numerical_columns 
            if col not in all_features and 
            not col.startswith('price') and  # Exclude target-related columns
            col not in ['id', 'listing_id', 'scraping_id']  # Exclude ID columns
        ]
        
        all_features.extend(additional_features)
        
        # Remove duplicates and sort
        feature_columns = sorted(list(set(all_features)))
        
        self.logger.info(f"Selected {len(feature_columns)} feature columns")
        return feature_columns
    
    def _create_city_split(self, city_data: pd.DataFrame, city: str,
                          feature_columns: List[str], target_column: str,
                          config: SplitConfiguration) -> Optional[DatasetSplit]:
        """Create train/val/test split for a specific city"""
        
        try:
            # Prepare features and target
            X = city_data[feature_columns].copy()
            y = city_data[target_column].copy()
            
            # Handle missing values in features
            X = self._handle_missing_values(X)
            
            # Stratification variable (if specified)
            stratify = None
            if config.stratify_column and config.stratify_column in city_data.columns:
                stratify = city_data[config.stratify_column]
            
            # First split: train + val vs test
            X_temp, X_test, y_temp, y_test, indices_temp, indices_test = train_test_split(
                X, y, city_data.index,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=stratify
            )
            
            # Second split: train vs val
            val_size_adjusted = config.val_size / (1 - config.test_size)
            X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(
                X_temp, y_temp, indices_temp,
                test_size=val_size_adjusted,
                random_state=config.random_state + 1
            )
            
            # Create DataFrames with original indices
            train_data = city_data.loc[indices_train].copy()
            val_data = city_data.loc[indices_val].copy()
            test_data = city_data.loc[indices_test].copy()
            
            # Calculate split statistics
            split_stats = self._calculate_split_stats(
                train_data, val_data, test_data, target_column
            )
            
            return DatasetSplit(
                city=city,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                feature_columns=feature_columns,
                target_column=target_column,
                split_stats=split_stats
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create split for {city}: {e}")
            return None
    
    def _create_combined_split(self, df: pd.DataFrame, feature_columns: List[str],
                              target_column: str, config: SplitConfiguration) -> DatasetSplit:
        """Create combined dataset split across all cities"""
        
        # Prepare features and target
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Stratify by city to ensure representation
        stratify = df['city_cleaned']
        
        # Create splits
        X_temp, X_test, y_temp, y_test, indices_temp, indices_test = train_test_split(
            X, y, df.index,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=stratify
        )
        
        # Get stratification for remaining data
        stratify_temp = df.loc[indices_temp, 'city_cleaned']
        val_size_adjusted = config.val_size / (1 - config.test_size)
        
        X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(
            X_temp, y_temp, indices_temp,
            test_size=val_size_adjusted,
            random_state=config.random_state + 1,
            stratify=stratify_temp
        )
        
        # Create DataFrames
        train_data = df.loc[indices_train].copy()
        val_data = df.loc[indices_val].copy()
        test_data = df.loc[indices_test].copy()
        
        # Calculate statistics
        split_stats = self._calculate_split_stats(
            train_data, val_data, test_data, target_column
        )
        
        return DatasetSplit(
            city='combined',
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            feature_columns=feature_columns,
            target_column=target_column,
            split_stats=split_stats
        )
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in feature matrix"""
        
        # Numerical columns: fill with median
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            X[col] = X[col].fillna(X[col].median())
        
        # Categorical columns: fill with mode or 'unknown'
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            mode_value = X[col].mode()
            fill_value = mode_value[0] if len(mode_value) > 0 else 'unknown'
            X[col] = X[col].fillna(fill_value)
        
        return X
    
    def _calculate_split_stats(self, train_data: pd.DataFrame, val_data: pd.DataFrame,
                              test_data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Calculate statistics for data splits"""
        
        stats = {
            'total_samples': len(train_data) + len(val_data) + len(test_data),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'train_ratio': len(train_data) / (len(train_data) + len(val_data) + len(test_data)),
            'val_ratio': len(val_data) / (len(train_data) + len(val_data) + len(test_data)),
            'test_ratio': len(test_data) / (len(train_data) + len(val_data) + len(test_data)),
            'target_stats': {
                'train_mean': train_data[target_column].mean(),
                'val_mean': val_data[target_column].mean(),
                'test_mean': test_data[target_column].mean(),
                'train_std': train_data[target_column].std(),
                'val_std': val_data[target_column].std(),
                'test_std': test_data[target_column].std(),
            }
        }
        
        return stats
    
    def _save_splits_to_disk(self, city_splits: Dict[str, DatasetSplit]):
        """Save dataset splits to disk"""
        
        self.logger.info("Saving dataset splits to disk")
        
        for city, split in city_splits.items():
            city_dir = self.output_dir / city
            city_dir.mkdir(exist_ok=True)
            
            # Save CSV files
            split.train_data.to_csv(city_dir / 'train.csv', index=False)
            split.val_data.to_csv(city_dir / 'val.csv', index=False)
            split.test_data.to_csv(city_dir / 'test.csv', index=False)
            
            # Save feature columns
            with open(city_dir / 'feature_columns.txt', 'w') as f:
                f.write('\n'.join(split.feature_columns))
            
            # Save split metadata
            metadata = {
                'city': split.city,
                'target_column': split.target_column,
                'feature_columns': split.feature_columns,
                'split_stats': split.split_stats
            }
            
            with open(city_dir / 'metadata.pickle', 'wb') as f:
                pickle.dump(metadata, f)
            
            self.logger.info(f"Saved {city} dataset split to {city_dir}")
    
    def _generate_split_report(self, city_splits: Dict[str, DatasetSplit]):
        """Generate comprehensive split report"""
        
        report = {
            'total_cities': len(city_splits) - 1,  # Exclude combined
            'cities': list(city_splits.keys()),
            'split_summary': {},
            'feature_summary': {
                'total_features': len(city_splits[list(city_splits.keys())[0]].feature_columns),
                'feature_groups': self.feature_groups
            }
        }
        
        # Add statistics for each city
        for city, split in city_splits.items():
            report['split_summary'][city] = split.split_stats
        
        # Save report
        report_path = self.output_dir / 'split_report.pickle'
        with open(report_path, 'wb') as f:
            pickle.dump(report, f)
        
        # Save human-readable summary
        summary_path = self.output_dir / 'split_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Dataset Split Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for city, split in city_splits.items():
                f.write(f"{city.upper()}:\n")
                f.write(f"  Total samples: {split.split_stats['total_samples']}\n")
                f.write(f"  Train: {split.split_stats['train_samples']} ({split.split_stats['train_ratio']:.2%})\n")
                f.write(f"  Val: {split.split_stats['val_samples']} ({split.split_stats['val_ratio']:.2%})\n")
                f.write(f"  Test: {split.split_stats['test_samples']} ({split.split_stats['test_ratio']:.2%})\n")
                f.write(f"  Target mean: {split.split_stats['target_stats']['train_mean']:,.0f} MAD\n")
                f.write("\n")
        
        self.logger.info(f"Split report saved to {report_path}")
    
    def load_city_split(self, city: str) -> Optional[DatasetSplit]:
        """Load a specific city split from disk"""
        
        city_dir = self.output_dir / city
        
        if not city_dir.exists():
            self.logger.error(f"No split found for city: {city}")
            return None
        
        try:
            # Load data
            train_data = pd.read_csv(city_dir / 'train.csv')
            val_data = pd.read_csv(city_dir / 'val.csv')
            test_data = pd.read_csv(city_dir / 'test.csv')
            
            # Load metadata
            with open(city_dir / 'metadata.pickle', 'rb') as f:
                metadata = pickle.load(f)
            
            return DatasetSplit(
                city=metadata['city'],
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                feature_columns=metadata['feature_columns'],
                target_column=metadata['target_column'],
                split_stats=metadata['split_stats']
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load split for {city}: {e}")
            return None
    
    def get_available_cities(self) -> List[str]:
        """Get list of available city splits"""
        
        cities = []
        for city_dir in self.output_dir.iterdir():
            if city_dir.is_dir() and (city_dir / 'train.csv').exists():
                cities.append(city_dir.name)
        
        return sorted(cities)
    
    def balance_city_datasets(self, city_splits: Dict[str, DatasetSplit],
                             min_samples: int = 500, max_samples: int = 5000) -> Dict[str, DatasetSplit]:
        """Balance datasets across cities for fair comparison"""
        
        self.logger.info("Balancing city datasets")
        
        balanced_splits = {}
        
        for city, split in city_splits.items():
            if city == 'combined':
                balanced_splits[city] = split
                continue
            
            # Calculate target sample size
            current_train_size = len(split.train_data)
            
            if current_train_size > max_samples:
                # Downsample
                train_balanced = split.train_data.sample(n=max_samples, random_state=42)
                val_balanced = split.val_data.sample(
                    n=min(len(split.val_data), max_samples // 4), random_state=42
                )
                test_balanced = split.test_data.sample(
                    n=min(len(split.test_data), max_samples // 4), random_state=42
                )
            elif current_train_size < min_samples:
                # Skip cities with insufficient data
                self.logger.warning(f"Skipping {city} - insufficient data ({current_train_size} samples)")
                continue
            else:
                # Keep as is
                train_balanced = split.train_data
                val_balanced = split.val_data
                test_balanced = split.test_data
            
            # Recalculate statistics
            split_stats = self._calculate_split_stats(
                train_balanced, val_balanced, test_balanced, split.target_column
            )
            
            balanced_splits[city] = DatasetSplit(
                city=city,
                train_data=train_balanced,
                val_data=val_balanced,
                test_data=test_balanced,
                feature_columns=split.feature_columns,
                target_column=split.target_column,
                split_stats=split_stats
            )
        
        self.logger.info(f"Balanced {len(balanced_splits)} city datasets")
        return balanced_splits