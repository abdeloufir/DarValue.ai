"""
Categorical encoding module for real estate data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json
from loguru import logger

from src.utils.monitoring import get_logger


@dataclass
class EncodingConfig:
    """Configuration for categorical encoding"""
    column_name: str
    encoding_type: str  # 'label', 'onehot', 'ordinal', 'target', 'frequency'
    handle_unknown: str = 'ignore'  # 'error', 'ignore', 'infrequent_if_exist'
    min_frequency: Optional[int] = None
    max_categories: Optional[int] = None


@dataclass
class EncodingResult:
    """Result of categorical encoding"""
    original_column: str
    encoded_columns: List[str]
    encoding_type: str
    encoder: Any
    mapping: Optional[Dict[str, Any]] = None
    feature_names: Optional[List[str]] = None


class CategoricalEncoder:
    """Handles categorical encoding for real estate data"""
    
    def __init__(self):
        self.logger = get_logger('categorical_encoder')
        self.encoders = {}
        self.encoding_results = {}
        
        # Predefined ordinal encodings for real estate data
        self.ordinal_mappings = {
            'condition': ['poor', 'average', 'good', 'excellent', 'new'],
            'price_category': ['budget', 'mid_range', 'premium', 'luxury'],
            'price_per_m2_category': ['below_market', 'market_rate', 'above_market', 'premium_market'],
            'floor_level': ['basement', 'ground_floor', 'floor_1', 'floor_2', 'floor_3', 
                           'floor_4', 'floor_5', 'floor_6+', 'top_floor']
        }
        
        # Default encoding strategies for different column types
        self.default_encoding_strategies = {
            # High cardinality - use label encoding or frequency encoding
            'neighborhood': 'frequency',
            'address': 'label',
            
            # Low-medium cardinality - use one-hot encoding
            'city': 'onehot',
            'property_type': 'onehot',
            'source_platform': 'onehot',
            
            # Ordinal features - use ordinal encoding
            'condition': 'ordinal',
            'price_category': 'ordinal',
            'price_per_m2_category': 'ordinal',
            
            # Boolean features - already encoded
            'parking': 'boolean',
            'elevator': 'boolean',
            'pool': 'boolean',
            'garden': 'boolean',
            'furnished': 'boolean'
        }
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  custom_configs: Optional[List[EncodingConfig]] = None) -> pd.DataFrame:
        """Encode categorical features in the DataFrame"""
        self.logger.info(f"Encoding categorical features for {len(df)} records")
        
        df_encoded = df.copy()
        
        # Identify categorical columns
        categorical_columns = self._identify_categorical_columns(df_encoded)
        self.logger.info(f"Found {len(categorical_columns)} categorical columns")
        
        # Process each categorical column
        for column in categorical_columns:
            if column not in df_encoded.columns:
                continue
                
            # Get encoding configuration
            config = self._get_encoding_config(column, df_encoded[column], custom_configs)
            
            if config.encoding_type == 'skip':
                continue
                
            # Apply encoding
            encoded_data = self._apply_encoding(df_encoded[column], config)
            
            # Store results
            self.encoding_results[column] = encoded_data
            
            # Add encoded columns to DataFrame
            if config.encoding_type == 'onehot':
                # Add one-hot encoded columns
                for i, col_name in enumerate(encoded_data.encoded_columns):
                    df_encoded[col_name] = encoded_data.encoder.transform(df_encoded[[column]]).toarray()[:, i]
            else:
                # Add single encoded column
                encoded_col_name = f"{column}_encoded"
                df_encoded[encoded_col_name] = encoded_data.encoder.transform(df_encoded[[column]])
                encoded_data.encoded_columns = [encoded_col_name]
            
            self.logger.info(f"Encoded {column} using {config.encoding_type} method")
        
        # Handle text features separately
        df_encoded = self._encode_text_features(df_encoded)
        
        # Create feature engineering from encoded categories
        df_encoded = self._create_categorical_features(df_encoded)
        
        self.logger.info("Categorical encoding complete")
        return df_encoded
    
    def _identify_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify categorical columns in the DataFrame"""
        categorical_columns = []
        
        for column in df.columns:
            # Skip already encoded columns and numerical columns
            if column.endswith('_encoded') or column.endswith('_onehot'):
                continue
                
            column_data = df[column]
            
            # Skip list columns - they need special handling
            if self._is_list_column(column_data):
                continue
                
            # Check if column is categorical
            if (column_data.dtype == 'object' or 
                column_data.dtype.name == 'category' or
                column in self.default_encoding_strategies):
                
                # Skip if too many unique values for certain encoding types
                unique_ratio = column_data.nunique() / len(column_data)
                if unique_ratio < 0.95:  # Not mostly unique
                    categorical_columns.append(column)
        
        return categorical_columns
    
    def _is_list_column(self, data: pd.Series) -> bool:
        """Check if a column contains list-like data"""
        # Check a sample of non-null values
        sample_data = data.dropna().head(10)
        if len(sample_data) == 0:
            return False
            
        for value in sample_data:
            if isinstance(value, (list, tuple, set)):
                return True
        
        return False
    
    def _get_encoding_config(self, column: str, data: pd.Series, 
                           custom_configs: Optional[List[EncodingConfig]]) -> EncodingConfig:
        """Get encoding configuration for a column"""
        
        # Check for custom configuration
        if custom_configs:
            for config in custom_configs:
                if config.column_name == column:
                    return config
        
        # Use default strategy
        encoding_type = self.default_encoding_strategies.get(column, 'auto')
        
        if encoding_type == 'auto':
            encoding_type = self._choose_automatic_encoding(data)
        
        return EncodingConfig(
            column_name=column,
            encoding_type=encoding_type,
            handle_unknown='ignore',
            min_frequency=5 if encoding_type == 'frequency' else None,
            max_categories=50 if encoding_type == 'onehot' else None
        )
    
    def _choose_automatic_encoding(self, data: pd.Series) -> str:
        """Automatically choose encoding strategy based on data characteristics"""
        unique_count = data.nunique()
        total_count = len(data)
        
        # Very high cardinality - use frequency or label encoding
        if unique_count > 100:
            return 'frequency'
        
        # High cardinality - use label encoding
        elif unique_count > 20:
            return 'label'
        
        # Medium cardinality - use one-hot encoding
        elif unique_count > 2:
            return 'onehot'
        
        # Binary - treat as boolean
        else:
            return 'boolean'
    
    def _apply_encoding(self, data: pd.Series, config: EncodingConfig) -> EncodingResult:
        """Apply specific encoding to data"""
        
        if config.encoding_type == 'label':
            return self._apply_label_encoding(data, config)
        elif config.encoding_type == 'onehot':
            return self._apply_onehot_encoding(data, config)
        elif config.encoding_type == 'ordinal':
            return self._apply_ordinal_encoding(data, config)
        elif config.encoding_type == 'frequency':
            return self._apply_frequency_encoding(data, config)
        elif config.encoding_type == 'target':
            return self._apply_target_encoding(data, config)
        elif config.encoding_type == 'boolean':
            return self._apply_boolean_encoding(data, config)
        else:
            raise ValueError(f"Unknown encoding type: {config.encoding_type}")
    
    def _apply_label_encoding(self, data: pd.Series, config: EncodingConfig) -> EncodingResult:
        """Apply label encoding"""
        encoder = LabelEncoder()
        
        # Handle missing values
        data_filled = data.fillna('MISSING')
        
        # Fit encoder
        encoder.fit(data_filled)
        
        return EncodingResult(
            original_column=config.column_name,
            encoded_columns=[f"{config.column_name}_encoded"],
            encoding_type='label',
            encoder=encoder,
            mapping=dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        )
    
    def _apply_onehot_encoding(self, data: pd.Series, config: EncodingConfig) -> EncodingResult:
        """Apply one-hot encoding"""
        encoder = OneHotEncoder(
            handle_unknown=config.handle_unknown,
            sparse_output=False,
            max_categories=config.max_categories
        )
        
        # Handle missing values
        data_filled = data.fillna('MISSING')
        data_reshaped = data_filled.values.reshape(-1, 1)
        
        # Fit encoder
        encoder.fit(data_reshaped)
        
        # Generate feature names
        feature_names = [f"{config.column_name}_{cat}" for cat in encoder.categories_[0]]
        
        return EncodingResult(
            original_column=config.column_name,
            encoded_columns=feature_names,
            encoding_type='onehot',
            encoder=encoder,
            feature_names=feature_names
        )
    
    def _apply_ordinal_encoding(self, data: pd.Series, config: EncodingConfig) -> EncodingResult:
        """Apply ordinal encoding"""
        
        # Get predefined categories or create from data
        if config.column_name in self.ordinal_mappings:
            categories = self.ordinal_mappings[config.column_name]
        else:
            # Create ordinal mapping from data frequency
            value_counts = data.value_counts()
            categories = value_counts.index.tolist()
        
        encoder = OrdinalEncoder(
            categories=[categories],
            handle_unknown=config.handle_unknown
        )
        
        # Handle missing values
        data_filled = data.fillna('MISSING')
        data_reshaped = data_filled.values.reshape(-1, 1)
        
        # Fit encoder
        encoder.fit(data_reshaped)
        
        return EncodingResult(
            original_column=config.column_name,
            encoded_columns=[f"{config.column_name}_encoded"],
            encoding_type='ordinal',
            encoder=encoder,
            mapping=dict(zip(categories, range(len(categories))))
        )
    
    def _apply_frequency_encoding(self, data: pd.Series, config: EncodingConfig) -> EncodingResult:
        """Apply frequency encoding"""
        
        # Calculate frequency mapping
        frequency_map = data.value_counts().to_dict()
        
        # Handle infrequent categories
        if config.min_frequency:
            infrequent_categories = [cat for cat, freq in frequency_map.items() 
                                   if freq < config.min_frequency]
            for cat in infrequent_categories:
                frequency_map[cat] = config.min_frequency - 1
        
        # Create encoder function
        def frequency_encoder(series):
            return series.map(frequency_map).fillna(0)
        
        return EncodingResult(
            original_column=config.column_name,
            encoded_columns=[f"{config.column_name}_encoded"],
            encoding_type='frequency',
            encoder=frequency_encoder,
            mapping=frequency_map
        )
    
    def _apply_target_encoding(self, data: pd.Series, config: EncodingConfig) -> EncodingResult:
        """Apply target encoding (requires target variable - placeholder for now)"""
        # This would require the target variable (price) for proper implementation
        # For now, use frequency encoding as fallback
        return self._apply_frequency_encoding(data, config)
    
    def _apply_boolean_encoding(self, data: pd.Series, config: EncodingConfig) -> EncodingResult:
        """Apply boolean encoding"""
        
        class BooleanEncoder:
            def __init__(self):
                self.mapping = {}
            
            def fit(self, data):
                unique_values = data.dropna().unique()
                for val in unique_values:
                    if str(val).lower() in ['true', '1', 'yes', 'y', 'oui']:
                        self.mapping[val] = 1
                    else:
                        self.mapping[val] = 0
                return self
            
            def transform(self, data):
                if hasattr(data, 'values'):
                    # Handle Series or DataFrame input
                    if len(data.shape) > 1:
                        data = data.iloc[:, 0]  # Get first column if DataFrame
                    return data.map(self.mapping).fillna(0).astype(int)
                else:
                    # Handle array-like input
                    return pd.Series(data).map(self.mapping).fillna(0).astype(int)
            
            def fit_transform(self, data):
                return self.fit(data).transform(data)
        
        encoder = BooleanEncoder()
        encoder.fit(data)
        
        return EncodingResult(
            original_column=config.column_name,
            encoded_columns=[f"{config.column_name}_encoded"],
            encoding_type='boolean',
            encoder=encoder,
            mapping=encoder.mapping
        )
    
    def _encode_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode text features using TF-IDF"""
        self.logger.info("Encoding text features")
        
        text_columns = ['title_cleaned', 'description_cleaned']
        
        for column in text_columns:
            if column not in df.columns:
                continue
                
            # Clean and prepare text data
            text_data = df[column].fillna('').astype(str)
            
            # Skip if mostly empty
            if text_data.str.len().mean() < 10:
                continue
            
            # Apply TF-IDF vectorization
            vectorizer = TfidfVectorizer(
                max_features=50,  # Limit features for memory efficiency
                stop_words='english',
                ngram_range=(1, 2),
                min_df=5,  # Minimum document frequency
                max_df=0.95  # Maximum document frequency
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(text_data)
                
                # Add TF-IDF features to DataFrame
                feature_names = [f"{column}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=feature_names,
                    index=df.index
                )
                
                df = pd.concat([df, tfidf_df], axis=1)
                
                # Store encoder
                self.encoders[f"{column}_tfidf"] = vectorizer
                
                self.logger.info(f"Created {len(feature_names)} TF-IDF features for {column}")
                
            except Exception as e:
                self.logger.warning(f"Failed to encode {column} with TF-IDF: {e}")
        
        return df
    
    def _create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from categorical encodings"""
        self.logger.info("Creating categorical feature interactions")
        
        # City-Property Type interactions
        if 'city_encoded' in df.columns and 'property_type_encoded' in df.columns:
            df['city_property_interaction'] = (
                df['city_encoded'].astype(str) + '_' + df['property_type_encoded'].astype(str)
            )
        
        # Neighborhood frequency by city
        if 'neighborhood_encoded' in df.columns and 'city_encoded' in df.columns:
            neighborhood_city_counts = df.groupby(['city_encoded', 'neighborhood_encoded']).size()
            df['neighborhood_frequency_in_city'] = df.apply(
                lambda row: neighborhood_city_counts.get((row['city_encoded'], row['neighborhood_encoded']), 0),
                axis=1
            )
        
        # Property type distribution in city
        if 'property_type_encoded' in df.columns and 'city_encoded' in df.columns:
            property_city_counts = df.groupby(['city_encoded', 'property_type_encoded']).size()
            df['property_frequency_in_city'] = df.apply(
                lambda row: property_city_counts.get((row['city_encoded'], row['property_type_encoded']), 0),
                axis=1
            )
        
        # Feature combination flags
        feature_combinations = [
            ('parking', 'elevator'),
            ('pool', 'garden'),
            ('furnished', 'elevator')
        ]
        
        for feat1, feat2 in feature_combinations:
            if feat1 in df.columns and feat2 in df.columns:
                df[f"{feat1}_and_{feat2}"] = (df[feat1] == 1) & (df[feat2] == 1)
        
        return df
    
    def get_feature_importance_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Get feature importance and mapping information"""
        feature_info = {}
        
        for column, result in self.encoding_results.items():
            feature_info[column] = {
                'encoding_type': result.encoding_type,
                'encoded_columns': result.encoded_columns,
                'mapping': result.mapping,
                'feature_count': len(result.encoded_columns)
            }
        
        return feature_info
    
    def save_encoders(self, filepath: str):
        """Save all encoders to file"""
        encoder_data = {
            'encoders': self.encoders,
            'encoding_results': self.encoding_results,
            'ordinal_mappings': self.ordinal_mappings
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(encoder_data, f)
        
        self.logger.info(f"Encoders saved to {filepath}")
    
    def load_encoders(self, filepath: str):
        """Load encoders from file"""
        with open(filepath, 'rb') as f:
            encoder_data = pickle.load(f)
        
        self.encoders = encoder_data['encoders']
        self.encoding_results = encoder_data['encoding_results']
        self.ordinal_mappings = encoder_data['ordinal_mappings']
        
        self.logger.info(f"Encoders loaded from {filepath}")
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted encoders"""
        self.logger.info(f"Transforming new data with {len(df)} records")
        
        df_transformed = df.copy()
        
        for column, result in self.encoding_results.items():
            if column not in df_transformed.columns:
                continue
            
            # Apply the stored encoder
            if result.encoding_type == 'onehot':
                encoded_data = result.encoder.transform(df_transformed[[column]])
                for i, col_name in enumerate(result.encoded_columns):
                    df_transformed[col_name] = encoded_data.toarray()[:, i]
            else:
                encoded_col_name = result.encoded_columns[0]
                if result.encoding_type == 'frequency':
                    df_transformed[encoded_col_name] = result.encoder(df_transformed[column])
                else:
                    df_transformed[encoded_col_name] = result.encoder.transform(df_transformed[[column]])
        
        return df_transformed
    
    def generate_encoding_report(self) -> Dict[str, Any]:
        """Generate comprehensive encoding report"""
        report = {
            'total_features_created': 0,
            'encoding_methods_used': {},
            'feature_details': {}
        }
        
        for column, result in self.encoding_results.items():
            method = result.encoding_type
            feature_count = len(result.encoded_columns)
            
            report['total_features_created'] += feature_count
            
            if method not in report['encoding_methods_used']:
                report['encoding_methods_used'][method] = 0
            report['encoding_methods_used'][method] += 1
            
            report['feature_details'][column] = {
                'encoding_method': method,
                'features_created': feature_count,
                'feature_names': result.encoded_columns,
                'unique_categories': len(result.mapping) if result.mapping else None
            }
        
        return report