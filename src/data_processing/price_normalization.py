"""
Price normalization and standardization module for real estate data
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, date
import requests
from loguru import logger

from src.utils.monitoring import get_logger


@dataclass
class PriceNormalizationResult:
    """Result of price normalization"""
    original_price: Any
    normalized_price_mad: Optional[float]
    original_currency: str
    confidence: float
    normalization_method: str
    exchange_rate_used: Optional[float] = None
    price_per_m2: Optional[float] = None


class CurrencyConverter:
    """Handles currency conversion for price normalization"""
    
    def __init__(self):
        self.logger = get_logger('currency_converter')
        
        # Fixed exchange rates (as of recent data) - in production, use live API
        self.exchange_rates = {
            'USD': 10.0,   # 1 USD = 10 MAD (approximate)
            'EUR': 11.0,   # 1 EUR = 11 MAD (approximate)
            'GBP': 12.5,   # 1 GBP = 12.5 MAD (approximate)
            'DH': 1.0,     # Moroccan Dirham variants
            'MAD': 1.0,    # Moroccan Dirham
            'DHS': 1.0,    # Dirham plural
        }
        
        # Common currency patterns in real estate listings
        self.currency_patterns = {
            r'(\d+(?:[,\s]\d{3})*(?:\.\d{2})?)\s*(?:mad|dh|dhs|dirham)': ('MAD', 1.0),
            r'(\d+(?:[,\s]\d{3})*(?:\.\d{2})?)\s*€': ('EUR', self.exchange_rates['EUR']),
            r'(\d+(?:[,\s]\d{3})*(?:\.\d{2})?)\s*\$': ('USD', self.exchange_rates['USD']),
            r'(\d+(?:[,\s]\d{3})*(?:\.\d{2})?)\s*£': ('GBP', self.exchange_rates['GBP']),
            r'€\s*(\d+(?:[,\s]\d{3})*(?:\.\d{2})?)': ('EUR', self.exchange_rates['EUR']),
            r'\$\s*(\d+(?:[,\s]\d{3})*(?:\.\d{2})?)': ('USD', self.exchange_rates['USD']),
            r'£\s*(\d+(?:[,\s]\d{3})*(?:\.\d{2})?)': ('GBP', self.exchange_rates['GBP']),
        }
    
    def convert_to_mad(self, price_text: str, fallback_currency: str = 'MAD') -> Tuple[Optional[float], str, float, float]:
        """Convert price text to MAD"""
        if pd.isna(price_text) or not str(price_text).strip():
            return None, 'UNKNOWN', 0.0, 1.0
        
        price_str = str(price_text).lower().strip()
        
        # Try to match currency patterns
        for pattern, (currency, rate) in self.currency_patterns.items():
            match = re.search(pattern, price_str, re.IGNORECASE)
            if match:
                # Extract numeric value
                numeric_str = match.group(1).replace(',', '').replace(' ', '')
                try:
                    price_value = float(numeric_str)
                    mad_price = price_value * rate
                    return mad_price, currency, 0.9, rate
                except ValueError:
                    continue
        
        # Try to extract any numeric value and assume default currency
        numeric_match = re.search(r'(\d+(?:[,\s]\d{3})*(?:\.\d{2})?)', price_str)
        if numeric_match:
            numeric_str = numeric_match.group(1).replace(',', '').replace(' ', '')
            try:
                price_value = float(numeric_str)
                rate = self.exchange_rates.get(fallback_currency, 1.0)
                mad_price = price_value * rate
                return mad_price, fallback_currency, 0.5, rate
            except ValueError:
                pass
        
        return None, 'UNKNOWN', 0.0, 1.0


class PriceNormalizer:
    """Normalizes and standardizes real estate prices"""
    
    def __init__(self):
        self.logger = get_logger('price_normalizer')
        self.currency_converter = CurrencyConverter()
        
        # Price validation thresholds for Moroccan market
        self.price_thresholds = {
            'min_price_mad': 50000,      # 50K MAD minimum
            'max_price_mad': 100000000,  # 100M MAD maximum
            'min_price_per_m2': 1000,    # 1K MAD/m² minimum
            'max_price_per_m2': 200000   # 200K MAD/m² maximum
        }
    
    def normalize_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all prices in the DataFrame to MAD"""
        self.logger.info(f"Normalizing prices for {len(df)} records")
        
        df_normalized = df.copy()
        
        # Initialize new columns
        df_normalized['price_mad'] = None
        df_normalized['original_currency'] = 'UNKNOWN'
        df_normalized['price_confidence'] = 0.0
        df_normalized['exchange_rate_used'] = 1.0
        df_normalized['price_per_m2'] = None
        df_normalized['price_normalization_method'] = 'unknown'
        
        normalization_results = []
        successful_normalizations = 0
        
        for idx, row in df_normalized.iterrows():
            # Get price from various possible columns
            price_text = self._extract_price_text(row)
            
            if not price_text:
                continue
            
            # Convert to MAD
            mad_price, currency, confidence, exchange_rate = self.currency_converter.convert_to_mad(price_text)
            
            if mad_price:
                # Validate price
                if self._is_valid_price(mad_price):
                    df_normalized.at[idx, 'price_mad'] = mad_price
                    df_normalized.at[idx, 'original_currency'] = currency
                    df_normalized.at[idx, 'price_confidence'] = confidence
                    df_normalized.at[idx, 'exchange_rate_used'] = exchange_rate
                    df_normalized.at[idx, 'price_normalization_method'] = 'automatic'
                    
                    # Calculate price per m² if surface is available
                    if pd.notna(row.get('surface_m2')) and row.get('surface_m2', 0) > 0:
                        price_per_m2 = mad_price / row['surface_m2']
                        if self._is_valid_price_per_m2(price_per_m2):
                            df_normalized.at[idx, 'price_per_m2'] = price_per_m2
                    
                    successful_normalizations += 1
                    
                    # Store result for analysis
                    normalization_results.append(PriceNormalizationResult(
                        original_price=price_text,
                        normalized_price_mad=mad_price,
                        original_currency=currency,
                        confidence=confidence,
                        normalization_method='automatic',
                        exchange_rate_used=exchange_rate,
                        price_per_m2=df_normalized.at[idx, 'price_per_m2']
                    ))
            
            if idx % 1000 == 0:
                self.logger.info(f"Normalized {idx} prices, {successful_normalizations} successful")
        
        self.logger.info(f"Price normalization complete: {successful_normalizations}/{len(df)} successful")
        
        # Apply additional price corrections
        df_normalized = self._apply_price_corrections(df_normalized)
        
        return df_normalized
    
    def _extract_price_text(self, row: pd.Series) -> Optional[str]:
        """Extract price text from various possible columns"""
        price_columns = ['price', 'price_text', 'prix', 'cost', 'amount']
        
        for col in price_columns:
            if col in row and pd.notna(row[col]):
                return str(row[col])
        
        return None
    
    def _is_valid_price(self, price: float) -> bool:
        """Validate if price is within reasonable range"""
        return (
            self.price_thresholds['min_price_mad'] <= price <= self.price_thresholds['max_price_mad']
        )
    
    def _is_valid_price_per_m2(self, price_per_m2: float) -> bool:
        """Validate if price per m² is within reasonable range"""
        return (
            self.price_thresholds['min_price_per_m2'] <= price_per_m2 <= self.price_thresholds['max_price_per_m2']
        )
    
    def _apply_price_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply various price corrections and validations"""
        self.logger.info("Applying price corrections")
        
        df_corrected = df.copy()
        corrections_applied = 0
        
        for idx, row in df_corrected.iterrows():
            if pd.isna(row['price_mad']):
                continue
            
            # Check for obvious scaling errors (prices too high or too low)
            corrected_price, correction_method = self._detect_and_correct_scaling_errors(
                row['price_mad'], row.get('surface_m2')
            )
            
            if corrected_price != row['price_mad']:
                df_corrected.at[idx, 'price_mad'] = corrected_price
                df_corrected.at[idx, 'price_normalization_method'] = correction_method
                df_corrected.at[idx, 'price_confidence'] = 0.7  # Lower confidence for corrected prices
                corrections_applied += 1
                
                # Recalculate price per m²
                if pd.notna(row.get('surface_m2')) and row.get('surface_m2', 0) > 0:
                    df_corrected.at[idx, 'price_per_m2'] = corrected_price / row['surface_m2']
        
        self.logger.info(f"Applied {corrections_applied} price corrections")
        return df_corrected
    
    def _detect_and_correct_scaling_errors(self, price: float, surface: Optional[float]) -> Tuple[float, str]:
        """Detect and correct common scaling errors in prices"""
        original_price = price
        
        # Check if price seems too high (might be in cents instead of dirhams)
        if price > 50000000:  # 50M MAD
            # Try dividing by 100
            corrected_price = price / 100
            if self._is_valid_price(corrected_price):
                return corrected_price, 'divided_by_100'
        
        # Check if price seems too low (might be in thousands)
        if price < 100000:  # 100K MAD
            # Try multiplying by 1000
            corrected_price = price * 1000
            if self._is_valid_price(corrected_price):
                # Validate against surface if available
                if surface and surface > 0:
                    price_per_m2 = corrected_price / surface
                    if self._is_valid_price_per_m2(price_per_m2):
                        return corrected_price, 'multiplied_by_1000'
                else:
                    return corrected_price, 'multiplied_by_1000'
        
        # Check price per m² ratio for reasonableness
        if surface and surface > 0:
            price_per_m2 = price / surface
            
            # If price per m² is too high, price might be inflated
            if price_per_m2 > 500000:  # 500K MAD/m² is unrealistic
                corrected_price = price / 10
                if self._is_valid_price(corrected_price):
                    new_price_per_m2 = corrected_price / surface
                    if self._is_valid_price_per_m2(new_price_per_m2):
                        return corrected_price, 'divided_by_10'
            
            # If price per m² is too low, price might need scaling up
            if price_per_m2 < 500:  # 500 MAD/m² is very low
                corrected_price = price * 10
                if self._is_valid_price(corrected_price):
                    new_price_per_m2 = corrected_price / surface
                    if self._is_valid_price_per_m2(new_price_per_m2):
                        return corrected_price, 'multiplied_by_10'
        
        return original_price, 'no_correction'
    
    def calculate_price_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional price-related metrics"""
        self.logger.info("Calculating price metrics")
        
        df_metrics = df.copy()
        
        # Price per room
        df_metrics['price_per_room'] = None
        room_mask = (df_metrics['rooms'].notna()) & (df_metrics['rooms'] > 0) & (df_metrics['price_mad'].notna())
        df_metrics.loc[room_mask, 'price_per_room'] = df_metrics.loc[room_mask, 'price_mad'] / df_metrics.loc[room_mask, 'rooms']
        
        # Price categories
        df_metrics['price_category'] = None
        price_mask = df_metrics['price_mad'].notna()
        
        df_metrics.loc[price_mask & (df_metrics['price_mad'] < 500000), 'price_category'] = 'budget'
        df_metrics.loc[price_mask & (df_metrics['price_mad'] >= 500000) & (df_metrics['price_mad'] < 1500000), 'price_category'] = 'mid_range'
        df_metrics.loc[price_mask & (df_metrics['price_mad'] >= 1500000) & (df_metrics['price_mad'] < 5000000), 'price_category'] = 'premium'
        df_metrics.loc[price_mask & (df_metrics['price_mad'] >= 5000000), 'price_category'] = 'luxury'
        
        # Price per m² categories by city
        if 'city' in df_metrics.columns:
            df_metrics['price_per_m2_category'] = None
            
            city_thresholds = {
                'casablanca': {'low': 15000, 'mid': 30000, 'high': 50000},
                'rabat': {'low': 12000, 'mid': 25000, 'high': 40000},
                'marrakech': {'low': 20000, 'mid': 40000, 'high': 70000},
                'tangier': {'low': 10000, 'mid': 20000, 'high': 35000},
                'fes': {'low': 8000, 'mid': 15000, 'high': 25000},
                'agadir': {'low': 15000, 'mid': 30000, 'high': 50000}
            }
            
            for city, thresholds in city_thresholds.items():
                city_mask = (df_metrics['city'].str.lower() == city) & df_metrics['price_per_m2'].notna()
                
                df_metrics.loc[city_mask & (df_metrics['price_per_m2'] < thresholds['low']), 'price_per_m2_category'] = 'below_market'
                df_metrics.loc[city_mask & (df_metrics['price_per_m2'] >= thresholds['low']) & (df_metrics['price_per_m2'] < thresholds['mid']), 'price_per_m2_category'] = 'market_rate'
                df_metrics.loc[city_mask & (df_metrics['price_per_m2'] >= thresholds['mid']) & (df_metrics['price_per_m2'] < thresholds['high']), 'price_per_m2_category'] = 'above_market'
                df_metrics.loc[city_mask & (df_metrics['price_per_m2'] >= thresholds['high']), 'price_per_m2_category'] = 'premium_market'
        
        # Relative price within city (percentile ranking)
        if 'city' in df_metrics.columns:
            df_metrics['price_percentile_in_city'] = None
            
            for city in df_metrics['city'].unique():
                if pd.isna(city):
                    continue
                    
                city_mask = (df_metrics['city'] == city) & df_metrics['price_mad'].notna()
                city_data = df_metrics.loc[city_mask, 'price_mad']
                
                if len(city_data) > 10:  # Need sufficient data for percentiles
                    percentiles = city_data.rank(pct=True)
                    df_metrics.loc[city_mask, 'price_percentile_in_city'] = percentiles
        
        self.logger.info("Price metrics calculation complete")
        return df_metrics
    
    def validate_price_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate price consistency across the dataset"""
        self.logger.info("Validating price consistency")
        
        validation_results = {
            'total_records': len(df),
            'records_with_price': df['price_mad'].notna().sum(),
            'records_with_price_per_m2': df['price_per_m2'].notna().sum(),
            'price_range': {
                'min': df['price_mad'].min() if df['price_mad'].notna().any() else None,
                'max': df['price_mad'].max() if df['price_mad'].notna().any() else None,
                'median': df['price_mad'].median() if df['price_mad'].notna().any() else None
            },
            'currency_distribution': df['original_currency'].value_counts().to_dict(),
            'price_confidence_stats': {
                'mean': df['price_confidence'].mean() if df['price_confidence'].notna().any() else None,
                'min': df['price_confidence'].min() if df['price_confidence'].notna().any() else None,
                'max': df['price_confidence'].max() if df['price_confidence'].notna().any() else None
            },
            'outliers_detected': 0,
            'inconsistencies': []
        }
        
        # Check for price inconsistencies
        if 'price_per_m2' in df.columns and 'surface_m2' in df.columns:
            # Check if calculated price_per_m2 matches stored value
            calculated_price_per_m2 = df['price_mad'] / df['surface_m2']
            price_diff = abs(calculated_price_per_m2 - df['price_per_m2'])
            
            # Allow 1% tolerance
            inconsistent_mask = (price_diff > df['price_per_m2'] * 0.01) & df['price_per_m2'].notna()
            inconsistent_count = inconsistent_mask.sum()
            
            if inconsistent_count > 0:
                validation_results['inconsistencies'].append(
                    f"{inconsistent_count} records with price/surface inconsistency"
                )
        
        # Check for extreme outliers
        if df['price_mad'].notna().any():
            Q1 = df['price_mad'].quantile(0.25)
            Q3 = df['price_mad'].quantile(0.75)
            IQR = Q3 - Q1
            outlier_threshold_high = Q3 + 3 * IQR
            outlier_threshold_low = Q1 - 3 * IQR
            
            outliers = df[
                (df['price_mad'] > outlier_threshold_high) | 
                (df['price_mad'] < outlier_threshold_low)
            ]
            validation_results['outliers_detected'] = len(outliers)
        
        self.logger.info(f"Price validation complete: {validation_results['records_with_price']}/{validation_results['total_records']} records have valid prices")
        
        return validation_results