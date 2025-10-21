"""
Text cleaning and standardization module for real estate data
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
import unicodedata
from loguru import logger

from src.utils.monitoring import get_logger


@dataclass
class TextCleaningResult:
    """Result of text cleaning operation"""
    original_text: str
    cleaned_text: str
    cleaning_operations: List[str]
    confidence: float


class TextStandardizer:
    """Standardizes and cleans text fields in real estate data"""
    
    def __init__(self):
        self.logger = get_logger('text_standardizer')
        
        # Moroccan city name standardization
        self.city_mappings = {
            'casa': 'casablanca',
            'casa blanca': 'casablanca',
            'dar el beida': 'casablanca',
            'dar el bayda': 'casablanca',
            'rabat-sale': 'rabat',
            'rabat salé': 'rabat',
            'rbat': 'rabat',
            'marrakesh': 'marrakech',
            'marrakech menara': 'marrakech',
            'tanger': 'tangier',
            'tangiers': 'tangier',
            'fez': 'fes',
            'fès': 'fes',
            'agadir-ida': 'agadir',
            'agadir ida ou tanane': 'agadir',
            'meknes': 'meknes',
            'meknès': 'meknes',
            'oujda-angad': 'oujda',
            'tetouan': 'tetouan',
            'tétouan': 'tetouan',
            'mohammedia': 'mohammedia',
            'mohammédia': 'mohammedia',
            'kenitra': 'kenitra',
            'kénitra': 'kenitra',
            'el jadida': 'el jadida',
            'jadida': 'el jadida',
            'safi': 'safi',
            'beni mellal': 'beni mellal',
            'béni mellal': 'beni mellal',
            'khouribga': 'khouribga',
            'nador': 'nador',
        }
        
        # Property type standardization
        self.property_type_mappings = {
            # Apartments
            'appartement': 'apartment',
            'appart': 'apartment',
            'app': 'apartment',
            'flat': 'apartment',
            'studio': 'studio',
            'duplex': 'duplex',
            'triplex': 'triplex',
            'penthouse': 'penthouse',
            'loft': 'loft',
            
            # Houses
            'maison': 'house',
            'villa': 'villa',
            'riad': 'riad',
            'townhouse': 'townhouse',
            'cottage': 'house',
            'pavillon': 'house',
            
            # Commercial
            'bureau': 'office',
            'local commercial': 'commercial',
            'commerce': 'commercial',
            'boutique': 'shop',
            'magasin': 'shop',
            'entrepôt': 'warehouse',
            'depot': 'warehouse',
            'usine': 'industrial',
            
            # Land
            'terrain': 'land',
            'terrain agricole': 'agricultural_land',
            'terrain constructible': 'buildable_land',
            'parcelle': 'land'
        }
        
        # Neighborhood/area common terms
        self.neighborhood_terms = {
            'centre ville': 'city_center',
            'centre-ville': 'city_center',
            'downtown': 'city_center',
            'medina': 'medina',
            'nouvelle ville': 'new_city',
            'ville nouvelle': 'new_city',
            'hay': 'neighborhood',
            'quartier': 'neighborhood',
            'zone': 'zone',
            'secteur': 'sector',
            'residence': 'residence',
            'résidence': 'residence'
        }
        
        # Common Arabic/French terms in descriptions
        self.description_terms = {
            'très bon état': 'excellent_condition',
            'bon état': 'good_condition',
            'état moyen': 'average_condition',
            'rénové': 'renovated',
            'neuf': 'new',
            'ancien': 'old',
            'moderne': 'modern',
            'traditionnel': 'traditional',
            'lumineux': 'bright',
            'calme': 'quiet',
            'proche': 'close_to',
            'vue mer': 'sea_view',
            'vue montagne': 'mountain_view',
            'jardin': 'garden',
            'piscine': 'swimming_pool',
            'garage': 'garage',
            'parking': 'parking',
            'ascenseur': 'elevator',
            'climatisation': 'air_conditioning',
            'chauffage': 'heating'
        }
    
    def clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize all text fields in the DataFrame"""
        self.logger.info(f"Cleaning text fields for {len(df)} records")
        
        df_cleaned = df.copy()
        
        # Clean city names
        if 'city' in df_cleaned.columns:
            df_cleaned['city_cleaned'] = df_cleaned['city'].apply(self._clean_city_name)
            
        # Clean property types
        if 'property_type' in df_cleaned.columns:
            df_cleaned['property_type_cleaned'] = df_cleaned['property_type'].apply(self._clean_property_type)
            
        # Clean neighborhood names
        if 'neighborhood' in df_cleaned.columns:
            df_cleaned['neighborhood_cleaned'] = df_cleaned['neighborhood'].apply(self._clean_neighborhood_name)
            
        # Clean titles
        if 'title' in df_cleaned.columns:
            df_cleaned['title_cleaned'] = df_cleaned['title'].apply(self._clean_title)
            
        # Clean descriptions
        if 'description' in df_cleaned.columns:
            df_cleaned['description_cleaned'] = df_cleaned['description'].apply(self._clean_description)
            df_cleaned['description_features'] = df_cleaned['description'].apply(self._extract_features_from_description)
            
        # Clean addresses
        if 'address' in df_cleaned.columns:
            df_cleaned['address_cleaned'] = df_cleaned['address'].apply(self._clean_address)
        
        self.logger.info("Text field cleaning complete")
        return df_cleaned
    
    def _clean_city_name(self, city_text: Any) -> Optional[str]:
        """Clean and standardize city names"""
        if pd.isna(city_text) or not str(city_text).strip():
            return None
            
        # Basic cleaning
        city = str(city_text).lower().strip()
        city = self._remove_accents(city)
        city = re.sub(r'[^\w\s-]', '', city)  # Remove special characters except hyphens
        city = re.sub(r'\s+', ' ', city)      # Normalize whitespace
        
        # Apply city mappings
        if city in self.city_mappings:
            return self.city_mappings[city]
        
        # Check for partial matches
        for variant, standard in self.city_mappings.items():
            if variant in city or city in variant:
                return standard
        
        return city
    
    def _clean_property_type(self, property_type_text: Any) -> Optional[str]:
        """Clean and standardize property types"""
        if pd.isna(property_type_text) or not str(property_type_text).strip():
            return None
            
        # Basic cleaning
        prop_type = str(property_type_text).lower().strip()
        prop_type = self._remove_accents(prop_type)
        prop_type = re.sub(r'[^\w\s]', ' ', prop_type)  # Replace special chars with spaces
        prop_type = re.sub(r'\s+', ' ', prop_type).strip()
        
        # Apply property type mappings
        if prop_type in self.property_type_mappings:
            return self.property_type_mappings[prop_type]
        
        # Check for partial matches
        for variant, standard in self.property_type_mappings.items():
            if variant in prop_type:
                return standard
        
        return prop_type
    
    def _clean_neighborhood_name(self, neighborhood_text: Any) -> Optional[str]:
        """Clean and standardize neighborhood names"""
        if pd.isna(neighborhood_text) or not str(neighborhood_text).strip():
            return None
            
        # Basic cleaning
        neighborhood = str(neighborhood_text).strip()
        neighborhood = self._remove_accents(neighborhood)
        
        # Remove common prefixes/suffixes
        neighborhood = re.sub(r'^(hay|quartier|zone|secteur)\s+', '', neighborhood, flags=re.IGNORECASE)
        neighborhood = re.sub(r'\s+(hay|quartier|zone|secteur)$', '', neighborhood, flags=re.IGNORECASE)
        
        # Clean special characters
        neighborhood = re.sub(r'[^\w\s-]', '', neighborhood)
        neighborhood = re.sub(r'\s+', ' ', neighborhood).strip()
        
        # Convert to title case
        neighborhood = neighborhood.title()
        
        return neighborhood if neighborhood else None
    
    def _clean_title(self, title_text: Any) -> Optional[str]:
        """Clean and standardize property titles"""
        if pd.isna(title_text) or not str(title_text).strip():
            return None
            
        title = str(title_text).strip()
        
        # Remove excessive punctuation
        title = re.sub(r'[!]{2,}', '!', title)
        title = re.sub(r'[?]{2,}', '?', title)
        title = re.sub(r'[.]{2,}', '...', title)
        
        # Remove promotional language
        promotional_patterns = [
            r'\b(urgent|urgente)\b',
            r'\b(prix négociable|négociable)\b',
            r'\b(occasion|affaire)\b',
            r'\b(contactez|appelez|tel|tél)\b.*',
            r'\b(whatsapp|whatapp)\b.*',
            r'\d{10,}',  # Remove phone numbers
        ]
        
        for pattern in promotional_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # Clean whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Capitalize properly
        title = title.title()
        
        return title if title else None
    
    def _clean_description(self, description_text: Any) -> Optional[str]:
        """Clean and standardize property descriptions"""
        if pd.isna(description_text) or not str(description_text).strip():
            return None
            
        description = str(description_text).strip()
        
        # Remove HTML tags if present
        description = re.sub(r'<[^>]+>', '', description)
        
        # Remove excessive punctuation
        description = re.sub(r'[!]{3,}', '!!!', description)
        description = re.sub(r'[?]{3,}', '???', description)
        description = re.sub(r'[.]{4,}', '...', description)
        
        # Remove phone numbers and contact info
        description = re.sub(r'\b\d{10,}\b', '[PHONE]', description)
        description = re.sub(r'\b[\w._%+-]+@[\w.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', description)
        
        # Remove excessive whitespace
        description = re.sub(r'\s+', ' ', description)
        description = re.sub(r'\n+', '\n', description)
        
        return description.strip() if description.strip() else None
    
    def _clean_address(self, address_text: Any) -> Optional[str]:
        """Clean and standardize addresses"""
        if pd.isna(address_text) or not str(address_text).strip():
            return None
            
        address = str(address_text).strip()
        address = self._remove_accents(address)
        
        # Standardize common address terms
        address_replacements = {
            r'\bavenue\b': 'av',
            r'\bboulevard\b': 'bd',
            r'\brue\b': 'rue',
            r'\bplace\b': 'pl',
            r'\bquartier\b': 'qt',
        }
        
        for pattern, replacement in address_replacements.items():
            address = re.sub(pattern, replacement, address, flags=re.IGNORECASE)
        
        # Clean and normalize
        address = re.sub(r'[^\w\s,.-]', '', address)
        address = re.sub(r'\s+', ' ', address).strip()
        
        return address if address else None
    
    def _extract_features_from_description(self, description_text: Any) -> Dict[str, bool]:
        """Extract features from property description"""
        features = {}
        
        if pd.isna(description_text) or not str(description_text).strip():
            return features
            
        description = str(description_text).lower()
        
        # Feature keywords to look for
        feature_keywords = {
            'parking': ['parking', 'garage', 'place de parking'],
            'pool': ['piscine', 'pool', 'swimming pool'],
            'garden': ['jardin', 'garden', 'espace vert'],
            'elevator': ['ascenseur', 'elevator', 'lift'],
            'air_conditioning': ['climatisation', 'clim', 'air conditioning', 'ac'],
            'heating': ['chauffage', 'heating', 'radiateur'],
            'furnished': ['meublé', 'furnished', 'équipé'],
            'balcony': ['balcon', 'balcony', 'terrasse'],
            'terrace': ['terrasse', 'terrace', 'roof top'],
            'sea_view': ['vue mer', 'sea view', 'vue océan'],
            'mountain_view': ['vue montagne', 'mountain view'],
            'new_construction': ['neuf', 'new', 'nouvelle construction'],
            'renovated': ['rénové', 'renovated', 'refait à neuf'],
            'security': ['sécurité', 'security', 'gardien', 'surveillance'],
            'internet': ['internet', 'wifi', 'fibre']
        }
        
        for feature, keywords in feature_keywords.items():
            features[feature] = any(keyword in description for keyword in keywords)
        
        return features
    
    def _remove_accents(self, text: str) -> str:
        """Remove accents from text while preserving Arabic characters"""
        if not text:
            return text
            
        # Normalize Unicode characters
        text = unicodedata.normalize('NFD', text)
        
        # Remove diacritics but keep Arabic characters
        result = []
        for char in text:
            # Keep Arabic characters (U+0600 to U+06FF) and Latin characters without diacritics
            if unicodedata.category(char) != 'Mn' or '\u0600' <= char <= '\u06FF':
                result.append(char)
        
        return ''.join(result)
    
    def standardize_categorical_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize categorical values across the dataset"""
        self.logger.info("Standardizing categorical values")
        
        df_standardized = df.copy()
        
        # Standardize boolean-like values
        boolean_columns = ['furnished', 'parking', 'elevator', 'pool', 'garden']
        for col in boolean_columns:
            if col in df_standardized.columns:
                df_standardized[col] = self._standardize_boolean_values(df_standardized[col])
        
        # Standardize condition/state values
        if 'condition' in df_standardized.columns:
            df_standardized['condition_standardized'] = df_standardized['condition'].apply(
                self._standardize_condition_values
            )
        
        # Standardize floor values
        if 'floor' in df_standardized.columns:
            df_standardized['floor_standardized'] = df_standardized['floor'].apply(
                self._standardize_floor_values
            )
        
        self.logger.info("Categorical value standardization complete")
        return df_standardized
    
    def _standardize_boolean_values(self, series: pd.Series) -> pd.Series:
        """Standardize boolean-like values"""
        def standardize_bool(value):
            if pd.isna(value):
                return None
                
            value_str = str(value).lower().strip()
            
            # True values
            true_values = ['oui', 'yes', 'true', '1', 'disponible', 'inclus', 'avec']
            if any(tv in value_str for tv in true_values):
                return True
            
            # False values
            false_values = ['non', 'no', 'false', '0', 'sans', 'aucun']
            if any(fv in value_str for fv in false_values):
                return False
            
            return None
        
        return series.apply(standardize_bool)
    
    def _standardize_condition_values(self, condition_text: Any) -> Optional[str]:
        """Standardize property condition values"""
        if pd.isna(condition_text) or not str(condition_text).strip():
            return None
            
        condition = str(condition_text).lower().strip()
        condition = self._remove_accents(condition)
        
        # Condition mappings
        condition_mappings = {
            'excellent': ['excellent', 'très bon', 'parfait', 'impeccable'],
            'good': ['bon', 'good', 'correct', 'satisfaisant'],
            'average': ['moyen', 'acceptable', 'passable'],
            'poor': ['mauvais', 'poor', 'à rénover', 'dégradé'],
            'new': ['neuf', 'new', 'nouvelle construction'],
            'renovated': ['rénové', 'refait', 'renovated']
        }
        
        for standard, variants in condition_mappings.items():
            if any(variant in condition for variant in variants):
                return standard
        
        return condition
    
    def _standardize_floor_values(self, floor_text: Any) -> Optional[str]:
        """Standardize floor values"""
        if pd.isna(floor_text) or not str(floor_text).strip():
            return None
            
        floor_str = str(floor_text).lower().strip()
        
        # Extract numeric floor
        floor_match = re.search(r'(\d+)', floor_str)
        if floor_match:
            floor_num = int(floor_match.group(1))
            return f"floor_{floor_num}"
        
        # Special floor types
        if any(term in floor_str for term in ['rdc', 'ground', 'rez']):
            return 'ground_floor'
        elif any(term in floor_str for term in ['sous-sol', 'basement', 'cave']):
            return 'basement'
        elif any(term in floor_str for term in ['dernier', 'top', 'penthouse']):
            return 'top_floor'
        
        return floor_str
    
    def generate_text_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistics about text field quality"""
        self.logger.info("Generating text field statistics")
        
        stats = {
            'total_records': len(df),
            'text_fields': {},
            'cleaning_impact': {}
        }
        
        text_columns = ['title', 'description', 'city', 'neighborhood', 'property_type', 'address']
        
        for col in text_columns:
            if col in df.columns:
                original_col = col
                cleaned_col = f"{col}_cleaned"
                
                stats['text_fields'][col] = {
                    'total_values': df[original_col].notna().sum(),
                    'empty_values': df[original_col].isna().sum(),
                    'avg_length': df[original_col].astype(str).str.len().mean(),
                    'unique_values': df[original_col].nunique()
                }
                
                if cleaned_col in df.columns:
                    # Compare before and after cleaning
                    original_unique = df[original_col].nunique()
                    cleaned_unique = df[cleaned_col].nunique()
                    
                    stats['cleaning_impact'][col] = {
                        'original_unique': original_unique,
                        'cleaned_unique': cleaned_unique,
                        'consolidation_rate': (original_unique - cleaned_unique) / original_unique if original_unique > 0 else 0
                    }
        
        return stats