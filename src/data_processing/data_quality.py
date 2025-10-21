"""
Data quality validation and reporting module for real estate data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from pathlib import Path
from loguru import logger

from ..utils.monitoring import get_logger


@dataclass
class QualityCheck:
    """Individual quality check result"""
    check_name: str
    status: str  # 'pass', 'fail', 'warning'
    score: float  # 0-1 scale
    details: Dict[str, Any]
    recommendations: List[str]


@dataclass
class QualityReport:
    """Comprehensive data quality report"""
    overall_score: float
    total_records: int
    quality_checks: List[QualityCheck]
    summary_stats: Dict[str, Any]
    recommendations: List[str]
    timestamp: str


class DataQualityValidator:
    """Comprehensive data quality validation for real estate data"""
    
    def __init__(self, output_dir: str = "reports"):
        self.logger = get_logger('data_quality_validator')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality thresholds
        self.thresholds = {
            'completeness': {
                'critical_fields': 0.95,  # 95% completeness for critical fields
                'important_fields': 0.80,  # 80% completeness for important fields
                'optional_fields': 0.50   # 50% completeness for optional fields
            },
            'validity': {
                'price_range': 0.95,      # 95% prices within valid range
                'coordinate_accuracy': 0.90, # 90% coordinates valid
                'text_quality': 0.85      # 85% text fields clean
            },
            'consistency': {
                'price_surface_ratio': 0.90,  # 90% price/surface ratios reasonable
                'duplicates': 0.05,           # Less than 5% duplicates
                'outliers': 0.10              # Less than 10% outliers
            }
        }
        
        # Field categories
        self.field_categories = {
            'critical': ['price_mad', 'surface_m2', 'city', 'property_type'],
            'important': ['rooms', 'neighborhood', 'latitude', 'longitude'],
            'optional': ['bathrooms', 'floor', 'parking', 'elevator', 'description']
        }
        
        # Expected data types
        self.expected_dtypes = {
            'price_mad': 'float64',
            'surface_m2': 'float64',
            'rooms': 'int64',
            'bathrooms': 'float64',
            'latitude': 'float64',
            'longitude': 'float64',
            'city': 'object',
            'property_type': 'object',
            'neighborhood': 'object'
        }
    
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
    
    def validate_data_quality(self, df: pd.DataFrame) -> QualityReport:
        """Perform comprehensive data quality validation"""
        
        self.logger.info(f"Starting data quality validation for {len(df)} records")
        
        quality_checks = []
        
        # 1. Completeness checks
        completeness_checks = self._check_completeness(df)
        quality_checks.extend(completeness_checks)
        
        # 2. Validity checks
        validity_checks = self._check_validity(df)
        quality_checks.extend(validity_checks)
        
        # 3. Consistency checks
        consistency_checks = self._check_consistency(df)
        quality_checks.extend(consistency_checks)
        
        # 4. Accuracy checks
        accuracy_checks = self._check_accuracy(df)
        quality_checks.extend(accuracy_checks)
        
        # 5. Uniqueness checks
        uniqueness_checks = self._check_uniqueness(df)
        quality_checks.extend(uniqueness_checks)
        
        # 6. Integrity checks
        integrity_checks = self._check_integrity(df)
        quality_checks.extend(integrity_checks)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(quality_checks)
        
        # Generate summary statistics
        summary_stats = self._generate_summary_stats(df)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(quality_checks)
        
        report = QualityReport(
            overall_score=overall_score,
            total_records=len(df),
            quality_checks=quality_checks,
            summary_stats=summary_stats,
            recommendations=recommendations,
            timestamp=pd.Timestamp.now().isoformat()
        )
        
        self.logger.info(f"Data quality validation complete. Overall score: {overall_score:.2f}")
        return report
    
    def _check_completeness(self, df: pd.DataFrame) -> List[QualityCheck]:
        """Check data completeness"""
        
        checks = []
        
        for category, fields in self.field_categories.items():
            available_fields = [f for f in fields if f in df.columns]
            
            if not available_fields:
                continue
            
            completeness_rates = []
            field_details = {}
            
            for field in available_fields:
                completeness = (df[field].notna().sum() / len(df))
                completeness_rates.append(completeness)
                field_details[field] = {
                    'completeness': completeness,
                    'missing_count': df[field].isna().sum(),
                    'missing_percentage': (df[field].isna().sum() / len(df)) * 100
                }
            
            avg_completeness = np.mean(completeness_rates)
            threshold = self.thresholds['completeness'][f'{category}_fields']
            
            status = 'pass' if avg_completeness >= threshold else 'fail'
            if avg_completeness >= threshold - 0.1:  # Within 10% of threshold
                status = 'warning'
            
            recommendations = []
            if status != 'pass':
                poor_fields = [f for f in available_fields 
                             if field_details[f]['completeness'] < threshold]
                recommendations.append(f"Improve data collection for {category} fields: {poor_fields}")
            
            checks.append(QualityCheck(
                check_name=f"completeness_{category}_fields",
                status=status,
                score=min(avg_completeness / threshold, 1.0),
                details=field_details,
                recommendations=recommendations
            ))
        
        return checks
    
    def _check_validity(self, df: pd.DataFrame) -> List[QualityCheck]:
        """Check data validity"""
        
        checks = []
        
        # Price validity
        if 'price_mad' in df.columns:
            price_data = df['price_mad'].dropna()
            valid_prices = price_data[
                (price_data >= 50000) & (price_data <= 100000000)
            ]
            validity_rate = len(valid_prices) / len(price_data) if len(price_data) > 0 else 0
            
            status = 'pass' if validity_rate >= self.thresholds['validity']['price_range'] else 'fail'
            
            checks.append(QualityCheck(
                check_name="price_validity",
                status=status,
                score=validity_rate,
                details={
                    'total_prices': len(price_data),
                    'valid_prices': len(valid_prices),
                    'validity_rate': validity_rate,
                    'price_range': {'min': price_data.min(), 'max': price_data.max()}
                },
                recommendations=["Remove or correct invalid price values"] if status != 'pass' else []
            ))
        
        # Coordinate validity
        if 'latitude' in df.columns and 'longitude' in df.columns:
            coord_data = df[['latitude', 'longitude']].dropna()
            
            # Morocco bounds
            valid_coords = coord_data[
                (coord_data['latitude'] >= 21.0) & (coord_data['latitude'] <= 36.0) &
                (coord_data['longitude'] >= -17.5) & (coord_data['longitude'] <= -1.0)
            ]
            
            validity_rate = len(valid_coords) / len(coord_data) if len(coord_data) > 0 else 0
            status = 'pass' if validity_rate >= self.thresholds['validity']['coordinate_accuracy'] else 'fail'
            
            checks.append(QualityCheck(
                check_name="coordinate_validity",
                status=status,
                score=validity_rate,
                details={
                    'total_coordinates': len(coord_data),
                    'valid_coordinates': len(valid_coords),
                    'validity_rate': validity_rate
                },
                recommendations=["Validate and correct coordinate data"] if status != 'pass' else []
            ))
        
        # Surface area validity
        if 'surface_m2' in df.columns:
            surface_data = df['surface_m2'].dropna()
            valid_surface = surface_data[
                (surface_data >= 15) & (surface_data <= 2000)
            ]
            validity_rate = len(valid_surface) / len(surface_data) if len(surface_data) > 0 else 0
            
            status = 'pass' if validity_rate >= 0.95 else 'fail'
            
            checks.append(QualityCheck(
                check_name="surface_validity",
                status=status,
                score=validity_rate,
                details={
                    'total_surface_values': len(surface_data),
                    'valid_surface_values': len(valid_surface),
                    'validity_rate': validity_rate
                },
                recommendations=["Review and correct surface area values"] if status != 'pass' else []
            ))
        
        return checks
    
    def _check_consistency(self, df: pd.DataFrame) -> List[QualityCheck]:
        """Check data consistency"""
        
        checks = []
        
        # Price per square meter consistency
        if 'price_mad' in df.columns and 'surface_m2' in df.columns:
            data_with_both = df[['price_mad', 'surface_m2']].dropna()
            
            if len(data_with_both) > 0:
                price_per_m2 = data_with_both['price_mad'] / data_with_both['surface_m2']
                
                # Reasonable price per m2 range (1000 - 200000 MAD/m2)
                reasonable_ratio = price_per_m2[
                    (price_per_m2 >= 1000) & (price_per_m2 <= 200000)
                ]
                
                consistency_rate = len(reasonable_ratio) / len(price_per_m2)
                status = 'pass' if consistency_rate >= self.thresholds['consistency']['price_surface_ratio'] else 'fail'
                
                checks.append(QualityCheck(
                    check_name="price_surface_consistency",
                    status=status,
                    score=consistency_rate,
                    details={
                        'total_records': len(price_per_m2),
                        'consistent_records': len(reasonable_ratio),
                        'consistency_rate': consistency_rate,
                        'price_per_m2_stats': {
                            'min': price_per_m2.min(),
                            'max': price_per_m2.max(),
                            'median': price_per_m2.median()
                        }
                    },
                    recommendations=["Review price and surface area data for consistency"] if status != 'pass' else []
                ))
        
        # Room count consistency
        if 'rooms' in df.columns and 'surface_m2' in df.columns:
            room_surface_data = df[['rooms', 'surface_m2']].dropna()
            
            if len(room_surface_data) > 0:
                # Reasonable room size (average 10-100 m2 per room)
                avg_room_size = room_surface_data['surface_m2'] / room_surface_data['rooms']
                reasonable_room_size = avg_room_size[
                    (avg_room_size >= 10) & (avg_room_size <= 100)
                ]
                
                consistency_rate = len(reasonable_room_size) / len(avg_room_size)
                status = 'pass' if consistency_rate >= 0.85 else 'fail'
                
                checks.append(QualityCheck(
                    check_name="room_surface_consistency",
                    status=status,
                    score=consistency_rate,
                    details={
                        'total_records': len(avg_room_size),
                        'consistent_records': len(reasonable_room_size),
                        'consistency_rate': consistency_rate
                    },
                    recommendations=["Review room count and surface area relationship"] if status != 'pass' else []
                ))
        
        return checks
    
    def _check_accuracy(self, df: pd.DataFrame) -> List[QualityCheck]:
        """Check data accuracy"""
        
        checks = []
        
        # City name accuracy (check against known Moroccan cities)
        if 'city' in df.columns or 'city_cleaned' in df.columns:
            city_col = 'city_cleaned' if 'city_cleaned' in df.columns else 'city'
            city_data = df[city_col].dropna()
            
            known_cities = {
                'casablanca', 'rabat', 'marrakech', 'tangier', 'fes', 'agadir',
                'meknes', 'oujda', 'kenitra', 'tetouan', 'safi', 'mohammedia',
                'el jadida', 'beni mellal', 'khouribga', 'nador'
            }
            
            recognized_cities = city_data[city_data.str.lower().isin(known_cities)]
            accuracy_rate = len(recognized_cities) / len(city_data) if len(city_data) > 0 else 0
            
            status = 'pass' if accuracy_rate >= 0.80 else 'warning'
            
            checks.append(QualityCheck(
                check_name="city_name_accuracy",
                status=status,
                score=accuracy_rate,
                details={
                    'total_cities': len(city_data),
                    'recognized_cities': len(recognized_cities),
                    'accuracy_rate': accuracy_rate,
                    'unrecognized_cities': city_data[~city_data.str.lower().isin(known_cities)].value_counts().head(10).to_dict()
                },
                recommendations=["Standardize city names using official Moroccan city list"] if status != 'pass' else []
            ))
        
        # Property type accuracy
        if 'property_type' in df.columns or 'property_type_cleaned' in df.columns:
            prop_col = 'property_type_cleaned' if 'property_type_cleaned' in df.columns else 'property_type'
            prop_data = df[prop_col].dropna()
            
            known_types = {
                'apartment', 'house', 'villa', 'studio', 'duplex', 'triplex',
                'penthouse', 'loft', 'riad', 'office', 'commercial', 'land'
            }
            
            recognized_types = prop_data[prop_data.str.lower().isin(known_types)]
            accuracy_rate = len(recognized_types) / len(prop_data) if len(prop_data) > 0 else 0
            
            status = 'pass' if accuracy_rate >= 0.75 else 'warning'
            
            checks.append(QualityCheck(
                check_name="property_type_accuracy",
                status=status,
                score=accuracy_rate,
                details={
                    'total_types': len(prop_data),
                    'recognized_types': len(recognized_types),
                    'accuracy_rate': accuracy_rate
                },
                recommendations=["Standardize property type categories"] if status != 'pass' else []
            ))
        
        return checks
    
    def _check_uniqueness(self, df: pd.DataFrame) -> List[QualityCheck]:
        """Check for duplicate records"""
        
        checks = []
        
        # Filter out list columns for duplicate checking
        df_for_duplicates = df.copy()
        list_columns = []
        
        for col in df_for_duplicates.columns:
            if self._is_list_column(df_for_duplicates[col]):
                list_columns.append(col)
                
        # Remove list columns from duplicate checking
        if list_columns:
            df_for_duplicates = df_for_duplicates.drop(columns=list_columns)
        
        # Exact duplicates
        exact_duplicates = df_for_duplicates.duplicated().sum()
        duplicate_rate = exact_duplicates / len(df)
        
        status = 'pass' if duplicate_rate <= self.thresholds['consistency']['duplicates'] else 'fail'
        
        checks.append(QualityCheck(
            check_name="exact_duplicates",
            status=status,
            score=1 - duplicate_rate,
            details={
                'total_records': len(df),
                'duplicate_records': exact_duplicates,
                'duplicate_rate': duplicate_rate,
                'excluded_list_columns': list_columns
            },
            recommendations=["Remove exact duplicate records"] if status != 'pass' else []
        ))
        
        # Near duplicates (if coordinate data available)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            coord_data = df[['latitude', 'longitude']].dropna()
            
            # Count records with same coordinates (rounded to 4 decimal places)
            coord_rounded = coord_data.round(4)
            near_duplicates = coord_rounded.duplicated().sum()
            near_duplicate_rate = near_duplicates / len(coord_data) if len(coord_data) > 0 else 0
            
            status = 'pass' if near_duplicate_rate <= 0.15 else 'warning'  # Allow some coordinate overlap
            
            checks.append(QualityCheck(
                check_name="coordinate_near_duplicates",
                status=status,
                score=1 - near_duplicate_rate,
                details={
                    'total_coordinates': len(coord_data),
                    'near_duplicate_coordinates': near_duplicates,
                    'near_duplicate_rate': near_duplicate_rate
                },
                recommendations=["Review records with identical coordinates"] if status != 'pass' else []
            ))
        
        return checks
    
    def _check_integrity(self, df: pd.DataFrame) -> List[QualityCheck]:
        """Check referential integrity and data types"""
        
        checks = []
        
        # Data type consistency
        type_issues = {}
        for column, expected_type in self.expected_dtypes.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if actual_type != expected_type:
                    type_issues[column] = {
                        'expected': expected_type,
                        'actual': actual_type
                    }
        
        integrity_score = 1 - (len(type_issues) / len(self.expected_dtypes))
        status = 'pass' if integrity_score >= 0.80 else 'warning'
        
        checks.append(QualityCheck(
            check_name="data_type_integrity",
            status=status,
            score=integrity_score,
            details={
                'type_issues': type_issues,
                'columns_checked': len(self.expected_dtypes),
                'issues_found': len(type_issues)
            },
            recommendations=["Fix data type inconsistencies"] if type_issues else []
        ))
        
        # Value range integrity
        range_issues = []
        
        # Check room counts
        if 'rooms' in df.columns:
            invalid_rooms = df[(df['rooms'] < 1) | (df['rooms'] > 20)]['rooms'].count()
            if invalid_rooms > 0:
                range_issues.append(f"Invalid room counts: {invalid_rooms} records")
        
        # Check bathroom counts
        if 'bathrooms' in df.columns:
            invalid_bathrooms = df[(df['bathrooms'] < 0) | (df['bathrooms'] > 10)]['bathrooms'].count()
            if invalid_bathrooms > 0:
                range_issues.append(f"Invalid bathroom counts: {invalid_bathrooms} records")
        
        range_integrity_score = 1 - (len(range_issues) / 5)  # Assume 5 potential range checks
        status = 'pass' if len(range_issues) == 0 else 'warning'
        
        checks.append(QualityCheck(
            check_name="value_range_integrity",
            status=status,
            score=max(0, range_integrity_score),
            details={
                'range_issues': range_issues,
                'issues_count': len(range_issues)
            },
            recommendations=["Fix out-of-range values"] if range_issues else []
        ))
        
        return checks
    
    def _calculate_overall_score(self, quality_checks: List[QualityCheck]) -> float:
        """Calculate overall quality score"""
        
        if not quality_checks:
            return 0.0
        
        # Weight different check types
        weights = {
            'completeness': 0.3,
            'validity': 0.25,
            'consistency': 0.20,
            'accuracy': 0.15,
            'uniqueness': 0.05,
            'integrity': 0.05
        }
        
        weighted_scores = []
        
        for check in quality_checks:
            # Extract check category from name
            category = check.check_name.split('_')[0]
            weight = weights.get(category, 0.1)
            weighted_scores.append(check.score * weight)
        
        return min(1.0, sum(weighted_scores))
    
    def _generate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics"""
        
        return {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_data_percentage': (df.isnull().sum().sum() / df.size) * 100,
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'duplicate_rows': df.duplicated().sum()
        }
    
    def _generate_recommendations(self, quality_checks: List[QualityCheck]) -> List[str]:
        """Generate overall recommendations"""
        
        all_recommendations = []
        for check in quality_checks:
            all_recommendations.extend(check.recommendations)
        
        # Remove duplicates and sort by priority
        unique_recommendations = list(set(all_recommendations))
        
        # Add general recommendations based on overall quality
        scores = [check.score for check in quality_checks]
        avg_score = np.mean(scores) if scores else 0
        
        if avg_score < 0.7:
            unique_recommendations.insert(0, "Overall data quality is below acceptable threshold - immediate action required")
        elif avg_score < 0.85:
            unique_recommendations.insert(0, "Data quality improvements recommended before model training")
        
        return unique_recommendations
    
    def generate_quality_visualizations(self, df: pd.DataFrame, report: QualityReport, 
                                      save_path: Optional[str] = None) -> plt.Figure:
        """Generate comprehensive quality visualizations"""
        
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # Create a grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Overall quality score gauge
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_quality_gauge(ax1, report.overall_score)
        
        # 2. Quality check status
        ax2 = fig.add_subplot(gs[0, 1:3])
        self._plot_quality_checks(ax2, report.quality_checks)
        
        # 3. Missing data heatmap
        ax3 = fig.add_subplot(gs[0, 3])
        self._plot_missing_data(ax3, df)
        
        # 4. Price distribution
        ax4 = fig.add_subplot(gs[1, 0:2])
        if 'price_mad' in df.columns:
            df['price_mad'].dropna().hist(bins=50, ax=ax4, alpha=0.7)
            ax4.set_title('Price Distribution')
            ax4.set_xlabel('Price (MAD)')
            ax4.set_ylabel('Frequency')
        
        # 5. Surface area distribution
        ax5 = fig.add_subplot(gs[1, 2:4])
        if 'surface_m2' in df.columns:
            df['surface_m2'].dropna().hist(bins=50, ax=ax5, alpha=0.7)
            ax5.set_title('Surface Area Distribution')
            ax5.set_xlabel('Surface (m²)')
            ax5.set_ylabel('Frequency')
        
        # 6. City distribution
        ax6 = fig.add_subplot(gs[2, 0:2])
        if 'city_cleaned' in df.columns:
            city_counts = df['city_cleaned'].value_counts().head(10)
            city_counts.plot(kind='bar', ax=ax6)
            ax6.set_title('Top 10 Cities by Property Count')
            ax6.tick_params(axis='x', rotation=45)
        
        # 7. Property type distribution
        ax7 = fig.add_subplot(gs[2, 2:4])
        if 'property_type_cleaned' in df.columns:
            type_counts = df['property_type_cleaned'].value_counts()
            type_counts.plot(kind='pie', ax=ax7, autopct='%1.1f%%')
            ax7.set_title('Property Type Distribution')
        
        # 8. Correlation heatmap
        ax8 = fig.add_subplot(gs[3, :])
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]  # Limit to first 10
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax8)
            ax8.set_title('Feature Correlation Matrix')
        
        plt.suptitle(f'Data Quality Report - Overall Score: {report.overall_score:.2f}', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Quality visualizations saved to {save_path}")
        
        return fig
    
    def _plot_quality_gauge(self, ax, score):
        """Plot quality score as a gauge"""
        
        # Create a simple gauge visualization
        angles = np.linspace(0, np.pi, 100)
        
        # Background arc
        ax.plot(np.cos(angles), np.sin(angles), 'lightgray', linewidth=10)
        
        # Score arc
        score_angles = angles[:int(score * 100)]
        if score >= 0.8:
            color = 'green'
        elif score >= 0.6:
            color = 'orange'
        else:
            color = 'red'
        
        ax.plot(np.cos(score_angles), np.sin(score_angles), color, linewidth=10)
        
        # Score text
        ax.text(0, -0.2, f'{score:.2f}', ha='center', va='center', fontsize=20, fontweight='bold')
        ax.text(0, -0.4, 'Quality Score', ha='center', va='center', fontsize=12)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Overall Quality')
    
    def _plot_quality_checks(self, ax, quality_checks):
        """Plot quality check results"""
        
        check_names = [check.check_name.replace('_', ' ').title() for check in quality_checks]
        scores = [check.score for check in quality_checks]
        statuses = [check.status for check in quality_checks]
        
        # Color mapping
        colors = {'pass': 'green', 'warning': 'orange', 'fail': 'red'}
        bar_colors = [colors[status] for status in statuses]
        
        bars = ax.barh(check_names, scores, color=bar_colors, alpha=0.7)
        
        # Add score labels
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.2f}', ha='left', va='center', fontsize=8)
        
        ax.set_xlim(0, 1.0)
        ax.set_xlabel('Quality Score')
        ax.set_title('Quality Check Results')
        ax.tick_params(axis='y', labelsize=8)
    
    def _plot_missing_data(self, ax, df):
        """Plot missing data visualization"""
        
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        # Show top 10 columns with most missing data
        top_missing = missing_percentages.nlargest(10)
        
        if len(top_missing) > 0:
            top_missing.plot(kind='bar', ax=ax, color='coral')
            ax.set_title('Missing Data by Column')
            ax.set_ylabel('Missing %')
            ax.tick_params(axis='x', rotation=45, labelsize=8)
        else:
            ax.text(0.5, 0.5, 'No Missing Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Missing Data by Column')
    
    def save_report(self, report: QualityReport, filepath: str):
        """Save quality report to file"""
        
        import json
        from dataclasses import asdict
        
        # Convert report to dictionary
        report_dict = asdict(report)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        self.logger.info(f"Quality report saved to {filepath}")
    
    def load_report(self, filepath: str) -> QualityReport:
        """Load quality report from file"""
        
        import json
        
        with open(filepath, 'r') as f:
            report_dict = json.load(f)
        
        # Reconstruct QualityCheck objects
        quality_checks = []
        for check_dict in report_dict['quality_checks']:
            quality_checks.append(QualityCheck(**check_dict))
        
        report_dict['quality_checks'] = quality_checks
        
        return QualityReport(**report_dict)