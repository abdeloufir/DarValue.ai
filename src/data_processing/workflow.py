"""
Automated data processing workflow orchestration
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import json
from datetime import datetime
import traceback
from loguru import logger

from ..utils.monitoring import get_logger
from .deduplication import ListingDeduplicator
from .outlier_detection import OutlierDetector, DataQualityAssessment
from .coordinate_validation import CoordinateValidator, GeocodingService, CoordinateEnrichment
from .price_normalization import PriceNormalizer
from .text_cleaning import TextStandardizer
from .categorical_encoding import CategoricalEncoder
from .dataset_splitting import CityDatasetSplitter, SplitConfiguration
from .data_quality import DataQualityValidator
from .feature_engineering import FeatureEngineer


@dataclass
class ProcessingStepResult:
    """Result of a processing step"""
    step_name: str
    status: str  # 'success', 'failed', 'skipped'
    duration_seconds: float
    input_records: int
    output_records: int
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]


@dataclass
class WorkflowConfiguration:
    """Configuration for the data processing workflow"""
    # Step toggles
    enable_deduplication: bool = True
    enable_outlier_detection: bool = True
    enable_coordinate_validation: bool = True
    enable_price_normalization: bool = True
    enable_text_cleaning: bool = True
    enable_categorical_encoding: bool = True
    enable_feature_engineering: bool = True
    enable_dataset_splitting: bool = True
    enable_quality_validation: bool = True
    
    # Processing parameters
    outlier_removal_strategy: str = 'remove'  # 'remove', 'cap', 'transform'
    duplicate_threshold: float = 0.8
    geocoding_enabled: bool = True
    feature_selection_k: int = 50
    train_test_split_config: Optional[SplitConfiguration] = None
    
    # Output configuration
    output_dir: str = "data/processed"
    save_intermediate_results: bool = True
    generate_reports: bool = True
    create_visualizations: bool = True


class DataProcessingWorkflow:
    """Orchestrates the complete data cleaning and preprocessing pipeline"""
    
    def __init__(self, config: WorkflowConfiguration = WorkflowConfiguration()):
        self.config = config
        self.logger = get_logger('data_processing_workflow')
        
        # Initialize processing components
        self.deduplicator = ListingDeduplicator()
        self.outlier_detector = OutlierDetector()
        self.coordinate_validator = CoordinateValidator()
        self.geocoding_service = GeocodingService()
        self.coordinate_enricher = CoordinateEnrichment()
        self.price_normalizer = PriceNormalizer()
        self.text_standardizer = TextStandardizer()
        self.categorical_encoder = CategoricalEncoder()
        self.feature_engineer = FeatureEngineer()
        self.dataset_splitter = CityDatasetSplitter(self.config.output_dir)
        self.quality_validator = DataQualityValidator(f"{self.config.output_dir}/reports")
        self.quality_assessor = DataQualityAssessment()
        
        # Results tracking
        self.step_results = []
        self.workflow_start_time = None
        self.intermediate_data = {}
        
        # Setup output directories
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "intermediate").mkdir(exist_ok=True)
    
    def run_complete_workflow(self, df: pd.DataFrame, target_column: str = 'price_mad') -> Dict[str, Any]:
        """Run the complete data processing workflow"""
        
        self.workflow_start_time = datetime.now()
        self.logger.info(f"Starting complete data processing workflow with {len(df)} records")
        
        try:
            current_data = df.copy()
            
            # Step 1: Deduplication
            if self.config.enable_deduplication:
                current_data = self._run_deduplication_step(current_data)
            
            # Step 2: Outlier Detection and Removal
            if self.config.enable_outlier_detection:
                current_data = self._run_outlier_detection_step(current_data)
            
            # Step 3: Coordinate Validation and Geocoding
            if self.config.enable_coordinate_validation:
                current_data = self._run_coordinate_validation_step(current_data)
            
            # Step 4: Price Normalization
            if self.config.enable_price_normalization:
                current_data = self._run_price_normalization_step(current_data)
            
            # Step 5: Text Cleaning and Standardization
            if self.config.enable_text_cleaning:
                current_data = self._run_text_cleaning_step(current_data)
            
            # Step 6: Categorical Encoding
            if self.config.enable_categorical_encoding:
                current_data = self._run_categorical_encoding_step(current_data)
            
            # Step 7: Feature Engineering
            if self.config.enable_feature_engineering:
                current_data = self._run_feature_engineering_step(current_data, target_column)
            
            # Step 8: Data Quality Validation
            if self.config.enable_quality_validation:
                quality_report = self._run_quality_validation_step(current_data)
            
            # Step 9: Dataset Splitting
            city_splits = None
            if self.config.enable_dataset_splitting:
                city_splits = self._run_dataset_splitting_step(current_data, target_column)
            
            # Generate final workflow report
            workflow_report = self._generate_workflow_report(df, current_data, city_splits)
            
            # Save final results
            self._save_final_results(current_data, workflow_report)
            
            self.logger.info("Complete data processing workflow finished successfully")
            
            return {
                'final_data': current_data,
                'city_splits': city_splits,
                'workflow_report': workflow_report,
                'step_results': self.step_results
            }
            
        except Exception as e:
            self.logger.error(f"Workflow failed with error: {e}")
            self.logger.error(traceback.format_exc())
            
            # Save error report
            error_report = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'completed_steps': len(self.step_results),
                'step_results': self.step_results
            }
            
            error_path = self.output_dir / "error_report.json"
            with open(error_path, 'w') as f:
                json.dump(error_report, f, indent=2, default=str)
            
            raise
    
    def _run_deduplication_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run deduplication step"""
        step_start_time = datetime.now()
        step_name = "deduplication"
        
        try:
            self.logger.info(f"Starting {step_name} step")
            
            # Find duplicates
            duplicates = self.deduplicator.find_duplicates(df)
            
            # Remove duplicates
            df_deduplicated = self.deduplicator.remove_duplicates(
                df, duplicates, threshold=self.config.duplicate_threshold
            )
            
            # Calculate metrics
            removed_count = len(df) - len(df_deduplicated)
            metrics = {
                'duplicates_found': len(duplicates),
                'duplicates_removed': removed_count,
                'duplicate_rate': removed_count / len(df) if len(df) > 0 else 0
            }
            
            duration = (datetime.now() - step_start_time).total_seconds()
            
            self.step_results.append(ProcessingStepResult(
                step_name=step_name,
                status='success',
                duration_seconds=duration,
                input_records=len(df),
                output_records=len(df_deduplicated),
                errors=[],
                warnings=[],
                metrics=metrics
            ))
            
            if self.config.save_intermediate_results:
                self.intermediate_data[step_name] = df_deduplicated.copy()
            
            self.logger.info(f"Deduplication complete: removed {removed_count} duplicates")
            return df_deduplicated
            
        except Exception as e:
            duration = (datetime.now() - step_start_time).total_seconds()
            
            self.step_results.append(ProcessingStepResult(
                step_name=step_name,
                status='failed',
                duration_seconds=duration,
                input_records=len(df),
                output_records=len(df),
                errors=[str(e)],
                warnings=[],
                metrics={}
            ))
            
            self.logger.error(f"Deduplication step failed: {e}")
            return df
    
    def _run_outlier_detection_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run outlier detection and removal step"""
        step_start_time = datetime.now()
        step_name = "outlier_detection"
        
        try:
            self.logger.info(f"Starting {step_name} step")
            
            # Detect outliers
            outlier_results = self.outlier_detector.detect_outliers(df)
            
            # Remove outliers based on strategy
            df_clean = self.outlier_detector.remove_outliers(
                df, outlier_results, strategy=self.config.outlier_removal_strategy
            )
            
            # Calculate metrics
            removed_count = len(df) - len(df_clean)
            total_outliers = sum(len(result.outlier_indices) for result in outlier_results)
            
            metrics = {
                'outliers_detected': total_outliers,
                'records_removed': removed_count,
                'outlier_detection_methods': len(set(r.detection_method for r in outlier_results)),
                'outlier_rate': total_outliers / len(df) if len(df) > 0 else 0
            }
            
            duration = (datetime.now() - step_start_time).total_seconds()
            
            self.step_results.append(ProcessingStepResult(
                step_name=step_name,
                status='success',
                duration_seconds=duration,
                input_records=len(df),
                output_records=len(df_clean),
                errors=[],
                warnings=[],
                metrics=metrics
            ))
            
            if self.config.save_intermediate_results:
                self.intermediate_data[step_name] = df_clean.copy()
            
            self.logger.info(f"Outlier detection complete: processed {total_outliers} outliers")
            return df_clean
            
        except Exception as e:
            duration = (datetime.now() - step_start_time).total_seconds()
            
            self.step_results.append(ProcessingStepResult(
                step_name=step_name,
                status='failed',
                duration_seconds=duration,
                input_records=len(df),
                output_records=len(df),
                errors=[str(e)],
                warnings=[],
                metrics={}
            ))
            
            self.logger.error(f"Outlier detection step failed: {e}")
            return df
    
    def _run_coordinate_validation_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run coordinate validation and geocoding step"""
        step_start_time = datetime.now()
        step_name = "coordinate_validation"
        
        try:
            self.logger.info(f"Starting {step_name} step")
            
            # Validate existing coordinates
            validation_results = self.coordinate_validator.validate_coordinates(df)
            
            # Count valid coordinates
            valid_coords = sum(1 for r in validation_results.values() if r.is_valid)
            
            df_with_coords = df.copy()
            
            # Geocode missing coordinates if enabled
            geocoded_count = 0
            if self.config.geocoding_enabled:
                df_with_coords = self.geocoding_service.geocode_missing_coordinates(df_with_coords)
                
                # Apply fallback geocoding
                df_with_coords = self.geocoding_service.fallback_city_geocoding(df_with_coords)
                
                # Count successful geocodes
                geocoded_count = df_with_coords['geocoding_source'].notna().sum()
            
            # Add coordinate enrichment
            df_with_coords = self.coordinate_enricher.calculate_distance_features(df_with_coords)
            
            # Calculate metrics
            metrics = {
                'valid_coordinates': valid_coords,
                'invalid_coordinates': len(validation_results) - valid_coords,
                'geocoded_records': geocoded_count,
                'coordinate_validation_rate': valid_coords / len(validation_results) if validation_results else 0
            }
            
            duration = (datetime.now() - step_start_time).total_seconds()
            
            self.step_results.append(ProcessingStepResult(
                step_name=step_name,
                status='success',
                duration_seconds=duration,
                input_records=len(df),
                output_records=len(df_with_coords),
                errors=[],
                warnings=[],
                metrics=metrics
            ))
            
            if self.config.save_intermediate_results:
                self.intermediate_data[step_name] = df_with_coords.copy()
            
            self.logger.info(f"Coordinate validation complete: {valid_coords} valid, {geocoded_count} geocoded")
            return df_with_coords
            
        except Exception as e:
            duration = (datetime.now() - step_start_time).total_seconds()
            
            self.step_results.append(ProcessingStepResult(
                step_name=step_name,
                status='failed',
                duration_seconds=duration,
                input_records=len(df),
                output_records=len(df),
                errors=[str(e)],
                warnings=[],
                metrics={}
            ))
            
            self.logger.error(f"Coordinate validation step failed: {e}")
            return df
    
    def _run_price_normalization_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run price normalization step"""
        step_start_time = datetime.now()
        step_name = "price_normalization"
        
        try:
            self.logger.info(f"Starting {step_name} step")
            
            # Normalize prices
            df_normalized = self.price_normalizer.normalize_prices(df)
            
            # Calculate price metrics
            df_normalized = self.price_normalizer.calculate_price_metrics(df_normalized)
            
            # Validate price consistency
            validation_results = self.price_normalizer.validate_price_consistency(df_normalized)
            
            # Calculate metrics
            metrics = {
                'records_with_price': validation_results['records_with_price'],
                'records_with_price_per_m2': validation_results['records_with_price_per_m2'],
                'price_normalization_rate': validation_results['records_with_price'] / len(df) if len(df) > 0 else 0,
                'currency_distribution': validation_results['currency_distribution'],
                'average_confidence': validation_results['price_confidence_stats']['mean']
            }
            
            duration = (datetime.now() - step_start_time).total_seconds()
            
            self.step_results.append(ProcessingStepResult(
                step_name=step_name,
                status='success',
                duration_seconds=duration,
                input_records=len(df),
                output_records=len(df_normalized),
                errors=[],
                warnings=[],
                metrics=metrics
            ))
            
            if self.config.save_intermediate_results:
                self.intermediate_data[step_name] = df_normalized.copy()
            
            self.logger.info(f"Price normalization complete: {metrics['records_with_price']} prices normalized")
            return df_normalized
            
        except Exception as e:
            duration = (datetime.now() - step_start_time).total_seconds()
            
            self.step_results.append(ProcessingStepResult(
                step_name=step_name,
                status='failed',
                duration_seconds=duration,
                input_records=len(df),
                output_records=len(df),
                errors=[str(e)],
                warnings=[],
                metrics={}
            ))
            
            self.logger.error(f"Price normalization step failed: {e}")
            return df
    
    def _run_text_cleaning_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run text cleaning and standardization step"""
        step_start_time = datetime.now()
        step_name = "text_cleaning"
        
        try:
            self.logger.info(f"Starting {step_name} step")
            
            # Clean text fields
            df_cleaned = self.text_standardizer.clean_text_fields(df)
            
            # Standardize categorical values
            df_cleaned = self.text_standardizer.standardize_categorical_values(df_cleaned)
            
            # Generate text statistics
            text_stats = self.text_standardizer.generate_text_statistics(df_cleaned)
            
            # Calculate metrics
            metrics = {
                'text_fields_processed': len(text_stats['text_fields']),
                'cleaning_impact': text_stats['cleaning_impact'],
                'total_records': text_stats['total_records']
            }
            
            duration = (datetime.now() - step_start_time).total_seconds()
            
            self.step_results.append(ProcessingStepResult(
                step_name=step_name,
                status='success',
                duration_seconds=duration,
                input_records=len(df),
                output_records=len(df_cleaned),
                errors=[],
                warnings=[],
                metrics=metrics
            ))
            
            if self.config.save_intermediate_results:
                self.intermediate_data[step_name] = df_cleaned.copy()
            
            self.logger.info(f"Text cleaning complete: {metrics['text_fields_processed']} fields processed")
            return df_cleaned
            
        except Exception as e:
            duration = (datetime.now() - step_start_time).total_seconds()
            
            self.step_results.append(ProcessingStepResult(
                step_name=step_name,
                status='failed',
                duration_seconds=duration,
                input_records=len(df),
                output_records=len(df),
                errors=[str(e)],
                warnings=[],
                metrics={}
            ))
            
            self.logger.error(f"Text cleaning step failed: {e}")
            return df
    
    def _run_categorical_encoding_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run categorical encoding step"""
        step_start_time = datetime.now()
        step_name = "categorical_encoding"
        
        try:
            self.logger.info(f"Starting {step_name} step")
            
            # Encode categorical features
            df_encoded = self.categorical_encoder.encode_categorical_features(df)
            
            # Generate encoding report
            encoding_report = self.categorical_encoder.generate_encoding_report()
            
            # Calculate metrics
            metrics = {
                'total_features_created': encoding_report['total_features_created'],
                'encoding_methods_used': encoding_report['encoding_methods_used'],
                'original_columns': len(df.columns),
                'encoded_columns': len(df_encoded.columns)
            }
            
            duration = (datetime.now() - step_start_time).total_seconds()
            
            self.step_results.append(ProcessingStepResult(
                step_name=step_name,
                status='success',
                duration_seconds=duration,
                input_records=len(df),
                output_records=len(df_encoded),
                errors=[],
                warnings=[],
                metrics=metrics
            ))
            
            if self.config.save_intermediate_results:
                self.intermediate_data[step_name] = df_encoded.copy()
                
                # Save encoders
                encoder_path = self.output_dir / "intermediate" / "categorical_encoders.pkl"
                self.categorical_encoder.save_encoders(str(encoder_path))
            
            self.logger.info(f"Categorical encoding complete: {metrics['total_features_created']} features created")
            return df_encoded
            
        except Exception as e:
            duration = (datetime.now() - step_start_time).total_seconds()
            
            self.step_results.append(ProcessingStepResult(
                step_name=step_name,
                status='failed',
                duration_seconds=duration,
                input_records=len(df),
                output_records=len(df),
                errors=[str(e)],
                warnings=[],
                metrics={}
            ))
            
            self.logger.error(f"Categorical encoding step failed: {e}")
            return df
    
    def _run_feature_engineering_step(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Run feature engineering step"""
        step_start_time = datetime.now()
        step_name = "feature_engineering"
        
        try:
            self.logger.info(f"Starting {step_name} step")
            
            # Engineer features
            engineered_result = self.feature_engineer.engineer_features(df, target_column)
            
            # Select top features if configured
            if self.config.feature_selection_k:
                df_selected, selected_features = self.feature_engineer.select_top_features(
                    engineered_result.data, target_column, self.config.feature_selection_k
                )
                engineered_result.data = df_selected
            
            # Generate feature summary
            feature_summary = self.feature_engineer.get_feature_summary(engineered_result)
            
            # Calculate metrics
            metrics = {
                'original_features': len(engineered_result.original_features),
                'engineered_features': len(engineered_result.engineered_features),
                'final_features': engineered_result.data.shape[1],
                'feature_types': feature_summary['feature_types'],
                'memory_usage_mb': feature_summary['memory_usage_mb']
            }
            
            duration = (datetime.now() - step_start_time).total_seconds()
            
            self.step_results.append(ProcessingStepResult(
                step_name=step_name,
                status='success',
                duration_seconds=duration,
                input_records=len(df),
                output_records=len(engineered_result.data),
                errors=[],
                warnings=[],
                metrics=metrics
            ))
            
            if self.config.save_intermediate_results:
                self.intermediate_data[step_name] = engineered_result.data.copy()
                
                # Save feature metadata
                feature_metadata_path = self.output_dir / "intermediate" / "feature_metadata.json"
                with open(feature_metadata_path, 'w') as f:
                    json.dump(feature_summary, f, indent=2, default=str)
            
            self.logger.info(f"Feature engineering complete: {metrics['engineered_features']} features created")
            return engineered_result.data
            
        except Exception as e:
            duration = (datetime.now() - step_start_time).total_seconds()
            
            self.step_results.append(ProcessingStepResult(
                step_name=step_name,
                status='failed',
                duration_seconds=duration,
                input_records=len(df),
                output_records=len(df),
                errors=[str(e)],
                warnings=[],
                metrics={}
            ))
            
            self.logger.error(f"Feature engineering step failed: {e}")
            return df
    
    def _run_quality_validation_step(self, df: pd.DataFrame) -> Any:
        """Run data quality validation step"""
        step_start_time = datetime.now()
        step_name = "quality_validation"
        
        try:
            self.logger.info(f"Starting {step_name} step")
            
            # Validate data quality
            quality_report = self.quality_validator.validate_data_quality(df)
            
            # Generate visualizations if configured
            if self.config.create_visualizations:
                viz_path = self.output_dir / "reports" / "quality_visualizations.png"
                self.quality_validator.generate_quality_visualizations(
                    df, quality_report, str(viz_path)
                )
            
            # Save quality report
            if self.config.generate_reports:
                report_path = self.output_dir / "reports" / "quality_report.json"
                self.quality_validator.save_report(quality_report, str(report_path))
            
            # Calculate metrics
            metrics = {
                'overall_quality_score': quality_report.overall_score,
                'total_quality_checks': len(quality_report.quality_checks),
                'passed_checks': len([c for c in quality_report.quality_checks if c.status == 'pass']),
                'failed_checks': len([c for c in quality_report.quality_checks if c.status == 'fail']),
                'warning_checks': len([c for c in quality_report.quality_checks if c.status == 'warning'])
            }
            
            duration = (datetime.now() - step_start_time).total_seconds()
            
            self.step_results.append(ProcessingStepResult(
                step_name=step_name,
                status='success',
                duration_seconds=duration,
                input_records=len(df),
                output_records=len(df),
                errors=[],
                warnings=[],
                metrics=metrics
            ))
            
            self.logger.info(f"Quality validation complete: score {quality_report.overall_score:.2f}")
            return quality_report
            
        except Exception as e:
            duration = (datetime.now() - step_start_time).total_seconds()
            
            self.step_results.append(ProcessingStepResult(
                step_name=step_name,
                status='failed',
                duration_seconds=duration,
                input_records=len(df),
                output_records=len(df),
                errors=[str(e)],
                warnings=[],
                metrics={}
            ))
            
            self.logger.error(f"Quality validation step failed: {e}")
            return None
    
    def _run_dataset_splitting_step(self, df: pd.DataFrame, target_column: str) -> Any:
        """Run dataset splitting step"""
        step_start_time = datetime.now()
        step_name = "dataset_splitting"
        
        try:
            self.logger.info(f"Starting {step_name} step")
            
            # Use provided configuration or default
            split_config = self.config.train_test_split_config or SplitConfiguration()
            
            # Split datasets
            city_splits = self.dataset_splitter.split_datasets(df, split_config, target_column)
            
            # Calculate metrics
            total_train = sum(split.split_stats['train_samples'] for split in city_splits.values())
            total_val = sum(split.split_stats['val_samples'] for split in city_splits.values())
            total_test = sum(split.split_stats['test_samples'] for split in city_splits.values())
            
            metrics = {
                'cities_processed': len(city_splits) - 1,  # Exclude 'combined'
                'total_train_samples': total_train,
                'total_val_samples': total_val,
                'total_test_samples': total_test,
                'average_train_size': total_train / max(1, len(city_splits) - 1)
            }
            
            duration = (datetime.now() - step_start_time).total_seconds()
            
            self.step_results.append(ProcessingStepResult(
                step_name=step_name,
                status='success',
                duration_seconds=duration,
                input_records=len(df),
                output_records=total_train + total_val + total_test,
                errors=[],
                warnings=[],
                metrics=metrics
            ))
            
            self.logger.info(f"Dataset splitting complete: {metrics['cities_processed']} cities processed")
            return city_splits
            
        except Exception as e:
            duration = (datetime.now() - step_start_time).total_seconds()
            
            self.step_results.append(ProcessingStepResult(
                step_name=step_name,
                status='failed',
                duration_seconds=duration,
                input_records=len(df),
                output_records=len(df),
                errors=[str(e)],
                warnings=[],
                metrics={}
            ))
            
            self.logger.error(f"Dataset splitting step failed: {e}")
            return None
    
    def _generate_workflow_report(self, original_df: pd.DataFrame, 
                                 final_df: pd.DataFrame, city_splits: Any) -> Dict[str, Any]:
        """Generate comprehensive workflow report"""
        
        total_duration = (datetime.now() - self.workflow_start_time).total_seconds()
        
        report = {
            'workflow_metadata': {
                'start_time': self.workflow_start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration_seconds': total_duration,
                'configuration': asdict(self.config)
            },
            'data_transformation': {
                'original_records': len(original_df),
                'final_records': len(final_df),
                'original_columns': len(original_df.columns),
                'final_columns': len(final_df.columns),
                'data_reduction_rate': (len(original_df) - len(final_df)) / len(original_df) if len(original_df) > 0 else 0,
                'feature_expansion_rate': (len(final_df.columns) - len(original_df.columns)) / len(original_df.columns) if len(original_df.columns) > 0 else 0
            },
            'step_summary': {
                'total_steps': len(self.step_results),
                'successful_steps': len([s for s in self.step_results if s.status == 'success']),
                'failed_steps': len([s for s in self.step_results if s.status == 'failed']),
                'total_processing_time': sum(s.duration_seconds for s in self.step_results)
            },
            'step_details': [asdict(step) for step in self.step_results],
            'city_splits_summary': {},
            'recommendations': []
        }
        
        # Add city splits summary
        if city_splits:
            report['city_splits_summary'] = {
                'cities_available': len(city_splits) - 1,  # Exclude combined
                'total_training_samples': sum(split.split_stats['train_samples'] for split in city_splits.values()),
                'cities': list(city_splits.keys())
            }
        
        # Generate recommendations based on results
        report['recommendations'] = self._generate_workflow_recommendations()
        
        return report
    
    def _generate_workflow_recommendations(self) -> List[str]:
        """Generate recommendations based on workflow results"""
        
        recommendations = []
        
        # Check for failed steps
        failed_steps = [s for s in self.step_results if s.status == 'failed']
        if failed_steps:
            recommendations.append(f"Address {len(failed_steps)} failed processing steps before proceeding to model training")
        
        # Check data quality
        quality_step = next((s for s in self.step_results if s.step_name == 'quality_validation'), None)
        if quality_step and 'overall_quality_score' in quality_step.metrics:
            quality_score = quality_step.metrics['overall_quality_score']
            if quality_score < 0.7:
                recommendations.append("Data quality score is below 0.7 - consider additional data cleaning")
            elif quality_score < 0.85:
                recommendations.append("Data quality is acceptable but could be improved")
        
        # Check data reduction
        dedup_step = next((s for s in self.step_results if s.step_name == 'deduplication'), None)
        if dedup_step and 'duplicate_rate' in dedup_step.metrics:
            if dedup_step.metrics['duplicate_rate'] > 0.2:
                recommendations.append("High duplicate rate detected - review data collection process")
        
        # Check feature engineering
        feature_step = next((s for s in self.step_results if s.step_name == 'feature_engineering'), None)
        if feature_step and 'engineered_features' in feature_step.metrics:
            if feature_step.metrics['engineered_features'] < 10:
                recommendations.append("Consider adding more engineered features for better model performance")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Data processing completed successfully - ready for model training")
        
        return recommendations
    
    def _save_final_results(self, final_df: pd.DataFrame, workflow_report: Dict[str, Any]):
        """Save final processing results"""
        
        # Save final processed dataset
        final_data_path = self.output_dir / "final_processed_data.csv"
        final_df.to_csv(final_data_path, index=False)
        
        # Save workflow report
        report_path = self.output_dir / "reports" / "workflow_report.json"
        with open(report_path, 'w') as f:
            json.dump(workflow_report, f, indent=2, default=str)
        
        # Save intermediate results if configured
        if self.config.save_intermediate_results and self.intermediate_data:
            for step_name, data in self.intermediate_data.items():
                step_path = self.output_dir / "intermediate" / f"{step_name}_data.csv"
                data.to_csv(step_path, index=False)
        
        self.logger.info(f"Final results saved to {self.output_dir}")
    
    def get_step_result(self, step_name: str) -> Optional[ProcessingStepResult]:
        """Get result for a specific processing step"""
        
        for step in self.step_results:
            if step.step_name == step_name:
                return step
        return None
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing workflow"""
        
        if not self.step_results:
            return {"status": "not_started"}
        
        successful_steps = [s for s in self.step_results if s.status == 'success']
        failed_steps = [s for s in self.step_results if s.status == 'failed']
        
        return {
            "total_steps": len(self.step_results),
            "successful_steps": len(successful_steps),
            "failed_steps": len(failed_steps),
            "success_rate": len(successful_steps) / len(self.step_results) if self.step_results else 0,
            "total_duration": sum(s.duration_seconds for s in self.step_results),
            "status": "completed" if not failed_steps else "completed_with_errors"
        }