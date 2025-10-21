"""
Logging and monitoring setup for DarValue.ai
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any
import json
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import functools


class LoggingManager:
    """Manages application logging configuration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._configured = False
    
    def setup_logging(self):
        """Configure loguru logging"""
        if self._configured:
            return
        
        # Remove default handler
        logger.remove()
        
        # Get configuration
        log_level = self.config.get('level', 'INFO')
        log_format = self.config.get('format', 
            "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}")
        log_file = self.config.get('log_file', 'logs/darvalue.log')
        rotation = self.config.get('rotation', '1 week')
        retention = self.config.get('retention', '30 days')
        
        # Create logs directory
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Console handler
        logger.add(
            sys.stderr,
            format=log_format,
            level=log_level,
            colorize=True
        )
        
        # File handler with rotation
        logger.add(
            log_file,
            format=log_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        # JSON file handler for structured logs
        json_log_file = log_file.replace('.log', '_structured.jsonl')
        logger.add(
            json_log_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name} | {function} | {line} | {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            serialize=True  # JSON format
        )
        
        self._configured = True
        logger.info("Logging system initialized")
    
    def get_logger(self, name: str):
        """Get a logger instance for a specific module"""
        return logger.bind(name=name)


class MetricsManager:
    """Manages Prometheus metrics for monitoring"""
    
    def __init__(self):
        # Scraping metrics
        self.listings_scraped = Counter(
            'darvalue_listings_scraped_total',
            'Total number of listings scraped',
            ['platform', 'city', 'status']
        )
        
        self.scraping_duration = Histogram(
            'darvalue_scraping_duration_seconds',
            'Time spent scraping',
            ['platform', 'city']
        )
        
        self.scraping_errors = Counter(
            'darvalue_scraping_errors_total',
            'Total scraping errors',
            ['platform', 'city', 'error_type']
        )
        
        # Database metrics
        self.database_operations = Counter(
            'darvalue_database_operations_total',
            'Database operations',
            ['operation', 'table', 'status']
        )
        
        self.database_query_duration = Histogram(
            'darvalue_database_query_duration_seconds',
            'Database query duration',
            ['operation', 'table']
        )
        
        # Image processing metrics
        self.images_processed = Counter(
            'darvalue_images_processed_total',
            'Images processed',
            ['status', 'room_type']
        )
        
        self.image_processing_duration = Histogram(
            'darvalue_image_processing_duration_seconds',
            'Image processing duration'
        )
        
        # System metrics
        self.active_scrapers = Gauge(
            'darvalue_active_scrapers',
            'Number of active scrapers'
        )
        
        self.pipeline_runs = Counter(
            'darvalue_pipeline_runs_total',
            'Total pipeline runs',
            ['status']
        )
        
        # Enrichment metrics
        self.enrichment_operations = Counter(
            'darvalue_enrichment_operations_total',
            'Enrichment operations',
            ['type', 'status']
        )
        
        self.api_calls = Counter(
            'darvalue_external_api_calls_total',
            'External API calls',
            ['service', 'status']
        )
    
    def start_metrics_server(self, port: int = 8000):
        """Start Prometheus metrics server"""
        try:
            start_http_server(port)
            logger.info(f"Metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")


class PerformanceMonitor:
    """Performance monitoring and profiling"""
    
    def __init__(self, metrics_manager: MetricsManager):
        self.metrics = metrics_manager
        self._operation_timers = {}
    
    def time_operation(self, operation_name: str, labels: Optional[Dict] = None):
        """Decorator to time operations"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                operation_id = f"{operation_name}_{id(start_time)}"
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Record success metrics
                    if operation_name.startswith('scraping'):
                        self.metrics.scraping_duration.labels(
                            platform=labels.get('platform', 'unknown'),
                            city=labels.get('city', 'unknown')
                        ).observe(duration)
                    
                    elif operation_name.startswith('database'):
                        self.metrics.database_query_duration.labels(
                            operation=labels.get('operation', 'unknown'),
                            table=labels.get('table', 'unknown')
                        ).observe(duration)
                        
                        self.metrics.database_operations.labels(
                            operation=labels.get('operation', 'unknown'),
                            table=labels.get('table', 'unknown'),
                            status='success'
                        ).inc()
                    
                    elif operation_name.startswith('image'):
                        self.metrics.image_processing_duration.observe(duration)
                    
                    logger.debug(f"{operation_name} completed in {duration:.2f}s")
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    
                    # Record error metrics
                    if operation_name.startswith('scraping'):
                        self.metrics.scraping_errors.labels(
                            platform=labels.get('platform', 'unknown'),
                            city=labels.get('city', 'unknown'),
                            error_type=type(e).__name__
                        ).inc()
                    
                    elif operation_name.startswith('database'):
                        self.metrics.database_operations.labels(
                            operation=labels.get('operation', 'unknown'),
                            table=labels.get('table', 'unknown'),
                            status='error'
                        ).inc()
                    
                    logger.error(f"{operation_name} failed after {duration:.2f}s: {e}")
                    raise
                    
            return wrapper
        return decorator
    
    def record_scraping_result(self, platform: str, city: str, status: str, count: int = 1):
        """Record scraping results"""
        self.metrics.listings_scraped.labels(
            platform=platform,
            city=city,
            status=status
        ).inc(count)
    
    def record_image_processing(self, status: str, room_type: str = 'unknown'):
        """Record image processing results"""
        self.metrics.images_processed.labels(
            status=status,
            room_type=room_type
        ).inc()
    
    def record_enrichment_operation(self, enrichment_type: str, status: str):
        """Record enrichment operations"""
        self.metrics.enrichment_operations.labels(
            type=enrichment_type,
            status=status
        ).inc()
    
    def record_api_call(self, service: str, status: str):
        """Record external API calls"""
        self.metrics.api_calls.labels(
            service=service,
            status=status
        ).inc()
    
    def set_active_scrapers(self, count: int):
        """Set number of active scrapers"""
        self.metrics.active_scrapers.set(count)
    
    def record_pipeline_run(self, status: str):
        """Record pipeline execution"""
        self.metrics.pipeline_runs.labels(status=status).inc()


class AlertManager:
    """Simple alerting system for critical issues"""
    
    def __init__(self):
        self.alert_thresholds = {
            'error_rate': 0.1,  # 10% error rate
            'scraping_failures': 5,  # 5 consecutive failures
            'database_errors': 3,  # 3 database errors
            'disk_space': 0.9  # 90% disk usage
        }
        self.alert_counts = {}
    
    def check_error_rate(self, platform: str, errors: int, total: int):
        """Check if error rate exceeds threshold"""
        if total == 0:
            return
        
        error_rate = errors / total
        if error_rate > self.alert_thresholds['error_rate']:
            self._send_alert(
                'HIGH_ERROR_RATE',
                f"High error rate for {platform}: {error_rate:.1%} ({errors}/{total})"
            )
    
    def check_scraping_failures(self, platform: str, city: str):
        """Track consecutive scraping failures"""
        key = f"{platform}_{city}"
        self.alert_counts[key] = self.alert_counts.get(key, 0) + 1
        
        if self.alert_counts[key] >= self.alert_thresholds['scraping_failures']:
            self._send_alert(
                'SCRAPING_FAILURES',
                f"Consecutive scraping failures for {platform} in {city}: {self.alert_counts[key]}"
            )
    
    def reset_failure_count(self, platform: str, city: str):
        """Reset failure count on success"""
        key = f"{platform}_{city}"
        self.alert_counts[key] = 0
    
    def _send_alert(self, alert_type: str, message: str):
        """Send alert (placeholder for actual alerting)"""
        logger.critical(f"ALERT [{alert_type}]: {message}")
        
        # In production, you would integrate with:
        # - Email notifications
        # - Slack/Discord webhooks
        # - SMS alerts
        # - PagerDuty
        # - etc.


class MonitoringSystem:
    """Main monitoring system that integrates all components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.logging_manager = LoggingManager(self.config.get('logging', {}))
        self.metrics_manager = MetricsManager()
        self.performance_monitor = PerformanceMonitor(self.metrics_manager)
        self.alert_manager = AlertManager()
        
        self._initialized = False
    
    def initialize(self):
        """Initialize the monitoring system"""
        if self._initialized:
            return
        
        # Setup logging
        self.logging_manager.setup_logging()
        
        # Start metrics server if enabled
        if self.config.get('enable_monitoring', False):
            metrics_port = self.config.get('prometheus_port', 8000)
            self.metrics_manager.start_metrics_server(metrics_port)
        
        self._initialized = True
        logger.info("Monitoring system initialized")
    
    def get_logger(self, name: str):
        """Get a logger for a module"""
        return self.logging_manager.get_logger(name)
    
    def get_performance_monitor(self) -> PerformanceMonitor:
        """Get performance monitor"""
        return self.performance_monitor
    
    def get_alert_manager(self) -> AlertManager:
        """Get alert manager"""
        return self.alert_manager
    
    def shutdown(self):
        """Shutdown monitoring system"""
        logger.info("Shutting down monitoring system")


# Global monitoring instance
_monitoring_system: Optional[MonitoringSystem] = None


def initialize_monitoring(config: Optional[Dict[str, Any]] = None):
    """Initialize global monitoring system"""
    global _monitoring_system
    _monitoring_system = MonitoringSystem(config)
    _monitoring_system.initialize()


def get_monitoring() -> MonitoringSystem:
    """Get global monitoring system"""
    if _monitoring_system is None:
        raise RuntimeError("Monitoring system not initialized. Call initialize_monitoring() first.")
    return _monitoring_system


def get_logger(name: str):
    """Get a logger instance"""
    if _monitoring_system:
        return _monitoring_system.get_logger(name)
    else:
        # Fallback to basic loguru logger
        return logger.bind(name=name)


def monitor_performance(operation_name: str, **labels):
    """Decorator for performance monitoring"""
    if _monitoring_system:
        return _monitoring_system.get_performance_monitor().time_operation(operation_name, labels)
    else:
        # No-op decorator if monitoring not initialized
        def decorator(func):
            return func
        return decorator