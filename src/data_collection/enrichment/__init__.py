"""
Enrichment package initialization
"""

from .geospatial_enricher import GeospatialEnricher, MoroccanDataEnricher, LocationEnrichment

__all__ = [
    'GeospatialEnricher',
    'MoroccanDataEnricher', 
    'LocationEnrichment'
]