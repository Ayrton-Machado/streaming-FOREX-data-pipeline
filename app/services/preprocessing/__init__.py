"""
Preprocessing Module - Pipeline de preprocessamento avançado

Etapa 4: Módulo completo de preprocessamento para pipeline ML-ready
Inclui normalização, feature engineering, filtros de mercado e data quality

Filosofia SLC:
- Simple: Interfaces claras e uso direto
- Lovable: Pipeline otimizado para FOREX
- Complete: Cobertura total do preprocessamento
"""

from .normalizer import DataNormalizer, get_data_normalizer
from .feature_engineer import FeatureEngineer, get_feature_engineer
from .market_filters import MarketFilters, get_market_filters
from .data_quality import DataQualityService, get_data_quality_service
from .pipeline import PreprocessingPipeline, get_preprocessing_pipeline

__all__ = [
    'DataNormalizer',
    'get_data_normalizer',
    'FeatureEngineer', 
    'get_feature_engineer',
    'MarketFilters',
    'get_market_filters',
    'DataQualityService',
    'get_data_quality_service',
    'PreprocessingPipeline',
    'get_preprocessing_pipeline'
]