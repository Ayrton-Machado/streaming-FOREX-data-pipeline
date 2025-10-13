"""
Complete Preprocessing Pipeline - Orchestração de todo o preprocessamento
Etapa 4: Pipeline completo integrando todos os serviços de preprocessamento
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from app.domain.schemas import ValidatedOHLCV, MLReadyCandle
from app.services.preprocessing.data_quality import DataQualityService, DataQualityMetrics
from app.services.preprocessing.normalizer import DataNormalizer
from app.services.preprocessing.feature_engineer import FeatureEngineer
from app.services.preprocessing.market_filters import MarketFilters

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Pipeline completo de preprocessamento para dados FOREX
    
    Filosofia SLC:
    - Simple: Interface única para todo o preprocessamento
    - Lovable: Pipeline otimizado e configurável
    - Complete: Cobertura total com relatórios detalhados
    
    Pipeline Flow:
    1. Data Quality → Validação e limpeza
    2. Market Filters → Features de contexto de mercado
    3. Feature Engineer → Indicadores técnicos
    4. Data Normalizer → Normalização para ML
    """
    
    def __init__(self):
        self.data_quality = DataQualityService()
        self.market_filters = MarketFilters()
        self.feature_engineer = FeatureEngineer()
        self.data_normalizer = DataNormalizer()
        
        # Estatísticas do pipeline
        self.pipeline_stats = {
            'total_processed': 0,
            'quality_issues_found': 0,
            'features_engineered': 0,
            'normalization_applied': 0,
            'processing_time_ms': 0
        }
        
        logger.info("PreprocessingPipeline inicializado com todos os serviços")
    
    def process_full_pipeline(
        self,
        candles: List[ValidatedOHLCV],
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[MLReadyCandle], Dict[str, Any]]:
        """
        Executa pipeline completo de preprocessamento
        
        Args:
            candles: Lista de velas validadas
            config: Configurações opcionais do pipeline
            
        Returns:
            Tupla (ml_ready_candles, pipeline_report)
        """
        start_time = datetime.now()
        
        if not candles:
            logger.warning("Lista de velas vazia")
            return [], {"error": "Lista vazia"}
        
        # Configuração padrão
        if config is None:
            config = self._get_default_config()
        
        logger.info(f"🚀 Iniciando pipeline completo: {len(candles)} velas")
        
        pipeline_report = {
            'input_count': len(candles),
            'stages_completed': [],
            'stages_failed': [],
            'quality_metrics': None,
            'processing_time_ms': 0,
            'final_count': 0
        }
        
        try:
            # Estágio 1: Data Quality
            if config.get('enable_data_quality', True):
                candles, quality_metrics = self._run_data_quality_stage(
                    candles, config.get('data_quality', {})
                )
                pipeline_report['quality_metrics'] = quality_metrics
                pipeline_report['stages_completed'].append('data_quality')
                logger.info(f"✅ Stage 1: Data Quality - {len(candles)} velas válidas")
            
            # Estágio 2: Market Filters
            if config.get('enable_market_filters', True):
                ml_candles = self._run_market_filters_stage(
                    candles, config.get('market_filters', {})
                )
                pipeline_report['stages_completed'].append('market_filters')
                logger.info(f"✅ Stage 2: Market Filters - Features de mercado adicionadas")
            else:
                # Converte para MLReadyCandle sem market features
                ml_candles = self._convert_to_ml_ready(candles)
            
            # Estágio 3: Feature Engineering
            if config.get('enable_feature_engineering', True):
                ml_candles = self._run_feature_engineering_stage(
                    ml_candles, config.get('feature_engineering', {})
                )
                pipeline_report['stages_completed'].append('feature_engineering')
                logger.info(f"✅ Stage 3: Feature Engineering - Indicadores técnicos calculados")
            
            # Estágio 4: Normalization
            if config.get('enable_normalization', True):
                ml_candles = self._run_normalization_stage(
                    ml_candles, config.get('normalization', {})
                )
                pipeline_report['stages_completed'].append('normalization')
                logger.info(f"✅ Stage 4: Normalization - Dados normalizados para ML")
            
            # Finalização
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000
            
            pipeline_report.update({
                'final_count': len(ml_candles),
                'processing_time_ms': processing_time,
                'success_rate': len(ml_candles) / len(candles) * 100,
                'avg_features_per_candle': self._count_avg_features(ml_candles)
            })
            
            # Atualiza estatísticas
            self.pipeline_stats['total_processed'] += len(candles)
            self.pipeline_stats['processing_time_ms'] += processing_time
            
            logger.info(f"🎉 Pipeline completo finalizado: {len(ml_candles)} MLReadyCandles")
            logger.info(f"⚡ Tempo de processamento: {processing_time:.1f}ms")
            
            return ml_candles, pipeline_report
            
        except Exception as e:
            logger.error(f"❌ Erro no pipeline: {str(e)}")
            pipeline_report['stages_failed'].append(f"Pipeline error: {str(e)}")
            return [], pipeline_report
    
    def process_data_quality_only(
        self,
        candles: List[ValidatedOHLCV],
        auto_correct: bool = True,
        strict_mode: bool = False
    ) -> Tuple[List[ValidatedOHLCV], DataQualityMetrics]:
        """Executa apenas o estágio de Data Quality"""
        return self.data_quality.validate_and_clean(candles, auto_correct, strict_mode)
    
    def process_market_features_only(
        self,
        candles: List[ValidatedOHLCV],
        filters: Optional[List[str]] = None
    ) -> List[MLReadyCandle]:
        """Executa apenas Market Filters"""
        return self.market_filters.apply_market_filters(candles, filters)
    
    def process_technical_indicators_only(
        self,
        ml_candles: List[MLReadyCandle],
        indicators: Optional[List[str]] = None
    ) -> List[MLReadyCandle]:
        """Executa apenas Feature Engineering"""
        return self.feature_engineer.engineer_features(ml_candles, indicators)
    
    def process_normalization_only(
        self,
        ml_candles: List[MLReadyCandle],
        method: str = 'minmax',
        feature_columns: Optional[List[str]] = None
    ) -> List[MLReadyCandle]:
        """Executa apenas Normalization"""
        return self.data_normalizer.normalize_candles(ml_candles, method, feature_columns)
    
    def get_pipeline_summary(self, candles: List[ValidatedOHLCV]) -> Dict[str, Any]:
        """
        Retorna resumo completo do que o pipeline faria
        sem executar o processamento
        """
        summary = {
            'input_analysis': {
                'total_candles': len(candles),
                'time_range': {
                    'start': candles[0].timestamp.isoformat() if candles else None,
                    'end': candles[-1].timestamp.isoformat() if candles else None
                },
                'data_sources': list(set(c.data_source for c in candles)),
                'symbols': list(set(c.raw_symbol for c in candles))
            },
            'pipeline_stages': {
                'data_quality': {
                    'description': 'Validação e limpeza de dados',
                    'expected_outputs': ['quality_metrics', 'cleaned_data']
                },
                'market_filters': {
                    'description': 'Features de contexto de mercado',
                    'expected_outputs': ['session_features', 'liquidity_features', 'time_features']
                },
                'feature_engineering': {
                    'description': 'Indicadores técnicos',
                    'expected_outputs': ['technical_indicators', 'trend_features']
                },
                'normalization': {
                    'description': 'Normalização para ML',
                    'expected_outputs': ['normalized_features', 'ml_ready_data']
                }
            },
            'estimated_features': self._estimate_feature_count(),
            'estimated_processing_time_ms': len(candles) * 2.5  # Estimativa baseada em benchmarks
        }
        
        return summary
    
    def _run_data_quality_stage(
        self,
        candles: List[ValidatedOHLCV],
        config: Dict[str, Any]
    ) -> Tuple[List[ValidatedOHLCV], DataQualityMetrics]:
        """Executa estágio de Data Quality"""
        auto_correct = config.get('auto_correct', True)
        strict_mode = config.get('strict_mode', False)
        
        cleaned_candles, metrics = self.data_quality.validate_and_clean(
            candles, auto_correct, strict_mode
        )
        
        self.pipeline_stats['quality_issues_found'] += len(self.data_quality.issues_found)
        return cleaned_candles, metrics
    
    def _run_market_filters_stage(
        self,
        candles: List[ValidatedOHLCV],
        config: Dict[str, Any]
    ) -> List[MLReadyCandle]:
        """Executa estágio de Market Filters"""
        filters = config.get('filters', None)
        return self.market_filters.apply_market_filters(candles, filters)
    
    def _run_feature_engineering_stage(
        self,
        ml_candles: List[MLReadyCandle],
        config: Dict[str, Any]
    ) -> List[MLReadyCandle]:
        """Executa estágio de Feature Engineering"""
        indicators = config.get('indicators', None)
        enhanced_candles = self.feature_engineer.engineer_features(ml_candles, indicators)
        
        self.pipeline_stats['features_engineered'] += len(enhanced_candles)
        return enhanced_candles
    
    def _run_normalization_stage(
        self,
        ml_candles: List[MLReadyCandle],
        config: Dict[str, Any]
    ) -> List[MLReadyCandle]:
        """Executa estágio de Normalization"""
        method = config.get('method', 'minmax')
        feature_columns = config.get('feature_columns', None)
        
        normalized_candles = self.data_normalizer.normalize_candles(
            ml_candles, method, feature_columns
        )
        
        self.pipeline_stats['normalization_applied'] += len(normalized_candles)
        return normalized_candles
    
    def _convert_to_ml_ready(self, candles: List[ValidatedOHLCV]) -> List[MLReadyCandle]:
        """Converte ValidatedOHLCV para MLReadyCandle básico"""
        ml_candles = []
        
        for candle in candles:
            ml_candle = MLReadyCandle(
                timestamp=candle.timestamp,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
                data_source=candle.data_source,
                raw_symbol=candle.raw_symbol,
                validation=candle.validation,
                is_outlier=candle.is_outlier,
                is_gap_fill=candle.is_gap_fill,
                normalized_features={},
                technical_indicators={},
                market_features={}
            )
            ml_candles.append(ml_candle)
        
        return ml_candles
    
    def _count_avg_features(self, ml_candles: List[MLReadyCandle]) -> float:
        """Conta média de features por candle"""
        if not ml_candles:
            return 0.0
        
        total_features = 0
        for candle in ml_candles:
            total_features += len(candle.normalized_features)
            total_features += len(candle.technical_indicators)
            total_features += len(candle.market_features)
        
        return total_features / len(ml_candles)
    
    def _estimate_feature_count(self) -> Dict[str, int]:
        """Estima quantidade de features que serão geradas"""
        return {
            'normalized_features': 6,  # OHLCV + return
            'technical_indicators': 15,  # SMA, EMA, RSI, MACD, etc.
            'market_features': 20,  # Sessões, liquidez, tempo, etc.
            'total_estimated': 41
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuração padrão do pipeline"""
        return {
            'enable_data_quality': True,
            'enable_market_filters': True,
            'enable_feature_engineering': True,
            'enable_normalization': True,
            'data_quality': {
                'auto_correct': True,
                'strict_mode': False
            },
            'market_filters': {
                'filters': ['market_sessions', 'volatility_regime', 'gap_detection', 'weekend_filter']
            },
            'feature_engineering': {
                'indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr']
            },
            'normalization': {
                'method': 'minmax',
                'feature_columns': None
            }
        }
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do pipeline"""
        return {
            **self.pipeline_stats,
            'avg_processing_time_per_candle': (
                self.pipeline_stats['processing_time_ms'] / 
                max(1, self.pipeline_stats['total_processed'])
            ),
            'services_status': {
                'data_quality': 'active',
                'market_filters': 'active', 
                'feature_engineer': 'active',
                'data_normalizer': 'active'
            }
        }
    
    def reset_stats(self):
        """Reset das estatísticas do pipeline"""
        self.pipeline_stats = {
            'total_processed': 0,
            'quality_issues_found': 0,
            'features_engineered': 0,
            'normalization_applied': 0,
            'processing_time_ms': 0
        }
        logger.info("📊 Estatísticas do pipeline resetadas")


# Factory function
def get_preprocessing_pipeline() -> PreprocessingPipeline:
    """Retorna instância de PreprocessingPipeline"""
    return PreprocessingPipeline()