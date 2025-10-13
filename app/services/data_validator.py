"""
Data Validator Service - Validação de qualidade dos dados FOREX
Detecta gaps, outliers, valores faltantes e calcula quality score

Filosofia SLC:
- Simple: Interface clara, fácil de usar e extender
- Lovable: Feedback detalhado sobre qualidade dos dados
- Complete: Validação abrangente para garantir dados confiáveis para ML
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import logging
from statistics import median

from app.core.config import settings
from app.domain.schemas import (
    RawOHLCV, 
    ValidatedOHLCV, 
    DataValidationResult,
    TechnicalIndicators
)
from app.domain.enums import (
    DataQuality, 
    ValidationError, 
    DataSource,
    Granularity
)

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Erro durante validação de dados"""
    pass


class DataValidator:
    """
    Serviço responsável por validar qualidade dos dados FOREX
    
    Funcionalidades:
    - Detecta valores faltantes (NaN, zeros inválidos)
    - Identifica outliers usando Z-Score e IQR
    - Verifica gaps temporais
    - Valida consistência OHLC
    - Calcula score de qualidade geral
    - Gera relatório detalhado de validação
    """
    
    def __init__(self):
        self.outlier_zscore_threshold = settings.outlier_zscore_threshold
        self.min_quality_score = 0.7  # Mínimo aceitável
        
        logger.info("DataValidator inicializado")
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        granularity: Granularity = Granularity.ONE_HOUR,
        data_source: DataSource = DataSource.YAHOO_FINANCE
    ) -> Tuple[List[ValidatedOHLCV], DataValidationResult]:
        """
        Valida um DataFrame completo de dados OHLCV
        
        Args:
            df: DataFrame com dados OHLCV (index: DatetimeIndex)
            granularity: Granularidade dos dados
            data_source: Fonte dos dados
            
        Returns:
            Tuple[List[ValidatedOHLCV], DataValidationResult]
            - Lista de velas validadas
            - Resultado consolidado da validação
        """
        if df is None or df.empty:
            logger.warning("DataFrame vazio fornecido para validação")
            return [], self._create_empty_validation_result()
        
        logger.info(f"Validando {len(df)} velas ({granularity})")
        
        # Inicializa resultado de validação
        validation_result = DataValidationResult(
            quality_score=1.0,  # Começamos com score perfeito
            quality_level=DataQuality.EXCELLENT,
            validation_errors=[],
            metadata={
                "total_candles": len(df),
                "granularity": granularity,
                "data_source": data_source,
                "validation_timestamp": datetime.utcnow()
            }
        )
        
        # Lista de velas validadas
        validated_candles: List[ValidatedOHLCV] = []
        
        try:
            # 1. Validação de valores faltantes
            missing_info = self._check_missing_values(df)
            if missing_info["has_missing"]:
                validation_result.has_missing_values = True
                validation_result.missing_values_count = missing_info["count"]
                validation_result.validation_errors.append(ValidationError.MISSING_VALUES)
                validation_result.quality_score -= 0.1
                logger.warning(f"Encontrados {missing_info['count']} valores faltantes")
            
            # 2. Validação de duplicatas
            duplicates_info = self._check_duplicates(df)
            if duplicates_info["has_duplicates"]:
                validation_result.has_duplicates = True
                validation_result.validation_errors.append(ValidationError.DUPLICATE_TIMESTAMPS)
                validation_result.quality_score -= 0.05
                logger.warning(f"Encontradas {duplicates_info['count']} duplicatas")
            
            # 3. Validação de gaps temporais
            gaps_info = self._check_temporal_gaps(df, granularity)
            if gaps_info["has_gaps"]:
                validation_result.has_gaps = True
                validation_result.gap_percentage = gaps_info["gap_percentage"]
                if gaps_info["gap_percentage"] > 10:  # Mais de 10% de gaps
                    validation_result.validation_errors.append(ValidationError.TIMESTAMP_GAPS)
                    validation_result.quality_score -= 0.15
                logger.info(f"Gaps temporais: {gaps_info['gap_percentage']:.2f}%")
            
            # 4. Detecção de outliers
            outliers_info = self._detect_outliers(df)
            if outliers_info["has_outliers"]:
                validation_result.has_outliers = True
                validation_result.outlier_count = outliers_info["count"]
                validation_result.validation_errors.append(ValidationError.OUTLIERS_DETECTED)
                # Penalidade baseada na porcentagem de outliers
                outlier_percentage = outliers_info["count"] / len(df) * 100
                if outlier_percentage > 5:
                    validation_result.quality_score -= 0.2
                elif outlier_percentage > 2:
                    validation_result.quality_score -= 0.1
                logger.info(f"Outliers detectados: {outliers_info['count']} ({outlier_percentage:.2f}%)")
            
            # 5. Validação de consistência OHLC
            consistency_info = self._check_ohlc_consistency(df)
            if not consistency_info["is_consistent"]:
                validation_result.validation_errors.append(ValidationError.INVALID_OHLC)
                validation_result.quality_score -= 0.3  # Penalidade alta para dados inconsistentes
                logger.error(f"Inconsistências OHLC: {consistency_info['issues']}")
            
            # 6. Verifica se há dados suficientes
            if len(df) < 10:  # Muito poucos dados
                validation_result.validation_errors.append(ValidationError.INSUFFICIENT_DATA)
                validation_result.quality_score -= 0.2
                logger.warning(f"Dados insuficientes: apenas {len(df)} velas")
            
            # Garante que quality_score não seja negativo
            validation_result.quality_score = max(0.0, validation_result.quality_score)
            
            # Atualiza quality_level baseado no score final
            validation_result.quality_level = DataQuality.from_score(validation_result.quality_score)
            
            # 7. Converte DataFrame para lista de ValidatedOHLCV
            validated_candles = self._dataframe_to_validated_ohlcv(
                df, 
                validation_result,
                outliers_info.get("outlier_indices", set()),
                data_source
            )
            
            logger.info(
                f"Validação concluída: {len(validated_candles)} velas, "
                f"score: {validation_result.quality_score:.3f} ({validation_result.quality_level})"
            )
            
            return validated_candles, validation_result
            
        except Exception as e:
            logger.error(f"Erro durante validação: {str(e)}")
            raise DataValidationError(f"Falha na validação dos dados: {str(e)}")
    
    def validate_single_candle(self, raw_candle: RawOHLCV) -> ValidatedOHLCV:
        """
        Valida uma única vela OHLCV
        Útil para validação de dados em tempo real
        """
        logger.debug(f"Validando vela única: {raw_candle.timestamp}")
        
        # Cria validação básica para vela única
        validation_result = DataValidationResult(
            quality_score=1.0,
            quality_level=DataQuality.EXCELLENT,
            validation_errors=[],
            metadata={
                "total_candles": 1,
                "single_candle_validation": True,
                "validation_timestamp": datetime.utcnow()
            }
        )
        
        # Valida consistência OHLC básica
        if not self._validate_single_ohlc(raw_candle):
            validation_result.validation_errors.append(ValidationError.INVALID_OHLC)
            validation_result.quality_score = 0.5
            validation_result.quality_level = DataQuality.POOR
        
        # Cria ValidatedOHLCV
        validated_candle = ValidatedOHLCV(
            timestamp=raw_candle.timestamp,
            open=raw_candle.open,
            high=raw_candle.high,
            low=raw_candle.low,
            close=raw_candle.close,
            volume=raw_candle.volume,
            data_source=raw_candle.data_source,
            raw_symbol=raw_candle.raw_symbol,
            validation=validation_result,
            is_outlier=False,  # Não podemos detectar outlier em vela única
            is_gap_fill=False
        )
        
        return validated_candle
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Verifica valores faltantes (NaN, zeros inválidos)"""
        critical_cols = ["Open", "High", "Low", "Close"]
        
        # Conta NaNs
        nan_count = df[critical_cols].isna().sum().sum()
        
        # Conta zeros inválidos (preços não podem ser zero)
        zero_count = (df[critical_cols] <= 0).sum().sum()
        
        total_missing = nan_count + zero_count
        
        return {
            "has_missing": total_missing > 0,
            "count": total_missing,
            "nan_count": nan_count,
            "zero_count": zero_count,
            "details": df[critical_cols].isna().sum().to_dict()
        }
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Verifica timestamps duplicados"""
        duplicate_count = df.index.duplicated().sum()
        
        return {
            "has_duplicates": duplicate_count > 0,
            "count": duplicate_count
        }
    
    def _check_temporal_gaps(self, df: pd.DataFrame, granularity: Granularity) -> Dict[str, Any]:
        """
        Verifica gaps temporais baseado na granularidade
        """
        if len(df) <= 1:
            return {"has_gaps": False, "gap_percentage": 0.0}
        
        # Define intervalo esperado baseado na granularidade
        interval_mapping = {
            Granularity.ONE_MIN: timedelta(minutes=1),
            Granularity.FIVE_MIN: timedelta(minutes=5),
            Granularity.FIFTEEN_MIN: timedelta(minutes=15),
            Granularity.ONE_HOUR: timedelta(hours=1),
            Granularity.ONE_DAY: timedelta(days=1)
        }
        
        expected_interval = interval_mapping.get(granularity, timedelta(hours=1))
        
        # Calcula diferenças entre timestamps consecutivos
        time_diffs = pd.Series(df.index).diff().dropna()
        
        # Identifica gaps (diferenças maiores que o esperado)
        tolerance = expected_interval * 1.5  # 50% de tolerância
        gaps = time_diffs[time_diffs > tolerance]
        
        # Calcula período total e período esperado
        total_period = df.index[-1] - df.index[0]
        expected_periods = int(total_period / expected_interval) + 1
        actual_periods = len(df)
        
        # Porcentagem de gaps
        gap_percentage = max(0, (expected_periods - actual_periods) / expected_periods * 100)
        
        return {
            "has_gaps": len(gaps) > 0 or gap_percentage > 5,
            "gap_percentage": gap_percentage,
            "gap_count": len(gaps),
            "expected_periods": expected_periods,
            "actual_periods": actual_periods
        }
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detecta outliers usando Z-Score e IQR
        Combina ambos métodos para maior robustez
        """
        outlier_indices = set()
        
        # Colunas para análise de outliers
        price_cols = ["Open", "High", "Low", "Close"]
        
        for col in price_cols:
            if col not in df.columns:
                continue
                
            series = df[col].dropna()
            if len(series) < 10:  # Dados insuficientes
                continue
            
            # Método 1: Z-Score
            z_scores = np.abs((series - series.mean()) / series.std())
            zscore_outliers = df.index[z_scores > self.outlier_zscore_threshold].tolist()
            
            # Método 2: IQR (Interquartile Range)
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = df.index[(series < lower_bound) | (series > upper_bound)].tolist()
            
            # Combina outliers de ambos métodos
            outlier_indices.update(zscore_outliers)
            outlier_indices.update(iqr_outliers)
        
        return {
            "has_outliers": len(outlier_indices) > 0,
            "count": len(outlier_indices),
            "outlier_indices": outlier_indices,
            "percentage": len(outlier_indices) / len(df) * 100 if len(df) > 0 else 0
        }
    
    def _check_ohlc_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Verifica consistência dos dados OHLC"""
        issues = []
        
        # Verifica se High >= max(Open, Close)
        high_invalid = df['High'] < df[['Open', 'Close']].max(axis=1)
        if high_invalid.any():
            issues.append(f"High < max(Open,Close) em {high_invalid.sum()} velas")
        
        # Verifica se Low <= min(Open, Close)
        low_invalid = df['Low'] > df[['Open', 'Close']].min(axis=1)
        if low_invalid.any():
            issues.append(f"Low > min(Open,Close) em {low_invalid.sum()} velas")
        
        # Verifica se High >= Low
        high_low_invalid = df['High'] < df['Low']
        if high_low_invalid.any():
            issues.append(f"High < Low em {high_low_invalid.sum()} velas")
        
        return {
            "is_consistent": len(issues) == 0,
            "issues": issues,
            "invalid_count": len(issues)
        }
    
    def _validate_single_ohlc(self, candle: RawOHLCV) -> bool:
        """Valida consistência OHLC de uma única vela"""
        try:
            # High deve ser >= max(Open, Close)
            if candle.high < max(candle.open, candle.close):
                return False
            
            # Low deve ser <= min(Open, Close)
            if candle.low > min(candle.open, candle.close):
                return False
                
            # High deve ser >= Low
            if candle.high < candle.low:
                return False
                
            return True
        except Exception:
            return False
    
    def _dataframe_to_validated_ohlcv(
        self,
        df: pd.DataFrame,
        validation_result: DataValidationResult,
        outlier_indices: set,
        data_source: DataSource
    ) -> List[ValidatedOHLCV]:
        """Converte DataFrame para lista de ValidatedOHLCV"""
        validated_candles = []
        
        for idx, row in df.iterrows():
            try:
                # Verifica se é outlier
                is_outlier = idx in outlier_indices
                
                validated_candle = ValidatedOHLCV(
                    timestamp=idx.to_pydatetime(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row.get('Volume', 0)),
                    data_source=data_source,
                    raw_symbol="EURUSD=X",
                    validation=validation_result,
                    is_outlier=is_outlier,
                    is_gap_fill=False  # Por enquanto não preenchemos gaps
                )
                
                validated_candles.append(validated_candle)
                
            except Exception as e:
                logger.warning(f"Erro ao processar vela {idx}: {str(e)}")
                continue
        
        return validated_candles
    
    def _create_empty_validation_result(self) -> DataValidationResult:
        """Cria resultado de validação para dados vazios"""
        return DataValidationResult(
            quality_score=0.0,
            quality_level=DataQuality.POOR,
            validation_errors=[ValidationError.INSUFFICIENT_DATA],
            metadata={
                "total_candles": 0,
                "validation_timestamp": datetime.utcnow(),
                "empty_dataset": True
            }
        )
    
    def get_validation_summary(self, validation_result: DataValidationResult) -> str:
        """
        Gera resumo textual do resultado de validação
        Útil para logging e debugging
        """
        summary = [
            f"📊 VALIDAÇÃO DE DADOS - Score: {validation_result.quality_score:.3f} ({validation_result.quality_level})",
            f"   Total de velas: {validation_result.metadata.get('total_candles', 0)}"
        ]
        
        if validation_result.has_missing_values:
            summary.append(f"   ⚠️  Valores faltantes: {validation_result.missing_values_count}")
        
        if validation_result.has_outliers:
            summary.append(f"   ⚠️  Outliers: {validation_result.outlier_count}")
        
        if validation_result.has_gaps:
            summary.append(f"   ⚠️  Gaps temporais: {validation_result.gap_percentage:.2f}%")
        
        if validation_result.has_duplicates:
            summary.append(f"   ⚠️  Timestamps duplicados encontrados")
        
        if validation_result.validation_errors:
            summary.append(f"   🚨 Erros: {', '.join(validation_result.validation_errors)}")
        else:
            summary.append(f"   ✅ Nenhum erro encontrado")
        
        return "\n".join(summary)


# Factory function
def get_data_validator() -> DataValidator:
    """Retorna instância de DataValidator"""
    return DataValidator()