"""
Data Quality Service - Validação e limpeza de dados
Etapa 4: Detecção de anomalias, correção de dados, relatórios de qualidade
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from app.domain.schemas import ValidatedOHLCV, MLReadyCandle
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Métricas de qualidade dos dados"""
    total_records: int
    valid_records: int
    invalid_records: int
    duplicate_records: int
    missing_records: int
    outlier_records: int
    quality_score: float  # 0-100%
    
    # Detalhes específicos
    price_inconsistencies: int
    volume_anomalies: int
    timestamp_gaps: int
    negative_prices: int
    zero_volume_records: int
    
    # Relatório temporal
    start_timestamp: datetime
    end_timestamp: datetime
    expected_records: int
    data_completeness: float  # %


@dataclass
class DataQualityIssue:
    """Representa um problema de qualidade identificado"""
    timestamp: datetime
    issue_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    suggested_action: str
    raw_value: Any
    corrected_value: Optional[Any] = None


class DataQualityService:
    """
    Serviço para validação e melhoria da qualidade dos dados
    
    Filosofia SLC:
    - Simple: Validações claras baseadas em regras de mercado
    - Lovable: Correções automáticas quando possível
    - Complete: Cobertura total com relatórios detalhados
    """
    
    def __init__(self):
        # Configurações de validação para FOREX EUR/USD
        self.validation_config = {
            'price_range': {'min': 0.8, 'max': 2.0},      # EUR/USD range realista
            'daily_change_limit': 0.05,                    # 5% máximo em 24h
            'spike_threshold': 0.02,                       # 2% spike threshold
            'volume_min': 0,                               # Volume mínimo
            'max_gap_minutes': 60,                         # Gap máximo entre velas
            'decimal_places': 5                            # Precisão FOREX
        }
        
        # Contadores de issues
        self.issues_found = []
        self.corrections_applied = 0
        
        logger.info("DataQualityService inicializado para FOREX EUR/USD")
    
    def validate_and_clean(
        self,
        candles: List[ValidatedOHLCV],
        auto_correct: bool = True,
        strict_mode: bool = False
    ) -> Tuple[List[ValidatedOHLCV], DataQualityMetrics]:
        """
        Valida e limpa dados com relatório de qualidade
        
        Args:
            candles: Lista de velas a validar
            auto_correct: Se deve tentar corrigir automaticamente
            strict_mode: Se deve ser mais rigoroso na validação
            
        Returns:
            Tupla (velas_limpas, métricas_qualidade)
        """
        if not candles:
            logger.warning("Lista de velas vazia")
            return [], self._empty_metrics()
        
        logger.info(f"🔍 Iniciando validação de qualidade: {len(candles)} velas")
        
        # Reset contadores
        self.issues_found = []
        self.corrections_applied = 0
        
        # Converte para DataFrame para análise
        df = self._candles_to_dataframe(candles)
        original_count = len(df)
        
        # 1. Validação básica de estrutura
        df = self._validate_basic_structure(df)
        
        # 2. Validação de preços
        df = self._validate_prices(df, auto_correct)
        
        # 3. Validação de volume
        df = self._validate_volume(df, auto_correct)
        
        # 4. Validação temporal
        df = self._validate_timestamps(df, auto_correct)
        
        # 5. Detecção de duplicatas
        df = self._remove_duplicates(df)
        
        # 6. Detecção de outliers
        df = self._detect_outliers(df, strict_mode)
        
        # 7. Validação de consistência OHLC
        df = self._validate_ohlc_consistency(df, auto_correct)
        
        # 8. Fill gaps se necessário
        if auto_correct:
            df = self._fill_data_gaps(df)
        
        # Converte de volta para ValidatedOHLCV
        cleaned_candles = self._dataframe_to_candles(df, candles)
        
        # Calcula métricas
        metrics = self._calculate_metrics(original_count, cleaned_candles, candles)
        
        logger.info(f"✅ Validação concluída: {metrics.valid_records}/{original_count} velas válidas")
        logger.info(f"🔧 Correções aplicadas: {self.corrections_applied}")
        logger.info(f"📊 Score de qualidade: {metrics.quality_score:.1f}%")
        
        return cleaned_candles, metrics
    
    def _validate_basic_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validação básica de estrutura dos dados"""
        
        # Verifica colunas obrigatórias
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                issue_type='missing_columns',
                severity='critical',
                description=f"Colunas obrigatórias ausentes: {missing_columns}",
                suggested_action='Verificar fonte de dados',
                raw_value=missing_columns
            )
            self.issues_found.append(issue)
            logger.error(f"❌ Colunas obrigatórias ausentes: {missing_columns}")
        
        # Remove linhas com valores NaN em colunas críticas
        initial_count = len(df)
        df = df.dropna(subset=['timestamp', 'open', 'high', 'low', 'close'])
        
        if len(df) < initial_count:
            removed_count = initial_count - len(df)
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                issue_type='missing_values',
                severity='medium',
                description=f"Removidas {removed_count} linhas com valores ausentes",
                suggested_action='Verificar integridade da fonte',
                raw_value=removed_count
            )
            self.issues_found.append(issue)
            logger.warning(f"⚠️ Removidas {removed_count} linhas com valores NaN")
        
        return df
    
    def _validate_prices(self, df: pd.DataFrame, auto_correct: bool) -> pd.DataFrame:
        """Validação de preços"""
        
        # 1. Preços negativos
        negative_mask = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        if negative_mask.any():
            count = negative_mask.sum()
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                issue_type='negative_prices',
                severity='critical',
                description=f"Encontrados {count} registros com preços ≤ 0",
                suggested_action='Remover ou corrigir registros',
                raw_value=count
            )
            self.issues_found.append(issue)
            
            if auto_correct:
                df = df[~negative_mask]
                self.corrections_applied += count
                logger.warning(f"🔧 Removidos {count} registros com preços ≤ 0")
        
        # 2. Preços fora do range realista
        price_cols = ['open', 'high', 'low', 'close']
        out_of_range_mask = (
            (df[price_cols] < self.validation_config['price_range']['min']) |
            (df[price_cols] > self.validation_config['price_range']['max'])
        ).any(axis=1)
        
        if out_of_range_mask.any():
            count = out_of_range_mask.sum()
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                issue_type='price_out_of_range',
                severity='high',
                description=f"Preços fora do range {self.validation_config['price_range']}: {count} registros",
                suggested_action='Verificar se são dados reais ou erros',
                raw_value=count
            )
            self.issues_found.append(issue)
            
            if auto_correct:
                df = df[~out_of_range_mask]
                self.corrections_applied += count
                logger.warning(f"🔧 Removidos {count} registros com preços fora do range")
        
        # 3. Spikes de preço irreais
        df = self._detect_price_spikes(df, auto_correct)
        
        return df
    
    def _detect_price_spikes(self, df: pd.DataFrame, auto_correct: bool) -> pd.DataFrame:
        """Detecta spikes de preço irreais"""
        if len(df) < 3:
            return df
        
        # Calcula mudanças percentuais
        df['price_change'] = df['close'].pct_change()
        
        # Identifica spikes
        spike_threshold = self.validation_config['spike_threshold']
        spike_mask = abs(df['price_change']) > spike_threshold
        
        if spike_mask.any():
            count = spike_mask.sum()
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                issue_type='price_spikes',
                severity='medium',
                description=f"Detectados {count} spikes de preço >{spike_threshold*100}%",
                suggested_action='Verificar se são movimentos reais do mercado',
                raw_value=count
            )
            self.issues_found.append(issue)
            
            if auto_correct:
                # Para spikes extremos (>10%), remove
                extreme_spikes = abs(df['price_change']) > 0.10
                if extreme_spikes.any():
                    extreme_count = extreme_spikes.sum()
                    df = df[~extreme_spikes]
                    self.corrections_applied += extreme_count
                    logger.warning(f"🔧 Removidos {extreme_count} spikes extremos")
        
        # Remove coluna temporária
        df = df.drop('price_change', axis=1, errors='ignore')
        
        return df
    
    def _validate_volume(self, df: pd.DataFrame, auto_correct: bool) -> pd.DataFrame:
        """Validação de volume"""
        
        # Volume negativo
        negative_volume_mask = df['volume'] < 0
        if negative_volume_mask.any():
            count = negative_volume_mask.sum()
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                issue_type='negative_volume',
                severity='high',
                description=f"Volume negativo encontrado: {count} registros",
                suggested_action='Corrigir para 0 ou valor absoluto',
                raw_value=count
            )
            self.issues_found.append(issue)
            
            if auto_correct:
                df.loc[negative_volume_mask, 'volume'] = 0
                self.corrections_applied += count
                logger.warning(f"🔧 Corrigidos {count} volumes negativos para 0")
        
        # Volume zero (pode ser normal em FOREX)
        zero_volume_count = (df['volume'] == 0).sum()
        if zero_volume_count > len(df) * 0.5:  # Mais de 50%
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                issue_type='excessive_zero_volume',
                severity='low',
                description=f"Muitos registros com volume zero: {zero_volume_count}/{len(df)}",
                suggested_action='Normal para FOREX, mas verificar fonte',
                raw_value=zero_volume_count
            )
            self.issues_found.append(issue)
        
        return df
    
    def _validate_timestamps(self, df: pd.DataFrame, auto_correct: bool) -> pd.DataFrame:
        """Validação temporal"""
        
        # Ordena por timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Detecta timestamps duplicados
        duplicate_timestamps = df['timestamp'].duplicated()
        if duplicate_timestamps.any():
            count = duplicate_timestamps.sum()
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                issue_type='duplicate_timestamps',
                severity='medium',
                description=f"Timestamps duplicados: {count}",
                suggested_action='Manter apenas o primeiro ou fazer média',
                raw_value=count
            )
            self.issues_found.append(issue)
            
            if auto_correct:
                # Mantém apenas o primeiro
                df = df[~duplicate_timestamps]
                self.corrections_applied += count
                logger.warning(f"🔧 Removidos {count} timestamps duplicados")
        
        # Detecta gaps grandes
        if len(df) > 1:
            time_diffs = df['timestamp'].diff()
            max_gap = timedelta(minutes=self.validation_config['max_gap_minutes'])
            large_gaps = time_diffs > max_gap
            
            if large_gaps.any():
                count = large_gaps.sum()
                issue = DataQualityIssue(
                    timestamp=datetime.now(),
                    issue_type='large_time_gaps',
                    severity='low',
                    description=f"Gaps grandes detectados: {count} (>{self.validation_config['max_gap_minutes']}min)",
                    suggested_action='Pode ser normal (fim de semana, feriados)',
                    raw_value=count
                )
                self.issues_found.append(issue)
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove registros duplicados"""
        initial_count = len(df)
        
        # Remove duplicatas completas
        df = df.drop_duplicates()
        
        removed_count = initial_count - len(df)
        if removed_count > 0:
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                issue_type='duplicate_records',
                severity='medium',
                description=f"Removidas {removed_count} duplicatas completas",
                suggested_action='Verificar processo de coleta',
                raw_value=removed_count
            )
            self.issues_found.append(issue)
            self.corrections_applied += removed_count
            logger.warning(f"🔧 Removidas {removed_count} duplicatas")
        
        return df
    
    def _detect_outliers(self, df: pd.DataFrame, strict_mode: bool) -> pd.DataFrame:
        """Detecta outliers usando métodos estatísticos"""
        if len(df) < 10:
            return df
        
        # Método Z-score para preços
        price_cols = ['open', 'high', 'low', 'close']
        z_threshold = 3 if not strict_mode else 2.5
        
        outlier_mask = pd.Series([False] * len(df))
        
        for col in price_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            col_outliers = z_scores > z_threshold
            outlier_mask |= col_outliers
        
        if outlier_mask.any():
            count = outlier_mask.sum()
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                issue_type='statistical_outliers',
                severity='low',
                description=f"Outliers detectados (Z-score>{z_threshold}): {count}",
                suggested_action='Verificar se são movimentos reais ou erros',
                raw_value=count
            )
            self.issues_found.append(issue)
            
            # Marca outliers mas não remove automaticamente
            df['is_outlier_detected'] = outlier_mask
        
        return df
    
    def _validate_ohlc_consistency(self, df: pd.DataFrame, auto_correct: bool) -> pd.DataFrame:
        """Valida consistência OHLC (High ≥ Open,Close,Low; Low ≤ Open,Close,High)"""
        
        # High deve ser >= Open, Close, Low
        high_invalid = (
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['high'] < df['low'])
        )
        
        # Low deve ser <= Open, Close, High  
        low_invalid = (
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['low'] > df['high'])
        )
        
        inconsistent_mask = high_invalid | low_invalid
        
        if inconsistent_mask.any():
            count = inconsistent_mask.sum()
            issue = DataQualityIssue(
                timestamp=datetime.now(),
                issue_type='ohlc_inconsistency',
                severity='high',
                description=f"Inconsistências OHLC: {count} registros",
                suggested_action='Corrigir High/Low para manter consistência',
                raw_value=count
            )
            self.issues_found.append(issue)
            
            if auto_correct:
                # Corrige High e Low
                for idx in df[inconsistent_mask].index:
                    prices = [df.loc[idx, 'open'], df.loc[idx, 'close']]
                    df.loc[idx, 'high'] = max(df.loc[idx, 'high'], max(prices))
                    df.loc[idx, 'low'] = min(df.loc[idx, 'low'], min(prices))
                
                self.corrections_applied += count
                logger.warning(f"🔧 Corrigidas {count} inconsistências OHLC")
        
        return df
    
    def _fill_data_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preenche gaps de dados pequenos com interpolação"""
        if len(df) < 2:
            return df
        
        # Detecta gaps de até 5 minutos
        df = df.sort_values('timestamp').reset_index(drop=True)
        time_diffs = df['timestamp'].diff()
        small_gaps = (time_diffs > timedelta(minutes=1)) & (time_diffs <= timedelta(minutes=5))
        
        if small_gaps.any():
            gap_count = small_gaps.sum()
            logger.info(f"🔄 Preenchendo {gap_count} pequenos gaps com interpolação")
            
            # Cria range temporal completo
            full_range = pd.date_range(
                start=df['timestamp'].min(),
                end=df['timestamp'].max(),
                freq='1min'
            )
            
            # Reindexes e interpola
            df = df.set_index('timestamp').reindex(full_range)
            
            # Interpola apenas preços, mantém volume como 0
            price_cols = ['open', 'high', 'low', 'close']
            df[price_cols] = df[price_cols].interpolate(method='linear')
            df['volume'] = df['volume'].fillna(0)
            
            # Marca registros interpolados
            df['is_gap_fill'] = df['volume'].isna()
            df['volume'] = df['volume'].fillna(0)
            
            # Reset index
            df = df.reset_index().rename(columns={'index': 'timestamp'})
            
            filled_count = df['is_gap_fill'].sum()
            if filled_count > 0:
                self.corrections_applied += filled_count
                logger.info(f"✅ Preenchidos {filled_count} gaps com interpolação")
        
        return df
    
    def _calculate_metrics(
        self,
        original_count: int,
        cleaned_candles: List[ValidatedOHLCV],
        original_candles: List[ValidatedOHLCV]
    ) -> DataQualityMetrics:
        """Calcula métricas de qualidade"""
        
        valid_count = len(cleaned_candles)
        invalid_count = original_count - valid_count
        
        # Conta tipos específicos de issues
        price_issues = len([i for i in self.issues_found if 'price' in i.issue_type])
        volume_issues = len([i for i in self.issues_found if 'volume' in i.issue_type])
        timestamp_issues = len([i for i in self.issues_found if 'timestamp' in i.issue_type or 'time' in i.issue_type])
        
        # Score de qualidade (0-100)
        quality_score = (valid_count / original_count) * 100 if original_count > 0 else 0
        
        # Aplica penalidades por tipos de issue
        critical_issues = len([i for i in self.issues_found if i.severity == 'critical'])
        high_issues = len([i for i in self.issues_found if i.severity == 'high'])
        
        quality_score -= (critical_issues * 10) + (high_issues * 5)
        quality_score = max(0, min(100, quality_score))
        
        # Timestamps
        start_ts = original_candles[0].timestamp if original_candles else datetime.now()
        end_ts = original_candles[-1].timestamp if original_candles else datetime.now()
        
        # Calcula completeness esperada (assumindo dados de 1 minuto)
        expected_duration = (end_ts - start_ts).total_seconds() / 60
        expected_records = int(expected_duration) if expected_duration > 0 else original_count
        data_completeness = (valid_count / expected_records) * 100 if expected_records > 0 else 100
        
        return DataQualityMetrics(
            total_records=original_count,
            valid_records=valid_count,
            invalid_records=invalid_count,
            duplicate_records=len([i for i in self.issues_found if 'duplicate' in i.issue_type]),
            missing_records=len([i for i in self.issues_found if 'missing' in i.issue_type]),
            outlier_records=len([i for i in self.issues_found if 'outlier' in i.issue_type]),
            quality_score=quality_score,
            price_inconsistencies=price_issues,
            volume_anomalies=volume_issues,
            timestamp_gaps=timestamp_issues,
            negative_prices=len([i for i in self.issues_found if i.issue_type == 'negative_prices']),
            zero_volume_records=0,  # Calculado separadamente se necessário
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            expected_records=expected_records,
            data_completeness=min(100, data_completeness)
        )
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Retorna relatório detalhado de qualidade"""
        
        # Agrupa issues por tipo e severidade
        issues_by_type = {}
        issues_by_severity = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for issue in self.issues_found:
            if issue.issue_type not in issues_by_type:
                issues_by_type[issue.issue_type] = []
            issues_by_type[issue.issue_type].append(issue)
            issues_by_severity[issue.severity] += 1
        
        return {
            "total_issues": len(self.issues_found),
            "corrections_applied": self.corrections_applied,
            "issues_by_type": {
                issue_type: len(issues) for issue_type, issues in issues_by_type.items()
            },
            "issues_by_severity": issues_by_severity,
            "detailed_issues": [
                {
                    "timestamp": issue.timestamp.isoformat(),
                    "type": issue.issue_type,
                    "severity": issue.severity,
                    "description": issue.description,
                    "action": issue.suggested_action,
                    "raw_value": str(issue.raw_value),
                    "corrected": issue.corrected_value is not None
                }
                for issue in self.issues_found
            ]
        }
    
    def _candles_to_dataframe(self, candles: List[ValidatedOHLCV]) -> pd.DataFrame:
        """Converte lista de velas para DataFrame"""
        data = []
        for candle in candles:
            data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume,
                'data_source': candle.data_source,
                'raw_symbol': candle.raw_symbol,
                'is_outlier': candle.is_outlier,
                'is_gap_fill': candle.is_gap_fill
            })
        
        return pd.DataFrame(data)
    
    def _dataframe_to_candles(
        self,
        df: pd.DataFrame,
        original_candles: List[ValidatedOHLCV]
    ) -> List[ValidatedOHLCV]:
        """Converte DataFrame de volta para ValidatedOHLCV"""
        candles = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            # Pega dados originais se disponível
            if i < len(original_candles):
                original = original_candles[i]
                data_source = original.data_source
                raw_symbol = original.raw_symbol
                validation = original.validation
            else:
                # Dados interpolados
                data_source = "interpolated"
                raw_symbol = "EURUSD"
                validation = {"interpolated": True}
            
            candle = ValidatedOHLCV(
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                data_source=data_source,
                raw_symbol=raw_symbol,
                validation=validation,
                is_outlier=row.get('is_outlier_detected', False),
                is_gap_fill=row.get('is_gap_fill', False)
            )
            candles.append(candle)
        
        return candles
    
    def _empty_metrics(self) -> DataQualityMetrics:
        """Retorna métricas vazias"""
        return DataQualityMetrics(
            total_records=0,
            valid_records=0,
            invalid_records=0,
            duplicate_records=0,
            missing_records=0,
            outlier_records=0,
            quality_score=0.0,
            price_inconsistencies=0,
            volume_anomalies=0,
            timestamp_gaps=0,
            negative_prices=0,
            zero_volume_records=0,
            start_timestamp=datetime.now(),
            end_timestamp=datetime.now(),
            expected_records=0,
            data_completeness=0.0
        )


# Factory function
def get_data_quality_service() -> DataQualityService:
    """Retorna instância de DataQualityService"""
    return DataQualityService()