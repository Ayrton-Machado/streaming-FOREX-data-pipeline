"""
Schemas Pydantic para o Pipeline FOREX EUR/USD
Contratos de dados que garantem type safety e validação automática

Filosofia SLC:
- Simple: Schemas claros e auto-documentados
- Lovable: Validação automática com mensagens úteis
- Complete: Cobertura de todos os casos de uso (raw → validated → ML-ready)
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from decimal import Decimal

from .enums import (
    Granularity, 
    NormalizationMethod, 
    DataSource, 
    DataQuality,
    ValidationError,
    TechnicalIndicatorType
)


class TimestampMixin(BaseModel):
    """Mixin para campos de timestamp padronizados"""
    timestamp: datetime = Field(
        ..., 
        description="Timestamp UTC da vela/cotação"
    )

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp_utc(cls, v: datetime) -> datetime:
        """Garante que timestamp está em UTC"""
        if v.tzinfo is None:
            # Se não tem timezone, assume UTC
            import pytz
            v = v.replace(tzinfo=pytz.UTC)
        elif str(v.tzinfo) != 'UTC':
            # Converte para UTC se não estiver
            import pytz
            v = v.astimezone(pytz.UTC)
        return v


class OHLCVMixin(BaseModel):
    """Mixin para campos OHLCV básicos"""
    open: float = Field(..., gt=0, description="Preço de abertura")
    high: float = Field(..., gt=0, description="Preço máximo")
    low: float = Field(..., gt=0, description="Preço mínimo") 
    close: float = Field(..., gt=0, description="Preço de fechamento")
    volume: int = Field(default=0, ge=0, description="Volume negociado")

    @field_validator('high')
    @classmethod
    def validate_high_price(cls, v: float, info) -> float:
        """Valida que high >= low"""
        if 'low' in info.data and v < info.data['low']:
            raise ValueError(f'High price ({v}) must be >= Low price ({info.data["low"]})')
        return v

    @field_validator('low')
    @classmethod  
    def validate_low_price(cls, v: float, info) -> float:
        """Valida que low <= open, close"""
        if 'open' in info.data and v > info.data['open']:
            raise ValueError(f'Low price ({v}) must be <= Open price ({info.data["open"]})')
        if 'close' in info.data and v > info.data['close']:
            raise ValueError(f'Low price ({v}) must be <= Close price ({info.data["close"]})')
        return v

    @model_validator(mode='after')
    def validate_ohlc_consistency(self) -> 'OHLCVMixin':
        """Validações gerais de consistência OHLC"""
        if self.high < max(self.open, self.close):
            raise ValueError('High must be >= max(Open, Close)')
        if self.low > min(self.open, self.close):
            raise ValueError('Low must be <= min(Open, Close)')
        return self


# ===== SCHEMAS DE ENTRADA (Dados Brutos) =====

class RawOHLCV(TimestampMixin, OHLCVMixin):
    """
    Dados OHLCV brutos vindos diretamente das fontes externas
    Primeira camada de validação - apenas estrutura básica
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )

    # Metadados de origem
    data_source: DataSource = Field(
        default=DataSource.YAHOO_FINANCE,
        description="Fonte dos dados"
    )
    raw_symbol: str = Field(
        default="EURUSD=X",
        description="Símbolo original da fonte"
    )


# ===== SCHEMAS INTERMEDIÁRIOS (Dados Validados) =====

class DataValidationResult(BaseModel):
    """Resultado da validação de qualidade dos dados"""
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Score de qualidade (0-1)")
    quality_level: DataQuality = Field(..., description="Nível de qualidade")
    
    # Flags de problemas
    has_missing_values: bool = Field(default=False)
    has_outliers: bool = Field(default=False)
    has_gaps: bool = Field(default=False)
    has_duplicates: bool = Field(default=False)
    
    # Estatísticas
    outlier_count: int = Field(default=0, ge=0)
    gap_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    missing_values_count: int = Field(default=0, ge=0)
    
    # Erros encontrados
    validation_errors: List[ValidationError] = Field(default_factory=list)
    
    # Metadados adicionais
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('quality_level', mode='before')
    @classmethod
    def set_quality_level_from_score(cls, v, info) -> DataQuality:
        """Auto-determina quality_level baseado no score"""
        if 'quality_score' in info.data:
            return DataQuality.from_score(info.data['quality_score'])
        return v or DataQuality.POOR


class ValidatedOHLCV(RawOHLCV):
    """
    Dados OHLCV que passaram por validação de qualidade
    Segunda camada - dados limpos e confiáveis
    """
    
    # Resultado da validação
    validation: DataValidationResult = Field(
        ..., 
        description="Resultado da validação de qualidade"
    )
    
    # Flags calculados
    is_outlier: bool = Field(
        default=False, 
        description="Esta vela foi identificada como outlier"
    )
    is_gap_fill: bool = Field(
        default=False,
        description="Esta vela foi preenchida para resolver gap"
    )
    
    # Dados corrigidos (se necessário)
    original_values: Optional[Dict[str, float]] = Field(
        default=None,
        description="Valores originais se houve correção"
    )


# ===== SCHEMAS DE INDICATORS TÉCNICOS =====

class MovingAverages(BaseModel):
    """Médias móveis calculadas"""
    sma_20: Optional[float] = Field(None, description="Simple Moving Average 20 períodos")
    sma_50: Optional[float] = Field(None, description="Simple Moving Average 50 períodos")  
    sma_200: Optional[float] = Field(None, description="Simple Moving Average 200 períodos")
    ema_12: Optional[float] = Field(None, description="Exponential Moving Average 12 períodos")
    ema_26: Optional[float] = Field(None, description="Exponential Moving Average 26 períodos")
    wma_14: Optional[float] = Field(None, description="Weighted Moving Average 14 períodos")


class Oscillators(BaseModel):
    """Osciladores técnicos"""
    rsi_14: Optional[float] = Field(None, ge=0, le=100, description="RSI 14 períodos")
    stoch_k: Optional[float] = Field(None, ge=0, le=100, description="Stochastic %K")
    stoch_d: Optional[float] = Field(None, ge=0, le=100, description="Stochastic %D")
    
    
class MACD(BaseModel):
    """Indicador MACD completo"""
    macd_line: Optional[float] = Field(None, description="Linha MACD (EMA12 - EMA26)")
    macd_signal: Optional[float] = Field(None, description="Sinal MACD (EMA9 do MACD)")
    macd_histogram: Optional[float] = Field(None, description="Histograma MACD (MACD - Signal)")


class BollingerBands(BaseModel):
    """Bandas de Bollinger"""
    bb_upper: Optional[float] = Field(None, description="Banda superior")
    bb_middle: Optional[float] = Field(None, description="Banda média (SMA)")
    bb_lower: Optional[float] = Field(None, description="Banda inferior")
    bb_width: Optional[float] = Field(None, ge=0, description="Largura das bandas")
    bb_position: Optional[float] = Field(None, ge=0, le=1, description="Posição do preço nas bandas (0-1)")


class TechnicalIndicators(BaseModel):
    """
    Conjunto completo de indicadores técnicos calculados
    Usado tanto em ValidatedOHLCV quanto em MLReadyCandle
    """
    
    # Médias móveis
    moving_averages: MovingAverages = Field(default_factory=MovingAverages)
    
    # Osciladores
    oscillators: Oscillators = Field(default_factory=Oscillators)
    
    # MACD
    macd: MACD = Field(default_factory=MACD)
    
    # Bollinger Bands
    bollinger: BollingerBands = Field(default_factory=BollingerBands)
    
    # Metadados
    indicators_calculated: List[TechnicalIndicatorType] = Field(
        default_factory=list,
        description="Lista de indicadores que foram calculados"
    )
    calculation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Quando os indicadores foram calculados"
    )


# ===== SCHEMAS DE NORMALIZAÇÃO =====

class NormalizationParams(BaseModel):
    """Parâmetros usados na normalização"""
    method: NormalizationMethod = Field(..., description="Método de normalização usado")
    
    # Parâmetros MinMax
    min_values: Optional[Dict[str, float]] = Field(None, description="Valores mínimos (MinMax)")
    max_values: Optional[Dict[str, float]] = Field(None, description="Valores máximos (MinMax)")
    
    # Parâmetros Z-Score
    mean_values: Optional[Dict[str, float]] = Field(None, description="Médias (Z-Score)")
    std_values: Optional[Dict[str, float]] = Field(None, description="Desvios padrão (Z-Score)")
    
    # Parâmetros Robust
    median_values: Optional[Dict[str, float]] = Field(None, description="Medianas (Robust)")
    iqr_values: Optional[Dict[str, float]] = Field(None, description="IQRs (Robust)")
    
    # Metadados
    applied_to_columns: List[str] = Field(default_factory=list, description="Colunas normalizadas")
    normalization_timestamp: datetime = Field(default_factory=datetime.utcnow)


class NormalizedOHLCV(BaseModel):
    """OHLCV normalizado para ML"""
    open_norm: float = Field(..., description="Open normalizado")
    high_norm: float = Field(..., description="High normalizado")
    low_norm: float = Field(..., description="Low normalizado")
    close_norm: float = Field(..., description="Close normalizado")
    volume_norm: float = Field(..., description="Volume normalizado")


# ===== SCHEMAS DE SAÍDA (ML-Ready) =====

class MLReadyCandle(TimestampMixin, OHLCVMixin):
    """
    Dados completamente processados e prontos para Machine Learning
    Camada final - tudo incluído para treinar modelos
    """
    
    # Dados normalizados
    normalized: NormalizedOHLCV = Field(
        ..., 
        description="Valores OHLCV normalizados"
    )
    
    # Indicadores técnicos
    indicators: TechnicalIndicators = Field(
        default_factory=TechnicalIndicators,
        description="Indicadores técnicos calculados"
    )
    
    # Informações de qualidade
    validation: DataValidationResult = Field(
        ...,
        description="Resultado da validação de qualidade"
    )
    
    # Parâmetros de normalização
    normalization_params: NormalizationParams = Field(
        ...,
        description="Parâmetros usados na normalização"
    )
    
    # Metadados ML
    granularity: Granularity = Field(..., description="Granularidade dos dados")
    data_source: DataSource = Field(default=DataSource.YAHOO_FINANCE)
    processing_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Quando o processamento foi finalizado"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2025-01-15T14:30:00Z",
                "open": 1.0850,
                "high": 1.0865,
                "low": 1.0845,
                "close": 1.0860,
                "volume": 125000,
                "normalized": {
                    "open_norm": 0.523,
                    "high_norm": 0.651,
                    "low_norm": 0.412,
                    "close_norm": 0.589,
                    "volume_norm": 0.345
                },
                "indicators": {
                    "moving_averages": {
                        "sma_20": 1.0855,
                        "ema_12": 1.0862
                    },
                    "oscillators": {
                        "rsi_14": 62.3
                    },
                    "macd": {
                        "macd_line": 0.0012,
                        "macd_signal": 0.0008,
                        "macd_histogram": 0.0004
                    }
                },
                "validation": {
                    "quality_score": 0.98,
                    "quality_level": "excellent"
                },
                "granularity": "1h"
            }
        }
    )


# ===== SCHEMAS DE API (Responses) =====

class BasicQuote(TimestampMixin, OHLCVMixin):
    """
    Cotação básica para endpoints rápidos
    Versão simplificada sem indicadores ou normalização
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2025-01-15T14:30:00Z",
                "open": 1.0850,
                "high": 1.0865,
                "low": 1.0845,
                "close": 1.0860,
                "volume": 125000
            }
        }
    )


class DataRange(BaseModel):
    """Informações sobre intervalo de dados"""
    start: datetime = Field(..., description="Data/hora de início")
    end: datetime = Field(..., description="Data/hora de fim") 
    total_periods: int = Field(..., ge=0, description="Total de períodos")
    granularity: Granularity = Field(..., description="Granularidade")


class DataStats(BaseModel):
    """Estatísticas úteis sobre o dataset"""
    total_candles: int = Field(..., ge=0)
    date_range: DataRange = Field(...)
    
    # Estatísticas de preço
    price_stats: Dict[str, float] = Field(
        default_factory=dict,
        description="Min, max, mean, std dos preços"
    )
    
    # Qualidade dos dados
    average_quality_score: float = Field(..., ge=0, le=1)
    quality_distribution: Dict[DataQuality, int] = Field(default_factory=dict)
    
    # Indicadores disponíveis
    available_indicators: List[TechnicalIndicatorType] = Field(default_factory=list)
    
    # Normalização aplicada
    normalization_method: Optional[NormalizationMethod] = None
    normalization_params_summary: Dict[str, Any] = Field(default_factory=dict)


class MLReadyResponse(BaseModel):
    """Response completa do endpoint /ml-ready"""
    data: List[MLReadyCandle] = Field(
        ..., 
        description="Lista de velas processadas"
    )
    stats: DataStats = Field(
        ..., 
        description="Estatísticas do dataset"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadados adicionais"
    )


class QuotesResponse(BaseModel):
    """Response do endpoint /quotes"""
    data: List[BasicQuote] = Field(..., description="Lista de cotações")
    total: int = Field(..., ge=0, description="Total de registros")
    granularity: Granularity = Field(..., description="Granularidade")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ===== SCHEMAS DE REQUEST (Query Parameters) =====

class BaseDataRequest(BaseModel):
    """Base para requests de dados"""
    granularity: Granularity = Field(
        default=Granularity.ONE_HOUR,
        description="Granularidade dos dados"
    )
    start_date: datetime = Field(..., description="Data de início (UTC)")
    end_date: datetime = Field(..., description="Data de fim (UTC)")

    @model_validator(mode='after')
    def validate_date_range(self) -> 'BaseDataRequest':
        """Valida que start_date < end_date"""
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        return self


class MLReadyRequest(BaseDataRequest):
    """Request para endpoint /ml-ready"""
    normalization: NormalizationMethod = Field(
        default=NormalizationMethod.MINMAX,
        description="Método de normalização"
    )
    include_indicators: bool = Field(
        default=True,
        description="Incluir indicadores técnicos"
    )
    remove_outliers: bool = Field(
        default=True,
        description="Remover outliers automaticamente"
    )
    min_quality_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Score mínimo de qualidade"
    )


class QuotesRequest(BaseModel):
    """Request para endpoint /quotes"""
    limit: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Número máximo de registros"
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Offset para paginação"
    )
    granularity: Granularity = Field(
        default=Granularity.ONE_DAY,
        description="Granularidade"
    )


# ===== SCHEMAS DE WEBSOCKET =====

class StreamMessage(BaseModel):
    """Mensagem WebSocket para streaming real-time"""
    event_type: str = Field(
        default="price_update",
        description="Tipo do evento (price_update, heartbeat, error)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp do evento"
    )
    data: Optional[BasicQuote] = Field(
        None,
        description="Dados da cotação (se event_type=price_update)"
    )
    message: Optional[str] = Field(
        None,
        description="Mensagem adicional"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StreamSubscription(BaseModel):
    """Configuração de subscription WebSocket"""
    action: str = Field(
        default="subscribe",
        description="Ação (subscribe/unsubscribe)"
    )
    granularity: Granularity = Field(
        default=Granularity.ONE_MIN,
        description="Granularidade para streaming"
    )
    include_indicators: bool = Field(
        default=False,
        description="Incluir indicadores no stream (mais lento)"
    )