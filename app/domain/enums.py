"""
Enumerações para o Pipeline FOREX EUR/USD
Tipos seguros e constantes que serão usadas em todo o sistema
"""

from enum import Enum
from typing import List


class Granularity(str, Enum):
    """
    Granularidades suportadas para dados OHLCV
    Valores correspondem aos aceitos pelo Yahoo Finance
    """
    ONE_MIN = "1min"
    FIVE_MIN = "5min"
    FIFTEEN_MIN = "15min"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"

    @classmethod
    def get_valid_values(cls) -> List[str]:
        """Retorna lista de valores válidos"""
        return [item.value for item in cls]
    
    @classmethod
    def to_yfinance_interval(cls, granularity: str) -> str:
        """Converte granularidade para formato Yahoo Finance"""
        mapping = {
            cls.ONE_MIN: "1m",
            cls.FIVE_MIN: "5m", 
            cls.FIFTEEN_MIN: "15m",
            cls.ONE_HOUR: "1h",
            cls.ONE_DAY: "1d"
        }
        return mapping.get(granularity, granularity)


class NormalizationMethod(str, Enum):
    """
    Métodos de normalização suportados para dados ML
    """
    MINMAX = "minmax"      # Min-Max scaling (0-1)
    ZSCORE = "zscore"      # Z-Score standardization (mean=0, std=1)
    ROBUST = "robust"      # Robust scaling (mediana e IQR)
    
    @classmethod
    def get_valid_values(cls) -> List[str]:
        """Retorna lista de métodos válidos"""
        return [item.value for item in cls]


class DataSource(str, Enum):
    """
    Fontes de dados suportadas
    """
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    MANUAL = "manual"  # Para dados inseridos manualmente ou processados
    
    @classmethod
    def get_primary_source(cls) -> str:
        """Retorna fonte primária padrão"""
        return cls.YAHOO_FINANCE


class DataQuality(str, Enum):
    """
    Níveis de qualidade dos dados
    """
    EXCELLENT = "excellent"    # quality_score >= 0.95
    GOOD = "good"             # quality_score >= 0.80
    FAIR = "fair"             # quality_score >= 0.60
    POOR = "poor"             # quality_score < 0.60
    
    @classmethod
    def from_score(cls, score: float) -> str:
        """Determina nível de qualidade baseado no score"""
        if score >= 0.95:
            return cls.EXCELLENT
        elif score >= 0.80:
            return cls.GOOD
        elif score >= 0.60:
            return cls.FAIR
        else:
            return cls.POOR


class ValidationError(str, Enum):
    """
    Tipos de erros de validação
    """
    MISSING_VALUES = "missing_values"
    INVALID_OHLC = "invalid_ohlc"      # High < Low, etc
    OUTLIERS_DETECTED = "outliers"
    INSUFFICIENT_DATA = "insufficient_data"
    TIMESTAMP_GAPS = "timestamp_gaps"
    DUPLICATE_TIMESTAMPS = "duplicates"
    
    
class TechnicalIndicatorType(str, Enum):
    """
    Tipos de indicadores técnicos suportados
    """
    # Moving Averages
    SMA = "sma"               # Simple Moving Average
    EMA = "ema"               # Exponential Moving Average
    WMA = "wma"               # Weighted Moving Average
    
    # Oscillators
    RSI = "rsi"               # Relative Strength Index
    STOCH = "stochastic"      # Stochastic Oscillator
    MACD = "macd"             # MACD
    
    # Bands
    BOLLINGER = "bollinger"   # Bollinger Bands
    
    # Volume
    VOLUME_SMA = "volume_sma" # Volume Simple Moving Average
    
    @classmethod
    def get_trend_indicators(cls) -> List[str]:
        """Retorna indicadores de tendência"""
        return [cls.SMA, cls.EMA, cls.WMA, cls.MACD]
    
    @classmethod
    def get_momentum_indicators(cls) -> List[str]:
        """Retorna indicadores de momentum"""
        return [cls.RSI, cls.STOCH]
    
    @classmethod
    def get_volatility_indicators(cls) -> List[str]:
        """Retorna indicadores de volatilidade"""
        return [cls.BOLLINGER]


# Constantes derivadas dos enums (para facilitar uso)
VALID_GRANULARITIES = Granularity.get_valid_values()
VALID_NORMALIZATIONS = NormalizationMethod.get_valid_values()
PRIMARY_DATA_SOURCE = DataSource.get_primary_source()

# Mapeamento para Yahoo Finance
GRANULARITY_TO_YFINANCE = {
    Granularity.ONE_MIN: "1m",
    Granularity.FIVE_MIN: "5m",
    Granularity.FIFTEEN_MIN: "15m", 
    Granularity.ONE_HOUR: "1h",
    Granularity.ONE_DAY: "1d"
}

# Thresholds para quality score
QUALITY_THRESHOLDS = {
    "excellent": 0.95,
    "good": 0.80,
    "fair": 0.60,
    "poor": 0.0
}