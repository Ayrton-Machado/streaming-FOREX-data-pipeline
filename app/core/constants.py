"""
Constantes do Pipeline FOREX EUR/USD
Valores fixos e mapeamentos usados em todo o sistema
"""

from typing import Dict

# ===== GRANULARIDADES SUPORTADAS =====
GRANULARITY_MAPPING: Dict[str, str] = {
    "1min": "1m",      # Yahoo Finance usa 'm' para minutos
    "5min": "5m",
    "15min": "15m",
    "1h": "1h",
    "1d": "1d",
}

# Granularidades válidas
VALID_GRANULARITIES = list(GRANULARITY_MAPPING.keys())

# ===== FOREX MARKET =====
EURUSD_TICKER = "EURUSD=X"
MARKET_NAME = "EUR/USD"
BASE_CURRENCY = "EUR"
QUOTE_CURRENCY = "USD"

# ===== DATA QUALITY =====
MIN_QUALITY_SCORE = 0.7  # Mínimo aceitável para considerar dados "bons"
MAX_GAP_PERCENTAGE = 5.0  # Máximo de gaps permitidos (%)
OUTLIER_ZSCORE_DEFAULT = 3.0

# ===== TECHNICAL INDICATORS =====
# Períodos padrão para indicadores técnicos
SMA_PERIODS = [20, 50, 200]
EMA_PERIODS = [12, 26]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# ===== API LIMITS =====
MAX_CANDLES_PER_REQUEST = 10000  # Limite de velas por request
DEFAULT_QUOTES_LIMIT = 100
MAX_QUOTES_LIMIT = 1000

# ===== CACHE KEYS =====
CACHE_KEY_PREFIX = "forex:eurusd"
CACHE_KEY_QUOTES = f"{CACHE_KEY_PREFIX}:quotes"
CACHE_KEY_ML_READY = f"{CACHE_KEY_PREFIX}:ml_ready"

# ===== TIMEZONE =====
FOREX_TIMEZONE = "UTC"  # Sempre trabalhamos em UTC

# ===== ERROR MESSAGES =====
ERROR_MESSAGES = {
    "invalid_granularity": f"Granularidade inválida. Use uma de: {VALID_GRANULARITIES}",
    "invalid_date_range": "Data de início deve ser anterior à data de fim",
    "max_range_exceeded": "Período solicitado excede o máximo permitido",
    "no_data_found": "Nenhum dado encontrado para o período especificado",
    "data_fetch_error": "Erro ao buscar dados da fonte externa",
    "validation_error": "Dados falharam na validação de qualidade",
}
