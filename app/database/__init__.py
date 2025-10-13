"""
Database Package - TimescaleDB integration
Etapa 4: Persistência de dados FOREX com otimizações para séries temporais
"""

from .connection import engine, SessionLocal, get_db
from .models import Base, OHLCVData, TechnicalIndicators, DataQualityLog
from .repository import OHLCVRepository

__all__ = [
    "engine",
    "SessionLocal", 
    "get_db",
    "Base",
    "OHLCVData",
    "TechnicalIndicators", 
    "DataQualityLog",
    "OHLCVRepository"
]