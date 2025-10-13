"""
Database Models - TimescaleDB optimized
Etapa 4: Models SQLAlchemy para dados FOREX com otimizações para séries temporais
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, DateTime, Numeric, Boolean, 
    Index, ForeignKey, Text, Float, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.database.connection import Base
from app.domain.enums import DataSource, DataQuality, Granularity


class OHLCVData(Base):
    """
    Tabela principal para dados OHLCV
    Otimizada para TimescaleDB com particionamento por tempo
    """
    __tablename__ = "ohlcv_data"
    
    # Primary key composta (timestamp + symbol + granularity)
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, default="EURUSD=X")
    granularity = Column(String(10), nullable=False)  # 1min, 5min, 1h, 1d
    
    # Dados OHLCV
    open = Column(Numeric(precision=10, scale=6), nullable=False)
    high = Column(Numeric(precision=10, scale=6), nullable=False)
    low = Column(Numeric(precision=10, scale=6), nullable=False)
    close = Column(Numeric(precision=10, scale=6), nullable=False)
    volume = Column(Integer, default=0)
    
    # Metadados de qualidade
    data_source = Column(String(50), nullable=False)
    quality_score = Column(Float, nullable=False)
    quality_level = Column(String(20), nullable=False)
    is_outlier = Column(Boolean, default=False)
    is_gap_fill = Column(Boolean, default=False)
    
    # Timestamps de controle
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relacionamentos
    technical_indicators = relationship("TechnicalIndicators", back_populates="ohlcv_data")
    
    # Índices otimizados para TimescaleDB
    __table_args__ = (
        # Índice composto principal (usado pelo TimescaleDB)
        Index('ix_ohlcv_timestamp_symbol_granularity', 'timestamp', 'symbol', 'granularity'),
        # Índice para consultas por quality
        Index('ix_ohlcv_quality', 'quality_level', 'quality_score'),
        # Índice para consultas por data source
        Index('ix_ohlcv_source', 'data_source', 'timestamp'),
        # Unique constraint para evitar duplicatas
        Index('ix_ohlcv_unique', 'timestamp', 'symbol', 'granularity', unique=True),
    )
    
    def __repr__(self):
        return f"<OHLCVData({self.symbol}@{self.timestamp}: {self.close})>"
    
    def to_dict(self):
        """Converte para dict para serialização"""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "granularity": self.granularity,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": self.volume,
            "data_source": self.data_source,
            "quality_score": self.quality_score,
            "quality_level": self.quality_level,
            "is_outlier": self.is_outlier,
            "is_gap_fill": self.is_gap_fill,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class TechnicalIndicators(Base):
    """
    Tabela para indicadores técnicos calculados
    Relacionada com OHLCVData via foreign key
    """
    __tablename__ = "technical_indicators"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ohlcv_id = Column(UUID(as_uuid=True), ForeignKey('ohlcv_data.id'), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    
    # Moving Averages
    sma_5 = Column(Float)    # Simple Moving Average 5 períodos
    sma_10 = Column(Float)   # Simple Moving Average 10 períodos
    sma_20 = Column(Float)   # Simple Moving Average 20 períodos
    sma_50 = Column(Float)   # Simple Moving Average 50 períodos
    
    ema_5 = Column(Float)    # Exponential Moving Average 5 períodos
    ema_10 = Column(Float)   # Exponential Moving Average 10 períodos
    ema_20 = Column(Float)   # Exponential Moving Average 20 períodos
    ema_50 = Column(Float)   # Exponential Moving Average 50 períodos
    
    # Oscillators
    rsi_14 = Column(Float)   # Relative Strength Index 14 períodos
    macd = Column(Float)     # MACD Line
    macd_signal = Column(Float)  # MACD Signal Line
    macd_histogram = Column(Float)  # MACD Histogram
    
    # Volatility
    bb_upper = Column(Float)     # Bollinger Band Upper
    bb_middle = Column(Float)    # Bollinger Band Middle (SMA 20)
    bb_lower = Column(Float)     # Bollinger Band Lower
    bb_width = Column(Float)     # Bollinger Band Width
    
    atr_14 = Column(Float)       # Average True Range 14 períodos
    volatility_10 = Column(Float)  # Rolling volatility 10 períodos
    volatility_30 = Column(Float)  # Rolling volatility 30 períodos
    
    # Volume indicators (mesmo que FOREX tenha volume = 0)
    volume_sma_10 = Column(Float)
    volume_ratio = Column(Float)
    
    # Custom indicators
    price_change_pct = Column(Float)  # Mudança percentual do preço
    high_low_pct = Column(Float)      # (High-Low)/Close percentage
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relacionamento
    ohlcv_data = relationship("OHLCVData", back_populates="technical_indicators")
    
    # Índices
    __table_args__ = (
        Index('ix_tech_timestamp_symbol', 'timestamp', 'symbol'),
        Index('ix_tech_ohlcv_id', 'ohlcv_id'),
    )
    
    def __repr__(self):
        return f"<TechnicalIndicators({self.symbol}@{self.timestamp})>"


class DataQualityLog(Base):
    """
    Log de qualidade de dados para auditoria
    Registra problemas detectados e correções aplicadas
    """
    __tablename__ = "data_quality_log"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    granularity = Column(String(10), nullable=False)
    
    # Detalhes do problema
    issue_type = Column(String(50), nullable=False)  # outlier, gap, invalid_ohlc, etc
    severity = Column(String(20), nullable=False)    # low, medium, high, critical
    description = Column(Text)
    
    # Dados antes/depois da correção
    original_data = Column(JSON)
    corrected_data = Column(JSON)
    
    # Métrica de qualidade
    quality_score_before = Column(Float)
    quality_score_after = Column(Float)
    
    # Ação tomada
    action_taken = Column(String(100))  # removed, interpolated, flagged, etc
    
    # Metadados
    data_source = Column(String(50))
    processing_batch = Column(String(100))  # Para agrupar processamentos
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Índices
    __table_args__ = (
        Index('ix_quality_timestamp', 'timestamp'),
        Index('ix_quality_symbol_issue', 'symbol', 'issue_type'),
        Index('ix_quality_severity', 'severity'),
        Index('ix_quality_batch', 'processing_batch'),
    )
    
    def __repr__(self):
        return f"<DataQualityLog({self.symbol}: {self.issue_type})>"


class MarketStatus(Base):
    """
    Tabela para status do mercado
    Registra horários de abertura/fechamento, feriados, etc.
    """
    __tablename__ = "market_status"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    
    # Status do mercado
    is_open = Column(Boolean, nullable=False)
    market_session = Column(String(50))  # london, new_york, asian, etc
    
    # Horários
    open_time = Column(DateTime(timezone=True))
    close_time = Column(DateTime(timezone=True))
    
    # Observações
    is_holiday = Column(Boolean, default=False)
    holiday_name = Column(String(100))
    notes = Column(Text)
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_market_date_symbol', 'date', 'symbol'),
        Index('ix_market_status', 'is_open', 'date'),
    )
    
    def __repr__(self):
        return f"<MarketStatus({self.symbol}@{self.date}: {'Open' if self.is_open else 'Closed'})>"


# Trigger para auto-update de updated_at (PostgreSQL)
def create_updated_at_trigger():
    """
    Cria trigger PostgreSQL para atualizar updated_at automaticamente
    """
    return """
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = NOW();
        RETURN NEW;
    END;
    $$ language 'plpgsql';
    
    CREATE TRIGGER update_ohlcv_data_updated_at 
        BEFORE UPDATE ON ohlcv_data 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        
    CREATE TRIGGER update_technical_indicators_updated_at 
        BEFORE UPDATE ON technical_indicators 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """