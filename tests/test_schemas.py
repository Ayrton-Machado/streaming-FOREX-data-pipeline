"""
Testes para schemas Pydantic
"""

import pytest
from datetime import datetime
from app.domain.schemas import RawOHLCV, ValidatedOHLCV, BasicQuote
from app.domain.enums import DataSource, DataQuality


class TestRawOHLCV:
    """Testes para schema RawOHLCV"""
    
    def test_valid_ohlcv_creation(self):
        """Testa criação de OHLCV válido"""
        candle = RawOHLCV(
            timestamp=datetime.now(),
            open=1.0850,
            high=1.0870,
            low=1.0840,
            close=1.0860,
            volume=125000,
            data_source=DataSource.YAHOO_FINANCE,
            raw_symbol="EURUSD=X"
        )
        
        assert candle.open == 1.0850
        assert candle.high == 1.0870
        assert candle.low == 1.0840
        assert candle.close == 1.0860
        assert candle.volume == 125000
        assert candle.data_source == DataSource.YAHOO_FINANCE
    
    def test_invalid_high_low(self):
        """Testa que high deve ser >= low"""
        with pytest.raises(ValueError):
            RawOHLCV(
                timestamp=datetime.now(),
                open=1.0850,
                high=1.0840,  # High < Low - inválido
                low=1.0850,
                close=1.0845,
                volume=0
            )
    
    def test_negative_prices(self):
        """Testa que preços não podem ser negativos"""
        with pytest.raises(ValueError):
            RawOHLCV(
                timestamp=datetime.now(),
                open=-1.0,  # Preço negativo - inválido
                high=1.0870,
                low=1.0840,
                close=1.0860,
                volume=0
            )
    
    def test_default_values(self):
        """Testa valores padrão"""
        candle = RawOHLCV(
            timestamp=datetime.now(),
            open=1.0850,
            high=1.0870,
            low=1.0840,
            close=1.0860
        )
        
        assert candle.volume == 0  # Volume padrão
        assert candle.data_source == DataSource.YAHOO_FINANCE  # Source padrão
        assert candle.raw_symbol == "EURUSD=X"  # Symbol padrão


class TestBasicQuote:
    """Testes para schema BasicQuote"""
    
    def test_basic_quote_creation(self):
        """Testa criação de BasicQuote"""
        quote = BasicQuote(
            timestamp=datetime.now(),
            open=1.0850,
            high=1.0870,
            low=1.0840,
            close=1.0860,
            volume=125000
        )
        
        assert quote.close == 1.0860
        assert quote.volume == 125000
    
    def test_ohlc_consistency(self):
        """Testa validação de consistência OHLC"""
        # Este deve passar
        BasicQuote(
            timestamp=datetime.now(),
            open=1.0850,
            high=1.0870,  # High >= max(Open, Close) ✓
            low=1.0840,   # Low <= min(Open, Close) ✓
            close=1.0860,
            volume=0
        )
        
        # Este deve falhar
        with pytest.raises(ValueError):
            BasicQuote(
                timestamp=datetime.now(),
                open=1.0850,
                high=1.0830,  # High < Open - inválido
                low=1.0840,
                close=1.0860,
                volume=0
            )