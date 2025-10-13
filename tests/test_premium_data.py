"""
Testes para Premium Data Sources (Etapa 6)
Testa integração com fontes de dados institucionais
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.premium_data_provider import (
    PremiumDataProvider,
    TickData,
    OrderBookData,
    OrderBookLevel,
    DataSource,
    DataQuality
)
from app.services.alpha_vantage_client import AlphaVantageClient
from app.services.polygon_client import PolygonIOClient


class TestPremiumDataProvider:
    """Testes para PremiumDataProvider"""
    
    @pytest.fixture
    def provider(self):
        """Fixture para PremiumDataProvider"""
        return PremiumDataProvider()
    
    def test_initialization(self, provider):
        """Testa inicialização do provider"""
        assert provider is not None
        assert provider.active_sources is not None
        assert provider.sources is not None
        assert provider.failover_enabled is True
        assert provider.quality_threshold == 0.95
    
    @pytest.mark.asyncio
    async def test_get_tick_data_with_mock(self, provider):
        """Testa obtenção de tick data com mock"""
        # Mock tick data
        mock_tick = TickData(
            timestamp=datetime.now(),
            symbol="EURUSD",
            bid=1.0500,
            ask=1.0501,
            last=1.05005,
            volume=1000000,
            spread=0.0001,
            source=DataSource.POLYGON_IO,
            quality=DataQuality.INSTITUTIONAL,
            latency_ms=25,
            metadata={"test": True}
        )
        
        # Mock polygon client
        with patch.object(provider, 'sources') as mock_sources:
            mock_client = AsyncMock()
            mock_client.get_tick_data.return_value = [mock_tick]
            mock_sources.__getitem__.return_value = mock_client
            provider.active_sources = [DataSource.POLYGON_IO]
            
            # Test
            result = await provider.get_tick_data("EURUSD", limit=1)
            
            assert len(result) == 1
            assert result[0].symbol == "EURUSD"
            assert result[0].bid == 1.0500
            assert result[0].latency_ms == 25
    
    @pytest.mark.asyncio
    async def test_get_order_book_with_mock(self, provider):
        """Testa obtenção de order book com mock"""
        # Mock order book data
        mock_order_book = OrderBookData(
            timestamp=datetime.now(),
            symbol="EURUSD",
            bids=[
                OrderBookLevel(price=1.0500, size=1000000, orders=5),
                OrderBookLevel(price=1.0499, size=2000000, orders=8)
            ],
            asks=[
                OrderBookLevel(price=1.0501, size=1500000, orders=6),
                OrderBookLevel(price=1.0502, size=1800000, orders=7)
            ],
            spread=0.0001,
            mid_price=1.05005,
            total_bid_volume=3000000,
            total_ask_volume=3300000,
            source=DataSource.POLYGON_IO
        )
        
        # Mock polygon client
        with patch.object(provider, 'sources') as mock_sources:
            mock_client = AsyncMock()
            mock_client.get_order_book.return_value = mock_order_book
            mock_sources.__getitem__.return_value = mock_client
            provider.active_sources = [DataSource.POLYGON_IO]
            
            # Test
            result = await provider.get_order_book("EURUSD", depth=5)
            
            assert result is not None
            assert result.symbol == "EURUSD"
            assert len(result.bids) == 2
            assert len(result.asks) == 2
            assert result.spread == 0.0001
    
    @pytest.mark.asyncio
    async def test_get_market_microstructure(self, provider):
        """Testa análise de microestrutura de mercado"""
        # Mock tick data for microstructure analysis
        mock_ticks = [
            TickData(
                timestamp=datetime.now() - timedelta(minutes=i),
                symbol="EURUSD",
                bid=1.0500 + (i * 0.00001),
                ask=1.0501 + (i * 0.00001),
                last=1.05005 + (i * 0.00001),
                volume=1000000,
                spread=0.0001,
                source=DataSource.POLYGON_IO,
                quality=DataQuality.INSTITUTIONAL,
                latency_ms=25 + i,
                metadata={}
            )
            for i in range(10)
        ]
        
        with patch.object(provider, 'get_tick_data', return_value=mock_ticks):
            with patch.object(provider, 'get_order_book', return_value=None):
                result = await provider.get_market_microstructure("EURUSD")
                
                assert 'avg_spread' in result
                assert 'tick_frequency' in result
                assert 'avg_latency_ms' in result
                assert 'liquidity_score' in result
                assert 'data_quality_score' in result
                
                # Verify calculations
                assert result['avg_spread'] == 0.0001
                assert result['avg_latency_ms'] > 0
                assert 0 <= result['liquidity_score'] <= 1
                assert 0 <= result['data_quality_score'] <= 1
    
    def test_validate_data_quality(self, provider):
        """Testa validação de qualidade dos dados"""
        # High quality data
        high_quality_ticks = [
            TickData(
                timestamp=datetime.now(),
                symbol="EURUSD",
                bid=1.0500,
                ask=1.0501,
                last=1.05005,
                volume=1000000,
                spread=0.0001,
                source=DataSource.POLYGON_IO,
                quality=DataQuality.INSTITUTIONAL,
                latency_ms=25,
                metadata={}
            )
            for _ in range(100)
        ]
        
        # Low quality data
        low_quality_ticks = [
            TickData(
                timestamp=datetime.now(),
                symbol="EURUSD",
                bid=1.0500,
                ask=1.0501,
                last=1.05005,
                volume=1000000,
                spread=0.0001,
                source=DataSource.YAHOO_FINANCE,
                quality=DataQuality.RETAIL,
                latency_ms=500,
                metadata={}
            )
            for _ in range(100)
        ]
        
        assert provider._validate_data_quality(high_quality_ticks) is True
        assert provider._validate_data_quality(low_quality_ticks) is False
        assert provider._validate_data_quality([]) is False
    
    def test_calculate_liquidity_score(self, provider):
        """Testa cálculo do score de liquidez"""
        import pandas as pd
        
        # Create test data
        df = pd.DataFrame({
            'spread': [0.0001, 0.0001, 0.0002, 0.0001],
            'volume': [1000000, 1500000, 800000, 1200000]
        })
        
        # Test without order book
        score = provider._calculate_liquidity_score(df, None)
        assert 0 <= score <= 1
        
        # Test with order book
        mock_order_book = OrderBookData(
            timestamp=datetime.now(),
            symbol="EURUSD",
            bids=[OrderBookLevel(price=1.0500, size=5000000, orders=10)],
            asks=[OrderBookLevel(price=1.0501, size=5000000, orders=10)],
            spread=0.0001,
            mid_price=1.05005,
            total_bid_volume=5000000,
            total_ask_volume=5000000,
            source=DataSource.POLYGON_IO
        )
        
        score_with_orderbook = provider._calculate_liquidity_score(df, mock_order_book)
        assert 0 <= score_with_orderbook <= 1
        assert score_with_orderbook >= score  # Should be higher with order book data
    
    @pytest.mark.asyncio
    async def test_data_sources_health(self, provider):
        """Testa verificação de saúde das fontes de dados"""
        # Mock successful responses
        with patch.object(provider.sources[DataSource.YAHOO_FINANCE], 'get_latest_quote') as mock_yahoo:
            mock_yahoo.return_value = {"close": 1.0500}
            
            health = await provider.get_data_sources_health()
            
            assert isinstance(health, dict)
            assert len(health) > 0
            
            # Check Yahoo Finance (always available)
            if 'yahoo_finance' in health:
                yahoo_health = health['yahoo_finance']
                assert 'status' in yahoo_health
                assert 'latency_ms' in yahoo_health
                assert 'last_update' in yahoo_health


class TestAlphaVantageClient:
    """Testes para AlphaVantageClient"""
    
    @pytest.fixture
    def client(self):
        """Fixture para AlphaVantageClient"""
        return AlphaVantageClient()
    
    @pytest.mark.asyncio
    async def test_make_request_without_api_key(self, client):
        """Testa requisição sem API key"""
        client.api_key = ""
        
        with pytest.raises(ValueError, match="Alpha Vantage API key not configured"):
            await client._make_request({"function": "CURRENCY_EXCHANGE_RATE"})
    
    @pytest.mark.asyncio
    async def test_get_latest_quote_mock(self, client):
        """Testa obtenção de cotação com mock"""
        if not client.api_key:
            # Mock the API response
            mock_response = {
                "Realtime Currency Exchange Rate": {
                    "5. Exchange Rate": "1.05000",
                    "6. Last Refreshed": "2024-01-01 12:00:00",
                    "8. Bid Price": "1.04995",
                    "9. Ask Price": "1.05005"
                }
            }
            
            with patch.object(client, '_make_request', return_value=mock_response):
                result = await client.get_latest_quote("EURUSD")
                
                assert result['symbol'] == "EURUSD"
                assert result['close'] == 1.05000
                assert result['bid'] == 1.04995
                assert result['ask'] == 1.05005
                assert 'timestamp' in result


class TestPolygonIOClient:
    """Testes para PolygonIOClient"""
    
    @pytest.fixture
    def client(self):
        """Fixture para PolygonIOClient"""
        return PolygonIOClient()
    
    @pytest.mark.asyncio
    async def test_make_request_without_api_key(self, client):
        """Testa requisição sem API key"""
        client.api_key = ""
        
        with pytest.raises(ValueError, match="Polygon.io API key not configured"):
            await client._make_request("/v1/test")
    
    @pytest.mark.asyncio
    async def test_get_forex_aggregates_mock(self, client):
        """Testa obtenção de dados agregados com mock"""
        if not client.api_key:
            # Mock the API response
            mock_response = {
                "results": [
                    {
                        "t": int(datetime.now().timestamp() * 1000),
                        "o": 1.0500,
                        "h": 1.0510,
                        "l": 1.0490,
                        "c": 1.0505,
                        "v": 1000000,
                        "vw": 1.0502,
                        "n": 150
                    }
                ]
            }
            
            with patch.object(client, '_make_request', return_value=mock_response):
                result = await client.get_forex_aggregates("C:EURUSD")
                
                assert len(result) == 1
                assert result.iloc[0]['open'] == 1.0500
                assert result.iloc[0]['close'] == 1.0505
                assert result.iloc[0]['source'] == 'polygon_io'


class TestIntegration:
    """Testes de integração entre componentes"""
    
    @pytest.mark.asyncio
    async def test_premium_provider_fallback_chain(self):
        """Testa cadeia de fallback do provider premium"""
        provider = PremiumDataProvider()
        
        # Test fallback to Yahoo Finance when premium sources fail
        with patch.object(provider, 'active_sources', [DataSource.YAHOO_FINANCE]):
            result = await provider.get_tick_data("EURUSD", limit=10)
            
            # Should get some data from fallback
            assert isinstance(result, list)
            # Note: May be empty if Yahoo Finance is not responding
    
    def test_data_quality_metrics_calculation(self):
        """Testa cálculo de métricas de qualidade"""
        provider = PremiumDataProvider()
        
        # Create test data with varying quality
        import pandas as pd
        
        df = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(10)],
            'bid': [1.0500 + (i * 0.00001) for i in range(10)],
            'ask': [1.0501 + (i * 0.00001) for i in range(10)],
            'last': [1.05005 + (i * 0.00001) for i in range(10)],
            'volume': [1000000 + (i * 100000) for i in range(10)],
            'spread': [0.0001] * 10,
            'latency_ms': [25 + i for i in range(10)]
        })
        
        metrics = provider._calculate_microstructure_metrics([], None)
        # Should handle empty data gracefully
        assert isinstance(metrics, dict)
        
        # Test with actual data (converting to TickData list)
        tick_data = []
        for _, row in df.iterrows():
            tick = TickData(
                timestamp=row['timestamp'],
                symbol="EURUSD",
                bid=row['bid'],
                ask=row['ask'],
                last=row['last'],
                volume=row['volume'],
                spread=row['spread'],
                source=DataSource.POLYGON_IO,
                quality=DataQuality.INSTITUTIONAL,
                latency_ms=row['latency_ms'],
                metadata={}
            )
            tick_data.append(tick)
        
        metrics = provider._calculate_microstructure_metrics(tick_data, None)
        
        assert 'avg_spread' in metrics
        assert 'tick_frequency' in metrics
        assert 'price_impact' in metrics
        assert 'liquidity_score' in metrics
        assert metrics['avg_spread'] == 0.0001
        assert metrics['avg_latency_ms'] > 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])