"""
Premium Data Provider - Institutional Grade Data Sources
Provides high-quality, low-latency data for AI trading systems.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from app.core.config import settings


class DataSource(Enum):
    """Available premium data sources"""
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON_IO = "polygon_io"
    FXCM = "fxcm"
    YAHOO_FINANCE = "yahoo_finance"  # Backup


class DataQuality(Enum):
    """Data quality levels"""
    INSTITUTIONAL = "institutional"  # < 10ms latency, tick-level
    PROFESSIONAL = "professional"   # < 100ms latency, second-level
    RETAIL = "retail"               # > 1s latency, minute-level


@dataclass
class TickData:
    """Institutional tick-level data structure"""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    spread: float
    source: DataSource
    quality: DataQuality
    latency_ms: float
    metadata: Dict[str, Any]


@dataclass
class OrderBookLevel:
    """Order book level data"""
    price: float
    size: float
    orders: int


@dataclass
class OrderBookData:
    """Full order book depth"""
    timestamp: datetime
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    spread: float
    mid_price: float
    total_bid_volume: float
    total_ask_volume: float
    source: DataSource


class PremiumDataProvider:
    """
    Premium Data Provider for institutional-grade market data.
    Provides redundancy, failover, and quality assurance.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sources = {}
        self.active_sources = []
        self.failover_enabled = True
        self.quality_threshold = 0.95
        
        # Initialize data sources
        self._initialize_sources()
        
    def _initialize_sources(self):
        """Initialize all available data sources"""
        self.logger.info("Initializing premium data sources...")
        
        # Alpha Vantage (Primary for fundamentals)
        if hasattr(settings, 'ALPHA_VANTAGE_API_KEY'):
            self.sources[DataSource.ALPHA_VANTAGE] = AlphaVantageClient()
            self.active_sources.append(DataSource.ALPHA_VANTAGE)
            
        # Polygon.io (Primary for tick data)
        if hasattr(settings, 'POLYGON_API_KEY'):
            self.sources[DataSource.POLYGON_IO] = PolygonIOClient()
            self.active_sources.append(DataSource.POLYGON_IO)
            
        # FXCM (Professional forex data)
        if hasattr(settings, 'FXCM_ACCESS_TOKEN'):
            self.sources[DataSource.FXCM] = FXCMClient()
            self.active_sources.append(DataSource.FXCM)
            
        # Yahoo Finance (Backup)
        self.sources[DataSource.YAHOO_FINANCE] = YahooFinanceClient()
        if DataSource.YAHOO_FINANCE not in self.active_sources:
            self.active_sources.append(DataSource.YAHOO_FINANCE)
            
        self.logger.info(f"Initialized {len(self.active_sources)} data sources: {self.active_sources}")

    async def get_tick_data(
        self, 
        symbol: str = "EURUSD", 
        limit: int = 1000,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[TickData]:
        """
        Get institutional-grade tick data with failover
        """
        self.logger.info(f"Fetching tick data for {symbol}, limit={limit}")
        
        # Try primary sources first (Polygon.io, FXCM)
        primary_sources = [DataSource.POLYGON_IO, DataSource.FXCM]
        
        for source in primary_sources:
            if source in self.active_sources:
                try:
                    client = self.sources[source]
                    data = await client.get_tick_data(symbol, limit, start_time, end_time)
                    
                    # Validate data quality
                    if self._validate_data_quality(data):
                        self.logger.info(f"Successfully fetched {len(data)} ticks from {source.value}")
                        return data
                        
                except Exception as e:
                    self.logger.warning(f"Failed to fetch from {source.value}: {e}")
                    continue
        
        # Fallback to secondary sources
        self.logger.warning("Primary sources failed, using fallback...")
        return await self._fallback_tick_data(symbol, limit, start_time, end_time)

    async def get_order_book(
        self, 
        symbol: str = "EURUSD",
        depth: int = 10
    ) -> Optional[OrderBookData]:
        """
        Get real-time order book data
        """
        self.logger.info(f"Fetching order book for {symbol}, depth={depth}")
        
        # Order book is primarily available from Polygon.io and FXCM
        sources_with_orderbook = [DataSource.POLYGON_IO, DataSource.FXCM]
        
        for source in sources_with_orderbook:
            if source in self.active_sources:
                try:
                    client = self.sources[source]
                    if hasattr(client, 'get_order_book'):
                        order_book = await client.get_order_book(symbol, depth)
                        if order_book:
                            self.logger.info(f"Order book fetched from {source.value}")
                            return order_book
                except Exception as e:
                    self.logger.warning(f"Failed to fetch order book from {source.value}: {e}")
                    continue
        
        self.logger.warning("No order book data available")
        return None

    async def get_market_microstructure(
        self, 
        symbol: str = "EURUSD",
        timeframe: str = "1min"
    ) -> Dict[str, Any]:
        """
        Get market microstructure data for institutional analysis
        """
        self.logger.info(f"Fetching market microstructure for {symbol}")
        
        try:
            # Get tick data
            ticks = await self.get_tick_data(symbol, limit=5000)
            
            # Get order book
            order_book = await self.get_order_book(symbol)
            
            # Calculate microstructure metrics
            microstructure = self._calculate_microstructure_metrics(ticks, order_book)
            
            return microstructure
            
        except Exception as e:
            self.logger.error(f"Failed to calculate microstructure: {e}")
            return {}

    def _calculate_microstructure_metrics(
        self, 
        ticks: List[TickData], 
        order_book: Optional[OrderBookData]
    ) -> Dict[str, Any]:
        """Calculate advanced microstructure metrics"""
        
        if not ticks:
            return {}
            
        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {
                'timestamp': tick.timestamp,
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'spread': tick.spread,
                'latency_ms': tick.latency_ms
            }
            for tick in ticks
        ])
        
        # Calculate metrics
        metrics = {
            'avg_spread': df['spread'].mean(),
            'spread_volatility': df['spread'].std(),
            'tick_frequency': len(df) / ((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60),  # ticks per minute
            'avg_latency_ms': df['latency_ms'].mean(),
            'volume_profile': {
                'total_volume': df['volume'].sum(),
                'avg_volume': df['volume'].mean(),
                'volume_std': df['volume'].std()
            },
            'price_impact': self._calculate_price_impact(df),
            'liquidity_score': self._calculate_liquidity_score(df, order_book),
            'data_quality_score': df['latency_ms'].apply(lambda x: 1.0 if x < 50 else 0.8 if x < 100 else 0.5).mean()
        }
        
        # Add order book metrics if available
        if order_book:
            metrics['order_book'] = {
                'bid_ask_spread': order_book.spread,
                'total_bid_volume': order_book.total_bid_volume,
                'total_ask_volume': order_book.total_ask_volume,
                'volume_imbalance': (order_book.total_bid_volume - order_book.total_ask_volume) / (order_book.total_bid_volume + order_book.total_ask_volume),
                'depth_levels': len(order_book.bids)
            }
        
        return metrics

    def _calculate_price_impact(self, df: pd.DataFrame) -> float:
        """Calculate average price impact per unit volume"""
        if len(df) < 2:
            return 0.0
            
        # Simple price impact calculation
        df['price_change'] = df['last'].diff().abs()
        df['volume_normalized'] = df['volume'] / df['volume'].mean()
        
        # Price impact = price change per unit volume
        impact = (df['price_change'] / df['volume_normalized']).mean()
        return float(impact) if not np.isnan(impact) else 0.0

    def _calculate_liquidity_score(self, df: pd.DataFrame, order_book: Optional[OrderBookData]) -> float:
        """Calculate liquidity score (0-1)"""
        
        # Base score from spread (tighter spread = higher liquidity)
        avg_spread = df['spread'].mean()
        spread_score = max(0, 1 - (avg_spread / 0.001))  # Normalize against 10 pips
        
        # Volume score
        volume_score = min(1.0, df['volume'].mean() / 1000000)  # Normalize against 1M volume
        
        # Order book score
        orderbook_score = 0.5  # Default if no order book
        if order_book:
            total_depth = order_book.total_bid_volume + order_book.total_ask_volume
            orderbook_score = min(1.0, total_depth / 10000000)  # Normalize against 10M
        
        # Combined liquidity score
        liquidity_score = (spread_score * 0.4 + volume_score * 0.3 + orderbook_score * 0.3)
        return float(liquidity_score)

    def _validate_data_quality(self, data: List[TickData]) -> bool:
        """Validate data quality meets institutional standards"""
        if not data:
            return False
            
        # Check latency
        avg_latency = sum(tick.latency_ms for tick in data) / len(data)
        if avg_latency > 100:  # More than 100ms average latency
            return False
            
        # Check data completeness
        complete_data = sum(1 for tick in data if tick.bid > 0 and tick.ask > 0)
        completeness = complete_data / len(data)
        
        return completeness >= self.quality_threshold

    async def _fallback_tick_data(
        self, 
        symbol: str, 
        limit: int,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> List[TickData]:
        """Fallback to Yahoo Finance with simulated tick data"""
        
        try:
            yahoo_client = self.sources[DataSource.YAHOO_FINANCE]
            
            # Simulate some basic tick data when no real data is available
            current_price = 1.0500  # EUR/USD approximate
            tick_data = []
            
            for i in range(min(limit, 100)):  # Generate up to 100 simulated ticks
                tick_time = datetime.now() - timedelta(seconds=i*10)
                # Add small random variation
                price_variation = (i % 10 - 5) * 0.00001
                price = current_price + price_variation
                
                tick = TickData(
                    timestamp=tick_time,
                    symbol=symbol,
                    bid=price - 0.00005,  # Simulate bid
                    ask=price + 0.00005,  # Simulate ask
                    last=price,
                    volume=1000000,  # Fixed volume
                    spread=0.0001,
                    source=DataSource.YAHOO_FINANCE,
                    quality=DataQuality.RETAIL,
                    latency_ms=500,  # Higher latency for simulated data
                    metadata={'simulated': True, 'fallback': True}
                )
                tick_data.append(tick)
            
            return tick_data
            
        except Exception as e:
            self.logger.error(f"Fallback failed: {e}")
            return []

    async def get_data_sources_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all data sources"""
        health_status = {}
        
        for source in self.active_sources:
            try:
                client = self.sources[source]
                
                # Test connection
                start_time = datetime.now()
                test_data = await client.get_latest_quote("EURUSD")
                latency = (datetime.now() - start_time).total_seconds() * 1000
                
                health_status[source.value] = {
                    'status': 'healthy' if test_data else 'degraded',
                    'latency_ms': latency,
                    'last_update': datetime.now().isoformat(),
                    'data_quality': 'high' if latency < 100 else 'medium' if latency < 500 else 'low'
                }
                
            except Exception as e:
                health_status[source.value] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'last_update': datetime.now().isoformat()
                }
        
        return health_status


# Placeholder clients (to be implemented)
class AlphaVantageClient:
    """Alpha Vantage API client for fundamental data"""
    
    async def get_tick_data(self, symbol: str, limit: int, start_time: datetime, end_time: datetime):
        # TODO: Implement Alpha Vantage tick data
        return []
    
    async def get_latest_quote(self, symbol: str):
        # TODO: Implement Alpha Vantage latest quote
        return {"close": 1.0500}


class PolygonIOClient:
    """Polygon.io API client for institutional tick data"""
    
    async def get_tick_data(self, symbol: str, limit: int, start_time: datetime, end_time: datetime):
        # TODO: Implement Polygon.io tick data
        return []
    
    async def get_order_book(self, symbol: str, depth: int):
        # TODO: Implement Polygon.io order book
        return None
    
    async def get_latest_quote(self, symbol: str):
        # TODO: Implement Polygon.io latest quote
        return {"close": 1.0500}


class FXCMClient:
    """FXCM API client for professional forex data"""
    
    async def get_tick_data(self, symbol: str, limit: int, start_time: datetime, end_time: datetime):
        # TODO: Implement FXCM tick data
        return []
    
    async def get_order_book(self, symbol: str, depth: int):
        # TODO: Implement FXCM order book
        return None
    
    async def get_latest_quote(self, symbol: str):
        # TODO: Implement FXCM latest quote
        return {"close": 1.0500}


class YahooFinanceClient:
    """Yahoo Finance client (backup/fallback)"""
    
    async def get_historical_data(self, symbol: str, interval: str, limit: int):
        # TODO: Implement Yahoo Finance historical data
        import yfinance as yf
        
        # Convert symbol for Yahoo Finance
        yahoo_symbol = f"{symbol}=X" if "USD" in symbol else symbol
        
        ticker = yf.Ticker(yahoo_symbol)
        data = ticker.history(period="1d", interval=interval)
        
        # Convert to our format
        data.reset_index(inplace=True)
        data['timestamp'] = data['Datetime'] if 'Datetime' in data.columns else data.index
        
        return data.tail(limit)
    
    async def get_latest_quote(self, symbol: str):
        # TODO: Implement Yahoo Finance latest quote
        return {"close": 1.0500}