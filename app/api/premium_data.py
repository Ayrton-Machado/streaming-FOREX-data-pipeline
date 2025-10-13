"""
Premium Data API Endpoints - Institutional Grade Data Access
Provides high-quality, low-latency data for AI trading systems
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

from app.services.premium_data_provider import (
    PremiumDataProvider, 
    TickData, 
    OrderBookData,
    DataSource,
    DataQuality
)


# Pydantic Models for API
class TickDataResponse(BaseModel):
    """Response model for tick data"""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    spread: float
    source: str
    quality: str
    latency_ms: float
    metadata: Dict[str, Any]


class OrderBookLevelResponse(BaseModel):
    """Order book level response"""
    price: float
    size: float
    orders: int


class OrderBookResponse(BaseModel):
    """Order book response model"""
    timestamp: datetime
    symbol: str
    bids: List[OrderBookLevelResponse]
    asks: List[OrderBookLevelResponse]
    spread: float
    mid_price: float
    total_bid_volume: float
    total_ask_volume: float
    source: str


class MarketMicrostructureResponse(BaseModel):
    """Market microstructure metrics response"""
    avg_spread: float
    spread_volatility: float
    tick_frequency: float
    avg_latency_ms: float
    volume_profile: Dict[str, float]
    price_impact: float
    liquidity_score: float
    data_quality_score: float
    order_book: Optional[Dict[str, Any]] = None


class DataSourceHealthResponse(BaseModel):
    """Data source health response"""
    status: str
    latency_ms: Optional[float] = None
    last_update: str
    data_quality: Optional[str] = None
    error: Optional[str] = None


class PremiumQuoteResponse(BaseModel):
    """Premium quote response"""
    symbol: str
    bid: float
    ask: float
    last: float
    spread: float
    timestamp: datetime
    source: str
    quality: str
    latency_ms: float


# Router
router = APIRouter(prefix="/premium", tags=["Premium Data"])
logger = logging.getLogger(__name__)

# Dependency to get premium data provider
async def get_premium_provider() -> PremiumDataProvider:
    """Dependency to provide premium data provider instance"""
    return PremiumDataProvider()


@router.get("/health", response_model=Dict[str, DataSourceHealthResponse])
async def get_data_sources_health(
    provider: PremiumDataProvider = Depends(get_premium_provider)
):
    """
    Get health status of all premium data sources
    
    Returns real-time status including:
    - Connection status
    - Latency metrics  
    - Data quality scores
    - Error information
    """
    try:
        health_data = await provider.get_data_sources_health()
        
        # Convert to response models
        response = {}
        for source, health in health_data.items():
            response[source] = DataSourceHealthResponse(**health)
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get data sources health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tick-data", response_model=List[TickDataResponse])
async def get_tick_data(
    symbol: str = Query(default="EURUSD", description="Forex pair symbol"),
    limit: int = Query(default=1000, ge=1, le=10000, description="Maximum number of ticks"),
    start_time: Optional[datetime] = Query(default=None, description="Start time (ISO format)"),
    end_time: Optional[datetime] = Query(default=None, description="End time (ISO format)"),
    provider: PremiumDataProvider = Depends(get_premium_provider)
):
    """
    Get institutional-grade tick data
    
    Features:
    - Sub-second timestamps
    - Bid/Ask spreads
    - Quality scoring
    - Latency tracking
    - Automatic failover
    """
    try:
        # Default time range if not provided
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(hours=1)
        
        tick_data = await provider.get_tick_data(symbol, limit, start_time, end_time)
        
        # Convert to response models
        response = []
        for tick in tick_data:
            response.append(TickDataResponse(
                timestamp=tick.timestamp,
                symbol=tick.symbol,
                bid=tick.bid,
                ask=tick.ask,
                last=tick.last,
                volume=tick.volume,
                spread=tick.spread,
                source=tick.source.value,
                quality=tick.quality.value,
                latency_ms=tick.latency_ms,
                metadata=tick.metadata
            ))
        
        logger.info(f"Returned {len(response)} ticks for {symbol}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to get tick data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/order-book", response_model=Optional[OrderBookResponse])
async def get_order_book(
    symbol: str = Query(default="EURUSD", description="Forex pair symbol"),
    depth: int = Query(default=10, ge=1, le=50, description="Order book depth"),
    provider: PremiumDataProvider = Depends(get_premium_provider)
):
    """
    Get real-time order book data (Level II)
    
    Features:
    - Bid/Ask levels with sizes
    - Order count per level
    - Volume imbalance metrics
    - Liquidity analysis
    """
    try:
        order_book = await provider.get_order_book(symbol, depth)
        
        if not order_book:
            return None
        
        # Convert to response model
        response = OrderBookResponse(
            timestamp=order_book.timestamp,
            symbol=order_book.symbol,
            bids=[
                OrderBookLevelResponse(
                    price=level.price,
                    size=level.size,
                    orders=level.orders
                ) for level in order_book.bids
            ],
            asks=[
                OrderBookLevelResponse(
                    price=level.price,
                    size=level.size,
                    orders=level.orders
                ) for level in order_book.asks
            ],
            spread=order_book.spread,
            mid_price=order_book.mid_price,
            total_bid_volume=order_book.total_bid_volume,
            total_ask_volume=order_book.total_ask_volume,
            source=order_book.source.value
        )
        
        logger.info(f"Returned order book for {symbol} with {depth} levels")
        return response
        
    except Exception as e:
        logger.error(f"Failed to get order book: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/microstructure", response_model=MarketMicrostructureResponse)
async def get_market_microstructure(
    symbol: str = Query(default="EURUSD", description="Forex pair symbol"),
    timeframe: str = Query(default="1min", description="Analysis timeframe"),
    provider: PremiumDataProvider = Depends(get_premium_provider)
):
    """
    Get market microstructure analysis
    
    Advanced metrics:
    - Spread dynamics
    - Liquidity scoring
    - Price impact analysis
    - Order flow metrics
    - Data quality assessment
    """
    try:
        microstructure = await provider.get_market_microstructure(symbol, timeframe)
        
        if not microstructure:
            raise HTTPException(status_code=404, detail="No microstructure data available")
        
        # Convert to response model
        response = MarketMicrostructureResponse(**microstructure)
        
        logger.info(f"Returned microstructure analysis for {symbol}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to get microstructure data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quote/latest", response_model=PremiumQuoteResponse)
async def get_premium_quote(
    symbol: str = Query(default="EURUSD", description="Forex pair symbol"),
    provider: PremiumDataProvider = Depends(get_premium_provider)
):
    """
    Get premium real-time quote
    
    Features:
    - Sub-second latency
    - Bid/Ask with spreads
    - Quality scoring
    - Source attribution
    - Automatic failover
    """
    try:
        # Get latest tick data (1 tick)
        tick_data = await provider.get_tick_data(symbol, limit=1)
        
        if not tick_data:
            raise HTTPException(status_code=404, detail="No quote data available")
        
        latest_tick = tick_data[0]
        
        response = PremiumQuoteResponse(
            symbol=latest_tick.symbol,
            bid=latest_tick.bid,
            ask=latest_tick.ask,
            last=latest_tick.last,
            spread=latest_tick.spread,
            timestamp=latest_tick.timestamp,
            source=latest_tick.source.value,
            quality=latest_tick.quality.value,
            latency_ms=latest_tick.latency_ms
        )
        
        logger.info(f"Returned premium quote for {symbol}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to get premium quote: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources", response_model=Dict[str, Any])
async def get_available_sources(
    provider: PremiumDataProvider = Depends(get_premium_provider)
):
    """
    Get information about available premium data sources
    
    Returns:
    - Active sources
    - Capabilities
    - Quality levels
    - Rate limits
    """
    try:
        sources_info = {
            "active_sources": [source.value for source in provider.active_sources],
            "total_sources": len(provider.sources),
            "failover_enabled": provider.failover_enabled,
            "quality_threshold": provider.quality_threshold,
            "capabilities": {
                "tick_data": ["polygon_io", "fxcm", "alpha_vantage"],
                "order_book": ["polygon_io", "fxcm"],
                "microstructure": ["polygon_io", "fxcm"],
                "economic_data": ["alpha_vantage"],
                "backup": ["yahoo_finance"]
            },
            "quality_levels": {
                "institutional": "< 10ms latency, tick-level",
                "professional": "< 100ms latency, second-level", 
                "retail": "> 1s latency, minute-level"
            }
        }
        
        return sources_info
        
    except Exception as e:
        logger.error(f"Failed to get sources info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=Dict[str, Any])
async def get_premium_data_stats(
    symbol: str = Query(default="EURUSD", description="Forex pair symbol"),
    hours: int = Query(default=24, ge=1, le=168, description="Hours to analyze"),
    provider: PremiumDataProvider = Depends(get_premium_provider)
):
    """
    Get premium data statistics and quality metrics
    
    Provides:
    - Data completeness
    - Average latency
    - Quality scores
    - Source performance
    """
    try:
        # Get recent tick data for analysis
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        tick_data = await provider.get_tick_data(symbol, limit=5000, start_time=start_time, end_time=end_time)
        
        if not tick_data:
            return {"error": "No data available for analysis"}
        
        # Calculate statistics
        total_ticks = len(tick_data)
        avg_latency = sum(tick.latency_ms for tick in tick_data) / total_ticks
        avg_spread = sum(tick.spread for tick in tick_data) / total_ticks
        
        # Source distribution
        sources = {}
        for tick in tick_data:
            source = tick.source.value
            sources[source] = sources.get(source, 0) + 1
        
        # Quality distribution
        qualities = {}
        for tick in tick_data:
            quality = tick.quality.value
            qualities[quality] = qualities.get(quality, 0) + 1
        
        stats = {
            "period": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "hours": hours
            },
            "data_summary": {
                "total_ticks": total_ticks,
                "avg_latency_ms": round(avg_latency, 2),
                "avg_spread": round(avg_spread, 6),
                "tick_frequency_per_minute": round(total_ticks / (hours * 60), 2)
            },
            "source_distribution": sources,
            "quality_distribution": qualities,
            "data_quality_score": sum(1 for tick in tick_data if tick.latency_ms < 100) / total_ticks
        }
        
        logger.info(f"Returned premium data stats for {symbol}")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get premium data stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))