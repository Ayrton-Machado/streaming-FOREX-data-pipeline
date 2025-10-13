"""
Alpha Vantage Client - Professional financial data
Provides fundamental data, economic indicators, and forex data
"""

import aiohttp
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

from app.core.config import settings


class AlphaVantageClient:
    """
    Alpha Vantage API client for fundamental and economic data
    Free tier: 25 requests/day
    Premium: unlimited requests
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = settings.alpha_vantage_api_key
        self.session = None
        
        if not self.api_key:
            self.logger.warning("Alpha Vantage API key not configured")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Make HTTP request to Alpha Vantage API"""
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not configured")
        
        params['apikey'] = self.api_key
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.get(self.BASE_URL, params=params) as response:
                data = await response.json()
                
                # Check for API errors
                if 'Error Message' in data:
                    raise Exception(f"Alpha Vantage Error: {data['Error Message']}")
                
                if 'Information' in data and 'call frequency' in data['Information']:
                    raise Exception("Alpha Vantage rate limit exceeded")
                
                return data
                
        except Exception as e:
            self.logger.error(f"Alpha Vantage request failed: {e}")
            raise
    
    async def get_forex_daily(self, from_symbol: str = "EUR", to_symbol: str = "USD") -> pd.DataFrame:
        """Get daily forex data"""
        params = {
            'function': 'FX_DAILY',
            'from_symbol': from_symbol,
            'to_symbol': to_symbol,
            'outputsize': 'full'
        }
        
        data = await self._make_request(params)
        
        # Parse response
        time_series_key = f"Time Series FX (Daily)"
        if time_series_key not in data:
            raise ValueError("Unexpected Alpha Vantage response format")
        
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        df_data = []
        for date_str, values in time_series.items():
            df_data.append({
                'timestamp': datetime.strptime(date_str, '%Y-%m-%d'),
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': 0,  # Forex doesn't have volume
                'source': 'alpha_vantage'
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    async def get_forex_intraday(
        self, 
        from_symbol: str = "EUR", 
        to_symbol: str = "USD",
        interval: str = "15min"
    ) -> pd.DataFrame:
        """Get intraday forex data"""
        params = {
            'function': 'FX_INTRADAY',
            'from_symbol': from_symbol,
            'to_symbol': to_symbol,
            'interval': interval,
            'outputsize': 'full'
        }
        
        data = await self._make_request(params)
        
        # Parse response
        time_series_key = f"Time Series FX ({interval})"
        if time_series_key not in data:
            raise ValueError("Unexpected Alpha Vantage response format")
        
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        df_data = []
        for datetime_str, values in time_series.items():
            df_data.append({
                'timestamp': datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S'),
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': 0,  # Forex doesn't have volume
                'source': 'alpha_vantage'
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    async def get_economic_indicator(self, indicator: str = "REAL_GDP") -> pd.DataFrame:
        """Get economic indicator data"""
        params = {
            'function': indicator,
            'interval': 'annual'
        }
        
        data = await self._make_request(params)
        
        # Economic indicators have different response formats
        # Handle the most common ones
        if 'data' in data:
            df_data = []
            for item in data['data']:
                df_data.append({
                    'date': item.get('date', ''),
                    'value': float(item.get('value', 0)) if item.get('value') else None
                })
            
            df = pd.DataFrame(df_data)
            df['timestamp'] = pd.to_datetime(df['date'])
            return df.sort_values('timestamp').reset_index(drop=True)
        
        return pd.DataFrame()
    
    async def get_latest_quote(self, symbol: str = "EURUSD") -> Dict[str, Any]:
        """Get latest forex quote"""
        # For EUR/USD
        from_symbol, to_symbol = symbol[:3], symbol[3:]
        
        params = {
            'function': 'CURRENCY_EXCHANGE_RATE',
            'from_currency': from_symbol,
            'to_currency': to_symbol
        }
        
        data = await self._make_request(params)
        
        # Parse response
        if 'Realtime Currency Exchange Rate' in data:
            rate_data = data['Realtime Currency Exchange Rate']
            return {
                'symbol': symbol,
                'close': float(rate_data['5. Exchange Rate']),
                'timestamp': datetime.strptime(
                    rate_data['6. Last Refreshed'], 
                    '%Y-%m-%d %H:%M:%S'
                ),
                'bid': float(rate_data['8. Bid Price']),
                'ask': float(rate_data['9. Ask Price']),
                'source': 'alpha_vantage'
            }
        
        raise ValueError("Unexpected Alpha Vantage quote response format")
    
    async def get_tick_data(
        self, 
        symbol: str, 
        limit: int = 1000,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Any]:
        """
        Get tick data (Alpha Vantage doesn't provide true tick data)
        Returns high-frequency intraday data instead
        """
        self.logger.warning("Alpha Vantage doesn't provide tick data, using 1min data")
        
        # Get 1-minute data as closest approximation
        from_symbol, to_symbol = symbol[:3], symbol[3:]
        df = await self.get_forex_intraday(from_symbol, to_symbol, "1min")
        
        # Convert to tick-like format
        tick_data = []
        for _, row in df.tail(limit).iterrows():
            # Simulate tick data from OHLC
            tick_data.extend([
                {
                    'timestamp': row['timestamp'],
                    'symbol': symbol,
                    'bid': row['close'] - 0.00005,  # Simulate bid
                    'ask': row['close'] + 0.00005,  # Simulate ask
                    'last': row['close'],
                    'volume': 0,
                    'spread': 0.0001,
                    'source': 'alpha_vantage',
                    'quality': 'professional',
                    'latency_ms': 1000,  # Alpha Vantage has higher latency
                    'metadata': {'simulated_from_1min': True}
                }
            ])
        
        return tick_data[-limit:] if len(tick_data) > limit else tick_data
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get global market status"""
        params = {
            'function': 'MARKET_STATUS'
        }
        
        try:
            data = await self._make_request(params)
            return data
        except Exception as e:
            self.logger.error(f"Failed to get market status: {e}")
            return {
                'markets': [],
                'error': str(e)
            }
    
    async def test_connection(self) -> bool:
        """Test API connection and key validity"""
        try:
            quote = await self.get_latest_quote("EURUSD")
            return bool(quote.get('close'))
        except Exception as e:
            self.logger.error(f"Alpha Vantage connection test failed: {e}")
            return False