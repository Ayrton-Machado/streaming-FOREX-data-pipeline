"""
Feature Engineer - Indicadores técnicos e feature engineering
Etapa 4: SMA, EMA, RSI, volatilidade e features customizadas
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from app.domain.schemas import ValidatedOHLCV, MLReadyCandle, TechnicalIndicators
from app.domain.enums import TechnicalIndicatorType
from app.core.config import settings

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Serviço para feature engineering e cálculo de indicadores técnicos
    
    Filosofia SLC:
    - Simple: Métodos bem definidos, parâmetros claros
    - Lovable: Indicadores populares, validação automática
    - Complete: Ampla gama de indicadores, features customizadas
    """
    
    def __init__(self):
        logger.info("FeatureEngineer inicializado")
    
    def engineer_features(
        self,
        candles: List[ValidatedOHLCV],
        indicators: List[TechnicalIndicatorType] = None
    ) -> List[MLReadyCandle]:
        """
        Aplica feature engineering completo em lista de velas
        
        Args:
            candles: Lista de velas validadas
            indicators: Lista de indicadores a calcular (None = todos)
            
        Returns:
            Lista de MLReadyCandle com features engineered
        """
        if not candles:
            logger.warning("Lista de velas vazia")
            return []
        
        if len(candles) < 50:
            logger.warning(f"Apenas {len(candles)} velas - alguns indicadores podem não ser calculados")
        
        # Converte para DataFrame
        df = self._candles_to_dataframe(candles)
        
        # Calcula todos os indicadores
        if indicators is None:
            indicators = list(TechnicalIndicatorType)
        
        # Moving Averages
        if TechnicalIndicatorType.SMA in indicators:
            df = self._calculate_sma(df)
        
        if TechnicalIndicatorType.EMA in indicators:
            df = self._calculate_ema(df)
        
        # Oscillators
        if TechnicalIndicatorType.RSI in indicators:
            df = self._calculate_rsi(df)
        
        if TechnicalIndicatorType.MACD in indicators:
            df = self._calculate_macd(df)
        
        # Volatility
        if TechnicalIndicatorType.BOLLINGER_BANDS in indicators:
            df = self._calculate_bollinger_bands(df)
        
        if TechnicalIndicatorType.ATR in indicators:
            df = self._calculate_atr(df)
        
        # Volume (mesmo que seja 0 para FOREX)
        if TechnicalIndicatorType.VOLUME_SMA in indicators:
            df = self._calculate_volume_indicators(df)
        
        # Custom features
        df = self._calculate_custom_features(df)
        
        # Converte de volta para MLReadyCandle
        ml_candles = self._dataframe_to_ml_candles(df, candles)
        
        logger.info(f"✅ Feature engineering completo: {len(ml_candles)} velas com {len(indicators)} indicadores")
        return ml_candles
    
    def _calculate_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula Simple Moving Averages"""
        periods = [5, 10, 20, 50, 100, 200]
        
        for period in periods:
            if len(df) >= period:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        return df
    
    def _calculate_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula Exponential Moving Averages"""
        periods = [5, 10, 20, 50, 100, 200]
        
        for period in periods:
            if len(df) >= period:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calcula Relative Strength Index"""
        if len(df) < period + 1:
            return df
        
        # Calcula mudanças de preço
        price_changes = df['close'].diff()
        
        # Separa gains e losses
        gains = price_changes.where(price_changes > 0, 0)
        losses = -price_changes.where(price_changes < 0, 0)
        
        # Calcula médias usando EMA
        avg_gains = gains.ewm(span=period).mean()
        avg_losses = losses.ewm(span=period).mean()
        
        # Calcula RS e RSI
        rs = avg_gains / avg_losses
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _calculate_macd(
        self, 
        df: pd.DataFrame, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> pd.DataFrame:
        """Calcula MACD (Moving Average Convergence Divergence)"""
        if len(df) < slow_period:
            return df
        
        # Calcula EMAs
        ema_fast = df['close'].ewm(span=fast_period).mean()
        ema_slow = df['close'].ewm(span=slow_period).mean()
        
        # MACD Line
        df['macd'] = ema_fast - ema_slow
        
        # Signal Line
        df['macd_signal'] = df['macd'].ewm(span=signal_period).mean()
        
        # MACD Histogram
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def _calculate_bollinger_bands(
        self, 
        df: pd.DataFrame, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """Calcula Bollinger Bands"""
        if len(df) < period:
            return df
        
        # SMA como linha central
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        df['bb_middle'] = sma
        df['bb_upper'] = sma + (std * std_dev)
        df['bb_lower'] = sma - (std * std_dev)
        
        # Bollinger Band Width
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # %B (posição dentro das bandas)
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calcula Average True Range"""
        if len(df) < 2:
            return df
        
        # True Range calculation
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # ATR usando EMA
        df['atr_14'] = true_range.ewm(span=period).mean()
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores de volume (mesmo que FOREX tenha volume = 0)"""
        periods = [10, 20, 50]
        
        for period in periods:
            if len(df) >= period:
                df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        
        # Volume ratio (volume atual vs média)
        if len(df) >= 20:
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            df['volume_ratio'] = df['volume_ratio'].fillna(1.0)  # Para FOREX
        
        return df
    
    def _calculate_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula features customizadas"""
        
        # Price changes
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change() * 100
        
        # High-Low percentage
        df['high_low_pct'] = ((df['high'] - df['low']) / df['close']) * 100
        
        # Open-Close relationship
        df['open_close_pct'] = ((df['close'] - df['open']) / df['open']) * 100
        
        # Rolling volatility (standard deviation of returns)
        periods = [10, 30, 60]
        for period in periods:
            if len(df) >= period:
                returns = df['close'].pct_change()
                df[f'volatility_{period}'] = returns.rolling(window=period).std() * 100
        
        # Price position relative to recent high/low
        periods = [10, 20, 50]
        for period in periods:
            if len(df) >= period:
                rolling_high = df['high'].rolling(window=period).max()
                rolling_low = df['low'].rolling(window=period).min()
                
                df[f'price_position_{period}'] = (
                    (df['close'] - rolling_low) / (rolling_high - rolling_low)
                ) * 100
        
        # Momentum indicators
        periods = [5, 10, 20]
        for period in periods:
            if len(df) >= period:
                df[f'momentum_{period}'] = (df['close'] / df['close'].shift(period) - 1) * 100
        
        # Moving average relationships
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            df['sma_ratio_20_50'] = df['sma_20'] / df['sma_50']
        
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            df['ema_ratio_12_26'] = df['ema_12'] / df['ema_26']
        
        # Support/Resistance levels (simplified)
        if len(df) >= 20:
            df['resistance_20'] = df['high'].rolling(window=20).max()
            df['support_20'] = df['low'].rolling(window=20).min()
            
            # Distance to support/resistance
            df['dist_to_resistance'] = ((df['resistance_20'] - df['close']) / df['close']) * 100
            df['dist_to_support'] = ((df['close'] - df['support_20']) / df['close']) * 100
        
        # Candlestick patterns (simplified)
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Doji detection
        df['is_doji'] = df['body_size'] < (df['high'] - df['low']) * 0.1
        
        return df
    
    def calculate_single_candle_features(
        self,
        current_candle: ValidatedOHLCV,
        historical_candles: List[ValidatedOHLCV]
    ) -> Dict[str, float]:
        """
        Calcula features para uma única vela usando histórico
        Útil para processamento em tempo real
        """
        if not historical_candles:
            return {}
        
        # Combina histórico + vela atual
        all_candles = historical_candles + [current_candle]
        
        # Calcula features
        ml_candles = self.engineer_features(all_candles)
        
        if not ml_candles:
            return {}
        
        # Retorna features da última vela (atual)
        return ml_candles[-1].technical_indicators
    
    def get_feature_importance_weights(self) -> Dict[str, float]:
        """
        Retorna pesos de importância para features
        Baseado em conhecimento de mercado FOREX
        """
        return {
            # Moving Averages (alta importância)
            'sma_20': 0.9,
            'sma_50': 0.8,
            'ema_20': 0.9,
            'ema_50': 0.8,
            
            # Oscillators (média-alta importância)
            'rsi_14': 0.8,
            'macd': 0.7,
            'macd_signal': 0.6,
            
            # Volatility (alta importância para FOREX)
            'atr_14': 0.9,
            'volatility_30': 0.8,
            'bb_width': 0.7,
            
            # Price action (muito alta importância)
            'price_change_pct': 1.0,
            'momentum_10': 0.8,
            'momentum_20': 0.7,
            
            # Support/Resistance
            'dist_to_resistance': 0.6,
            'dist_to_support': 0.6,
            'price_position_20': 0.7,
            
            # Volume (baixa para FOREX)
            'volume_ratio': 0.2,
            'volume_sma_20': 0.1,
        }
    
    def validate_features(self, ml_candles: List[MLReadyCandle]) -> Dict[str, Any]:
        """
        Valida qualidade das features calculadas
        """
        if not ml_candles:
            return {"error": "Lista vazia", "is_valid": False}
        
        # Extrai todas as features
        all_features = {}
        for candle in ml_candles:
            for feature_name, value in candle.technical_indicators.items():
                if feature_name not in all_features:
                    all_features[feature_name] = []
                all_features[feature_name].append(value)
        
        validation_result = {
            "total_candles": len(ml_candles),
            "total_features": len(all_features),
            "feature_stats": {},
            "issues": [],
            "is_valid": True
        }
        
        # Valida cada feature
        for feature_name, values in all_features.items():
            values_array = np.array(values)
            
            stats = {
                "count": len(values),
                "non_null_count": int(np.sum(~np.isnan(values_array))),
                "null_percentage": float(np.sum(np.isnan(values_array)) / len(values) * 100),
                "min": float(np.nanmin(values_array)) if len(values) > 0 else None,
                "max": float(np.nanmax(values_array)) if len(values) > 0 else None,
                "mean": float(np.nanmean(values_array)) if len(values) > 0 else None,
                "std": float(np.nanstd(values_array)) if len(values) > 0 else None,
                "has_inf": bool(np.any(np.isinf(values_array)))
            }
            
            validation_result["feature_stats"][feature_name] = stats
            
            # Verifica problemas
            if stats["null_percentage"] > 50:
                validation_result["issues"].append(f"{feature_name}: {stats['null_percentage']:.1f}% valores nulos")
                validation_result["is_valid"] = False
            
            if stats["has_inf"]:
                validation_result["issues"].append(f"{feature_name}: contém valores infinitos")
                validation_result["is_valid"] = False
        
        return validation_result
    
    def _candles_to_dataframe(self, candles: List[ValidatedOHLCV]) -> pd.DataFrame:
        """Converte lista de velas para DataFrame"""
        data = []
        for candle in candles:
            data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    def _dataframe_to_ml_candles(
        self, 
        df: pd.DataFrame, 
        original_candles: List[ValidatedOHLCV]
    ) -> List[MLReadyCandle]:
        """Converte DataFrame com features para MLReadyCandle"""
        ml_candles = []
        
        # Identifica colunas de indicadores técnicos
        feature_columns = [col for col in df.columns if col not in 
                         ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        for i, (_, row) in enumerate(df.iterrows()):
            # Extrai features técnicas
            technical_indicators = {}
            for col in feature_columns:
                value = row[col]
                if pd.notna(value):  # Só adiciona se não for NaN
                    technical_indicators[col] = float(value)
            
            # Cria MLReadyCandle
            ml_candle = MLReadyCandle(
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                data_source=original_candles[i].data_source,
                raw_symbol=original_candles[i].raw_symbol,
                validation=original_candles[i].validation,
                is_outlier=original_candles[i].is_outlier,
                is_gap_fill=original_candles[i].is_gap_fill,
                normalized_features={},  # Será preenchido pelo DataNormalizer
                technical_indicators=technical_indicators,
                market_features={}       # Será preenchido pelo MarketFilters
            )
            
            ml_candles.append(ml_candle)
        
        return ml_candles


# Factory function
def get_feature_engineer() -> FeatureEngineer:
    """Retorna instância de FeatureEngineer"""
    return FeatureEngineer()