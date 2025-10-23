"""
Advanced Feature Engineering Service
Expande o FeatureEngineer com indicadores técnicos avançados e análise de features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class IndicatorType(Enum):
    """Tipos de indicadores técnicos"""
    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OSCILLATOR = "oscillator"

@dataclass
class IndicatorConfig:
    """Configuração de indicador técnico"""
    name: str
    period: int
    type: IndicatorType
    parameters: Optional[Dict[str, Any]] = None
    enabled: bool = True

class AdvancedFeatureEngineer:
    """
    Feature Engineering Avançado para dados FOREX
    
    Funcionalidades:
    - Indicadores técnicos avançados
    - Rolling statistics
    - Correlation features
    - Feature selection automática
    - Pattern recognition básico
    """
    
    def __init__(self):
        self.indicator_configs = self._get_default_indicators()
        self.feature_importance_cache = {}
        
    def _get_default_indicators(self) -> List[IndicatorConfig]:
        """Configurações padrão de indicadores"""
        return [
            # Momentum Indicators
            IndicatorConfig("williams_r", 14, IndicatorType.MOMENTUM, {"lookback": 14}),
            IndicatorConfig("stochastic_k", 14, IndicatorType.MOMENTUM, {"smooth_k": 3}),
            IndicatorConfig("stochastic_d", 14, IndicatorType.MOMENTUM, {"smooth_d": 3}),
            IndicatorConfig("cci", 20, IndicatorType.MOMENTUM, {"constant": 0.015}),
            IndicatorConfig("roc", 12, IndicatorType.MOMENTUM),
            IndicatorConfig("momentum", 10, IndicatorType.MOMENTUM),
            
            # Trend Indicators  
            IndicatorConfig("adx", 14, IndicatorType.TREND),
            IndicatorConfig("aroon_up", 25, IndicatorType.TREND),
            IndicatorConfig("aroon_down", 25, IndicatorType.TREND),
            IndicatorConfig("psar", 14, IndicatorType.TREND, {"af": 0.02, "max_af": 0.2}),
            IndicatorConfig("supertrend", 10, IndicatorType.TREND, {"multiplier": 3.0}),
            
            # Volatility Indicators
            IndicatorConfig("keltner_upper", 20, IndicatorType.VOLATILITY, {"multiplier": 2.0}),
            IndicatorConfig("keltner_lower", 20, IndicatorType.VOLATILITY, {"multiplier": 2.0}),
            IndicatorConfig("donchian_upper", 20, IndicatorType.VOLATILITY),
            IndicatorConfig("donchian_lower", 20, IndicatorType.VOLATILITY),
            IndicatorConfig("true_range", 1, IndicatorType.VOLATILITY),
            
            # Volume Indicators (simulados para FOREX)
            IndicatorConfig("obv", 1, IndicatorType.VOLUME),
            IndicatorConfig("ad_line", 1, IndicatorType.VOLUME),
            IndicatorConfig("cmf", 20, IndicatorType.VOLUME),
            
            # Oscillators
            IndicatorConfig("ultimate_oscillator", 14, IndicatorType.OSCILLATOR, 
                          {"period1": 7, "period2": 14, "period3": 28}),
            IndicatorConfig("trix", 14, IndicatorType.OSCILLATOR),
        ]
    
    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Williams %R Indicator"""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        return williams_r.fillna(0)
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, 
                           d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator (%K and %D)"""
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        k_percent = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent.fillna(50), d_percent.fillna(50)
    
    def calculate_cci(self, df: pd.DataFrame, period: int = 20, 
                     constant: float = 0.015) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma_tp) / (constant * mad)
        return cci.fillna(0)
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = (high - high.shift(1)).where(
            (high - high.shift(1)) > (low.shift(1) - low), 0
        ).where((high - high.shift(1)) > 0, 0)
        
        dm_minus = (low.shift(1) - low).where(
            (low.shift(1) - low) > (high - high.shift(1)), 0
        ).where((low.shift(1) - low) > 0, 0)
        
        # Smoothed values
        tr_smooth = tr.rolling(window=period).mean()
        dm_plus_smooth = dm_plus.rolling(window=period).mean()
        dm_minus_smooth = dm_minus.rolling(window=period).mean()
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # ADX calculation
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return adx.fillna(0)
    
    def calculate_aroon(self, df: pd.DataFrame, period: int = 25) -> Tuple[pd.Series, pd.Series]:
        """Aroon Up and Aroon Down"""
        high = df['high']
        low = df['low']
        
        aroon_up = pd.Series(index=df.index, dtype=float)
        aroon_down = pd.Series(index=df.index, dtype=float)
        
        for i in range(period, len(df)):
            high_slice = high.iloc[i-period+1:i+1]
            low_slice = low.iloc[i-period+1:i+1]
            
            high_idx = high_slice.idxmax()
            low_idx = low_slice.idxmin()
            
            days_since_high = i - high_slice.index.get_loc(high_idx)
            days_since_low = i - low_slice.index.get_loc(low_idx)
            
            aroon_up.iloc[i] = ((period - days_since_high) / period) * 100
            aroon_down.iloc[i] = ((period - days_since_low) / period) * 100
        
        return aroon_up.fillna(50), aroon_down.fillna(50)
    
    def calculate_parabolic_sar(self, df: pd.DataFrame, af_start: float = 0.02, 
                               af_increment: float = 0.02, af_max: float = 0.2) -> pd.Series:
        """Parabolic SAR"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        psar = np.zeros(len(df))
        bull = True
        af = af_start
        ep = low[0] if bull else high[0]
        psar[0] = high[0] if bull else low[0]
        
        for i in range(1, len(df)):
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            
            if bull:
                if low[i] <= psar[i]:
                    bull = False
                    psar[i] = ep
                    af = af_start
                    ep = low[i]
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_increment, af_max)
            else:
                if high[i] >= psar[i]:
                    bull = True
                    psar[i] = ep
                    af = af_start
                    ep = high[i]
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_increment, af_max)
        
        return pd.Series(psar, index=df.index)
    
    def calculate_keltner_channels(self, df: pd.DataFrame, period: int = 20, 
                                 multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        middle_line = typical_price.rolling(window=period).mean()
        
        # True Range for channel width
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        upper_channel = middle_line + (multiplier * atr)
        lower_channel = middle_line - (multiplier * atr)
        
        return upper_channel, middle_line, lower_channel
    
    def calculate_donchian_channels(self, df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Donchian Channels"""
        upper_channel = df['high'].rolling(window=period).max()
        lower_channel = df['low'].rolling(window=period).min()
        
        return upper_channel, lower_channel
    
    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume (simulado para FOREX usando price changes)"""
        # Para FOREX, simularemos volume baseado em volatilidade
        price_change = df['close'].diff()
        volume_proxy = abs(df['high'] - df['low'])  # Range como proxy de volume
        
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = 0
        
        for i in range(1, len(df)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume_proxy.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume_proxy.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def calculate_ultimate_oscillator(self, df: pd.DataFrame, period1: int = 7, 
                                    period2: int = 14, period3: int = 28) -> pd.Series:
        """Ultimate Oscillator"""
        high = df['high']
        low = df['low']
        close = df['close']
        prior_close = close.shift(1)
        
        # True Range and Buying Pressure
        tr1 = high - low
        tr2 = abs(high - prior_close)
        tr3 = abs(low - prior_close)
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        buying_pressure = close - pd.concat([low, prior_close], axis=1).min(axis=1)
        
        # Average calculations for different periods
        avg_bp1 = buying_pressure.rolling(window=period1).sum()
        avg_tr1 = true_range.rolling(window=period1).sum()
        
        avg_bp2 = buying_pressure.rolling(window=period2).sum()
        avg_tr2 = true_range.rolling(window=period2).sum()
        
        avg_bp3 = buying_pressure.rolling(window=period3).sum()
        avg_tr3 = true_range.rolling(window=period3).sum()
        
        # Ultimate Oscillator calculation
        uo = 100 * (
            (4 * avg_bp1 / avg_tr1) + 
            (2 * avg_bp2 / avg_tr2) + 
            (avg_bp3 / avg_tr3)
        ) / (4 + 2 + 1)
        
        return uo.fillna(50)
    
    def calculate_rolling_statistics(self, df: pd.DataFrame, 
                                   windows: List[int] = [5, 10, 20, 50]) -> Dict[str, pd.Series]:
        """Rolling statistics para diferentes janelas"""
        stats = {}
        
        for window in windows:
            # Rolling returns statistics
            returns = df['close'].pct_change()
            stats[f'rolling_mean_{window}'] = returns.rolling(window).mean()
            stats[f'rolling_std_{window}'] = returns.rolling(window).std()
            stats[f'rolling_skew_{window}'] = returns.rolling(window).skew()
            stats[f'rolling_kurt_{window}'] = returns.rolling(window).kurt()
            
            # Rolling price statistics
            stats[f'rolling_max_{window}'] = df['close'].rolling(window).max()
            stats[f'rolling_min_{window}'] = df['close'].rolling(window).min()
            stats[f'rolling_range_{window}'] = stats[f'rolling_max_{window}'] - stats[f'rolling_min_{window}']
            
            # Rolling percentiles
            stats[f'rolling_q75_{window}'] = df['close'].rolling(window).quantile(0.75)
            stats[f'rolling_q25_{window}'] = df['close'].rolling(window).quantile(0.25)
            stats[f'rolling_median_{window}'] = df['close'].rolling(window).median()
        
        return stats
    
    def calculate_correlation_features(self, df: pd.DataFrame, 
                                     window: int = 20) -> Dict[str, pd.Series]:
        """Features baseadas em correlação entre OHLC"""
        corr_features = {}
        
        # Correlações rolling entre OHLC
        corr_features['high_low_corr'] = df['high'].rolling(window).corr(df['low'])
        corr_features['open_close_corr'] = df['open'].rolling(window).corr(df['close'])
        corr_features['high_volume_corr'] = df['high'].rolling(window).corr(df['volume']) if 'volume' in df.columns else pd.Series(0, index=df.index)
        
        # Correlações de returns
        returns = df['close'].pct_change()
        high_returns = df['high'].pct_change()
        low_returns = df['low'].pct_change()
        
        corr_features['returns_high_corr'] = returns.rolling(window).corr(high_returns)
        corr_features['returns_low_corr'] = returns.rolling(window).corr(low_returns)
        
        return corr_features
    
    def calculate_all_advanced_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calcula todos os indicadores avançados"""
        indicators = {}
        
        # Momentum indicators
        indicators['williams_r'] = self.calculate_williams_r(df)
        stoch_k, stoch_d = self.calculate_stochastic(df)
        indicators['stochastic_k'] = stoch_k
        indicators['stochastic_d'] = stoch_d
        indicators['cci'] = self.calculate_cci(df)
        
        # Trend indicators
        indicators['adx'] = self.calculate_adx(df)
        aroon_up, aroon_down = self.calculate_aroon(df)
        indicators['aroon_up'] = aroon_up
        indicators['aroon_down'] = aroon_down
        indicators['aroon_oscillator'] = aroon_up - aroon_down
        indicators['parabolic_sar'] = self.calculate_parabolic_sar(df)
        
        # Volatility indicators
        kelt_upper, kelt_middle, kelt_lower = self.calculate_keltner_channels(df)
        indicators['keltner_upper'] = kelt_upper
        indicators['keltner_middle'] = kelt_middle
        indicators['keltner_lower'] = kelt_lower
        indicators['keltner_squeeze'] = (kelt_upper - kelt_lower) / kelt_middle
        
        donch_upper, donch_lower = self.calculate_donchian_channels(df)
        indicators['donchian_upper'] = donch_upper
        indicators['donchian_lower'] = donch_lower
        indicators['donchian_middle'] = (donch_upper + donch_lower) / 2
        
        # Volume indicators (simulados)
        indicators['obv'] = self.calculate_obv(df)
        
        # Oscillators
        indicators['ultimate_oscillator'] = self.calculate_ultimate_oscillator(df)
        
        # Rolling statistics
        rolling_stats = self.calculate_rolling_statistics(df)
        indicators.update(rolling_stats)
        
        # Correlation features
        corr_features = self.calculate_correlation_features(df)
        indicators.update(corr_features)
        
        # Fill NaN values
        for key, series in indicators.items():
            # use pandas ffill() method directly instead of fillna(method='ffill')
            indicators[key] = series.ffill().fillna(0)
        
        return indicators
    
    def calculate_feature_importance(self, df: pd.DataFrame, 
                                   target_col: str = 'close') -> Dict[str, float]:
        """Calcula importância das features usando correlação e mutual information"""
        from sklearn.feature_selection import mutual_info_regression
        from scipy.stats import pearsonr
        
        # Calcula todas as features
        features = self.calculate_all_advanced_indicators(df)
        
        # Prepara target (returns futuros)
        target = df[target_col].pct_change().shift(-1).dropna()
        
        importance_scores = {}
        
        for feature_name, feature_series in features.items():
            # Alinha feature com target
            aligned_feature = feature_series.loc[target.index]
            
            if len(aligned_feature) > 0 and not aligned_feature.isna().all():
                # Correlação de Pearson
                try:
                    corr, p_value = pearsonr(aligned_feature.dropna(), 
                                           target.loc[aligned_feature.dropna().index])
                    importance_scores[f"{feature_name}_corr"] = abs(corr) if not np.isnan(corr) else 0
                except:
                    importance_scores[f"{feature_name}_corr"] = 0
                
                # Mutual Information (simplificado)
                try:
                    valid_idx = aligned_feature.notna() & target.notna()
                    if valid_idx.sum() > 10:  # Mínimo de pontos para MI
                        mi_score = mutual_info_regression(
                            aligned_feature[valid_idx].values.reshape(-1, 1),
                            target[valid_idx].values,
                            random_state=42
                        )[0]
                        importance_scores[f"{feature_name}_mi"] = mi_score
                    else:
                        importance_scores[f"{feature_name}_mi"] = 0
                except:
                    importance_scores[f"{feature_name}_mi"] = 0
        
        return importance_scores
    
    def select_top_features(self, importance_scores: Dict[str, float], 
                           top_n: int = 20) -> List[str]:
        """Seleciona as top N features baseado na importância"""
        # Combina scores de correlação e mutual information
        combined_scores = {}
        
        for key, score in importance_scores.items():
            if key.endswith('_corr'):
                feature_name = key.replace('_corr', '')
                if feature_name not in combined_scores:
                    combined_scores[feature_name] = 0
                combined_scores[feature_name] += score * 0.5
            elif key.endswith('_mi'):
                feature_name = key.replace('_mi', '')
                if feature_name not in combined_scores:
                    combined_scores[feature_name] = 0
                combined_scores[feature_name] += score * 0.5
        
        # Ordena por score combinado
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [feature for feature, score in sorted_features[:top_n]]
    
    def generate_feature_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gera relatório completo de features"""
        features = self.calculate_all_advanced_indicators(df)
        importance_scores = self.calculate_feature_importance(df)
        top_features = self.select_top_features(importance_scores)
        
        # Estatísticas das features
        feature_stats = {}
        for name, series in features.items():
            feature_stats[name] = {
                'mean': float(series.mean()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'missing_pct': float(series.isna().sum() / len(series) * 100)
            }
        
        return {
            'total_features': len(features),
            'top_features': top_features,
            'feature_importance': importance_scores,
            'feature_statistics': feature_stats,
            'indicator_types': {
                'momentum': 6,
                'trend': 5, 
                'volatility': 6,
                'volume': 1,
                'oscillator': 1,
                'rolling_stats': len([k for k in features.keys() if 'rolling_' in k]),
                'correlation': len([k for k in features.keys() if '_corr' in k])
            }
        }