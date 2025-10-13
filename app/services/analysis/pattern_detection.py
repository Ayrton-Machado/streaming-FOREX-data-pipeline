"""
Technical Pattern Detection Service
Sistema de detecção de padrões técnicos em dados FOREX
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class CandlestickPattern(Enum):
    """Padrões de candlestick"""
    DOJI = "doji"
    HAMMER = "hammer"
    HANGING_MAN = "hanging_man"
    INVERTED_HAMMER = "inverted_hammer"
    SHOOTING_STAR = "shooting_star"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"

class ChartPattern(Enum):
    """Padrões de gráfico"""
    SUPPORT = "support"
    RESISTANCE = "resistance"
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    BREAKOUT_UP = "breakout_up"
    BREAKOUT_DOWN = "breakout_down"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_shoulders"

class PatternConfidence(Enum):
    """Níveis de confiança do padrão"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class PatternSignal:
    """Sinal de padrão detectado"""
    pattern_type: Union[CandlestickPattern, ChartPattern]
    timestamp: pd.Timestamp
    confidence: PatternConfidence
    price_level: float
    signal_strength: float  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"{self.pattern_type.value.upper()} at {self.timestamp} (conf: {self.confidence.value})"

class TechnicalPatternDetector:
    """
    Detector de padrões técnicos para análise FOREX
    
    Funcionalidades:
    - Detecção de padrões candlestick
    - Identificação de suporte/resistência
    - Detecção de tendências
    - Identificação de breakouts
    - Padrões de reversão e continuação
    """
    
    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
        self.pattern_cache = {}
        
    def _calculate_body_size(self, row: pd.Series) -> float:
        """Calcula tamanho do corpo do candle"""
        return abs(row['close'] - row['open'])
    
    def _calculate_upper_shadow(self, row: pd.Series) -> float:
        """Calcula tamanho da sombra superior"""
        return row['high'] - max(row['open'], row['close'])
    
    def _calculate_lower_shadow(self, row: pd.Series) -> float:
        """Calcula tamanho da sombra inferior"""
        return min(row['open'], row['close']) - row['low']
    
    def _calculate_total_range(self, row: pd.Series) -> float:
        """Calcula range total do candle"""
        return row['high'] - row['low']
    
    def _is_bullish(self, row: pd.Series) -> bool:
        """Verifica se é candle bullish"""
        return row['close'] > row['open']
    
    def _is_bearish(self, row: pd.Series) -> bool:
        """Verifica se é candle bearish"""
        return row['close'] < row['open']
    
    def detect_doji(self, df: pd.DataFrame, threshold: float = 0.1) -> List[PatternSignal]:
        """Detecta padrões Doji"""
        signals = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            body_size = self._calculate_body_size(row)
            total_range = self._calculate_total_range(row)
            
            if total_range > 0:
                body_ratio = body_size / total_range
                
                # Doji: corpo muito pequeno em relação ao range total
                if body_ratio <= threshold:
                    confidence = PatternConfidence.HIGH if body_ratio <= threshold/2 else PatternConfidence.MEDIUM
                    
                    signals.append(PatternSignal(
                        pattern_type=CandlestickPattern.DOJI,
                        timestamp=df.index[i],
                        confidence=confidence,
                        price_level=row['close'],
                        signal_strength=1 - body_ratio,
                        metadata={
                            'body_ratio': body_ratio,
                            'total_range': total_range
                        }
                    ))
        
        return signals
    
    def detect_hammer(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detecta padrões Hammer"""
        signals = []
        
        for i in range(1, len(df)):  # Precisa de contexto anterior
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            body_size = self._calculate_body_size(row)
            lower_shadow = self._calculate_lower_shadow(row)
            upper_shadow = self._calculate_upper_shadow(row)
            total_range = self._calculate_total_range(row)
            
            if total_range > 0 and body_size > 0:
                # Hammer: sombra inferior longa, corpo pequeno, sombra superior pequena
                lower_ratio = lower_shadow / total_range
                upper_ratio = upper_shadow / total_range
                body_ratio = body_size / total_range
                
                # Critérios para Hammer
                if (lower_ratio >= 0.6 and  # Sombra inferior >= 60% do range
                    body_ratio <= 0.3 and   # Corpo <= 30% do range
                    upper_ratio <= 0.1 and  # Sombra superior <= 10% do range
                    self._is_bearish(prev_row)):  # Contexto bearish anterior
                    
                    confidence = PatternConfidence.HIGH if lower_ratio >= 0.7 else PatternConfidence.MEDIUM
                    
                    signals.append(PatternSignal(
                        pattern_type=CandlestickPattern.HAMMER,
                        timestamp=df.index[i],
                        confidence=confidence,
                        price_level=row['close'],
                        signal_strength=lower_ratio,
                        metadata={
                            'lower_shadow_ratio': lower_ratio,
                            'body_ratio': body_ratio,
                            'upper_shadow_ratio': upper_ratio
                        }
                    ))
        
        return signals
    
    def detect_engulfing_patterns(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detecta padrões Engulfing (bullish e bearish)"""
        signals = []
        
        for i in range(1, len(df)):
            curr_row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            curr_body_size = self._calculate_body_size(curr_row)
            prev_body_size = self._calculate_body_size(prev_row)
            
            # Bullish Engulfing
            if (self._is_bearish(prev_row) and self._is_bullish(curr_row) and
                curr_row['open'] < prev_row['close'] and
                curr_row['close'] > prev_row['open'] and
                curr_body_size > prev_body_size):
                
                strength = min(1.0, curr_body_size / (prev_body_size + 1e-6))
                confidence = PatternConfidence.HIGH if strength > 1.5 else PatternConfidence.MEDIUM
                
                signals.append(PatternSignal(
                    pattern_type=CandlestickPattern.BULLISH_ENGULFING,
                    timestamp=df.index[i],
                    confidence=confidence,
                    price_level=curr_row['close'],
                    signal_strength=min(1.0, strength),
                    metadata={
                        'size_ratio': curr_body_size / prev_body_size,
                        'prev_body_size': prev_body_size,
                        'curr_body_size': curr_body_size
                    }
                ))
            
            # Bearish Engulfing
            elif (self._is_bullish(prev_row) and self._is_bearish(curr_row) and
                  curr_row['open'] > prev_row['close'] and
                  curr_row['close'] < prev_row['open'] and
                  curr_body_size > prev_body_size):
                
                strength = min(1.0, curr_body_size / (prev_body_size + 1e-6))
                confidence = PatternConfidence.HIGH if strength > 1.5 else PatternConfidence.MEDIUM
                
                signals.append(PatternSignal(
                    pattern_type=CandlestickPattern.BEARISH_ENGULFING,
                    timestamp=df.index[i],
                    confidence=confidence,
                    price_level=curr_row['close'],
                    signal_strength=min(1.0, strength),
                    metadata={
                        'size_ratio': curr_body_size / prev_body_size,
                        'prev_body_size': prev_body_size,
                        'curr_body_size': curr_body_size
                    }
                ))
        
        return signals
    
    def detect_star_patterns(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detecta padrões Morning Star e Evening Star"""
        signals = []
        
        for i in range(2, len(df)):
            first = df.iloc[i-2]
            star = df.iloc[i-1]
            third = df.iloc[i]
            
            first_body = self._calculate_body_size(first)
            star_body = self._calculate_body_size(star)
            third_body = self._calculate_body_size(third)
            
            # Morning Star (padrão bullish)
            if (self._is_bearish(first) and  # Primeiro candle bearish
                star_body < first_body * 0.3 and  # Star com corpo pequeno
                self._is_bullish(third) and  # Terceiro candle bullish
                third['close'] > (first['open'] + first['close']) / 2):  # Recuperação significativa
                
                strength = min(1.0, third_body / (first_body + 1e-6))
                confidence = PatternConfidence.HIGH if strength > 0.8 else PatternConfidence.MEDIUM
                
                signals.append(PatternSignal(
                    pattern_type=CandlestickPattern.MORNING_STAR,
                    timestamp=df.index[i],
                    confidence=confidence,
                    price_level=third['close'],
                    signal_strength=strength,
                    metadata={
                        'star_body_ratio': star_body / first_body,
                        'recovery_level': (third['close'] - first['close']) / first_body
                    }
                ))
            
            # Evening Star (padrão bearish)
            elif (self._is_bullish(first) and  # Primeiro candle bullish
                  star_body < first_body * 0.3 and  # Star com corpo pequeno
                  self._is_bearish(third) and  # Terceiro candle bearish
                  third['close'] < (first['open'] + first['close']) / 2):  # Queda significativa
                
                strength = min(1.0, third_body / (first_body + 1e-6))
                confidence = PatternConfidence.HIGH if strength > 0.8 else PatternConfidence.MEDIUM
                
                signals.append(PatternSignal(
                    pattern_type=CandlestickPattern.EVENING_STAR,
                    timestamp=df.index[i],
                    confidence=confidence,
                    price_level=third['close'],
                    signal_strength=strength,
                    metadata={
                        'star_body_ratio': star_body / first_body,
                        'decline_level': (first['close'] - third['close']) / first_body
                    }
                ))
        
        return signals
    
    def detect_support_resistance(self, df: pd.DataFrame, window: int = 20, 
                                 min_touches: int = 2) -> List[PatternSignal]:
        """Detecta níveis de suporte e resistência"""
        signals = []
        
        # Calcula rolling min/max
        rolling_min = df['low'].rolling(window=window, center=True).min()
        rolling_max = df['high'].rolling(window=window, center=True).max()
        
        # Detecta suportes (mínimos locais)
        for i in range(window, len(df) - window):
            if df['low'].iloc[i] == rolling_min.iloc[i]:
                support_level = df['low'].iloc[i]
                
                # Conta quantas vezes o preço tocou este nível
                touches = 0
                for j in range(max(0, i - window * 2), min(len(df), i + window * 2)):
                    if abs(df['low'].iloc[j] - support_level) / support_level < 0.002:  # 0.2% tolerance
                        touches += 1
                
                if touches >= min_touches:
                    confidence = PatternConfidence.HIGH if touches >= 4 else PatternConfidence.MEDIUM
                    
                    signals.append(PatternSignal(
                        pattern_type=ChartPattern.SUPPORT,
                        timestamp=df.index[i],
                        confidence=confidence,
                        price_level=support_level,
                        signal_strength=min(1.0, touches / 5),
                        metadata={
                            'touches': touches,
                            'window': window
                        }
                    ))
        
        # Detecta resistências (máximos locais)
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == rolling_max.iloc[i]:
                resistance_level = df['high'].iloc[i]
                
                # Conta quantas vezes o preço tocou este nível
                touches = 0
                for j in range(max(0, i - window * 2), min(len(df), i + window * 2)):
                    if abs(df['high'].iloc[j] - resistance_level) / resistance_level < 0.002:  # 0.2% tolerance
                        touches += 1
                
                if touches >= min_touches:
                    confidence = PatternConfidence.HIGH if touches >= 4 else PatternConfidence.MEDIUM
                    
                    signals.append(PatternSignal(
                        pattern_type=ChartPattern.RESISTANCE,
                        timestamp=df.index[i],
                        confidence=confidence,
                        price_level=resistance_level,
                        signal_strength=min(1.0, touches / 5),
                        metadata={
                            'touches': touches,
                            'window': window
                        }
                    ))
        
        return signals
    
    def detect_trends(self, df: pd.DataFrame, window: int = 20) -> List[PatternSignal]:
        """Detecta tendências de alta e baixa"""
        signals = []
        
        # Calcula médias móveis para suavizar
        ma_short = df['close'].rolling(window=window//2).mean()
        ma_long = df['close'].rolling(window=window).mean()
        
        # Detecta mudanças de tendência
        trend_changes = []
        current_trend = None
        
        for i in range(window, len(df)):
            if ma_short.iloc[i] > ma_long.iloc[i]:
                new_trend = 'up'
            else:
                new_trend = 'down'
            
            # Detecta mudança de tendência
            if current_trend != new_trend and current_trend is not None:
                # Calcula força da tendência
                price_change = abs(df['close'].iloc[i] - df['close'].iloc[i-window])
                volatility = df['close'].iloc[i-window:i].std()
                
                if volatility > 0:
                    trend_strength = min(1.0, price_change / (volatility * window))
                    
                    if trend_strength > 0.3:  # Filtro de ruído
                        pattern_type = ChartPattern.UPTREND if new_trend == 'up' else ChartPattern.DOWNTREND
                        confidence = PatternConfidence.HIGH if trend_strength > 0.7 else PatternConfidence.MEDIUM
                        
                        signals.append(PatternSignal(
                            pattern_type=pattern_type,
                            timestamp=df.index[i],
                            confidence=confidence,
                            price_level=df['close'].iloc[i],
                            signal_strength=trend_strength,
                            metadata={
                                'price_change': price_change,
                                'volatility': volatility,
                                'ma_short': ma_short.iloc[i],
                                'ma_long': ma_long.iloc[i]
                            }
                        ))
                
                trend_changes.append((df.index[i], new_trend))
            
            current_trend = new_trend
        
        return signals
    
    def detect_breakouts(self, df: pd.DataFrame, window: int = 20, 
                        min_volume_increase: float = 1.5) -> List[PatternSignal]:
        """Detecta breakouts de suporte/resistência"""
        signals = []
        
        # Calcula Bollinger Bands como proxy para breakouts
        ma = df['close'].rolling(window=window).mean()
        std = df['close'].rolling(window=window).std()
        upper_band = ma + (2 * std)
        lower_band = ma - (2 * std)
        
        # Simula volume usando range (para FOREX)
        volume_proxy = (df['high'] - df['low']).rolling(window=5).mean()
        
        for i in range(window, len(df)):
            current_price = df['close'].iloc[i]
            current_volume = volume_proxy.iloc[i]
            avg_volume = volume_proxy.iloc[i-window:i].mean()
            
            # Breakout para cima
            if (current_price > upper_band.iloc[i] and
                df['close'].iloc[i-1] <= upper_band.iloc[i-1] and
                current_volume > avg_volume * min_volume_increase):
                
                strength = min(1.0, (current_price - upper_band.iloc[i]) / std.iloc[i])
                confidence = PatternConfidence.HIGH if strength > 0.5 else PatternConfidence.MEDIUM
                
                signals.append(PatternSignal(
                    pattern_type=ChartPattern.BREAKOUT_UP,
                    timestamp=df.index[i],
                    confidence=confidence,
                    price_level=current_price,
                    signal_strength=strength,
                    metadata={
                        'upper_band': upper_band.iloc[i],
                        'volume_ratio': current_volume / avg_volume,
                        'std_distance': strength
                    }
                ))
            
            # Breakout para baixo
            elif (current_price < lower_band.iloc[i] and
                  df['close'].iloc[i-1] >= lower_band.iloc[i-1] and
                  current_volume > avg_volume * min_volume_increase):
                
                strength = min(1.0, (lower_band.iloc[i] - current_price) / std.iloc[i])
                confidence = PatternConfidence.HIGH if strength > 0.5 else PatternConfidence.MEDIUM
                
                signals.append(PatternSignal(
                    pattern_type=ChartPattern.BREAKOUT_DOWN,
                    timestamp=df.index[i],
                    confidence=confidence,
                    price_level=current_price,
                    signal_strength=strength,
                    metadata={
                        'lower_band': lower_band.iloc[i],
                        'volume_ratio': current_volume / avg_volume,
                        'std_distance': strength
                    }
                ))
        
        return signals
    
    def detect_all_patterns(self, df: pd.DataFrame) -> Dict[str, List[PatternSignal]]:
        """Detecta todos os padrões disponíveis"""
        print("Detectando padrões candlestick...")
        patterns = {
            'doji': self.detect_doji(df),
            'hammer': self.detect_hammer(df),
            'engulfing': self.detect_engulfing_patterns(df),
            'star_patterns': self.detect_star_patterns(df)
        }
        
        print("Detectando padrões de gráfico...")
        patterns.update({
            'support_resistance': self.detect_support_resistance(df),
            'trends': self.detect_trends(df),
            'breakouts': self.detect_breakouts(df)
        })
        
        return patterns
    
    def filter_patterns_by_confidence(self, patterns: Dict[str, List[PatternSignal]],
                                    min_confidence: PatternConfidence = PatternConfidence.MEDIUM) -> Dict[str, List[PatternSignal]]:
        """Filtra padrões por nível mínimo de confiança"""
        confidence_order = [
            PatternConfidence.LOW,
            PatternConfidence.MEDIUM,
            PatternConfidence.HIGH,
            PatternConfidence.VERY_HIGH
        ]
        
        min_level = confidence_order.index(min_confidence)
        
        filtered_patterns = {}
        for pattern_type, signals in patterns.items():
            filtered_signals = [
                signal for signal in signals
                if confidence_order.index(signal.confidence) >= min_level
            ]
            filtered_patterns[pattern_type] = filtered_signals
        
        return filtered_patterns
    
    def generate_pattern_summary(self, patterns: Dict[str, List[PatternSignal]]) -> Dict[str, Any]:
        """Gera resumo dos padrões detectados"""
        total_patterns = sum(len(signals) for signals in patterns.values())
        
        # Conta por tipo de confiança
        confidence_counts = {conf.value: 0 for conf in PatternConfidence}
        for signals in patterns.values():
            for signal in signals:
                confidence_counts[signal.confidence.value] += 1
        
        # Conta por tipo de padrão
        pattern_counts = {pattern_type: len(signals) for pattern_type, signals in patterns.items()}
        
        # Últimos padrões detectados
        all_signals = []
        for signals in patterns.values():
            all_signals.extend(signals)
        
        # Ordena por timestamp
        all_signals.sort(key=lambda x: x.timestamp, reverse=True)
        recent_patterns = all_signals[:10]
        
        return {
            'total_patterns': total_patterns,
            'pattern_counts': pattern_counts,
            'confidence_distribution': confidence_counts,
            'recent_patterns': [
                {
                    'pattern': signal.pattern_type.value,
                    'timestamp': signal.timestamp.isoformat(),
                    'confidence': signal.confidence.value,
                    'price': signal.price_level,
                    'strength': signal.signal_strength
                }
                for signal in recent_patterns
            ],
            'analysis_quality': {
                'high_confidence_ratio': confidence_counts['high'] / max(total_patterns, 1),
                'pattern_diversity': len([count for count in pattern_counts.values() if count > 0]),
                'avg_signal_strength': np.mean([s.signal_strength for s in all_signals]) if all_signals else 0
            }
        }