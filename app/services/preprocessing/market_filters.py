"""
Market Filters - Filtros de mercado e features contextuais
Etapa 4: Horários de negociação, volatilidade, eventos de mercado
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timezone
from typing import List, Dict, Any, Optional, Tuple
import logging

from app.domain.schemas import ValidatedOHLCV, MLReadyCandle
from app.core.config import settings

logger = logging.getLogger(__name__)


class MarketFilters:
    """
    Serviço para filtros de mercado e features contextuais
    
    Filosofia SLC:
    - Simple: Filtros claros baseados em horários e volatilidade
    - Lovable: Considera características específicas do FOREX
    - Complete: Multiple market sessions, volatility regimes, gap detection
    """
    
    def __init__(self):
        # Definição das sessões de FOREX (UTC)
        self.market_sessions = {
            'sydney': {'start': time(22, 0), 'end': time(7, 0)},    # 22:00-07:00 UTC
            'tokyo': {'start': time(0, 0), 'end': time(9, 0)},     # 00:00-09:00 UTC  
            'london': {'start': time(8, 0), 'end': time(17, 0)},   # 08:00-17:00 UTC
            'new_york': {'start': time(13, 0), 'end': time(22, 0)} # 13:00-22:00 UTC
        }
        
        # Overlaps importantes (alta liquidez)
        self.session_overlaps = {
            'london_new_york': {'start': time(13, 0), 'end': time(17, 0)},  # Mais importante
            'sydney_tokyo': {'start': time(0, 0), 'end': time(7, 0)},
            'tokyo_london': {'start': time(8, 0), 'end': time(9, 0)}
        }
        
        logger.info("MarketFilters inicializado com sessões FOREX")
    
    def apply_market_filters(
        self,
        candles: List[ValidatedOHLCV],
        filters: Optional[List[str]] = None
    ) -> List[MLReadyCandle]:
        """
        Aplica filtros de mercado e adiciona features contextuais
        
        Args:
            candles: Lista de velas validadas
            filters: Lista de filtros a aplicar (None = todos)
            
        Returns:
            Lista de MLReadyCandle com market features
        """
        if not candles:
            logger.warning("Lista de velas vazia")
            return []
        
        # Converte para DataFrame
        df = self._candles_to_dataframe(candles)
        
        # Aplica filtros padrão se não especificado
        if filters is None:
            filters = ['market_sessions', 'volatility_regime', 'gap_detection', 'weekend_filter']
        
        # Adiciona features de sessão de mercado
        if 'market_sessions' in filters:
            df = self._add_market_session_features(df)
        
        # Adiciona regime de volatilidade
        if 'volatility_regime' in filters:
            df = self._add_volatility_regime(df)
        
        # Detecção de gaps de mercado
        if 'gap_detection' in filters:
            df = self._add_gap_detection(df)
        
        # Filtro de fim de semana
        if 'weekend_filter' in filters:
            df = self._add_weekend_features(df)
        
        # Features de tempo
        df = self._add_time_features(df)
        
        # Features de liquidez estimada
        df = self._add_liquidity_features(df)
        
        # Converte de volta para MLReadyCandle
        ml_candles = self._dataframe_to_ml_candles(df, candles)
        
        logger.info(f"✅ Market filters aplicados: {len(ml_candles)} velas com features de mercado")
        return ml_candles
    
    def _add_market_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de sessão de mercado"""
        
        # Extrai hora UTC
        df['hour_utc'] = df['timestamp'].dt.hour
        df['minute_utc'] = df['timestamp'].dt.minute
        
        # Identifica sessões ativas
        for session_name, session_times in self.market_sessions.items():
            df[f'is_{session_name}_session'] = self._is_in_session(
                df['hour_utc'], 
                session_times['start'], 
                session_times['end']
            )
        
        # Identifica overlaps de sessões
        for overlap_name, overlap_times in self.session_overlaps.items():
            df[f'is_{overlap_name}_overlap'] = self._is_in_session(
                df['hour_utc'],
                overlap_times['start'],
                overlap_times['end']
            )
        
        # Conta quantas sessões estão ativas
        session_columns = [col for col in df.columns if col.startswith('is_') and col.endswith('_session')]
        df['active_sessions_count'] = df[session_columns].sum(axis=1)
        
        # Identifica sessão principal
        df['primary_session'] = 'none'
        df.loc[df['is_sydney_session'], 'primary_session'] = 'sydney'
        df.loc[df['is_tokyo_session'], 'primary_session'] = 'tokyo'
        df.loc[df['is_london_session'], 'primary_session'] = 'london'
        df.loc[df['is_new_york_session'], 'primary_session'] = 'new_york'
        
        # Prioridade para overlaps
        df.loc[df['is_london_new_york_overlap'], 'primary_session'] = 'london_new_york'
        
        return df
    
    def _add_volatility_regime(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """Identifica regime de volatilidade"""
        if len(df) < lookback:
            df['volatility_regime'] = 'unknown'
            df['volatility_percentile'] = 50.0
            return df
        
        # Calcula volatilidade rolling (ATR normalizado)
        df['high_low_range'] = df['high'] - df['low']
        df['volatility_rolling'] = df['high_low_range'].rolling(window=lookback).mean()
        
        # Percentil de volatilidade
        df['volatility_percentile'] = df['volatility_rolling'].rolling(window=50).rank(pct=True) * 100
        
        # Classifica regime
        df['volatility_regime'] = 'normal'
        df.loc[df['volatility_percentile'] <= 20, 'volatility_regime'] = 'low'
        df.loc[df['volatility_percentile'] >= 80, 'volatility_regime'] = 'high'
        df.loc[df['volatility_percentile'] >= 95, 'volatility_regime'] = 'extreme'
        
        return df
    
    def _add_gap_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta gaps de mercado (aberturas com gaps)"""
        if len(df) < 2:
            df['has_gap'] = False
            df['gap_size_pct'] = 0.0
            return df
        
        # Calcula gap entre close anterior e open atual
        prev_close = df['close'].shift(1)
        current_open = df['open']
        
        # Gap em pontos e percentual
        gap_points = current_open - prev_close
        gap_pct = (gap_points / prev_close) * 100
        
        # Define threshold para gap significativo (0.1% para FOREX)
        gap_threshold = 0.1
        
        df['gap_size_pct'] = gap_pct.fillna(0)
        df['has_gap'] = abs(df['gap_size_pct']) > gap_threshold
        df['gap_direction'] = np.where(df['gap_size_pct'] > 0, 'up', 
                                     np.where(df['gap_size_pct'] < 0, 'down', 'none'))
        
        return df
    
    def _add_weekend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features relacionadas a fim de semana"""
        
        # Dia da semana (0=Monday, 6=Sunday)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # É fim de semana?
        df['is_weekend'] = df['day_of_week'].isin([5, 6])  # Saturday, Sunday
        
        # É próximo ao fim de semana?
        df['is_friday'] = df['day_of_week'] == 4
        df['is_sunday_open'] = (df['day_of_week'] == 6) & (df['timestamp'].dt.hour >= 22)
        
        # Distância do fim de semana em horas
        df['hours_to_weekend'] = np.where(
            df['day_of_week'] < 5,  # Segunda a Sexta
            (4 - df['day_of_week']) * 24 + (22 - df['timestamp'].dt.hour),
            0
        )
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features temporais"""
        
        # Features básicas de tempo
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        
        # Features cíclicas (para ML)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Classificação de horário
        df['time_of_day'] = 'night'
        df.loc[df['hour'].between(6, 11), 'time_of_day'] = 'morning'
        df.loc[df['hour'].between(12, 17), 'time_of_day'] = 'afternoon'
        df.loc[df['hour'].between(18, 21), 'time_of_day'] = 'evening'
        
        return df
    
    def _add_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estima features de liquidez"""
        
        # Score de liquidez baseado em sessões ativas
        df['liquidity_score'] = 0.0
        
        # Pontuação por sessão (baseado em volume típico)
        session_weights = {
            'sydney': 0.2,
            'tokyo': 0.4,
            'london': 0.8,
            'new_york': 0.9
        }
        
        for session, weight in session_weights.items():
            df.loc[df[f'is_{session}_session'], 'liquidity_score'] += weight
        
        # Bonus para overlaps
        overlap_bonus = {
            'london_new_york': 0.5,  # Maior liquidez
            'sydney_tokyo': 0.2,
            'tokyo_london': 0.3
        }
        
        for overlap, bonus in overlap_bonus.items():
            df.loc[df[f'is_{overlap}_overlap'], 'liquidity_score'] += bonus
        
        # Normaliza score (0-1)
        if df['liquidity_score'].max() > 0:
            df['liquidity_score'] = df['liquidity_score'] / df['liquidity_score'].max()
        
        # Classificação de liquidez
        df['liquidity_level'] = 'low'
        df.loc[df['liquidity_score'] >= 0.3, 'liquidity_level'] = 'medium'
        df.loc[df['liquidity_score'] >= 0.6, 'liquidity_level'] = 'high'
        df.loc[df['liquidity_score'] >= 0.8, 'liquidity_level'] = 'very_high'
        
        return df
    
    def _is_in_session(
        self, 
        hours: pd.Series, 
        start_time: time, 
        end_time: time
    ) -> pd.Series:
        """Verifica se horário está dentro de uma sessão"""
        start_hour = start_time.hour
        end_hour = end_time.hour
        
        if start_hour <= end_hour:
            # Sessão não cruza meia-noite
            return hours.between(start_hour, end_hour)
        else:
            # Sessão cruza meia-noite (ex: 22:00-07:00)
            return (hours >= start_hour) | (hours <= end_hour)
    
    def filter_trading_hours_only(
        self, 
        candles: List[ValidatedOHLCV],
        sessions: List[str] = None
    ) -> List[ValidatedOHLCV]:
        """
        Filtra velas para manter apenas horários de negociação
        
        Args:
            candles: Lista de velas
            sessions: Sessões a manter (None = todas)
            
        Returns:
            Lista filtrada de velas
        """
        if not candles:
            return []
        
        if sessions is None:
            sessions = ['london', 'new_york', 'tokyo']  # Principais sessões
        
        filtered_candles = []
        
        for candle in candles:
            hour_utc = candle.timestamp.hour
            is_trading_hour = False
            
            for session in sessions:
                if session in self.market_sessions:
                    session_times = self.market_sessions[session]
                    if self._is_in_session(
                        pd.Series([hour_utc]), 
                        session_times['start'], 
                        session_times['end']
                    ).iloc[0]:
                        is_trading_hour = True
                        break
            
            if is_trading_hour:
                filtered_candles.append(candle)
        
        logger.info(f"🕒 Filtro de horário: {len(filtered_candles)}/{len(candles)} velas mantidas")
        return filtered_candles
    
    def filter_high_liquidity_only(
        self,
        candles: List[ValidatedOHLCV]
    ) -> List[ValidatedOHLCV]:
        """
        Filtra velas para manter apenas períodos de alta liquidez
        """
        # Aplica market features primeiro
        ml_candles = self.apply_market_filters(candles)
        
        # Filtra por liquidez alta
        high_liquidity_candles = []
        
        for i, ml_candle in enumerate(ml_candles):
            liquidity_level = ml_candle.market_features.get('liquidity_level', 'low')
            if liquidity_level in ['high', 'very_high']:
                high_liquidity_candles.append(candles[i])
        
        logger.info(f"💧 Filtro de liquidez: {len(high_liquidity_candles)}/{len(candles)} velas mantidas")
        return high_liquidity_candles
    
    def get_market_summary(self, candles: List[ValidatedOHLCV]) -> Dict[str, Any]:
        """
        Retorna resumo das características de mercado
        """
        if not candles:
            return {"error": "Lista vazia"}
        
        # Aplica market features
        ml_candles = self.apply_market_filters(candles)
        
        # Estatísticas por sessão
        session_stats = {}
        for session in ['sydney', 'tokyo', 'london', 'new_york']:
            session_candles = [
                candle for candle in ml_candles 
                if candle.market_features.get(f'is_{session}_session', False)
            ]
            
            if session_candles:
                prices = [candle.close for candle in session_candles]
                session_stats[session] = {
                    "count": len(session_candles),
                    "avg_price": sum(prices) / len(prices),
                    "min_price": min(prices),
                    "max_price": max(prices),
                    "percentage": len(session_candles) / len(ml_candles) * 100
                }
        
        # Estatísticas de liquidez
        liquidity_distribution = {}
        for level in ['low', 'medium', 'high', 'very_high']:
            count = sum(1 for candle in ml_candles 
                       if candle.market_features.get('liquidity_level') == level)
            liquidity_distribution[level] = {
                "count": count,
                "percentage": count / len(ml_candles) * 100
            }
        
        # Gaps detectados
        gaps_count = sum(1 for candle in ml_candles 
                        if candle.market_features.get('has_gap', False))
        
        return {
            "total_candles": len(ml_candles),
            "session_distribution": session_stats,
            "liquidity_distribution": liquidity_distribution,
            "gaps_detected": gaps_count,
            "gap_percentage": gaps_count / len(ml_candles) * 100,
            "period": {
                "start": candles[0].timestamp.isoformat(),
                "end": candles[-1].timestamp.isoformat()
            }
        }
    
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
        """Converte DataFrame com market features para MLReadyCandle"""
        ml_candles = []
        
        # Identifica colunas de market features
        basic_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        market_feature_columns = [col for col in df.columns if col not in basic_columns]
        
        for i, (_, row) in enumerate(df.iterrows()):
            # Extrai market features
            market_features = {}
            for col in market_feature_columns:
                value = row[col]
                if pd.notna(value):  # Só adiciona se não for NaN
                    if isinstance(value, (bool, np.bool_)):
                        market_features[col] = bool(value)
                    elif isinstance(value, (int, np.integer)):
                        market_features[col] = int(value)
                    elif isinstance(value, (float, np.floating)):
                        market_features[col] = float(value)
                    else:
                        market_features[col] = str(value)
            
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
                normalized_features={},     # Será preenchido pelo DataNormalizer
                technical_indicators={},    # Será preenchido pelo FeatureEngineer
                market_features=market_features
            )
            
            ml_candles.append(ml_candle)
        
        return ml_candles


# Factory function
def get_market_filters() -> MarketFilters:
    """Retorna instância de MarketFilters"""
    return MarketFilters()