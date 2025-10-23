"""
Data Fetcher Service - Busca dados EUR/USD do Yahoo Finance
Etapa 2: Integrado com schemas Pydantic e validação de qualidade
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple
import logging

from app.core.config import settings
from app.core.constants import (
    GRANULARITY_MAPPING,
    VALID_GRANULARITIES,
    ERROR_MESSAGES,
    FOREX_TIMEZONE
)
from app.domain.schemas import (
    RawOHLCV,
    ValidatedOHLCV, 
    DataValidationResult,
    BasicQuote
)
from app.domain.enums import (
    Granularity,
    DataSource,
    DataQuality
)
from app.services.data_validator import DataValidator


# Setup logging
logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """Erro ao buscar dados de fontes externas"""
    pass


class DataFetcher:
    """
    Serviço responsável por buscar dados FOREX do Yahoo Finance
    
    Etapa 2: Agora retorna dados validados usando schemas Pydantic
    
    Filosofia SLC:
    - Simple: Interface clara, métodos auto-explicativos
    - Lovable: Mensagens de erro úteis, logging informativo, dados validados
    - Complete: Tratamento de erros robusto + validação de qualidade
    """
    
    def __init__(self):
        self.ticker_symbol = settings.forex_pair
        self.ticker = yf.Ticker(self.ticker_symbol)
        self.validator = DataValidator()
        logger.info(f"DataFetcher inicializado para {self.ticker_symbol} (com validação)")
    
    def get_historical_data_validated(
        self,
        start_date: datetime,
        end_date: datetime,
        granularity: Granularity = Granularity.ONE_HOUR
    ) -> Tuple[List[ValidatedOHLCV], DataValidationResult]:
        """
        Busca dados históricos EUR/USD do Yahoo Finance E VALIDA
        
        Args:
            start_date: Data de início (inclusive)
            end_date: Data de fim (inclusive)
            granularity: Granularidade dos dados (Enum)
            
        Returns:
            Tuple[List[ValidatedOHLCV], DataValidationResult]
            - Lista de velas validadas com schemas Pydantic
            - Resultado da validação de qualidade
            
        Raises:
            DataFetchError: Se falhar ao buscar ou validar dados
        """
        # Validação de entrada usando enum
        self._validate_inputs_v2(start_date, end_date, granularity)
        
        # Busca dados brutos (método original)
        df = self.get_historical_data(start_date, end_date, granularity.value)
        
        if df is None or df.empty:
            logger.warning("Nenhum dado retornado do Yahoo Finance")
            return [], self.validator._create_empty_validation_result()
        
        # Valida qualidade dos dados
        validated_candles, validation_result = self.validator.validate_dataframe(
            df, granularity, DataSource.YAHOO_FINANCE
        )
        
        # Log do resultado da validação
        summary = self.validator.get_validation_summary(validation_result)
        logger.info(f"Dados validados:\n{summary}")
        
        return validated_candles, validation_result
    
    def get_latest_quote_validated(self) -> Optional[ValidatedOHLCV]:
        """
        Busca a cotação mais recente EUR/USD E VALIDA
        
        Returns:
            ValidatedOHLCV ou None se não conseguir buscar
        """
        try:
            # Busca cotação bruta
            quote_dict = self.get_latest_quote()
            
            # Converte para RawOHLCV
            raw_candle = RawOHLCV(
                timestamp=quote_dict["timestamp"],
                open=quote_dict["open"],
                high=quote_dict["high"],
                low=quote_dict["low"],
                close=quote_dict["close"],
                volume=quote_dict["volume"],
                data_source=DataSource.YAHOO_FINANCE,
                raw_symbol=self.ticker_symbol
            )
            
            # Valida vela única
            validated_candle = self.validator.validate_single_candle(raw_candle)
            
            logger.debug(f"Cotação validada: {validated_candle.close} (score: {validated_candle.validation.quality_score:.3f})")
            return validated_candle
            
        except Exception as e:
            logger.error(f"Erro ao buscar/validar cotação recente: {str(e)}")
            return None
    
    def convert_dataframe_to_raw_ohlcv(self, df: pd.DataFrame) -> List[RawOHLCV]:
        """
        Converte DataFrame do Yahoo Finance para lista de RawOHLCV
        Útil para processamento manual
        """
        raw_candles = []
        
        for idx, row in df.iterrows():
            try:
                raw_candle = RawOHLCV(
                    timestamp=idx.to_pydatetime(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row.get('Volume', 0)),
                    data_source=DataSource.YAHOO_FINANCE,
                    raw_symbol=self.ticker_symbol
                )
                raw_candles.append(raw_candle)
                
            except Exception as e:
                logger.warning(f"Erro ao converter vela {idx}: {str(e)}")
                continue
        
        return raw_candles
    
    def convert_validated_to_basic_quotes(self, validated_candles: List[ValidatedOHLCV]) -> List[BasicQuote]:
        """
        Converte ValidatedOHLCV para BasicQuote (para endpoints simples)
        """
        basic_quotes = []
        
        for candle in validated_candles:
            try:
                basic_quote = BasicQuote(
                    timestamp=candle.timestamp,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume
                )
                basic_quotes.append(basic_quote)
                
            except Exception as e:
                logger.warning(f"Erro ao converter para BasicQuote: {str(e)}")
                continue
        
        return basic_quotes
    
    def get_validation_stats(self, validated_candles: List[ValidatedOHLCV]) -> Dict[str, Any]:
        """
        Calcula estatísticas úteis sobre dados validados
        Útil para metadados de response da API
        """
        if not validated_candles:
            return {
                "total_candles": 0,
                "quality_stats": {},
                "has_data": False
            }
        
        # Estatísticas de qualidade
        quality_scores = [candle.validation.quality_score for candle in validated_candles]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        # Contagem por nível de qualidade
        quality_levels = [candle.validation.quality_level for candle in validated_candles]
        quality_distribution = {}
        for level in DataQuality:
            quality_distribution[level] = quality_levels.count(level)
        
        # Contagem de problemas
        outlier_count = sum(1 for candle in validated_candles if candle.is_outlier)
        gap_fill_count = sum(1 for candle in validated_candles if candle.is_gap_fill)
        
        # Estatísticas de preço
        close_prices = [candle.close for candle in validated_candles]
        
        return {
            "total_candles": len(validated_candles),
            "has_data": True,
            "date_range": {
                "start": validated_candles[0].timestamp.isoformat(),
                "end": validated_candles[-1].timestamp.isoformat(),
            },
            "quality_stats": {
                "average_quality_score": avg_quality,
                "quality_distribution": quality_distribution,
                "outlier_count": outlier_count,
                "gap_fill_count": gap_fill_count,
            },
            "price_stats": {
                "min_close": min(close_prices),
                "max_close": max(close_prices),
                "mean_close": sum(close_prices) / len(close_prices),
            },
            "data_source": validated_candles[0].data_source,
            "validation_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _validate_inputs_v2(
        self,
        start_date: datetime,
        end_date: datetime,
        granularity: Granularity
    ) -> None:
        """Valida parâmetros de entrada usando Enums"""
        
        # Valida granularidade (agora é Enum, mas verificamos mesmo assim)
        if not isinstance(granularity, Granularity):
            raise ValueError(f"Granularity deve ser um Enum válido: {list(Granularity)}")
        
        # Valida intervalo de datas
        if start_date >= end_date:
            raise ValueError(ERROR_MESSAGES["invalid_date_range"])
        
        # Valida período máximo
        max_days = settings.max_historical_days
        delta_days = (end_date - start_date).days
        
        if delta_days > max_days:
            raise ValueError(
                f"{ERROR_MESSAGES['max_range_exceeded']}: "
                f"{delta_days} dias (máximo: {max_days})"
            )
    
    # ===== MÉTODOS LEGADOS (compatibilidade com Etapa 1) =====
    
    def get_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        granularity: str = "1h"
    ) -> pd.DataFrame:
        """
        Busca dados históricos EUR/USD do Yahoo Finance (método original)
        
        LEGADO: Mantido para compatibilidade com scripts existentes
        RECOMENDADO: Use get_historical_data_validated() para novos códigos
        
        Args:
            start_date: Data de início (inclusive)
            end_date: Data de fim (inclusive)
            granularity: Granularidade dos dados (string: "1min", "5min", etc.)
            
        Returns:
            DataFrame com colunas: Open, High, Low, Close, Volume
            Index: DatetimeIndex em UTC
            
        Raises:
            DataFetchError: Se falhar ao buscar ou validar dados
        """
        # Validação de entrada (método legado)
        self._validate_inputs_legacy(start_date, end_date, granularity)
        
        # Converte granularidade para formato Yahoo Finance
        yf_interval = GRANULARITY_MAPPING[granularity]
        
        logger.info(
            f"Buscando dados {self.ticker_symbol}: "
            f"{start_date.date()} até {end_date.date()} ({granularity})"
        )
        
        try:
            # Download dos dados
            df = yf.download(
                self.ticker_symbol,
                start=start_date,
                end=end_date,
                interval=yf_interval,
                progress=False,
                auto_adjust=False,  # Mantém preços originais
                prepost=False,      # Sem pre/post market (não aplicável a FOREX)
            )
            
            # Validação básica
            if df is None or df.empty:
                raise DataFetchError(ERROR_MESSAGES["no_data_found"])
            
            # Limpeza e padronização
            df = self._clean_dataframe(df)
            
            logger.info(f"✅ {len(df)} velas carregadas com sucesso")
            return df
            
        except Exception as e:
            if isinstance(e, DataFetchError):
                raise
            logger.error(f"Erro ao buscar dados: {str(e)}")
            raise DataFetchError(f"{ERROR_MESSAGES['data_fetch_error']}: {str(e)}")

    def get_latest_quote(self) -> Dict[str, Any]:
        """
        Busca a cotação mais recente EUR/USD (método original)
        
        LEGADO: Mantido para compatibilidade
        RECOMENDADO: Use get_latest_quote_validated() para novos códigos
        
        Returns:
            Dict com: timestamp, open, high, low, close, volume
        """
        logger.debug(f"Buscando cotação mais recente de {self.ticker_symbol}")
        
        try:
            # Busca último dia de dados com granularidade mínima
            now = datetime.now()
            start = now - timedelta(days=1)
            
            df = yf.download(
                self.ticker_symbol,
                start=start,
                end=now,
                interval="1m",
                progress=False
            )
            
            if df.empty:
                raise DataFetchError("Nenhuma cotação recente disponível")
            
            # Pega a última linha
            latest = df.iloc[-1]
            
            quote = {
                "timestamp": df.index[-1].to_pydatetime(),
                "open": float(latest["Open"].iloc[0]) if isinstance(latest["Open"], pd.Series) else float(latest["Open"]),
                "high": float(latest["High"].iloc[0]) if isinstance(latest["High"], pd.Series) else float(latest["High"]),
                "low": float(latest["Low"].iloc[0]) if isinstance(latest["Low"], pd.Series) else float(latest["Low"]),
                "close": float(latest["Close"].iloc[0]) if isinstance(latest["Close"], pd.Series) else float(latest["Close"]),
                "volume": int(latest["Volume"].iloc[0]) if isinstance(latest["Volume"], pd.Series) else int(latest["Volume"]) if "Volume" in latest else 0,
            }
            
            logger.debug(f"Cotação recente: {quote['close']} @ {quote['timestamp']}")
            return quote
            
        except Exception as e:
            logger.error(f"Erro ao buscar cotação recente: {str(e)}")
            raise DataFetchError(f"Falha ao obter cotação mais recente: {str(e)}")

    def _validate_inputs_legacy(
        self,
        start_date: datetime,
        end_date: datetime,
        granularity: str
    ) -> None:
        """Valida parâmetros de entrada (método legado)"""
        
        # Valida granularidade
        if granularity not in VALID_GRANULARITIES:
            raise ValueError(ERROR_MESSAGES["invalid_granularity"])
        
        # Valida intervalo de datas
        if start_date >= end_date:
            raise ValueError(ERROR_MESSAGES["invalid_date_range"])
        
        # Valida período máximo
        max_days = settings.max_historical_days
        delta_days = (end_date - start_date).days
        
        if delta_days > max_days:
            raise ValueError(
                f"{ERROR_MESSAGES['max_range_exceeded']}: "
                f"{delta_days} dias (máximo: {max_days})"
            )
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa e padroniza o DataFrame do Yahoo Finance
        
        Args:
            df: DataFrame bruto do yfinance
            
        Returns:
            DataFrame limpo e padronizado
        """
        # Garante que o index é DatetimeIndex em UTC
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Converte para UTC se necessário
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        elif str(df.index.tz) != "UTC":
            df.index = df.index.tz_convert("UTC")
        
        # Remove possíveis níveis múltiplos no index (yfinance às vezes retorna)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Padroniza nomes de colunas (uppercase)
        df.columns = [col.title() for col in df.columns]
        
        # Remove linhas com NaN em colunas críticas (OHLC)
        critical_cols = ["Open", "High", "Low", "Close"]
        df = df.dropna(subset=critical_cols)
        
        # Preenche Volume com 0 se estiver NaN (FOREX não tem volume real)
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].fillna(0).astype(int)
        else:
            df["Volume"] = 0
        
        # Ordena por timestamp (garantia)
        df = df.sort_index()
        
        # Remove duplicatas (por timestamp)
        df = df[~df.index.duplicated(keep="first")]
        
        return df
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Retorna informações úteis sobre o DataFrame
        Útil para debugging e metadados de response
        
        Args:
            df: DataFrame com dados OHLCV
            
        Returns:
            Dict com estatísticas e metadados
        """
        if df.empty:
            return {"total_candles": 0, "has_data": False}
        
        info = {
            "total_candles": len(df),
            "has_data": True,
            "date_range": {
                "start": df.index[0].isoformat(),
                "end": df.index[-1].isoformat(),
            },
            "price_stats": {
                "min_close": float(df["Close"].min()),
                "max_close": float(df["Close"].max()),
                "mean_close": float(df["Close"].mean()),
                "std_close": float(df["Close"].std()),
            },
            "missing_values": {
                col: int(df[col].isna().sum())
                for col in df.columns
            },
            "total_volume": int(df["Volume"].sum()) if "Volume" in df.columns else 0,
        }
        
        return info


# Factory function (útil para dependency injection futura)
def get_data_fetcher() -> DataFetcher:
    """Retorna instância de DataFetcher (singleton pattern futuro)"""
    return DataFetcher()