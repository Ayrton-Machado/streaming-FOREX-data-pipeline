"""
Database Persistence Endpoints - Endpoints para operações de persistência
Etapa 4: Endpoints FastAPI para salvar e consultar dados no TimescaleDB
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
from uuid import UUID

from fastapi import APIRouter, Query, HTTPException, Depends, Body
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.connection import get_db
from app.database.repository import OHLCVRepository
from app.services.preprocessing.pipeline import get_preprocessing_pipeline
from app.domain.schemas import ValidatedOHLCV, MLReadyCandle
from app.api.models import (
    QuotesResponse,
    HealthResponse,
    ErrorResponse
)

logger = logging.getLogger(__name__)

# Router para persistência
persistence_router = APIRouter(prefix="/api/v1/data", tags=["Data Persistence"])


# Modelos de request/response específicos para persistência
from pydantic import BaseModel, Field


class SaveDataRequest(BaseModel):
    """Request para salvar dados"""
    candles: List[ValidatedOHLCV]
    symbol: str = Field(default="EURUSD", description="Símbolo dos dados")
    apply_preprocessing: bool = Field(default=True, description="Aplicar pipeline de preprocessamento")
    preprocessing_config: Optional[Dict[str, Any]] = Field(default=None, description="Configuração do preprocessamento")


class SaveDataResponse(BaseModel):
    """Response do salvamento"""
    success: bool
    records_saved: int
    records_processed: int
    processing_time_ms: float
    preprocessing_applied: bool
    quality_score: Optional[float] = None
    message: str


class QueryDataRequest(BaseModel):
    """Request para consulta de dados"""
    symbol: str = Field(default="EURUSD")
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(default=1000, ge=1, le=10000)
    offset: int = Field(default=0, ge=0)
    include_technical_indicators: bool = Field(default=False)
    include_market_features: bool = Field(default=False)
    aggregation: Optional[str] = Field(default=None, description="Agregação: 'hourly', 'daily', 'weekly'")


class QueryDataResponse(BaseModel):
    """Response da consulta"""
    success: bool
    total_records: int
    returned_records: int
    data: List[Dict[str, Any]]
    query_time_ms: float
    has_more: bool
    next_offset: Optional[int] = None


class DataStatsResponse(BaseModel):
    """Response das estatísticas"""
    symbol: str
    total_records: int
    date_range: Dict[str, str]
    data_quality_score: float
    completeness_percentage: float
    last_update: datetime
    technical_indicators_available: bool
    market_features_available: bool


@persistence_router.post("/save", response_model=SaveDataResponse)
async def save_data(
    request: SaveDataRequest,
    db: Session = Depends(get_db)
):
    """
    Salva dados OHLCV no banco com preprocessamento opcional
    
    - **candles**: Lista de velas ValidatedOHLCV
    - **symbol**: Símbolo dos dados (default: EURUSD)
    - **apply_preprocessing**: Se deve aplicar pipeline de preprocessamento
    - **preprocessing_config**: Configuração personalizada do pipeline
    """
    start_time = datetime.now()
    
    try:
        if not request.candles:
            raise HTTPException(status_code=400, detail="Lista de velas não pode estar vazia")
        
        logger.info(f"💾 Salvando {len(request.candles)} velas para {request.symbol}")
        
        repository = OHLCVRepository(db)
        records_saved = 0
        quality_score = None
        preprocessing_applied = False
        
        if request.apply_preprocessing:
            # Aplica pipeline de preprocessamento
            pipeline = get_preprocessing_pipeline()
            
            ml_candles, pipeline_report = pipeline.process_full_pipeline(
                request.candles, 
                request.preprocessing_config
            )
            
            if not ml_candles:
                raise HTTPException(
                    status_code=422, 
                    detail="Processamento falhou - nenhuma vela válida após preprocessamento"
                )
            
            # Salva dados preprocessados
            for candle in ml_candles:
                try:
                    saved_id = repository.save_single_ohlcv(
                        symbol=request.symbol,
                        timestamp=candle.timestamp,
                        open_price=candle.open,
                        high_price=candle.high,
                        low_price=candle.low,
                        close_price=candle.close,
                        volume=candle.volume,
                        normalized_features=candle.normalized_features,
                        technical_indicators=candle.technical_indicators,
                        market_features=candle.market_features,
                        data_source=candle.data_source,
                        is_gap_fill=candle.is_gap_fill
                    )
                    if saved_id:
                        records_saved += 1
                except Exception as e:
                    logger.warning(f"Erro ao salvar vela {candle.timestamp}: {str(e)}")
                    continue
            
            preprocessing_applied = True
            quality_score = pipeline_report.get('quality_metrics', {}).get('quality_score')
            
        else:
            # Salva dados sem preprocessamento
            for candle in request.candles:
                try:
                    saved_id = repository.save_single_ohlcv(
                        symbol=request.symbol,
                        timestamp=candle.timestamp,
                        open_price=candle.open,
                        high_price=candle.high,
                        low_price=candle.low,
                        close_price=candle.close,
                        volume=candle.volume,
                        data_source=candle.data_source,
                        is_gap_fill=candle.is_gap_fill
                    )
                    if saved_id:
                        records_saved += 1
                except Exception as e:
                    logger.warning(f"Erro ao salvar vela {candle.timestamp}: {str(e)}")
                    continue
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"✅ Salvamento concluído: {records_saved}/{len(request.candles)} velas salvas")
        
        return SaveDataResponse(
            success=True,
            records_saved=records_saved,
            records_processed=len(request.candles),
            processing_time_ms=processing_time,
            preprocessing_applied=preprocessing_applied,
            quality_score=quality_score,
            message=f"Salvos {records_saved} registros com sucesso"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro no salvamento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@persistence_router.post("/query", response_model=QueryDataResponse) 
async def query_data(
    request: QueryDataRequest,
    db: Session = Depends(get_db)
):
    """
    Consulta dados OHLCV com filtros avançados
    
    - **symbol**: Símbolo a consultar
    - **start_time/end_time**: Range temporal
    - **limit/offset**: Paginação
    - **include_technical_indicators**: Incluir indicadores técnicos
    - **include_market_features**: Incluir features de mercado
    - **aggregation**: Agregação temporal (hourly, daily, weekly)
    """
    start_time = datetime.now()
    
    try:
        repository = OHLCVRepository(db)
        
        logger.info(f"🔍 Consultando dados: {request.symbol} [{request.start_time} - {request.end_time}]")
        
        # Define range padrão se não fornecido
        if not request.start_time:
            request.start_time = datetime.now() - timedelta(days=7)
        if not request.end_time:
            request.end_time = datetime.now()
        
        # Consulta dados
        if request.aggregation:
            # Consulta com agregação
            results = repository.get_aggregated_data(
                symbol=request.symbol,
                start_time=request.start_time,
                end_time=request.end_time,
                aggregation=request.aggregation,
                limit=request.limit,
                offset=request.offset
            )
        else:
            # Consulta normal
            results = repository.get_ohlcv_range(
                symbol=request.symbol,
                start_time=request.start_time,
                end_time=request.end_time,
                limit=request.limit,
                offset=request.offset
            )
        
        # Converte para formato de resposta
        data_list = []
        for record in results:
            item = {
                "id": str(record.id),
                "timestamp": record.timestamp.isoformat(),
                "symbol": record.symbol,
                "open": record.open_price,
                "high": record.high_price,
                "low": record.low_price,
                "close": record.close_price,
                "volume": record.volume,
                "data_source": record.data_source,
                "is_gap_fill": record.is_gap_fill
            }
            
            # Adiciona features opcionais
            if request.include_technical_indicators and record.technical_indicators:
                item["technical_indicators"] = record.technical_indicators
                
            if request.include_market_features and record.market_features:
                item["market_features"] = record.market_features
            
            data_list.append(item)
        
        # Conta total de registros para paginação
        total_count = repository.count_records(
            symbol=request.symbol,
            start_time=request.start_time,
            end_time=request.end_time
        )
        
        has_more = (request.offset + len(results)) < total_count
        next_offset = request.offset + len(results) if has_more else None
        
        query_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"✅ Consulta concluída: {len(results)} registros retornados")
        
        return QueryDataResponse(
            success=True,
            total_records=total_count,
            returned_records=len(results),
            data=data_list,
            query_time_ms=query_time,
            has_more=has_more,
            next_offset=next_offset
        )
        
    except Exception as e:
        logger.error(f"❌ Erro na consulta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@persistence_router.get("/stats/{symbol}", response_model=DataStatsResponse)
async def get_data_stats(
    symbol: str,
    db: Session = Depends(get_db)
):
    """
    Retorna estatísticas dos dados para um símbolo
    """
    try:
        repository = OHLCVRepository(db)
        
        # Estatísticas básicas
        stats = repository.get_statistics(symbol)
        
        if not stats:
            raise HTTPException(status_code=404, detail=f"Nenhum dado encontrado para {symbol}")
        
        # Verifica disponibilidade de features
        sample_record = repository.get_latest_record(symbol)
        has_technical = bool(sample_record and sample_record.technical_indicators)
        has_market = bool(sample_record and sample_record.market_features)
        
        return DataStatsResponse(
            symbol=symbol,
            total_records=stats["count"],
            date_range={
                "start": stats["start_time"].isoformat(),
                "end": stats["end_time"].isoformat()
            },
            data_quality_score=95.0,  # Placeholder - integrar com DataQualityService
            completeness_percentage=stats.get("completeness", 100.0),
            last_update=stats["end_time"],
            technical_indicators_available=has_technical,
            market_features_available=has_market
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao obter estatísticas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@persistence_router.delete("/cleanup/{symbol}")
async def cleanup_old_data(
    symbol: str,
    days_to_keep: int = Query(default=90, description="Dias de dados para manter"),
    db: Session = Depends(get_db)
):
    """
    Remove dados antigos para um símbolo
    """
    try:
        if days_to_keep < 1:
            raise HTTPException(status_code=400, detail="days_to_keep deve ser >= 1")
        
        repository = OHLCVRepository(db)
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        deleted_count = repository.delete_old_records(symbol, cutoff_date)
        
        logger.info(f"🗑️ Limpeza concluída: {deleted_count} registros removidos para {symbol}")
        
        return {
            "success": True,
            "deleted_records": deleted_count,
            "cutoff_date": cutoff_date.isoformat(),
            "message": f"Removidos dados anteriores a {cutoff_date.strftime('%Y-%m-%d')}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro na limpeza: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@persistence_router.get("/health")
async def persistence_health_check(db: Session = Depends(get_db)):
    """
    Health check específico para persistência
    """
    try:
        repository = OHLCVRepository(db)
        
        # Testa conexão com query simples
        total_records = repository.count_records()
        
        return {
            "status": "healthy",
            "database_connected": True,
            "total_records": total_records,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Health check falhou: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Database unavailable: {str(e)}")


# Endpoints adicionais para análise e agregação

@persistence_router.get("/analyze/{symbol}")
async def analyze_symbol_data(
    symbol: str,
    period_days: int = Query(default=30, description="Período em dias para análise"),
    db: Session = Depends(get_db)
):
    """
    Análise detalhada dos dados de um símbolo
    """
    try:
        repository = OHLCVRepository(db)
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=period_days)
        
        # Dados para análise
        records = repository.get_ohlcv_range(symbol, start_time, end_time, limit=10000)
        
        if not records:
            raise HTTPException(status_code=404, detail=f"Nenhum dado encontrado para {symbol}")
        
        # Cálculos de análise
        prices = [r.close_price for r in records]
        volumes = [r.volume for r in records]
        
        analysis = {
            "symbol": symbol,
            "period_days": period_days,
            "total_records": len(records),
            "price_analysis": {
                "min_price": min(prices),
                "max_price": max(prices),
                "avg_price": sum(prices) / len(prices),
                "price_range": max(prices) - min(prices),
                "volatility": _calculate_volatility(prices)
            },
            "volume_analysis": {
                "total_volume": sum(volumes),
                "avg_volume": sum(volumes) / len(volumes) if volumes else 0,
                "max_volume": max(volumes) if volumes else 0
            },
            "data_quality": {
                "completeness": _check_completeness(records, start_time, end_time),
                "gaps_detected": _detect_gaps(records),
                "outliers_detected": _detect_price_outliers(prices)
            },
            "technical_features": {
                "has_indicators": any(r.technical_indicators for r in records),
                "has_market_features": any(r.market_features for r in records),
                "feature_coverage": _calculate_feature_coverage(records)
            }
        }
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro na análise: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


# Funções auxiliares para análise

def _calculate_volatility(prices: List[float]) -> float:
    """Calcula volatilidade simples"""
    if len(prices) < 2:
        return 0.0
    
    returns = []
    for i in range(1, len(prices)):
        ret = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(ret)
    
    if not returns:
        return 0.0
    
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    return variance ** 0.5


def _check_completeness(records: List, start_time: datetime, end_time: datetime) -> float:
    """Verifica completeness dos dados (assumindo 1 minuto)"""
    expected_minutes = int((end_time - start_time).total_seconds() / 60)
    actual_records = len(records)
    return min(100.0, (actual_records / expected_minutes) * 100) if expected_minutes > 0 else 100.0


def _detect_gaps(records: List) -> int:
    """Detecta gaps nos dados"""
    if len(records) < 2:
        return 0
    
    gaps = 0
    for i in range(1, len(records)):
        time_diff = (records[i].timestamp - records[i-1].timestamp).total_seconds()
        if time_diff > 300:  # Gap > 5 minutos
            gaps += 1
    
    return gaps


def _detect_price_outliers(prices: List[float]) -> int:
    """Detecta outliers de preço usando Z-score"""
    if len(prices) < 10:
        return 0
    
    mean_price = sum(prices) / len(prices)
    std_price = (sum((p - mean_price) ** 2 for p in prices) / len(prices)) ** 0.5
    
    if std_price == 0:
        return 0
    
    outliers = 0
    for price in prices:
        z_score = abs((price - mean_price) / std_price)
        if z_score > 3:  # Z-score > 3 = outlier
            outliers += 1
    
    return outliers


def _calculate_feature_coverage(records: List) -> Dict[str, float]:
    """Calcula cobertura de features"""
    if not records:
        return {"technical": 0.0, "market": 0.0, "normalized": 0.0}
    
    total = len(records)
    technical_count = sum(1 for r in records if r.technical_indicators)
    market_count = sum(1 for r in records if r.market_features)
    normalized_count = sum(1 for r in records if r.normalized_features)
    
    return {
        "technical": (technical_count / total) * 100,
        "market": (market_count / total) * 100,
        "normalized": (normalized_count / total) * 100
    }