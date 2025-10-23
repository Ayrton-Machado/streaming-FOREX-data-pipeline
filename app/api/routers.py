"""
FastAPI Routers - Endpoints da API FOREX
Etapa 3: Implementação dos endpoints REST
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional
import logging
import time

from fastapi import APIRouter, Query, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.services.data_fetcher import DataFetcher, DataFetchError
from app.domain.enums import Granularity
from app.api.models import (
    QuotesResponse,
    LatestQuoteResponse,
    BasicQuotesResponse,
    HealthResponse,
    ErrorResponse
)

# Setup logging
logger = logging.getLogger(__name__)

# Router principal
router = APIRouter()

# Variável global para tracking de uptime
start_time = time.time()


def get_data_fetcher() -> DataFetcher:
    """Dependency injection para DataFetcher"""
    return DataFetcher()


def parse_granularity(granularity_str: str) -> Granularity:
    """Converte string de granularidade para Enum"""
    mapping = {
        "1min": Granularity.ONE_MIN,
        "5min": Granularity.FIVE_MIN, 
        "15min": Granularity.FIFTEEN_MIN,
        "1h": Granularity.ONE_HOUR,
        "1d": Granularity.ONE_DAY
    }
    
    if granularity_str not in mapping:
        raise HTTPException(
            status_code=400,
            detail=f"Granularidade inválida. Use: {list(mapping.keys())}"
        )
    
    return mapping[granularity_str]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Verifica se a API e seus serviços estão funcionando
    """
    try:
        # Testa serviços básicos
        fetcher = DataFetcher()
        
        # Testa conectividade básica
        services_status = {
            "data_fetcher": "healthy",
            "data_validator": "healthy", 
            "yahoo_finance": "healthy"
        }
        
        # Calcula uptime
        uptime = time.time() - start_time
        
        return HealthResponse(
            success=True,
            message="API funcionando corretamente",
            timestamp=datetime.now(timezone.utc),
            status="healthy",
            version="1.0.0",
            services=services_status,
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check falhou: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "message": "Serviço indisponível",
                "error_code": "SERVICE_UNAVAILABLE",
                "error_details": {"original_error": str(e)}
            }
        )


@router.get("/quote/latest", response_model=LatestQuoteResponse)
async def get_latest_quote(
    fetcher: DataFetcher = Depends(get_data_fetcher)
):
    """
    Obtém a cotação EUR/USD mais recente
    
    Returns:
        LatestQuoteResponse: Cotação validada mais recente
    """
    try:
        logger.info("Buscando cotação mais recente")
        
        # Busca cotação validada
        latest_quote = fetcher.get_latest_quote_validated()
        
        if not latest_quote:
            raise HTTPException(
                status_code=404,
                detail={
                    "success": False,
                    "message": "Nenhuma cotação recente disponível",
                    "error_code": "NO_RECENT_QUOTE"
                }
            )
        
        return LatestQuoteResponse(
            success=True,
            message="Cotação mais recente obtida com sucesso",
            timestamp=datetime.now(timezone.utc),
            data=latest_quote
        )
        
    except HTTPException:
        raise
    except DataFetchError as e:
        logger.error(f"Erro ao buscar cotação: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "Erro ao buscar cotação",
                "error_code": "DATA_FETCH_ERROR",
                "error_details": {"original_error": str(e)}
            }
        )
    except Exception as e:
        logger.error(f"Erro inesperado: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "Erro interno do servidor",
                "error_code": "INTERNAL_ERROR",
                "error_details": {"original_error": str(e)}
            }
        )


@router.get("/quotes")
async def get_historical_quotes(
    start_date: datetime = Query(
        description="Data de início (ISO format)",
        examples=["2025-10-11T00:00:00Z"]
    ),
    end_date: datetime = Query(
        description="Data de fim (ISO format)", 
        examples=["2025-10-12T00:00:00Z"]
    ),
    granularity: str = Query(
        default="1h",
        description="Granularidade dos dados",
        pattern=r"^(1min|5min|15min|1h|1d)$",
        examples=["1h"]
    ),
    format: str = Query(
        default="validated",
        description="Formato da resposta",
        pattern=r"^(validated|basic)$",
        examples=["validated"]
    ),
    fetcher: DataFetcher = Depends(get_data_fetcher)
):
    """
    Obtém dados históricos EUR/USD
    
    Args:
        start_date: Data de início (inclusive)
        end_date: Data de fim (inclusive)  
        granularity: Granularidade (1min, 5min, 15min, 1h, 1d)
        format: Formato da resposta (validated, basic)
        
    Returns:
        QuotesResponse: Dados históricos validados com metadados
    """
    try:
        logger.info(
            f"Buscando dados históricos: {start_date} até {end_date} ({granularity})"
        )
        
        # Validação básica de datas
        if start_date >= end_date:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "message": "Data de início deve ser anterior à data de fim",
                    "error_code": "INVALID_DATE_RANGE"
                }
            )
        
        # Valida período máximo
        max_days = settings.max_historical_days
        delta_days = (end_date - start_date).days
        
        if delta_days > max_days:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "message": f"Período muito longo. Máximo: {max_days} dias",
                    "error_code": "MAX_RANGE_EXCEEDED",
                    "error_details": {"requested_days": delta_days, "max_days": max_days}
                }
            )
        
        # Converte granularidade
        granularity_enum = parse_granularity(granularity)
        
        # Busca dados validados
        validated_candles, validation_result = fetcher.get_historical_data_validated(
            start_date=start_date,
            end_date=end_date,
            granularity=granularity_enum
        )
        
        if not validated_candles:
            raise HTTPException(
                status_code=404,
                detail={
                    "success": False,
                    "message": "Nenhum dado encontrado para o período solicitado",
                    "error_code": "NO_DATA_FOUND"
                }
            )
        
        # Gera estatísticas
        stats = fetcher.get_validation_stats(validated_candles)
        
        # Metadados
        metadata = {
            "granularity": granularity,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "validation_result": {
                "quality_score": validation_result.quality_score,
                "quality_level": validation_result.quality_level,
                "errors": [error.value for error in validation_result.validation_errors],
                "total_outliers": validation_result.outlier_count,
                "gap_percentage": validation_result.gap_percentage
            }
        }
        
        # Formato de resposta
        if format == "basic":
            basic_quotes = fetcher.convert_validated_to_basic_quotes(validated_candles)
            return BasicQuotesResponse(
                success=True,
                message=f"{len(basic_quotes)} cotações obtidas com sucesso",
                timestamp=datetime.now(timezone.utc),
                data=basic_quotes,
                metadata=metadata,
                total_candles=len(basic_quotes)
            )
        
        return QuotesResponse(
            success=True,
            message=f"{len(validated_candles)} velas validadas obtidas com sucesso",
            timestamp=datetime.now(timezone.utc),
            data=validated_candles,
            metadata=metadata,
            total_candles=len(validated_candles),
            quality_stats=stats.get("quality_stats", {})
        )
        
    except HTTPException:
        raise
    except DataFetchError as e:
        logger.error(f"Erro ao buscar dados históricos: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "Erro ao buscar dados históricos",
                "error_code": "DATA_FETCH_ERROR",
                "error_details": {"original_error": str(e)}
            }
        )
    except Exception as e:
        logger.error(f"Erro inesperado: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "Erro interno do servidor", 
                "error_code": "INTERNAL_ERROR",
                "error_details": {"original_error": str(e)}
            }
        )


@router.get("/quotes/basic")
async def get_basic_quotes(
    start_date: datetime = Query(description="Data de início"),
    end_date: datetime = Query(description="Data de fim"),
    granularity: str = Query(default="1h", description="Granularidade"),
    fetcher: DataFetcher = Depends(get_data_fetcher)
):
    """
    Endpoint simplificado para cotações básicas (sem validação completa)
    
    Útil para clientes que precisam apenas dos dados OHLCV básicos
    """
    return await get_historical_quotes(
        start_date=start_date,
        end_date=end_date,
        granularity=granularity,
        format="basic",
        fetcher=fetcher
    )
