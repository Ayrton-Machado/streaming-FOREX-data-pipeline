"""
Response Models - Schemas Pydantic para respostas da API
Etapa 3: Modelos padronizados para endpoints
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from app.domain.schemas import ValidatedOHLCV, BasicQuote
from app.domain.enums import DataQuality, DataSource


class APIResponse(BaseModel):
    """Resposta base para todos os endpoints"""
    
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        ser_json_timedelta='iso8601'
    )
    
    success: bool = Field(description="Se a requisição foi bem-sucedida")
    message: str = Field(description="Mensagem descritiva")
    timestamp: datetime = Field(description="Timestamp da resposta")


class QuotesResponse(APIResponse):
    """Resposta para endpoint /quotes - dados históricos"""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Dados históricos obtidos com sucesso",
                "timestamp": "2025-10-12T10:30:00Z",
                "data": [
                    {
                        "timestamp": "2025-10-12T10:00:00Z",
                        "open": 1.0850,
                        "high": 1.0870,
                        "low": 1.0840,
                        "close": 1.0860,
                        "volume": 0,
                        "data_source": "yahoo_finance",
                        "raw_symbol": "EURUSD=X",
                        "validation": {
                            "quality_score": 0.95,
                            "quality_level": "excellent",
                            "has_gaps": False,
                            "validation_errors": []
                        },
                        "is_outlier": False,
                        "is_gap_fill": False
                    }
                ],
                "metadata": {
                    "granularity": "1h",
                    "date_range": {
                        "start": "2025-10-11T10:00:00Z",
                        "end": "2025-10-12T10:00:00Z"
                    }
                },
                "total_candles": 24,
                "quality_stats": {
                    "average_quality_score": 0.95,
                    "quality_distribution": {
                        "excellent": 20,
                        "good": 4,
                        "fair": 0,
                        "poor": 0
                    }
                }
            }
        }
    )
    
    data: List[ValidatedOHLCV] = Field(description="Lista de velas validadas")
    metadata: Dict[str, Any] = Field(description="Metadados sobre os dados")
    
    # Estatísticas úteis
    total_candles: int = Field(description="Total de velas retornadas")
    quality_stats: Dict[str, Any] = Field(description="Estatísticas de qualidade")


class LatestQuoteResponse(APIResponse):
    """Resposta para endpoint /quote/latest - cotação atual"""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Cotação mais recente obtida com sucesso",
                "timestamp": "2025-10-12T10:30:00Z",
                "data": {
                    "timestamp": "2025-10-12T10:29:00Z",
                    "open": 1.0850,
                    "high": 1.0870,
                    "low": 1.0840,
                    "close": 1.0860,
                    "volume": 0,
                    "data_source": "yahoo_finance",
                    "raw_symbol": "EURUSD=X",
                    "validation": {
                        "quality_score": 0.98,
                        "quality_level": "excellent",
                        "has_gaps": False,
                        "validation_errors": []
                    },
                    "is_outlier": False,
                    "is_gap_fill": False
                }
            }
        }
    )
    
    data: ValidatedOHLCV = Field(description="Cotação mais recente validada")


class BasicQuotesResponse(APIResponse):
    """Resposta simplificada para clientes que não precisam de validação completa"""
    
    data: List[BasicQuote] = Field(description="Lista de cotações básicas")
    metadata: Dict[str, Any] = Field(description="Metadados básicos")
    total_candles: int = Field(description="Total de cotações")


class HealthResponse(APIResponse):
    """Resposta para endpoint /health - status da API"""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "API funcionando corretamente",
                "timestamp": "2025-10-12T10:30:00Z",
                "status": "healthy",
                "version": "1.0.0",
                "services": {
                    "yahoo_finance": "healthy",
                    "data_validator": "healthy",
                    "data_fetcher": "healthy"
                },
                "uptime_seconds": 3600.5
            }
        }
    )
    
    status: str = Field(description="Status da API (healthy/unhealthy)")
    version: str = Field(description="Versão da API")
    services: Dict[str, str] = Field(description="Status dos serviços")
    uptime_seconds: float = Field(description="Tempo de atividade em segundos")


class ErrorResponse(APIResponse):
    """Resposta para erros da API"""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": False,
                "message": "Erro ao buscar dados",
                "timestamp": "2025-10-12T10:30:00Z",
                "error_code": "DATA_FETCH_ERROR",
                "error_details": {
                    "original_error": "Connection timeout",
                    "retry_after": 60
                }
            }
        }
    )
    
    error_code: str = Field(description="Código do erro")
    error_details: Optional[Dict[str, Any]] = Field(description="Detalhes adicionais do erro")


# Modelos para parâmetros de query
class QuotesQueryParams(BaseModel):
    """Parâmetros para endpoint /quotes"""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start_date": "2025-10-11T00:00:00Z",
                "end_date": "2025-10-12T00:00:00Z", 
                "granularity": "1h",
                "format": "validated"
            }
        }
    )
    
    start_date: datetime = Field(description="Data de início (ISO format)")
    end_date: datetime = Field(description="Data de fim (ISO format)")
    granularity: str = Field(
        default="1h",
        description="Granularidade (1min, 5min, 15min, 1h, 1d)",
        pattern=r"^(1min|5min|15min|1h|1d)$"
    )
    format: str = Field(
        default="validated",
        description="Formato da resposta (validated, basic)",
        pattern=r"^(validated|basic)$"
    )
