"""
Main FastAPI Application - FOREX EUR/USD ML Pipeline
Etapa 3: API REST com endpoints para dados FOREX

Filosofia SLC:
- Simple: Endpoints claros e intuitivos
- Lovable: Documentação automática, respostas padronizadas
- Complete: Tratamento de erros, CORS, logging, health check
"""

import logging
import time
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from app.core.config import settings
from app.api.routers import router
from app.api.persistence import persistence_router
from app.api.models import ErrorResponse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação"""
    # Startup
    logger.info("🚀 Iniciando FOREX EUR/USD API...")
    logger.info(f"📊 Par FOREX: {settings.forex_pair}")
    logger.info(f"🔗 Documentação: http://localhost:8000/docs")
    logger.info(f"🌐 WebSocket Test Client: http://localhost:8000/ws/test-client")
    
    yield
    
    # Shutdown
    logger.info("⭐ Encerrando FOREX EUR/USD API...")
    
    # Cleanup WebSocket connections
    try:
        from app.services.websocket_manager import websocket_manager
        await websocket_manager.shutdown()
        logger.info("✅ WebSocket manager desligado com sucesso")
    except ImportError:
        logger.warning("⚠️ WebSocket manager não disponível para shutdown")
    except Exception as e:
        logger.error(f"❌ Erro ao desligar WebSocket manager: {e}")


# Cria aplicação FastAPI
app = FastAPI(
    title=settings.app_name,
    description="""
    ## 📈 FOREX EUR/USD ML Pipeline API - Data Provider Elite
    
    **Premium data provider for AI trading systems with institutional-grade streaming.**
    
    ### 🚀 Características:
    - **Dados premium** via Alpha Vantage, Polygon.io, FXCM
    - **WebSocket streaming** de alta frequência para sistemas IA
    - **8 canais especializados** para diferentes estratégias de trading
    - **Validação de qualidade** automática com fallback
    - **Microestrutura de mercado** e análise de liquidez
    - **ML-ready features** com vetores normalizados
    - **Schemas Pydantic** para type safety
    
    ### 📊 REST Endpoints:
    - `GET /quote/latest` - Cotação mais recente
    - `GET /quotes` - Dados históricos validados
    - `POST /api/v1/data/save` - Salvar dados com preprocessamento
    - `POST /api/v1/data/query` - Consultar dados com filtros avançados
    - `GET /api/v1/data/stats/{symbol}` - Estatísticas dos dados
    - `GET /api/v1/premium/*` - Endpoints de dados premium
    - `GET /health` - Status da API
    
    ### � WebSocket Streaming:
    - `WS /ws/stream` - Conexão para streaming em tempo real
    - `GET /ws/test-client` - Cliente de teste interativo
    - `GET /ws/stats` - Estatísticas de streaming
    - `GET /ws/streams` - Canais disponíveis
    
    ### 🎯 Canais de Streaming:
    1. **raw_ticks** (10Hz) - Dados tick de alta frequência para scalping
    2. **ml_features** (1Hz) - Vetores ML prontos para inferência
    3. **trading_signals** (2Hz) - Sinais com entry/exit e gestão de risco
    4. **pattern_alerts** - Alertas de padrões chartistas
    5. **technical_analysis** (1Hz) - Indicadores técnicos e níveis
    6. **order_book** (5Hz) - Livro de ordens L2 para market making
    7. **microstructure** - Métricas de microestrutura e liquidez
    8. **economic_events** - Eventos econômicos e impacto esperado
    
    ### 🤖 Casos de uso para IA:
    1. **Algoritmos de scalping** - raw_ticks + order_book
    2. **ML model inference** - ml_features com vetores normalizados
    3. **News trading bots** - economic_events + pattern_alerts
    4. **Market making** - microstructure + order_book
    5. **Multi-timeframe analysis** - technical_analysis + trading_signals
    """,
    version="1.0.0",
    contact={
        "name": "FOREX ML Pipeline",
        "url": "https://github.com/Ayrton-Machado/streaming-FOREX-data-pipeline",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan
)

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar domínios permitidos
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Middleware para logging de requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log todas as requisições HTTP"""
    start_time = time.time()
    
    # Log da requisição
    logger.info(
        f"📝 {request.method} {request.url.path} "
        f"({request.client.host if request.client else 'unknown'})"
    )
    
    # Processa requisição
    response = await call_next(request)
    
    # Log da resposta
    process_time = time.time() - start_time
    logger.info(
        f"✅ {request.method} {request.url.path} → "
        f"{response.status_code} ({process_time:.3f}s)"
    )
    
    return response


# Exception handlers globais
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler para HTTPExceptions padronizadas"""
    
    # Se o detail já é um dict (nosso formato), usa direto
    if isinstance(exc.detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    
    # Senão, formata no nosso padrão
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": str(exc.detail),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_code": f"HTTP_{exc.status_code}",
            "error_details": None
        }
    )


@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """Handler para erros internos não capturados"""
    logger.error(f"❌ Erro interno não capturado: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Erro interno do servidor",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_code": "INTERNAL_SERVER_ERROR",
            "error_details": {"original_error": str(exc)}
        }
    )


# Inclui routers
app.include_router(router, prefix="/api/v1", tags=["FOREX Data"])
app.include_router(persistence_router, tags=["Data Persistence"])

# Inclui router de features avançadas
try:
    from app.api.advanced_features import router as advanced_router
    app.include_router(advanced_router, tags=["Advanced Features"])
    logger.info("✅ Router de features avançadas carregado")
except ImportError as e:
    logger.warning(f"⚠️ Router de features avançadas não carregado: {e}")

# Inclui router de dados premium (Etapa 6)
try:
    from app.api.premium_data import router as premium_router
    app.include_router(premium_router, prefix="/api/v1", tags=["Premium Data"])
    logger.info("✅ Router de dados premium carregado")
except ImportError as e:
    logger.warning(f"⚠️ Router de dados premium não carregado: {e}")

# Inclui router de WebSocket streaming (Etapa 8)
try:
    from app.api.websocket_routes import router as websocket_router
    app.include_router(websocket_router, tags=["WebSocket Streaming"])
    logger.info("✅ Router de WebSocket streaming carregado")
except ImportError as e:
    logger.warning(f"⚠️ Router de WebSocket streaming não carregado: {e}")


# Endpoint raiz com informações da API
@app.get("/", tags=["Root"])
async def root():
    """
    Informações básicas da API
    """
    return {
        "app": settings.app_name,
        "version": "1.0.0",
        "status": "running",
        "forex_pair": settings.forex_pair,
        "docs_url": "/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "latest_quote": "/api/v1/quote/latest", 
            "historical_quotes": "/api/v1/quotes",
            "basic_quotes": "/api/v1/quotes/basic",
            "save_data": "/api/v1/data/save",
            "query_data": "/api/v1/data/query",
            "data_stats": "/api/v1/data/stats/{symbol}",
            "analyze_data": "/api/v1/data/analyze/{symbol}",
            "premium_health": "/api/v1/premium/health",
            "premium_tick_data": "/api/v1/premium/tick-data",
            "premium_order_book": "/api/v1/premium/order-book",
            "premium_microstructure": "/api/v1/premium/microstructure",
            "premium_quote": "/api/v1/premium/quote/latest",
            "websocket_stream": "/ws/stream",
            "websocket_stats": "/ws/stats",
            "websocket_test_client": "/ws/test-client",
            "available_streams": "/ws/streams"
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# Customiza OpenAPI schema
def custom_openapi():
    """Customiza a documentação OpenAPI"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Adiciona exemplos customizados
    openapi_schema["info"]["x-logo"] = {
        "url": "https://via.placeholder.com/120x120/1e40af/ffffff?text=FOREX"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Para debug local
if __name__ == "__main__":
    import uvicorn
    logger.info("🔥 Rodando em modo DEBUG")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )