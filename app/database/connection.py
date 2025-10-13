"""
Database Connection - TimescaleDB/PostgreSQL
Etapa 4: Configuração de conexão otimizada para séries temporais
"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import logging
from typing import Generator

from app.core.config import settings

# Setup logging
logger = logging.getLogger(__name__)

# Configuração do engine SQLAlchemy
engine = create_engine(
    settings.database_url,
    # Configurações específicas para TimescaleDB
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.debug,  # Log SQL queries em debug
    # Otimizações para séries temporais
    connect_args={
        "options": "-c timezone=utc"  # Força UTC
    }
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base para models
Base = declarative_base()

# Metadata para migrations
metadata = MetaData()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency injection para FastAPI
    Fornece sessão de database com auto-cleanup
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """
    Inicializa o database criando todas as tabelas
    e habilitando TimescaleDB extension
    """
    logger.info("🔄 Inicializando database...")
    
    try:
        # Importa todos os models para garantir que sejam registrados
        from app.database import models
        
        # Cria todas as tabelas
        Base.metadata.create_all(bind=engine)
        
        # Habilita TimescaleDB extension (se ainda não estiver)
        with engine.connect() as conn:
            # Verifica se TimescaleDB está disponível
            result = conn.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'"
            )
            
            if not result.fetchone():
                logger.warning("⚠️ TimescaleDB extension não encontrada. Usando PostgreSQL padrão.")
            else:
                logger.info("✅ TimescaleDB extension ativa")
                
                # Cria hypertable para OHLCV data (otimização TimescaleDB)
                try:
                    conn.execute(
                        "SELECT create_hypertable('ohlcv_data', 'timestamp', if_not_exists => TRUE)"
                    )
                    logger.info("✅ Hypertable 'ohlcv_data' criada")
                except Exception as e:
                    logger.warning(f"Hypertable já existe ou erro: {str(e)}")
        
        logger.info("✅ Database inicializado com sucesso")
        
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar database: {str(e)}")
        raise


def check_db_health() -> bool:
    """
    Verifica se o database está saudável
    Usado pelo health check da API
    """
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return False


def get_db_info() -> dict:
    """
    Retorna informações sobre o database
    Útil para debugging e monitoramento
    """
    try:
        with engine.connect() as conn:
            # Versão PostgreSQL
            pg_version = conn.execute("SELECT version()").fetchone()[0]
            
            # Verifica TimescaleDB
            timescale_result = conn.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'"
            ).fetchone()
            
            has_timescale = bool(timescale_result)
            
            # Informações de conexão
            info = {
                "database_url": settings.database_url.split('@')[-1],  # Remove credenciais
                "postgresql_version": pg_version.split()[0:2],
                "has_timescaledb": has_timescale,
                "pool_size": engine.pool.size(),
                "checked_in": engine.pool.checkedin(),
                "checked_out": engine.pool.checkedout(),
                "timezone": "UTC"
            }
            
            if has_timescale:
                # Versão TimescaleDB
                ts_version = conn.execute(
                    "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
                ).fetchone()
                if ts_version:
                    info["timescaledb_version"] = ts_version[0]
            
            return info
            
    except Exception as e:
        logger.error(f"Erro ao obter informações do database: {str(e)}")
        return {
            "error": str(e),
            "database_url": "connection_failed"
        }