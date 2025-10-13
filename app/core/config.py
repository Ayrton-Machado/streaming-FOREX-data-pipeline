"""
Configurações Centralizadas do Pipeline FOREX EUR/USD
Usa Pydantic BaseSettings para validação e type safety
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Literal
import os


class Settings(BaseSettings):
    """
    Configurações da aplicação carregadas de variáveis de ambiente
    Filosofia SLC: Validação automática + valores padrão sensatos
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ===== Application =====
    app_name: str = Field(
        default="FOREX EUR/USD ML Pipeline",
        description="Nome da aplicação"
    )
    app_version: str = Field(
        default="0.1.0",
        description="Versão da API"
    )
    debug: bool = Field(
        default=False,
        description="Modo debug (logs verbosos)"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Nível de logging"
    )
    
    # ===== API Configuration =====
    api_host: str = Field(
        default="0.0.0.0",
        description="Host da API"
    )
    api_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Porta da API"
    )
    api_prefix: str = Field(
        default="/api/v1",
        description="Prefixo base das rotas"
    )
    
    # ===== Data Sources =====
    forex_pair: str = Field(
        default="EURUSD=X",
        description="Ticker EUR/USD no Yahoo Finance"
    )
    default_granularity: str = Field(
        default="1h",
        description="Granularidade padrão para queries"
    )
    max_historical_days: int = Field(
        default=1825,  # ~5 anos
        ge=1,
        le=3650,  # máx 10 anos
        description="Máximo de dias históricos permitidos"
    )
    
    # ===== Premium Data Sources (Etapa 6) =====
    alpha_vantage_api_key: str = Field(
        default="",
        description="API Key do Alpha Vantage (dados fundamentais)"
    )
    polygon_api_key: str = Field(
        default="",
        description="API Key do Polygon.io (dados tick-level)"
    )
    fxcm_access_token: str = Field(
        default="",
        description="Token de acesso FXCM (dados profissionais)"
    )
    fxcm_environment: Literal["demo", "live"] = Field(
        default="demo",
        description="Ambiente FXCM (demo/live)"
    )
    
    # Data Quality Settings
    data_quality_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Limite mínimo de qualidade dos dados (0-1)"
    )
    max_latency_ms: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Latência máxima aceitável em milissegundos"
    )
    enable_failover: bool = Field(
        default=True,
        description="Habilita failover automático entre fontes de dados"
    )
    
    # Legacy settings (mantidos para compatibilidade)
    alpha_vantage_enabled: bool = Field(
        default=False,
        description="Habilita validação com Alpha Vantage (legacy)"
    )
    
    # ===== Database (Etapa 4) =====
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str = Field(default="forex_pipeline")
    postgres_user: str = Field(default="forex_user")
    postgres_password: str = Field(default="forex_password")
    
    @property
    def database_url(self) -> str:
        """Constrói URL de conexão PostgreSQL"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    # ===== Redis Cache (Etapa 7) =====
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379, ge=1, le=65535)
    redis_db: int = Field(default=0, ge=0, le=15)
    redis_password: str = Field(default="")
    cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
        description="TTL do cache em segundos (5min padrão)"
    )
    
    @property
    def redis_url(self) -> str:
        """Constrói URL de conexão Redis"""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    # ===== Data Processing =====
    remove_outliers: bool = Field(
        default=True,
        description="Remove outliers automaticamente"
    )
    outlier_zscore_threshold: float = Field(
        default=3.0,
        ge=1.0,
        le=5.0,
        description="Threshold Z-Score para detectar outliers"
    )
    default_normalization: Literal["minmax", "zscore", "robust"] = Field(
        default="minmax",
        description="Método de normalização padrão"
    )
    
    # ===== Rate Limiting =====
    max_requests_per_minute: int = Field(
        default=60,
        ge=1,
        description="Limite de requests por minuto por IP"
    )
    max_historical_days: int = Field(
        default=1825,
        ge=1,
        description="Máximo de dias históricos permitidos (~5 anos)"
    )
    
    @field_validator("forex_pair")
    @classmethod
    def validate_forex_pair(cls, v: str) -> str:
        """Valida que o par de moedas termina com =X (formato Yahoo Finance)"""
        if not v.endswith("=X"):
            raise ValueError("FOREX pair deve terminar com '=X' (formato Yahoo Finance)")
        return v.upper()
    
    @field_validator("api_prefix")
    @classmethod
    def validate_api_prefix(cls, v: str) -> str:
        """Garante que o prefixo começa com /"""
        if not v.startswith("/"):
            return f"/{v}"
        return v


# Instância global de configurações (Singleton pattern)
settings = Settings()


# Helper para debug
if __name__ == "__main__":
    print("=" * 60)
    print("⚙️  CONFIGURAÇÕES CARREGADAS")
    print("=" * 60)
    print(f"App: {settings.app_name} v{settings.app_version}")
    print(f"Debug: {settings.debug}")
    print(f"API: {settings.api_host}:{settings.api_port}{settings.api_prefix}")
    print(f"FOREX Pair: {settings.forex_pair}")
    print(f"Database URL: {settings.database_url}")
    print(f"Redis URL: {settings.redis_url}")
    print(f"Normalization: {settings.default_normalization}")
    print("=" * 60)
