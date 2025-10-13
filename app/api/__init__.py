"""
API Package - FastAPI routers e endpoints
Etapa 3: Implementação da API REST para dados FOREX
"""

from .routers import router
from .models import *

__all__ = ["router"]