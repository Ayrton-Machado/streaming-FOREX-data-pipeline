"""Tests package - Testes unitários e de integração"""

# Configuração pytest
import sys
from pathlib import Path

# Adiciona app ao path para imports
app_dir = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_dir))