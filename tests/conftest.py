"""
Configuração do pytest
"""

import pytest
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Configurações globais do pytest
pytest_plugins = []

# Fixtures globais disponíveis para todos os testes
@pytest.fixture(scope="session")
def test_data_dir():
    """Diretório de dados de teste"""
    return Path(__file__).parent / "data"

@pytest.fixture(scope="session") 
def sample_timestamps():
    """Timestamps de exemplo para testes"""
    import pandas as pd
    return pd.date_range('2025-01-01', periods=10, freq='1H', tz='UTC')