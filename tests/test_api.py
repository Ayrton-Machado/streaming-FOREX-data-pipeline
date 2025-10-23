"""
Script de Teste - Etapa 3
Valida que a API FastAPI está funcionando corretamente

Execução:
    python tests/test_api.py
"""

import sys
from pathlib import Path

# Adiciona o diretório raiz ao path para importar módulos app
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any
import logging

# FastAPI testing
from fastapi.testclient import TestClient
import pytest

# Imports do nosso projeto
from app.main import app
from app.core.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cliente de teste
client = TestClient(app)

def print_header(title: str):
    """Imprime cabeçalho formatado"""
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_subheader(title: str):
    """Imprime subcabeçalho formatado"""
    print(f"\n📝 {title}")

def print_success(message: str):
    """Imprime mensagem de sucesso"""
    print(f"   ✅ {message}")

def print_error(message: str):
    """Imprime mensagem de erro"""
    print(f"   ❌ {message}")

def print_info(message: str):
    """Imprime mensagem informativa"""
    print(f"   📊 {message}")

def validate_response_structure(response_data: Dict[str, Any], required_fields: list) -> bool:
    """Valida se a resposta tem a estrutura esperada"""
    for field in required_fields:
        if field not in response_data:
            print_error(f"Campo obrigatório '{field}' não encontrado na resposta")
            return False
    return True

def test_root_endpoint():
    """Testa o endpoint raiz"""
    print_subheader("Testando endpoint raiz GET /")
    
    try:
        response = client.get("/")
        
        assert response.status_code == 200, f"Status code esperado 200, recebido: {response.status_code}"
        
        data = response.json()
        required_fields = ["app", "version", "status", "forex_pair", "endpoints"]
        
        assert validate_response_structure(data, required_fields), "Resposta com estrutura inválida"
        
        print_success(f"App: {data['app']}")
        print_success(f"Status: {data['status']}")
        print_success(f"FOREX Pair: {data['forex_pair']}")
        print_success(f"Endpoints disponíveis: {len(data['endpoints'])}")
        
    except Exception as e:
        pytest.fail(f"Erro ao testar endpoint raiz: {e}")

def test_health_endpoint():
    """Testa o endpoint de health check"""
    print_subheader("Testando endpoint GET /api/v1/health")
    
    try:
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200, f"Status code esperado 200, recebido: {response.status_code}"
        
        data = response.json()
        required_fields = ["success", "message", "timestamp", "status", "version", "services", "uptime_seconds"]
        
        assert validate_response_structure(data, required_fields), "Resposta de health com estrutura inválida"
        
        print_success(f"Status: {data['status']}")
        print_success(f"Version: {data['version']}")
        print_success(f"Uptime: {data['uptime_seconds']:.2f}s")
        print_success(f"Services: {list(data['services'].keys())}")
        
        # Valida que todos os serviços estão healthy
        for service, status in data['services'].items():
            assert status == "healthy", f"Serviço {service} não está healthy: {status}"
            print_info(f"✓ {service}: {status}")
    except Exception as e:
        pytest.fail(f"Erro ao testar health endpoint: {e}")

def test_latest_quote_endpoint():
    """Testa o endpoint de cotação mais recente"""
    print_subheader("Testando endpoint GET /api/v1/quote/latest")
    
    try:
        response = client.get("/api/v1/quote/latest")
        
        assert response.status_code == 200, f"Status code esperado 200, recebido: {response.status_code}: {response.text}"
        
        data = response.json()
        required_fields = ["success", "message", "timestamp", "data"]
        
        assert validate_response_structure(data, required_fields), "Resposta latest quote com estrutura inválida"
        
        # Valida estrutura da cotação
        quote_data = data["data"]
        quote_fields = ["timestamp", "open", "high", "low", "close", "volume", "data_source", "validation"]
        
        assert validate_response_structure(quote_data, quote_fields), "Estrutura de quote inválida"
        
        print_success(f"Timestamp: {quote_data['timestamp']}")
        print_success(f"Close: {quote_data['close']}")
        print_success(f"Data Source: {quote_data['data_source']}")
        print_success(f"Quality Score: {quote_data['validation']['quality_score']}")
        print_success(f"Quality Level: {quote_data['validation']['quality_level']}")
        
        # Valida lógica OHLC
        o, h, l, c = quote_data['open'], quote_data['high'], quote_data['low'], quote_data['close']
        assert (l <= min(o, c) <= max(o, c) <= h), f"Lógica OHLC inválida: O={o}, H={h}, L={l}, C={c}"
    except Exception as e:
        pytest.fail(f"Erro ao testar latest quote endpoint: {e}")

def test_historical_quotes_endpoint():
    """Testa o endpoint de dados históricos"""
    print_subheader("Testando endpoint GET /api/v1/quotes")
    
    try:
        # Define período de teste (últimos 3 dias)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=3)
        
        params = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "granularity": "1d",
            "format": "validated"
        }
        
        response = client.get("/api/v1/quotes", params=params)
        
        assert response.status_code == 200, f"Status code esperado 200, recebido: {response.status_code}: {response.text}"
        
        data = response.json()
        required_fields = ["success", "message", "timestamp", "data", "metadata", "total_candles", "quality_stats"]
        
        assert validate_response_structure(data, required_fields), "Historical quotes estrutura inválida"
        
        quotes = data["data"]
        metadata = data["metadata"]
        
        print_success(f"Total de velas: {data['total_candles']}")
        print_success(f"Granularidade: {metadata['granularity']}")
        print_success(f"Período: {metadata['date_range']['start']} até {metadata['date_range']['end']}")
        
        if quotes:
            first_quote = quotes[0]
            print_success(f"Primeira vela: {first_quote['close']} @ {first_quote['timestamp']}")
            print_success(f"Quality Score média: {metadata['validation_result']['quality_score']:.3f}")
        
        # Testa formato básico
        params["format"] = "basic"
        response_basic = client.get("/api/v1/quotes", params=params)
        
        assert response_basic.status_code == 200, "Formato básico falhou"
    except Exception as e:
        pytest.fail(f"Erro ao testar historical quotes endpoint: {e}")

def test_quotes_basic_endpoint():
    """Testa o endpoint simplificado de cotações"""
    print_subheader("Testando endpoint GET /api/v1/quotes/basic")
    
    try:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=2)
        
        params = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "granularity": "1d"
        }
        
        response = client.get("/api/v1/quotes/basic", params=params)
        
        assert response.status_code == 200, f"Status code esperado 200, recebido: {response.status_code}"
        
        data = response.json()
        
        print_success(f"Total de cotações básicas: {data['total_candles']}")
        
        if data["data"]:
            basic_quote = data["data"][0]
            # BasicQuote deve ter apenas campos OHLCV básicos
            expected_fields = ["timestamp", "open", "high", "low", "close", "volume"]
            assert all(field in basic_quote for field in expected_fields), "Estrutura BasicQuote inválida"
    except Exception as e:
        pytest.fail(f"Erro ao testar quotes basic endpoint: {e}")

def test_error_handling():
    """Testa tratamento de erros da API"""
    print_subheader("Testando tratamento de erros")
    
    try:
        # Teste 1: Datas inválidas
        params = {
            "start_date": "2025-10-12T00:00:00Z",
            "end_date": "2025-10-10T00:00:00Z",  # End antes de start
            "granularity": "1h"
        }
        
        response = client.get("/api/v1/quotes", params=params)
        
        assert response.status_code == 400, f"Esperado status 400 para datas inválidas, recebido: {response.status_code}"
        
        # Teste 2: Granularidade inválida
        params = {
            "start_date": "2025-10-10T00:00:00Z",
            "end_date": "2025-10-12T00:00:00Z",
            "granularity": "invalid"
        }
        
        response = client.get("/api/v1/quotes", params=params)
        
        assert response.status_code == 422, f"Esperado status 422 para granularidade inválida, recebido: {response.status_code}"
        
        # Teste 3: Endpoint inexistente
        response = client.get("/api/v1/nonexistent")
        
        assert response.status_code == 404, f"Esperado status 404 para endpoint inexistente, recebido: {response.status_code}"
    except Exception as e:
        pytest.fail(f"Erro ao testar tratamento de erros: {e}")

def test_openapi_documentation():
    """Testa se a documentação OpenAPI está disponível"""
    print_subheader("Testando documentação OpenAPI")
    
    try:
        # Testa schema OpenAPI
        response = client.get("/openapi.json")
        
        assert response.status_code == 200, f"Schema OpenAPI não disponível: {response.status_code}"
        
        schema = response.json()
        
        assert "info" in schema and "paths" in schema, "Schema OpenAPI com estrutura inválida"
        print_success(f"Schema OpenAPI válido: {schema['info']['title']} v{schema['info']['version']}")
        print_success(f"Endpoints documentados: {len(schema['paths'])}")
        
        # Testa se Swagger UI está disponível
        response = client.get("/docs")
        
        assert response.status_code == 200, "Swagger UI não está disponível"
    except Exception as e:
        pytest.fail(f"Erro ao testar documentação: {e}")

def main():
    """Executa todos os testes da Etapa 3"""
    
    print_header("🚀 TESTE DA API - ETAPA 3")
    print("   FastAPI + Endpoints REST para dados FOREX")
    print_header("")
    
    print_info(f"⚙️ Configurações:")
    print_info(f"   - App: {settings.app_name}")
    print_info(f"   - FOREX Pair: {settings.forex_pair}")
    print_info(f"   - Max Historical Days: {settings.max_historical_days}")
    
    # Lista de testes
    tests = [
        ("Endpoint Raiz", test_root_endpoint),
        ("Health Check", test_health_endpoint),
        ("Latest Quote", test_latest_quote_endpoint),
        ("Historical Quotes", test_historical_quotes_endpoint),
        ("Basic Quotes", test_quotes_basic_endpoint),
        ("Error Handling", test_error_handling),
        ("OpenAPI Documentation", test_openapi_documentation),
    ]
    
    # Executa testes
    results = []
    
    for test_name, test_func in tests:
        print_header(f"TESTE: {test_name}")
        try:
            test_func()
            results.append((test_name, True))
            print_success(f"✅ {test_name} - PASSOU")
        except AssertionError as ae:
            print_error(f"❌ {test_name} - FALHOU: {ae}")
            results.append((test_name, False))
        except Exception as e:
            print_error(f"❌ {test_name} - ERRO: {e}")
            results.append((test_name, False))
    
    # Resultado final
    print_header("📊 RESULTADO FINAL")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{status} - {test_name}")
    
    print_header("")
    print(f"   Total: {passed}/{total} testes passaram")
    print_header("")
    
    if passed == total:
        print("🎉 ETAPA 3 COMPLETA! Todos os testes passaram!")
        print()
        print("📝 O que foi implementado:")
        print("   ✅ FastAPI app com estrutura completa")
        print("   ✅ Endpoint GET /api/v1/health (health check)")
        print("   ✅ Endpoint GET /api/v1/quote/latest (cotação atual)")
        print("   ✅ Endpoint GET /api/v1/quotes (dados históricos)")
        print("   ✅ Endpoint GET /api/v1/quotes/basic (formato simplificado)")
        print("   ✅ Middleware CORS + logging de requests")
        print("   ✅ Exception handlers padronizados")
        print("   ✅ Documentação automática (Swagger)")
        print("   ✅ Response models Pydantic")
        print()
        print("📝 Próximos passos:")
        print("   1. Etapa 4: TimescaleDB + Persistência")
        print("   2. Etapa 5: Feature Engineering (indicadores técnicos)")
        print("   3. Etapa 6: WebSocket + Streaming real-time")
        
        return True
    else:
        print(f"❌ {total - passed} teste(s) falharam. Verifique os logs acima.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)