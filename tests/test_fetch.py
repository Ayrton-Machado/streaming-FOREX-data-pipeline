"""
Script de Teste - Etapa 1
Valida que o DataFetcher está funcionando corretamente

Execução:
    python tests/test_fetch.py
"""

import sys
from pathlib import Path

# Adiciona o diretório raiz ao path para importar módulos app
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from datetime import datetime, timedelta
from app.services.data_fetcher import DataFetcher, DataFetchError
from app.core.config import settings
import logging
import pytest

# Setup logging bonito
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)


def print_separator(title: str = ""):
    """Print separador visual"""
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)
    else:
        print()


def test_basic_fetch():
    """Teste 1: Buscar dados dos últimos 7 dias (granularidade diária)"""
    print_separator("TESTE 1: Buscar dados EUR/USD últimos 7 dias (1d)")
    
    try:
        fetcher = DataFetcher()
        
        # Define período
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print(f"📅 Período: {start_date.date()} até {end_date.date()}")
        print(f"🎯 Granularidade: 1d")
        print(f"📊 Ticker: {settings.forex_pair}")
        print("\nBuscando dados...")
        
        # Busca dados
        df = fetcher.get_historical_data(
            start_date=start_date,
            end_date=end_date,
            granularity="1d"
        )
        
        # Mostra resultados
        print(f"\n✅ SUCESSO! {len(df)} velas carregadas")
        print(f"\n📋 Primeiras 3 linhas:")
        print(df.head(3).to_string())
        
        print(f"\n📋 Últimas 3 linhas:")
        print(df.tail(3).to_string())
        
        # Informações adicionais
        info = fetcher.get_data_info(df)
        print(f"\n📊 Estatísticas:")
        print(f"   - Total de velas: {info['total_candles']}")
        print(f"   - Preço mínimo: ${info['price_stats']['min_close']:.5f}")
        print(f"   - Preço máximo: ${info['price_stats']['max_close']:.5f}")
        print(f"   - Preço médio: ${info['price_stats']['mean_close']:.5f}")
        print(f"   - Desvio padrão: ${info['price_stats']['std_close']:.5f}")
        
    except DataFetchError as e:
        pytest.fail(f"Erro ao buscar dados: {e}")
    except Exception as e:
        pytest.fail(f"Erro inesperado: {e}")


def test_hourly_fetch():
    """Teste 2: Buscar dados horários (últimas 48h)"""
    print_separator("TESTE 2: Buscar dados EUR/USD últimas 48h (1h)")
    
    try:
        fetcher = DataFetcher()
        
        # Define período
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        print(f"📅 Período: {start_date.strftime('%Y-%m-%d %H:%M')} até {end_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"🎯 Granularidade: 1h")
        print("\nBuscando dados...")
        
        # Busca dados
        df = fetcher.get_historical_data(
            start_date=start_date,
            end_date=end_date,
            granularity="1h"
        )
        
        # Mostra resultados
        print(f"\n✅ SUCESSO! {len(df)} velas carregadas")
        print(f"\n📋 Últimas 5 linhas:")
        print(df.tail(5).to_string())
        
        # Verifica gaps (importante para ML)
        expected_hours = int((end_date - start_date).total_seconds() / 3600)
        actual_hours = len(df)
        gap_percentage = ((expected_hours - actual_hours) / expected_hours * 100) if expected_hours > 0 else 0
        
        print(f"\n🔍 Análise de Gaps:")
        print(f"   - Horas esperadas: {expected_hours}")
        print(f"   - Horas recebidas: {actual_hours}")
        print(f"   - Gaps: {gap_percentage:.2f}%")
        
        if gap_percentage > 5:
            print(f"   ⚠️  ATENÇÃO: Gaps acima de 5% (mercado fechado ou feriados)")
        else:
            print(f"   ✅ Gaps dentro do aceitável")
        
    except DataFetchError as e:
        pytest.fail(f"Erro ao buscar dados (hourly): {e}")
    except Exception as e:
        pytest.fail(f"Erro inesperado (hourly): {e}")


def test_latest_quote():
    """Teste 3: Buscar cotação mais recente"""
    print_separator("TESTE 3: Buscar cotação mais recente EUR/USD")
    
    try:
        fetcher = DataFetcher()
        
        print("Buscando cotação mais recente...")
        
        # Busca cotação
        quote = fetcher.get_latest_quote()
        
        # Mostra resultados
        print(f"\n✅ SUCESSO!")
        print(f"\n💹 Cotação Atual EUR/USD:")
        print(f"   - Timestamp: {quote['timestamp']}")
        print(f"   - Open:   ${quote['open']:.5f}")
        print(f"   - High:   ${quote['high']:.5f}")
        print(f"   - Low:    ${quote['low']:.5f}")
        print(f"   - Close:  ${quote['close']:.5f}")
        print(f"   - Volume: {quote['volume']:,}")
        
        # Cálculo simples
        change = quote['close'] - quote['open']
        change_pct = (change / quote['open'] * 100) if quote['open'] > 0 else 0
        
        print(f"\n📈 Variação:")
        print(f"   - Absoluta: ${change:+.5f}")
        print(f"   - Percentual: {change_pct:+.3f}%")
        
    except DataFetchError as e:
        pytest.fail(f"Erro ao buscar cotação: {e}")
    except Exception as e:
        pytest.fail(f"Erro inesperado ao buscar cotação: {e}")


def test_invalid_inputs():
    """Teste 4: Validação de inputs inválidos"""
    print_separator("TESTE 4: Validação de inputs inválidos")
    
    fetcher = DataFetcher()
    tests_passed = 0
    total_tests = 3
    
    # Teste 4.1: Granularidade inválida
    print("\n📝 Teste 4.1: Granularidade inválida")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        fetcher.get_historical_data(start_date, end_date, granularity="10s")
        print("   ❌ FALHOU: Deveria ter lançado ValueError")
    except ValueError as e:
        print(f"   ✅ PASSOU: {e}")
        tests_passed += 1
    
    # Teste 4.2: Data de início posterior à data de fim
    print("\n📝 Teste 4.2: Data de início > Data de fim")
    try:
        end_date = datetime.now()
        start_date = end_date + timedelta(days=1)
        fetcher.get_historical_data(start_date, end_date, granularity="1d")
        print("   ❌ FALHOU: Deveria ter lançado ValueError")
    except ValueError as e:
        print(f"   ✅ PASSOU: {e}")
        tests_passed += 1
    
    # Teste 4.3: Período muito longo (mais de 5 anos)
    print("\n📝 Teste 4.3: Período muito longo (>5 anos)")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2000)  # ~5.5 anos
        fetcher.get_historical_data(start_date, end_date, granularity="1d")
        print("   ❌ FALHOU: Deveria ter lançado ValueError")
    except ValueError as e:
        print(f"   ✅ PASSOU: {e}")
        tests_passed += 1
    
    print(f"\n📊 Resultado: {tests_passed}/{total_tests} testes de validação passaram")
    assert tests_passed == total_tests, f"{tests_passed}/{total_tests} validation subtests passed"


def main():
    """Executa todos os testes"""
    print_separator("🚀 TESTE DE VALIDAÇÃO - ETAPA 1")
    print("   DataFetcher Service - Yahoo Finance EUR/USD")
    print_separator()
    
    print("\n⚙️  Configurações carregadas:")
    print(f"   - App: {settings.app_name}")
    print(f"   - FOREX Pair: {settings.forex_pair}")
    print(f"   - Debug: {settings.debug}")
    
    # Executa testes
    results = []
    
    results.append(("Teste 1 - Dados Diários", test_basic_fetch()))
    results.append(("Teste 2 - Dados Horários", test_hourly_fetch()))
    results.append(("Teste 3 - Cotação Recente", test_latest_quote()))
    results.append(("Teste 4 - Validação Inputs", test_invalid_inputs()))
    
    # Resultado final
    print_separator("📊 RESULTADO FINAL")
    
    total = len(results)
    passed = sum(1 for _, result in results if result)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{status} - {test_name}")
    
    print("\n" + "=" * 70)
    print(f"   Total: {passed}/{total} testes passaram")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 ETAPA 1 COMPLETA! Todos os testes passaram!")
        print("\n📝 Próximos passos:")
        print("   1. Revisar código gerado")
        print("   2. Instalar dependências: pip install -r requirements.txt")
        print("   3. Criar arquivo .env baseado no .env.example")
        print("   4. Avançar para Etapa 2: Schemas Pydantic + Validação")
        return 0
    else:
        print("\n⚠️  Alguns testes falharam. Revise os erros acima.")
        return 1


if __name__ == "__main__":
    exit(main())
