"""
Script de Teste - Etapa 2
Valida schemas Pydantic + validação de qualidade de dados

Execução:
    python tests/test_validation.py
"""

import sys
from pathlib import Path

# Adiciona o diretório raiz ao path para importar módulos app
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from datetime import datetime, timedelta
import logging

# Imports do nosso projeto
from app.services.data_fetcher import DataFetcher, DataFetchError
from app.services.data_validator import DataValidator, DataValidationError
from app.domain.enums import (
    Granularity, 
    NormalizationMethod, 
    DataSource, 
    DataQuality
)
from app.domain.schemas import (
    RawOHLCV,
    ValidatedOHLCV,
    BasicQuote,
    TechnicalIndicators
)
from app.core.config import settings

# Setup logging bonito
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)


def print_separator(title: str = ""):
    """Print separador visual"""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)
    else:
        print()


def test_enums():
    """Teste 1: Verifica se os Enums estão funcionando"""
    print_separator("TESTE 1: Enums e Constantes")
    
    try:
        # Testa Granularity
        print("📝 Testando Granularity enum:")
        for granularity in Granularity:
            yf_interval = Granularity.to_yfinance_interval(granularity)
            print(f"   {granularity.value} → {yf_interval}")
        
        # Testa NormalizationMethod
        print(f"\n📝 Métodos de normalização: {NormalizationMethod.get_valid_values()}")
        
        # Testa DataSource
        print(f"📝 Fonte primária: {DataSource.get_primary_source()}")
        
        # Testa DataQuality
        print(f"\n📝 Testando níveis de qualidade:")
        test_scores = [0.98, 0.85, 0.65, 0.45]
        for score in test_scores:
            level = DataQuality.from_score(score)
            print(f"   Score {score} → {level}")
        
        print(f"\n✅ Todos os enums funcionando corretamente!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO ao testar enums: {e}")
        return False


def test_raw_ohlcv_schema():
    """Teste 2: Valida schema RawOHLCV"""
    print_separator("TESTE 2: Schema RawOHLCV")
    
    try:
        print("📝 Criando RawOHLCV válido:")
        
        # Dados válidos
        valid_candle = RawOHLCV(
            timestamp=datetime.now(),
            open=1.0850,
            high=1.0870,
            low=1.0840,
            close=1.0860,
            volume=125000,
            data_source=DataSource.YAHOO_FINANCE,
            raw_symbol="EURUSD=X"
        )
        
        print(f"   ✅ Vela criada: {valid_candle.close} @ {valid_candle.timestamp}")
        print(f"   📊 Fonte: {valid_candle.data_source}")
        
        # Testa validação automática
        print(f"\n📝 Testando validações automáticas:")
        
        # Teste 2.1: High < Low (deve falhar)
        print("   Teste 2.1: High < Low")
        try:
            invalid_candle = RawOHLCV(
                timestamp=datetime.now(),
                open=1.0850,
                high=1.0840,  # High menor que Low!
                low=1.0850,
                close=1.0845,
                volume=0
            )
            print("   ❌ FALHOU: Deveria ter lançado ValidationError")
            return False
        except Exception as e:
            print(f"   ✅ PASSOU: {e}")
        
        # Teste 2.2: Preços negativos (deve falhar)
        print("   Teste 2.2: Preços negativos")
        try:
            invalid_candle = RawOHLCV(
                timestamp=datetime.now(),
                open=-1.0,  # Preço negativo!
                high=1.0870,
                low=1.0840,
                close=1.0860,
                volume=0
            )
            print("   ❌ FALHOU: Deveria ter lançado ValidationError")
            return False
        except Exception as e:
            print(f"   ✅ PASSOU: {e}")
        
        print(f"\n✅ Schema RawOHLCV funcionando corretamente!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO ao testar RawOHLCV: {e}")
        return False


def test_data_validator():
    """Teste 3: Testa DataValidator com dados reais"""
    print_separator("TESTE 3: DataValidator com dados reais")
    
    try:
        # Busca dados reais para testar
        fetcher = DataFetcher()
        validator = DataValidator()
        
        print("📝 Buscando dados reais para validação...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Busca dados brutos
        df = fetcher.get_historical_data(
            start_date=start_date,
            end_date=end_date,
            granularity="1d"
        )
        
        print(f"   ✅ {len(df)} velas obtidas")
        
        # Valida os dados
        print(f"\n📝 Validando qualidade dos dados...")
        validated_candles, validation_result = validator.validate_dataframe(
            df, Granularity.ONE_DAY, DataSource.YAHOO_FINANCE
        )
        
        # Mostra resultado da validação
        print(f"\n📊 Resultado da Validação:")
        print(f"   - Total de velas: {len(validated_candles)}")
        print(f"   - Quality Score: {validation_result.quality_score:.3f}")
        print(f"   - Quality Level: {validation_result.quality_level}")
        print(f"   - Outliers: {validation_result.outlier_count}")
        print(f"   - Gaps: {validation_result.gap_percentage:.2f}%")
        print(f"   - Valores faltantes: {validation_result.missing_values_count}")
        print(f"   - Erros: {validation_result.validation_errors}")
        
        # Testa summary
        summary = validator.get_validation_summary(validation_result)
        print(f"\n📋 Summary da validação:")
        print(summary)
        
        # Verifica se temos ValidatedOHLCV válidos
        if validated_candles:
            first_candle = validated_candles[0]
            print(f"\n📊 Primeira vela validada:")
            print(f"   - Timestamp: {first_candle.timestamp}")
            print(f"   - Close: {first_candle.close}")
            print(f"   - Quality Score: {first_candle.validation.quality_score:.3f}")
            print(f"   - Is Outlier: {first_candle.is_outlier}")
            print(f"   - Data Source: {first_candle.data_source}")
        
        print(f"\n✅ DataValidator funcionando corretamente!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO ao testar DataValidator: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integrated_fetcher():
    """Teste 4: Testa DataFetcher integrado com validação"""
    print_separator("TESTE 4: DataFetcher integrado (novos métodos)")
    
    try:
        fetcher = DataFetcher()
        
        print("📝 Testando get_historical_data_validated()...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)
        
        # Usa novo método integrado
        validated_candles, validation_result = fetcher.get_historical_data_validated(
            start_date=start_date,
            end_date=end_date,
            granularity=Granularity.ONE_DAY  # Agora usa Enum!
        )
        
        print(f"   ✅ {len(validated_candles)} velas validadas")
        print(f"   📊 Quality Score: {validation_result.quality_score:.3f}")
        
        # Testa conversão para BasicQuote
        print(f"\n📝 Testando conversão para BasicQuote...")
        basic_quotes = fetcher.convert_validated_to_basic_quotes(validated_candles)
        
        print(f"   ✅ {len(basic_quotes)} cotações básicas convertidas")
        
        if basic_quotes:
            quote = basic_quotes[0]
            print(f"   📊 Primeira cotação: {quote.close} @ {quote.timestamp}")
        
        # Testa estatísticas
        print(f"\n📝 Testando get_validation_stats()...")
        stats = fetcher.get_validation_stats(validated_candles)
        
        print(f"   ✅ Stats geradas:")
        print(f"      - Total: {stats['total_candles']}")
        print(f"      - Quality média: {stats['quality_stats']['average_quality_score']:.3f}")
        print(f"      - Outliers: {stats['quality_stats']['outlier_count']}")
        
        print(f"\n✅ DataFetcher integrado funcionando!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO ao testar DataFetcher integrado: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_candle_validation():
    """Teste 5: Validação de vela única"""
    print_separator("TESTE 5: Validação de vela única")
    
    try:
        validator = DataValidator()
        
        print("📝 Criando vela única para validação...")
        
        # Cria RawOHLCV
        raw_candle = RawOHLCV(
            timestamp=datetime.now(),
            open=1.0850,
            high=1.0870,
            low=1.0840,
            close=1.0860,
            volume=125000,
            data_source=DataSource.YAHOO_FINANCE,
            raw_symbol="EURUSD=X"
        )
        
        # Valida vela única
        validated_candle = validator.validate_single_candle(raw_candle)
        
        print(f"   ✅ Vela validada:")
        print(f"      - Timestamp: {validated_candle.timestamp}")
        print(f"      - Close: {validated_candle.close}")
        print(f"      - Quality Score: {validated_candle.validation.quality_score:.3f}")
        print(f"      - Quality Level: {validated_candle.validation.quality_level}")
        print(f"      - Is Outlier: {validated_candle.is_outlier}")
        
        print(f"\n✅ Validação de vela única funcionando!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO ao testar vela única: {e}")
        return False


def test_backward_compatibility():
    """Teste 6: Compatibilidade com métodos legados"""
    print_separator("TESTE 6: Compatibilidade com Etapa 1")
    
    try:
        fetcher = DataFetcher()
        
        print("📝 Testando métodos legados da Etapa 1...")
        
        # Testa método legado
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        df = fetcher.get_historical_data(
            start_date=start_date,
            end_date=end_date,
            granularity="1d"  # String como antes
        )
        
        print(f"   ✅ get_historical_data() legado: {len(df)} velas")
        
        # Testa get_data_info legado
        info = fetcher.get_data_info(df)
        print(f"   ✅ get_data_info() legado: {info['total_candles']} velas")
        
        print(f"\n✅ Compatibilidade com Etapa 1 mantida!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO na compatibilidade: {e}")
        return False


def main():
    """Executa todos os testes da Etapa 2"""
    print_separator("🚀 TESTE DE VALIDAÇÃO - ETAPA 2")
    print("   Schemas Pydantic + Validação de Qualidade")
    print_separator()
    
    print("\n⚙️  Configurações carregadas:")
    print(f"   - App: {settings.app_name}")
    print(f"   - FOREX Pair: {settings.forex_pair}")
    print(f"   - Outlier Threshold: {settings.outlier_zscore_threshold}")
    
    # Executa testes
    results = []
    
    results.append(("Teste 1 - Enums", test_enums()))
    results.append(("Teste 2 - Schema RawOHLCV", test_raw_ohlcv_schema()))
    results.append(("Teste 3 - DataValidator", test_data_validator()))
    results.append(("Teste 4 - DataFetcher Integrado", test_integrated_fetcher()))
    results.append(("Teste 5 - Vela Única", test_single_candle_validation()))
    results.append(("Teste 6 - Compatibilidade", test_backward_compatibility()))
    
    # Resultado final
    print_separator("📊 RESULTADO FINAL")
    
    total = len(results)
    passed = sum(1 for _, result in results if result)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{status} - {test_name}")
    
    print("\n" + "=" * 80)
    print(f"   Total: {passed}/{total} testes passaram")
    print("=" * 80)
    
    if passed == total:
        print("\n🎉 ETAPA 2 COMPLETA! Todos os testes passaram!")
        print("\n📝 O que foi implementado:")
        print("   ✅ Enums tipados (Granularity, NormalizationMethod, DataSource, etc)")
        print("   ✅ Schemas Pydantic completos (RawOHLCV, ValidatedOHLCV, etc)")
        print("   ✅ DataValidator com detecção de outliers, gaps e quality score")
        print("   ✅ DataFetcher integrado com validação")
        print("   ✅ Compatibilidade com Etapa 1 mantida")
        print("   ✅ Validação de velas únicas para real-time")
        
        print("\n📝 Próximos passos:")
        print("   1. Etapa 3: FastAPI básico + Endpoint /quotes")
        print("   2. Etapa 4: TimescaleDB + Persistência")
        print("   3. Etapa 5: Feature Engineering (indicadores técnicos)")
        return 0
    else:
        print("\n⚠️  Alguns testes falharam. Revise os erros acima.")
        return 1


if __name__ == "__main__":
    exit(main())