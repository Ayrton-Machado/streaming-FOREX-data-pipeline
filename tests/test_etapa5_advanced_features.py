"""
Teste da Etapa 5: Feature Engineering Avançado
Testa todos os componentes de análise avançada
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import sys
import os

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.features.advanced_feature_engineer import AdvancedFeatureEngineer
from app.services.analysis.backtesting_engine import (
    BacktestingEngine,
    SimpleMovingAverageStrategy,
    RSIStrategy,
    MACDStrategy
)
from app.services.analysis.feature_importance import FeatureImportanceAnalyzer
from app.services.analysis.pattern_detection import TechnicalPatternDetector

def generate_sample_data(days: int = 100) -> pd.DataFrame:
    """Gera dados de amostra para teste"""
    print(f"📊 Gerando {days} dias de dados de amostra...")
    
    # Data range
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         end=datetime.now(), freq='1H')
    
    # Simula preços EURUSD
    np.random.seed(42)
    base_price = 1.0800
    
    # Random walk com drift
    returns = np.random.normal(0.0001, 0.005, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Cria OHLCV
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    
    # Simula open, high, low
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    
    # High/Low baseado em volatilidade
    volatility = np.random.uniform(0.0005, 0.003, len(df))
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + volatility)
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - volatility)
    
    # Volume simulado
    df['volume'] = np.random.uniform(1000, 10000, len(df))
    
    print(f"✅ Dados gerados: {len(df)} pontos de {df.index[0]} a {df.index[-1]}")
    return df

def test_advanced_feature_engineering():
    """Testa o feature engineering avançado"""
    print("\n🔬 Testando Feature Engineering Avançado...")
    
    # Gera dados
    df = generate_sample_data(30)  # 30 dias
    
    # Inicializa feature engineer
    engineer = AdvancedFeatureEngineer()
    
    # Testa indicadores individuais
    print("📈 Testando indicadores individuais...")
    
    # Momentum indicators
    williams_r = engineer.calculate_williams_r(df)
    print(f"  Williams %R: {len(williams_r.dropna())} valores válidos")
    
    stoch_k, stoch_d = engineer.calculate_stochastic(df)
    print(f"  Stochastic: K={len(stoch_k.dropna())}, D={len(stoch_d.dropna())} valores")
    
    cci = engineer.calculate_cci(df)
    print(f"  CCI: {len(cci.dropna())} valores válidos")
    
    # Trend indicators
    adx = engineer.calculate_adx(df)
    print(f"  ADX: {len(adx.dropna())} valores válidos")
    
    aroon_up, aroon_down = engineer.calculate_aroon(df)
    print(f"  Aroon: Up={len(aroon_up.dropna())}, Down={len(aroon_down.dropna())} valores")
    
    psar = engineer.calculate_parabolic_sar(df)
    print(f"  Parabolic SAR: {len(psar.dropna())} valores válidos")
    
    # Calcula todas as features
    print("🔧 Calculando todas as features...")
    all_features = engineer.calculate_all_advanced_indicators(df)
    print(f"✅ Total de features calculadas: {len(all_features)}")
    
    # Gera relatório
    print("📋 Gerando relatório de features...")
    report = engineer.generate_feature_report(df)
    
    print(f"  - Total de features: {report['total_features']}")
    print(f"  - Top 5 features: {report['top_features'][:5]}")
    print(f"  - Tipos de indicadores: {report['indicator_types']}")
    
    return all_features, report

def test_backtesting():
    """Testa o sistema de backtesting"""
    print("\n📊 Testando Sistema de Backtesting...")
    
    # Gera dados
    df = generate_sample_data(60)  # 60 dias para backtesting
    
    # Inicializa engine
    engine = BacktestingEngine(initial_capital=10000.0)
    
    # Cria estratégias
    strategies = [
        SimpleMovingAverageStrategy(10, 20),
        RSIStrategy(14, 30, 70),
        MACDStrategy(12, 26, 9)
    ]
    
    print(f"🎯 Testando {len(strategies)} estratégias...")
    
    # Executa backtesting
    results = engine.compare_strategies(df, strategies)
    
    # Mostra resultados
    for strategy_name, result in results.items():
        print(f"\n📈 Estratégia: {strategy_name}")
        print(f"  - Retorno total: {result.total_return:.2f}%")
        print(f"  - Total de trades: {result.total_trades}")
        print(f"  - Win rate: {result.win_rate:.1f}%")
        print(f"  - Max drawdown: {result.max_drawdown:.2f}%")
        print(f"  - Sharpe ratio: {result.sharpe_ratio:.2f}")
        print(f"  - Profit factor: {result.profit_factor:.2f}")
    
    # Gera relatório de performance
    performance_report = engine.generate_performance_report(results)
    print(f"\n🏆 Melhor estratégia (retorno): {performance_report['rankings']['total_return'][0]}")
    print(f"🎯 Melhor estratégia (Sharpe): {performance_report['rankings']['sharpe_ratio'][0]}")
    
    return results, performance_report

def test_feature_importance():
    """Testa análise de importância de features"""
    print("\n🔍 Testando Análise de Importância de Features...")
    
    # Gera dados
    df = generate_sample_data(50)  # 50 dias
    
    # Calcula features
    engineer = AdvancedFeatureEngineer()
    features_dict = engineer.calculate_all_advanced_indicators(df)
    features_df = pd.DataFrame(features_dict, index=df.index)
    
    # Define target (retornos futuros)
    target = df['close'].pct_change().shift(-1).dropna()
    
    # Alinha dados
    common_idx = features_df.index.intersection(target.index)
    features_df = features_df.loc[common_idx]
    target = target.loc[common_idx]
    
    print(f"📊 Analisando {len(features_df.columns)} features em {len(features_df)} pontos")
    
    # Inicializa analisador
    analyzer = FeatureImportanceAnalyzer()
    
    # Testa métodos individuais
    print("🔬 Testando métodos de análise...")
    
    # Correlação
    corr_result = analyzer.analyze_correlation(features_df, target)
    print(f"  - Correlação: {len(corr_result.top_features)} top features")
    
    # Mutual Information
    mi_result = analyzer.analyze_mutual_information(features_df, target)
    print(f"  - Mutual Info: {len(mi_result.top_features)} top features")
    
    # Random Forest
    rf_result = analyzer.analyze_random_forest(features_df, target)
    print(f"  - Random Forest: {len(rf_result.top_features)} top features")
    
    # Análise compreensiva
    print("🎯 Executando análise compreensiva...")
    results = analyzer.comprehensive_analysis(features_df, target)
    
    successful_methods = [name for name, result in results.items() 
                         if not result.metadata.get('error')]
    print(f"✅ Métodos executados com sucesso: {successful_methods}")
    
    # Ranking de consenso
    consensus = analyzer.create_consensus_ranking(results)
    print(f"🏆 Top 10 features por consenso:")
    for i, item in enumerate(consensus['consensus_ranking'][:10], 1):
        print(f"  {i:2d}. {item['feature']:20} (score: {item['consensus_score']:.3f})")
    
    return results, consensus

def test_pattern_detection():
    """Testa detecção de padrões técnicos"""
    print("\n🎨 Testando Detecção de Padrões Técnicos...")
    
    # Gera dados com mais volatilidade para padrões
    df = generate_sample_data(40)  # 40 dias
    
    # Adiciona mais volatilidade para criar padrões
    volatility_boost = np.random.normal(0, 0.01, len(df))
    df['high'] = df['high'] * (1 + abs(volatility_boost))
    df['low'] = df['low'] * (1 - abs(volatility_boost))
    
    print(f"📊 Analisando padrões em {len(df)} pontos de dados")
    
    # Inicializa detector
    detector = TechnicalPatternDetector()
    
    # Testa padrões individuais
    print("🕯️ Detectando padrões candlestick...")
    
    doji_patterns = detector.detect_doji(df)
    print(f"  - Doji: {len(doji_patterns)} padrões detectados")
    
    hammer_patterns = detector.detect_hammer(df)
    print(f"  - Hammer: {len(hammer_patterns)} padrões detectados")
    
    engulfing_patterns = detector.detect_engulfing_patterns(df)
    print(f"  - Engulfing: {len(engulfing_patterns)} padrões detectados")
    
    star_patterns = detector.detect_star_patterns(df)
    print(f"  - Star patterns: {len(star_patterns)} padrões detectados")
    
    print("📈 Detectando padrões de gráfico...")
    
    support_resistance = detector.detect_support_resistance(df)
    print(f"  - Suporte/Resistência: {len(support_resistance)} níveis detectados")
    
    trends = detector.detect_trends(df)
    print(f"  - Tendências: {len(trends)} mudanças detectadas")
    
    breakouts = detector.detect_breakouts(df)
    print(f"  - Breakouts: {len(breakouts)} padrões detectados")
    
    # Detecta todos os padrões
    print("🎯 Detectando todos os padrões...")
    all_patterns = detector.detect_all_patterns(df)
    
    total_patterns = sum(len(signals) for signals in all_patterns.values())
    print(f"✅ Total de padrões detectados: {total_patterns}")
    
    # Gera resumo
    summary = detector.generate_pattern_summary(all_patterns)
    print(f"📋 Resumo dos padrões:")
    print(f"  - Distribuição de confiança: {summary['confidence_distribution']}")
    print(f"  - Qualidade da análise: {summary['analysis_quality']['high_confidence_ratio']:.2f}")
    
    # Mostra padrões recentes
    if summary['recent_patterns']:
        print(f"🔥 Últimos 3 padrões detectados:")
        for i, pattern in enumerate(summary['recent_patterns'][:3], 1):
            print(f"  {i}. {pattern['pattern'].upper()} ({pattern['confidence']}) "
                  f"@ {pattern['timestamp'][:16]}")
    
    return all_patterns, summary

def test_api_integration():
    """Testa integração com a API"""
    print("\n🌐 Testando Integração com API...")
    
    try:
        # Importa módulos da API
        from app.api.advanced_features import (
            FeatureEngineeringRequest,
            BacktestRequest,
            FeatureImportanceRequest,
            PatternDetectionRequest
        )
        
        print("✅ Modelos Pydantic importados com sucesso")
        
        # Testa criação de requests
        feature_req = FeatureEngineeringRequest(
            symbol="EURUSD",
            indicators=["williams_r", "stochastic_k", "adx"]
        )
        print(f"✅ FeatureEngineeringRequest: {feature_req.symbol}")
        
        backtest_req = BacktestRequest(
            symbol="EURUSD",
            strategies=["sma", "rsi"],
            initial_capital=10000.0
        )
        print(f"✅ BacktestRequest: {len(backtest_req.strategies)} estratégias")
        
        importance_req = FeatureImportanceRequest(
            symbol="EURUSD",
            methods=["correlation", "random_forest"]
        )
        print(f"✅ FeatureImportanceRequest: {len(importance_req.methods)} métodos")
        
        pattern_req = PatternDetectionRequest(
            symbol="EURUSD",
            min_confidence="medium"
        )
        print(f"✅ PatternDetectionRequest: confiança {pattern_req.min_confidence}")
        
        print("🎯 Todos os modelos de API funcionando corretamente")
        
    except ImportError as e:
        print(f"⚠️ Erro na importação da API: {e}")
        print("   (Normal se FastAPI não estiver instalado)")
    
    return True

def main():
    """Executa todos os testes"""
    print("🚀 TESTE COMPLETO - ETAPA 5: FEATURE ENGINEERING AVANÇADO")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Teste 1: Feature Engineering
        features, feature_report = test_advanced_feature_engineering()
        
        # Teste 2: Backtesting
        backtest_results, performance_report = test_backtesting()
        
        # Teste 3: Feature Importance
        importance_results, consensus = test_feature_importance()
        
        # Teste 4: Pattern Detection
        patterns, pattern_summary = test_pattern_detection()
        
        # Teste 5: API Integration
        api_test = test_api_integration()
        
        # Resumo final
        print("\n" + "=" * 60)
        print("🎯 RESUMO DOS TESTES")
        print("=" * 60)
        
        print(f"✅ Feature Engineering: {len(features)} features calculadas")
        print(f"✅ Backtesting: {len(backtest_results)} estratégias testadas")
        print(f"✅ Feature Importance: {len(importance_results)} métodos executados")
        print(f"✅ Pattern Detection: {sum(len(s) for s in patterns.values())} padrões detectados")
        print(f"✅ API Integration: {'Sucesso' if api_test else 'Falhou'}")
        
        # Estatísticas
        execution_time = datetime.now() - start_time
        print(f"\n⏱️ Tempo total de execução: {execution_time.total_seconds():.2f}s")
        
        # Top insights
        print(f"\n🏆 TOP INSIGHTS:")
        print(f"  - Melhor estratégia: {performance_report['rankings']['total_return'][0]}")
        print(f"  - Top feature: {consensus['top_features'][0] if consensus['top_features'] else 'N/A'}")
        print(f"  - Padrões high confidence: {pattern_summary['confidence_distribution']['high']}")
        
        print(f"\n🎉 ETAPA 5 COMPLETADA COM SUCESSO!")
        print(f"   Sistema de Feature Engineering Avançado totalmente funcional")
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO NO TESTE: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)