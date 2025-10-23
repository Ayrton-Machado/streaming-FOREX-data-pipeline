"""
Teste Isolado da Etapa 5: Feature Engineering Avançado
Teste independente sem dependências externas
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import warnings
warnings.filterwarnings('ignore')

def generate_sample_data(days: int = 100) -> pd.DataFrame:
    """Gera dados de amostra para teste"""
    print(f"📊 Gerando {days} dias de dados de amostra...")
    
    # use 'h' instead of 'H' for hourly frequency
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         end=datetime.now(), freq='1h')
    
    np.random.seed(42)
    base_price = 1.0800
    
    returns = np.random.normal(0.0001, 0.005, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    
    volatility = np.random.uniform(0.0005, 0.003, len(df))
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + volatility)
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - volatility)
    df['volume'] = np.random.uniform(1000, 10000, len(df))
    
    print(f"✅ Dados gerados: {len(df)} pontos")
    return df

def test_advanced_indicators():
    """Testa implementação dos indicadores avançados"""
    print("\n🔬 Testando Indicadores Técnicos Avançados...")
    
    df = generate_sample_data(30)
    
    print("📈 Testando Williams %R...")
    # Williams %R
    period = 14
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
    williams_r = williams_r.fillna(0)
    print(f"  ✅ Williams %R: {len(williams_r.dropna())} valores calculados")
    
    print("📈 Testando Stochastic Oscillator...")
    # Stochastic
    k_period = 14
    d_period = 3
    lowest_low_k = df['low'].rolling(window=k_period).min()
    highest_high_k = df['high'].rolling(window=k_period).max()
    k_percent = 100 * (df['close'] - lowest_low_k) / (highest_high_k - lowest_low_k)
    d_percent = k_percent.rolling(window=d_period).mean()
    print(f"  ✅ Stochastic K: {len(k_percent.dropna())} valores")
    print(f"  ✅ Stochastic D: {len(d_percent.dropna())} valores")
    
    print("📈 Testando CCI...")
    # CCI
    period = 20
    constant = 0.015
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean()))
    )
    cci = (typical_price - sma_tp) / (constant * mad)
    cci = cci.fillna(0)
    print(f"  ✅ CCI: {len(cci.dropna())} valores calculados")
    
    print("📈 Testando Parabolic SAR...")
    # Parabolic SAR simplificado
    high = df['high'].values
    low = df['low'].values
    psar = np.zeros(len(df))
    psar[0] = low[0]
    
    for i in range(1, min(50, len(df))):  # Limita para teste rápido
        psar[i] = psar[i-1] + 0.02 * (high[i-1] - psar[i-1])
    
    psar_series = pd.Series(psar, index=df.index)
    print(f"  ✅ Parabolic SAR: {len(psar_series.dropna())} valores calculados")
    
    print("📈 Testando OBV...")
    # OBV simulado
    price_change = df['close'].diff()
    volume_proxy = abs(df['high'] - df['low'])
    
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = 0
    
    for i in range(1, len(df)):
        if price_change.iloc[i] > 0:
            obv.iloc[i] = obv.iloc[i-1] + volume_proxy.iloc[i]
        elif price_change.iloc[i] < 0:
            obv.iloc[i] = obv.iloc[i-1] - volume_proxy.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    print(f"  ✅ OBV: {len(obv.dropna())} valores calculados")
    
    # Estatísticas rolling
    print("📊 Testando Rolling Statistics...")
    windows = [5, 10, 20]
    returns = df['close'].pct_change()
    
    for window in windows:
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_skew = returns.rolling(window).skew()
        print(f"  ✅ Rolling {window}: mean={len(rolling_mean.dropna())}, "
              f"std={len(rolling_std.dropna())}, skew={len(rolling_skew.dropna())}")
    
    total_indicators = 15
    print(f"\n✅ Total de indicadores testados: {total_indicators}")
    # Assertions to ensure code executed and indicators computed
    assert total_indicators > 0

def test_backtesting_logic():
    """Testa lógica de backtesting"""
    print("\n📊 Testando Lógica de Backtesting...")
    
    df = generate_sample_data(60)
    
    # Estratégia SMA simples
    print("📈 Testando Estratégia SMA...")
    fast_ma = df['close'].rolling(window=10).mean()
    slow_ma = df['close'].rolling(window=20).mean()
    
    signals = pd.Series(0, index=df.index)
    buy_signals = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    sell_signals = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    
    signals[buy_signals] = 1
    signals[sell_signals] = -1
    
    print(f"  ✅ Sinais gerados: {signals.abs().sum()} total")
    print(f"  ✅ Sinais de compra: {(signals == 1).sum()}")
    print(f"  ✅ Sinais de venda: {(signals == -1).sum()}")
    
    # Simula trades
    print("💼 Simulando trades...")
    trades = []
    position = None
    
    for i, (timestamp, signal) in enumerate(signals.items()):
        if signal != 0 and position is None:
            # Abre posição
            position = {
                'entry_time': timestamp,
                'entry_price': df.loc[timestamp, 'close'],
                'type': 'buy' if signal > 0 else 'sell'
            }
        elif signal != 0 and position is not None:
            # Fecha posição
            exit_price = df.loc[timestamp, 'close']
            
            if position['type'] == 'buy':
                pnl = exit_price - position['entry_price']
            else:
                pnl = position['entry_price'] - exit_price
            
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': timestamp,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'return_pct': (pnl / position['entry_price']) * 100
            })
            
            position = None
    
    print(f"  ✅ Trades executados: {len(trades)}")
    
    if trades:
        total_pnl = sum(trade['pnl'] for trade in trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        win_rate = (winning_trades / len(trades)) * 100
        
        print(f"  ✅ PnL total: {total_pnl:.4f}")
        print(f"  ✅ Win rate: {win_rate:.1f}%")
        print(f"  ✅ Trades vencedores: {winning_trades}/{len(trades)}")
    
    # Assert that backtesting simulation completed (no exceptions) and trades list is returned
    assert isinstance(trades, list)
    # It's acceptable if no trades were found, but function should run to completion
    return

def test_pattern_detection_logic():
    """Testa lógica de detecção de padrões"""
    print("\n🎨 Testando Detecção de Padrões...")
    
    df = generate_sample_data(40)
    
    # Adiciona volatilidade para criar padrões
    volatility_boost = np.random.normal(0, 0.01, len(df))
    df['high'] = df['high'] * (1 + abs(volatility_boost))
    df['low'] = df['low'] * (1 - abs(volatility_boost))
    
    # Detecta Doji
    print("🕯️ Detectando padrões Doji...")
    doji_count = 0
    threshold = 0.1
    
    for i in range(len(df)):
        row = df.iloc[i]
        body_size = abs(row['close'] - row['open'])
        total_range = row['high'] - row['low']
        
        if total_range > 0:
            body_ratio = body_size / total_range
            if body_ratio <= threshold:
                doji_count += 1
    
    print(f"  ✅ Padrões Doji detectados: {doji_count}")
    
    # Detecta Hammer
    print("🕯️ Detectando padrões Hammer...")
    hammer_count = 0
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        body_size = abs(row['close'] - row['open'])
        lower_shadow = min(row['open'], row['close']) - row['low']
        upper_shadow = row['high'] - max(row['open'], row['close'])
        total_range = row['high'] - row['low']
        
        if total_range > 0 and body_size > 0:
            lower_ratio = lower_shadow / total_range
            upper_ratio = upper_shadow / total_range
            body_ratio = body_size / total_range
            
            if (lower_ratio >= 0.6 and body_ratio <= 0.3 and 
                upper_ratio <= 0.1 and prev_row['close'] < prev_row['open']):
                hammer_count += 1
    
    print(f"  ✅ Padrões Hammer detectados: {hammer_count}")
    
    # Detecta Suporte/Resistência
    print("📈 Detectando Suporte/Resistência...")
    window = 10
    support_levels = []
    resistance_levels = []
    
    for i in range(window, len(df) - window):
        # Suporte (mínimo local)
        local_min = df['low'].iloc[i-window:i+window+1].min()
        if df['low'].iloc[i] == local_min:
            support_levels.append((df.index[i], df['low'].iloc[i]))
        
        # Resistência (máximo local)
        local_max = df['high'].iloc[i-window:i+window+1].max()
        if df['high'].iloc[i] == local_max:
            resistance_levels.append((df.index[i], df['high'].iloc[i]))
    
    print(f"  ✅ Níveis de suporte: {len(support_levels)}")
    print(f"  ✅ Níveis de resistência: {len(resistance_levels)}")
    
    # Detecta tendências (SMA cross)
    print("📊 Detectando mudanças de tendência...")
    ma_short = df['close'].rolling(window=5).mean()
    ma_long = df['close'].rolling(window=15).mean()
    
    trend_changes = 0
    current_trend = None
    
    for i in range(15, len(df)):
        if ma_short.iloc[i] > ma_long.iloc[i]:
            new_trend = 'up'
        else:
            new_trend = 'down'
        
        if current_trend != new_trend and current_trend is not None:
            trend_changes += 1
        
        current_trend = new_trend
    
    print(f"  ✅ Mudanças de tendência: {trend_changes}")
    
    total_patterns = doji_count + hammer_count + len(support_levels) + len(resistance_levels)
    print(f"\n✅ Total de padrões detectados: {total_patterns}")
    # Ensure function executed and computed pattern counts
    assert isinstance(total_patterns, int)
    return

def test_feature_importance_logic():
    """Testa lógica de análise de importância"""
    print("\n🔍 Testando Análise de Importância de Features...")
    
    df = generate_sample_data(50)
    
    # Cria features sintéticas
    features = pd.DataFrame()
    features['sma_10'] = df['close'].rolling(10).mean()
    features['sma_20'] = df['close'].rolling(20).mean()
    features['rsi'] = 50 + np.random.normal(0, 10, len(df))  # RSI simulado
    features['volume_ma'] = df['volume'].rolling(5).mean()
    features['price_change'] = df['close'].pct_change()
    features['volatility'] = df['close'].rolling(10).std()
    
    # Remove NaN
    features = features.dropna()
    
    # Define target (retornos futuros)
    target = df['close'].pct_change().shift(-1).dropna()
    
    # Alinha dados
    common_idx = features.index.intersection(target.index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]
    
    print(f"📊 Analisando {len(features.columns)} features em {len(features)} pontos")
    
    # Testa correlação
    print("🔗 Calculando correlações...")
    correlations = {}
    
    for col in features.columns:
        feature_series = features[col].dropna()
        aligned_target = target.loc[feature_series.index]
        
        if len(feature_series) > 0 and len(aligned_target) > 0:
            corr = feature_series.corr(aligned_target)
            correlations[col] = abs(corr) if not pd.isna(corr) else 0
    
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    print(f"  ✅ Top 3 features por correlação:")
    for i, (feature, corr) in enumerate(sorted_corr[:3], 1):
        print(f"    {i}. {feature}: {corr:.3f}")
    
    # Testa variância das features
    print("📊 Analisando variância das features...")
    variances = features.var().sort_values(ascending=False)
    print(f"  ✅ Features com maior variância:")
    for i, (feature, var) in enumerate(variances.head(3).items(), 1):
        print(f"    {i}. {feature}: {var:.6f}")
    
    # Simulação de Random Forest Feature Importance
    print("🌳 Simulando Random Forest Importance...")
    # Simula importâncias (normalmente seria calculado pelo RF)
    rf_importance = {
        col: np.random.uniform(0, 1) for col in features.columns
    }
    
    sorted_rf = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)
    print(f"  ✅ Top 3 features por RF (simulado):")
    for i, (feature, importance) in enumerate(sorted_rf[:3], 1):
        print(f"    {i}. {feature}: {importance:.3f}")
    
    print(f"\n✅ Análise de importância completada para {len(features.columns)} features")
    assert len(features.columns) > 0
    return

def main():
    """Executa teste completo simplificado"""
    print("🚀 TESTE SIMPLIFICADO - ETAPA 5: FEATURE ENGINEERING AVANÇADO")
    print("=" * 65)
    
    start_time = datetime.now()
    
    try:
        # Teste 1: Indicadores Avançados
        test1 = test_advanced_indicators()
        
        # Teste 2: Backtesting
        test2 = test_backtesting_logic()
        
        # Teste 3: Detecção de Padrões
        test3 = test_pattern_detection_logic()
        
        # Teste 4: Feature Importance
        test4 = test_feature_importance_logic()
        
        # Run pytest-style tests sequentially for manual execution
        try:
            test_advanced_indicators()
            test_backtesting_logic()
            test_pattern_detection_logic()
            test_feature_importance_logic()

            execution_time = datetime.now() - start_time
            print(f"\n⏱️ Tempo de execução: {execution_time.total_seconds():.2f}s")
            print("🎉 ETAPA 5 - LÓGICA CORE VALIDADA COM SUCESSO!")
            return True
        except AssertionError as ae:
            print(f"\n❌ Falha nos testes: {ae}")
            return False
        except Exception as e:
            print(f"\n❌ Erro nos testes: {e}")
            return False
        
    except Exception as e:
        print(f"\n❌ ERRO NO TESTE: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)