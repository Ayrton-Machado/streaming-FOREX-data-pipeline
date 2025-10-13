"""
Backtesting System for FOREX Trading Strategies
Sistema de validação e análise de performance para indicadores técnicos
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

class TradeType(Enum):
    """Tipos de trade"""
    BUY = "buy"
    SELL = "sell"

class SignalType(Enum):
    """Tipos de sinal"""
    ENTRY = "entry"
    EXIT = "exit"
    HOLD = "hold"

@dataclass
class Trade:
    """Representa um trade individual"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    trade_type: TradeType
    size: float = 1.0
    commission: float = 0.0
    swap: float = 0.0
    
    @property
    def is_open(self) -> bool:
        return self.exit_time is None
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.exit_time:
            return self.exit_time - self.entry_time
        return None
    
    @property
    def pnl(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        
        if self.trade_type == TradeType.BUY:
            gross_pnl = (self.exit_price - self.entry_price) * self.size
        else:
            gross_pnl = (self.entry_price - self.exit_price) * self.size
        
        return gross_pnl - self.commission - self.swap
    
    @property
    def return_pct(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        
        if self.trade_type == TradeType.BUY:
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.exit_price) / self.entry_price) * 100

@dataclass
class BacktestResult:
    """Resultado do backtesting"""
    trades: List[Trade] = field(default_factory=list)
    initial_capital: float = 10000.0
    final_capital: float = 0.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    @property
    def total_trades(self) -> int:
        return len([t for t in self.trades if not t.is_open])
    
    @property
    def winning_trades(self) -> int:
        return len([t for t in self.trades if not t.is_open and t.pnl > 0])
    
    @property
    def losing_trades(self) -> int:
        return len([t for t in self.trades if not t.is_open and t.pnl < 0])
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades if t.pnl is not None)
    
    @property
    def total_return(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.final_capital - self.initial_capital) / self.initial_capital * 100
    
    @property
    def max_drawdown(self) -> float:
        """Calcula máximo drawdown"""
        if not self.trades:
            return 0.0
        
        capital_curve = [self.initial_capital]
        running_capital = self.initial_capital
        
        for trade in self.trades:
            if trade.pnl is not None:
                running_capital += trade.pnl
                capital_curve.append(running_capital)
        
        peak = capital_curve[0]
        max_dd = 0.0
        
        for capital in capital_curve:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    @property
    def sharpe_ratio(self) -> float:
        """Calcula Sharpe Ratio (simplificado)"""
        if not self.trades:
            return 0.0
        
        returns = [t.return_pct for t in self.trades if t.return_pct is not None]
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Assumindo risk-free rate de 2% anual
        risk_free_rate = 2.0
        return (mean_return - risk_free_rate) / std_return
    
    @property
    def profit_factor(self) -> float:
        """Calcula Profit Factor"""
        gross_profit = sum(t.pnl for t in self.trades if t.pnl is not None and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl is not None and t.pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss

class TradingStrategy:
    """Classe base para estratégias de trading"""
    
    def __init__(self, name: str):
        self.name = name
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Gera sinais de trading baseado nos dados
        Retorna: Series com valores 1 (buy), -1 (sell), 0 (hold)
        """
        raise NotImplementedError
    
    def __str__(self):
        return f"Strategy: {self.name}"

class SimpleMovingAverageStrategy(TradingStrategy):
    """Estratégia baseada em médias móveis simples"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        super().__init__(f"SMA_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Gera sinais baseados no cruzamento de médias móveis"""
        fast_ma = df['close'].rolling(window=self.fast_period).mean()
        slow_ma = df['close'].rolling(window=self.slow_period).mean()
        
        signals = pd.Series(0, index=df.index)
        
        # Sinal de compra quando MA rápida cruza acima da MA lenta
        buy_signals = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        
        # Sinal de venda quando MA rápida cruza abaixo da MA lenta
        sell_signals = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals

class RSIStrategy(TradingStrategy):
    """Estratégia baseada no RSI"""
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(f"RSI_{period}_{oversold}_{overbought}")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calcula RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.period).mean()
        loss = (-delta).where(delta < 0, 0).rolling(window=self.period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Gera sinais baseados em RSI oversold/overbought"""
        rsi = self.calculate_rsi(df['close'])
        signals = pd.Series(0, index=df.index)
        
        # Compra quando RSI sai de oversold
        buy_signals = (rsi > self.oversold) & (rsi.shift(1) <= self.oversold)
        
        # Venda quando RSI sai de overbought
        sell_signals = (rsi < self.overbought) & (rsi.shift(1) >= self.overbought)
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals

class MACDStrategy(TradingStrategy):
    """Estratégia baseada no MACD"""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(f"MACD_{fast}_{slow}_{signal}")
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula MACD, Signal Line e Histogram"""
        exp1 = prices.ewm(span=self.fast).mean()
        exp2 = prices.ewm(span=self.slow).mean()
        
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=self.signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Gera sinais baseados no cruzamento MACD"""
        macd_line, signal_line, histogram = self.calculate_macd(df['close'])
        
        signals = pd.Series(0, index=df.index)
        
        # Compra quando MACD cruza acima da signal line
        buy_signals = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        
        # Venda quando MACD cruza abaixo da signal line
        sell_signals = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals

class BacktestingEngine:
    """Engine principal para backtesting"""
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.0001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.results_cache = {}
    
    def run_backtest(self, df: pd.DataFrame, strategy: TradingStrategy, 
                    max_positions: int = 1) -> BacktestResult:
        """Executa backtesting de uma estratégia"""
        
        # Gera sinais
        signals = strategy.generate_signals(df)
        
        # Inicializa resultado
        result = BacktestResult(
            initial_capital=self.initial_capital,
            start_date=df.index[0] if len(df) > 0 else None,
            end_date=df.index[-1] if len(df) > 0 else None
        )
        
        # Estado do backtesting
        current_capital = self.initial_capital
        open_trades = []
        
        for i, (timestamp, signal) in enumerate(signals.items()):
            if i >= len(df):
                break
                
            current_price = df.loc[timestamp, 'close']
            
            # Processa sinal de entrada
            if signal != 0 and len(open_trades) < max_positions:
                trade_type = TradeType.BUY if signal > 0 else TradeType.SELL
                
                # Calcula size baseado no capital disponível
                trade_size = current_capital * 0.1  # 10% do capital por trade
                commission_cost = trade_size * self.commission
                
                trade = Trade(
                    entry_time=timestamp,
                    exit_time=None,
                    entry_price=current_price,
                    exit_price=None,
                    trade_type=trade_type,
                    size=trade_size,
                    commission=commission_cost
                )
                
                open_trades.append(trade)
                current_capital -= commission_cost
            
            # Processa saídas (estratégia simples: sai no sinal oposto)
            trades_to_close = []
            for trade in open_trades:
                should_close = False
                
                # Sinal oposto
                if ((trade.trade_type == TradeType.BUY and signal < 0) or 
                    (trade.trade_type == TradeType.SELL and signal > 0)):
                    should_close = True
                
                # Stop loss simples (5% loss)
                elif trade.trade_type == TradeType.BUY and current_price <= trade.entry_price * 0.95:
                    should_close = True
                elif trade.trade_type == TradeType.SELL and current_price >= trade.entry_price * 1.05:
                    should_close = True
                
                if should_close:
                    trade.exit_time = timestamp
                    trade.exit_price = current_price
                    trade.commission += trade.size * self.commission  # Commission na saída
                    
                    # Atualiza capital
                    if trade.pnl:
                        current_capital += trade.pnl
                    
                    trades_to_close.append(trade)
            
            # Remove trades fechados da lista de abertos
            for trade in trades_to_close:
                open_trades.remove(trade)
                result.trades.append(trade)
        
        # Fecha trades restantes no final
        final_price = df['close'].iloc[-1] if len(df) > 0 else 0
        for trade in open_trades:
            trade.exit_time = df.index[-1]
            trade.exit_price = final_price
            trade.commission += trade.size * self.commission
            
            if trade.pnl:
                current_capital += trade.pnl
            
            result.trades.append(trade)
        
        result.final_capital = current_capital
        
        # Cache resultado
        cache_key = f"{strategy.name}_{hash(str(df.index))}"
        self.results_cache[cache_key] = result
        
        return result
    
    def compare_strategies(self, df: pd.DataFrame, 
                          strategies: List[TradingStrategy]) -> Dict[str, BacktestResult]:
        """Compara múltiplas estratégias"""
        results = {}
        
        for strategy in strategies:
            results[strategy.name] = self.run_backtest(df, strategy)
        
        return results
    
    def generate_performance_report(self, results: Dict[str, BacktestResult]) -> Dict[str, Any]:
        """Gera relatório de performance comparativo"""
        report = {
            'summary': {},
            'detailed_metrics': {},
            'rankings': {}
        }
        
        # Métricas resumidas
        for strategy_name, result in results.items():
            report['summary'][strategy_name] = {
                'total_return': result.total_return,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'profit_factor': result.profit_factor
            }
        
        # Métricas detalhadas
        for strategy_name, result in results.items():
            trades = [t for t in result.trades if not t.is_open]
            
            if trades:
                returns = [t.return_pct for t in trades if t.return_pct is not None]
                durations = [t.duration.total_seconds() / 3600 for t in trades if t.duration]  # em horas
                
                report['detailed_metrics'][strategy_name] = {
                    'avg_return_per_trade': np.mean(returns) if returns else 0,
                    'std_return_per_trade': np.std(returns) if returns else 0,
                    'avg_trade_duration_hours': np.mean(durations) if durations else 0,
                    'best_trade': max(returns) if returns else 0,
                    'worst_trade': min(returns) if returns else 0,
                    'consecutive_wins': self._calculate_consecutive_wins(trades),
                    'consecutive_losses': self._calculate_consecutive_losses(trades)
                }
        
        # Rankings
        ranking_metrics = ['total_return', 'win_rate', 'sharpe_ratio', 'profit_factor']
        for metric in ranking_metrics:
            sorted_strategies = sorted(
                results.items(),
                key=lambda x: getattr(x[1], metric) if hasattr(x[1], metric) else 
                              report['summary'][x[0]][metric],
                reverse=True
            )
            report['rankings'][metric] = [name for name, _ in sorted_strategies]
        
        return report
    
    def _calculate_consecutive_wins(self, trades: List[Trade]) -> int:
        """Calcula máximo de vitórias consecutivas"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade.pnl and trade.pnl > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_consecutive_losses(self, trades: List[Trade]) -> int:
        """Calcula máximo de perdas consecutivas"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade.pnl and trade.pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def export_results(self, results: Dict[str, BacktestResult], 
                      filename: str = "backtest_results.json"):
        """Exporta resultados para JSON"""
        export_data = {}
        
        for strategy_name, result in results.items():
            trades_data = []
            for trade in result.trades:
                trades_data.append({
                    'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
                    'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'trade_type': trade.trade_type.value,
                    'size': trade.size,
                    'pnl': trade.pnl,
                    'return_pct': trade.return_pct
                })
            
            export_data[strategy_name] = {
                'initial_capital': result.initial_capital,
                'final_capital': result.final_capital,
                'total_return': result.total_return,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'profit_factor': result.profit_factor,
                'trades': trades_data
            }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return filename