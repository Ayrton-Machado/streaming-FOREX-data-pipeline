"""
Endpoint para Feature Engineering Avançado
API endpoints para análise de features, backtesting e detecção de padrões
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import asyncio
import json

from app.services.features.advanced_feature_engineer import AdvancedFeatureEngineer
from app.services.analysis.backtesting_engine import (
    BacktestingEngine, 
    SimpleMovingAverageStrategy,
    RSIStrategy,
    MACDStrategy,
    BacktestResult
)
from app.services.analysis.feature_importance import (
    FeatureImportanceAnalyzer,
    ImportanceMethod
)
from app.services.analysis.pattern_detection import (
    TechnicalPatternDetector,
    PatternConfidence
)
from app.core.database import get_db_connection

router = APIRouter(prefix="/api/v1/advanced-features", tags=["Advanced Features"])

class FeatureEngineeringRequest(BaseModel):
    """Request para feature engineering"""
    symbol: str = Field(default="EURUSD", description="Par de moedas")
    start_date: Optional[datetime] = Field(default=None, description="Data inicial")
    end_date: Optional[datetime] = Field(default=None, description="Data final")
    indicators: Optional[List[str]] = Field(default=None, description="Indicadores específicos")

class BacktestRequest(BaseModel):
    """Request para backtesting"""
    symbol: str = Field(default="EURUSD", description="Par de moedas")
    start_date: Optional[datetime] = Field(default=None, description="Data inicial")
    end_date: Optional[datetime] = Field(default=None, description="Data final")
    strategies: List[str] = Field(default=["sma", "rsi", "macd"], description="Estratégias a testar")
    initial_capital: float = Field(default=10000.0, description="Capital inicial")

class FeatureImportanceRequest(BaseModel):
    """Request para análise de importância"""
    symbol: str = Field(default="EURUSD", description="Par de moedas")
    start_date: Optional[datetime] = Field(default=None, description="Data inicial")
    end_date: Optional[datetime] = Field(default=None, description="Data final")
    methods: List[str] = Field(default=["correlation", "mutual_info", "random_forest"], description="Métodos de análise")
    target_column: str = Field(default="close", description="Coluna target")

class PatternDetectionRequest(BaseModel):
    """Request para detecção de padrões"""
    symbol: str = Field(default="EURUSD", description="Par de moedas")
    start_date: Optional[datetime] = Field(default=None, description="Data inicial")
    end_date: Optional[datetime] = Field(default=None, description="Data final")
    min_confidence: str = Field(default="medium", description="Confiança mínima")
    pattern_types: Optional[List[str]] = Field(default=None, description="Tipos de padrão específicos")

async def get_market_data(symbol: str, start_date: Optional[datetime], 
                         end_date: Optional[datetime]) -> pd.DataFrame:
    """Obtém dados de mercado do banco"""
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=30)
    
    try:
        async with get_db_connection() as conn:
            query = """
            SELECT timestamp, open, high, low, close, volume
            FROM forex_data 
            WHERE symbol = %s 
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            """
            
            df = pd.read_sql(query, conn, params=(symbol, start_date, end_date))
            
            if df.empty:
                raise HTTPException(status_code=404, detail="Dados não encontrados")
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            return df
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao obter dados: {str(e)}")

@router.post("/feature-engineering")
async def calculate_advanced_features(request: FeatureEngineeringRequest):
    """
    Calcula features avançadas para os dados
    """
    try:
        # Obtém dados
        df = await get_market_data(request.symbol, request.start_date, request.end_date)
        
        # Inicializa feature engineer
        engineer = AdvancedFeatureEngineer()
        
        # Calcula features
        if request.indicators:
            # Calcula indicadores específicos
            features = {}
            all_features = engineer.calculate_all_advanced_indicators(df)
            
            for indicator in request.indicators:
                if indicator in all_features:
                    features[indicator] = all_features[indicator].tolist()
        else:
            # Calcula todas as features
            features = engineer.calculate_all_advanced_indicators(df)
            # Converte Series para listas para JSON
            features = {name: series.tolist() for name, series in features.items()}
        
        # Gera relatório
        report = engineer.generate_feature_report(df)
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "data_points": len(df),
            "period": {
                "start": df.index[0].isoformat(),
                "end": df.index[-1].isoformat()
            },
            "features": features,
            "report": report,
            "metadata": {
                "total_features": len(features),
                "calculation_time": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no feature engineering: {str(e)}")

@router.post("/backtesting")
async def run_backtesting(request: BacktestRequest):
    """
    Executa backtesting de estratégias
    """
    try:
        # Obtém dados
        df = await get_market_data(request.symbol, request.start_date, request.end_date)
        
        # Inicializa engine de backtesting
        engine = BacktestingEngine(initial_capital=request.initial_capital)
        
        # Configura estratégias
        strategies = []
        for strategy_name in request.strategies:
            if strategy_name == "sma":
                strategies.append(SimpleMovingAverageStrategy(10, 20))
            elif strategy_name == "rsi":
                strategies.append(RSIStrategy(14, 30, 70))
            elif strategy_name == "macd":
                strategies.append(MACDStrategy(12, 26, 9))
        
        if not strategies:
            raise HTTPException(status_code=400, detail="Nenhuma estratégia válida especificada")
        
        # Executa backtesting
        results = engine.compare_strategies(df, strategies)
        
        # Gera relatório de performance
        performance_report = engine.generate_performance_report(results)
        
        # Converte resultados para formato JSON
        json_results = {}
        for strategy_name, result in results.items():
            json_results[strategy_name] = {
                "initial_capital": result.initial_capital,
                "final_capital": result.final_capital,
                "total_return": result.total_return,
                "total_trades": result.total_trades,
                "winning_trades": result.winning_trades,
                "losing_trades": result.losing_trades,
                "win_rate": result.win_rate,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "profit_factor": result.profit_factor,
                "trades": [
                    {
                        "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
                        "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
                        "entry_price": trade.entry_price,
                        "exit_price": trade.exit_price,
                        "trade_type": trade.trade_type.value,
                        "pnl": trade.pnl,
                        "return_pct": trade.return_pct
                    }
                    for trade in result.trades[:10]  # Limita a 10 trades para response
                ]
            }
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "backtesting_period": {
                "start": df.index[0].isoformat(),
                "end": df.index[-1].isoformat(),
                "data_points": len(df)
            },
            "strategies_tested": len(strategies),
            "results": json_results,
            "performance_report": performance_report,
            "metadata": {
                "execution_time": datetime.now().isoformat(),
                "initial_capital": request.initial_capital
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no backtesting: {str(e)}")

@router.post("/feature-importance")
async def analyze_feature_importance(request: FeatureImportanceRequest):
    """
    Analisa importância de features
    """
    try:
        # Obtém dados
        df = await get_market_data(request.symbol, request.start_date, request.end_date)
        
        # Calcula features
        engineer = AdvancedFeatureEngineer()
        features_dict = engineer.calculate_all_advanced_indicators(df)
        features_df = pd.DataFrame(features_dict, index=df.index)
        
        # Define target (retornos futuros)
        target = df[request.target_column].pct_change().shift(-1).dropna()
        
        # Alinha features com target
        common_idx = features_df.index.intersection(target.index)
        features_df = features_df.loc[common_idx]
        target = target.loc[common_idx]
        
        # Inicializa analisador
        analyzer = FeatureImportanceAnalyzer()
        
        # Mapeia métodos
        method_map = {
            "correlation": ImportanceMethod.CORRELATION,
            "mutual_info": ImportanceMethod.MUTUAL_INFO,
            "random_forest": ImportanceMethod.RANDOM_FOREST,
            "lasso": ImportanceMethod.LASSO,
            "f_statistic": ImportanceMethod.F_STATISTIC,
            "rfe": ImportanceMethod.RFE
        }
        
        methods = [method_map[m] for m in request.methods if m in method_map]
        
        # Executa análise
        results = analyzer.comprehensive_analysis(features_df, target, methods)
        
        # Cria ranking de consenso
        consensus = analyzer.create_consensus_ranking(results)
        
        # Converte resultados para JSON
        json_results = {}
        for method_name, result in results.items():
            if not result.metadata.get('error'):
                json_results[method_name] = {
                    "top_features": result.top_features,
                    "total_analyzed": len(result.scores),
                    "analysis_params": result.analysis_params,
                    "metadata": result.metadata,
                    "scores": [
                        {
                            "feature": score.name,
                            "score": score.score,
                            "rank": score.rank
                        }
                        for score in result.scores[:20]  # Top 20
                    ]
                }
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "analysis_period": {
                "start": df.index[0].isoformat(),
                "end": df.index[-1].isoformat(),
                "data_points": len(df)
            },
            "target_column": request.target_column,
            "methods_executed": len(json_results),
            "results_by_method": json_results,
            "consensus_ranking": consensus,
            "recommendations": analyzer._generate_recommendations(consensus, {}),
            "metadata": {
                "total_features": len(features_df.columns),
                "analysis_time": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na análise de importância: {str(e)}")

@router.post("/pattern-detection")
async def detect_patterns(request: PatternDetectionRequest):
    """
    Detecta padrões técnicos nos dados
    """
    try:
        # Obtém dados
        df = await get_market_data(request.symbol, request.start_date, request.end_date)
        
        # Inicializa detector
        detector = TechnicalPatternDetector()
        
        # Mapeia confiança mínima
        confidence_map = {
            "low": PatternConfidence.LOW,
            "medium": PatternConfidence.MEDIUM,
            "high": PatternConfidence.HIGH,
            "very_high": PatternConfidence.VERY_HIGH
        }
        
        min_confidence = confidence_map.get(request.min_confidence, PatternConfidence.MEDIUM)
        
        # Detecta padrões
        patterns = detector.detect_all_patterns(df)
        
        # Filtra por confiança
        filtered_patterns = detector.filter_patterns_by_confidence(patterns, min_confidence)
        
        # Gera resumo
        summary = detector.generate_pattern_summary(filtered_patterns)
        
        # Converte para JSON
        json_patterns = {}
        for pattern_type, signals in filtered_patterns.items():
            json_patterns[pattern_type] = [
                {
                    "pattern": signal.pattern_type.value,
                    "timestamp": signal.timestamp.isoformat(),
                    "confidence": signal.confidence.value,
                    "price_level": signal.price_level,
                    "signal_strength": signal.signal_strength,
                    "metadata": signal.metadata
                }
                for signal in signals
            ]
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "analysis_period": {
                "start": df.index[0].isoformat(),
                "end": df.index[-1].isoformat(),
                "data_points": len(df)
            },
            "min_confidence": request.min_confidence,
            "patterns_detected": json_patterns,
            "summary": summary,
            "metadata": {
                "total_patterns": summary["total_patterns"],
                "detection_time": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na detecção de padrões: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check do módulo de features avançadas"""
    try:
        # Testa conexão com banco
        async with get_db_connection() as conn:
            result = await conn.fetch("SELECT 1")
        
        return {
            "status": "healthy",
            "module": "advanced_features",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "services": {
                "feature_engineering": "available",
                "backtesting": "available", 
                "feature_importance": "available",
                "pattern_detection": "available"
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "module": "advanced_features",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/supported-indicators")
async def get_supported_indicators():
    """Lista indicadores técnicos suportados"""
    engineer = AdvancedFeatureEngineer()
    
    indicators_by_type = {
        "momentum": [
            "williams_r", "stochastic_k", "stochastic_d", "cci", 
            "roc", "momentum"
        ],
        "trend": [
            "adx", "aroon_up", "aroon_down", "aroon_oscillator",
            "parabolic_sar", "supertrend"
        ],
        "volatility": [
            "keltner_upper", "keltner_middle", "keltner_lower",
            "keltner_squeeze", "donchian_upper", "donchian_lower",
            "donchian_middle", "true_range"
        ],
        "volume": [
            "obv", "ad_line", "cmf"
        ],
        "oscillator": [
            "ultimate_oscillator", "trix"
        ],
        "statistical": [
            "rolling_mean", "rolling_std", "rolling_skew", "rolling_kurt",
            "rolling_max", "rolling_min", "rolling_range",
            "rolling_q75", "rolling_q25", "rolling_median"
        ],
        "correlation": [
            "high_low_corr", "open_close_corr", "high_volume_corr",
            "returns_high_corr", "returns_low_corr"
        ]
    }
    
    return {
        "status": "success",
        "total_indicators": sum(len(indicators) for indicators in indicators_by_type.values()),
        "indicators_by_type": indicators_by_type,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0"
        }
    }