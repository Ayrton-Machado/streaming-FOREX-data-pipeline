"""
Data Normalizer - Normalização avançada para ML
Etapa 4: MinMax, Z-score, Robust scaling e transformações
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging

from app.domain.schemas import ValidatedOHLCV, MLReadyCandle
from app.domain.enums import NormalizationMethod
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class NormalizationParams:
    """Parâmetros de normalização para um dataset"""
    method: NormalizationMethod
    feature_name: str
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    median: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class DataNormalizer:
    """
    Serviço de normalização avançada para dados FOREX
    
    Filosofia SLC:
    - Simple: Interface clara, métodos auto-explicativos
    - Lovable: Preserva parâmetros para reverter normalização
    - Complete: Múltiplos métodos, handling de outliers, validação
    """
    
    def __init__(self):
        self.normalization_params: Dict[str, NormalizationParams] = {}
        logger.info("DataNormalizer inicializado")
    
    def normalize_candles(
        self,
        candles: List[ValidatedOHLCV],
        method: NormalizationMethod = NormalizationMethod.MINMAX,
        features: Optional[List[str]] = None,
        fit_on_data: bool = True
    ) -> Tuple[List[MLReadyCandle], Dict[str, NormalizationParams]]:
        """
        Normaliza lista de velas validadas para ML
        
        Args:
            candles: Lista de velas validadas
            method: Método de normalização
            features: Features específicas para normalizar (None = todas)
            fit_on_data: Se deve calcular parâmetros nos dados ou usar existentes
            
        Returns:
            Tuple[List[MLReadyCandle], Dict[str, NormalizationParams]]
        """
        if not candles:
            logger.warning("Lista de velas vazia")
            return [], {}
        
        # Converte para DataFrame para processamento
        df = self._candles_to_dataframe(candles)
        
        # Define features a normalizar
        if features is None:
            features = ['open', 'high', 'low', 'close', 'volume']
        
        # Normaliza cada feature
        normalized_df = df.copy()
        used_params = {}
        
        for feature in features:
            if feature in df.columns:
                normalized_values, params = self._normalize_feature(
                    df[feature].values, 
                    feature, 
                    method, 
                    fit_on_data
                )
                normalized_df[f"{feature}_normalized"] = normalized_values
                used_params[feature] = params
        
        # Converte de volta para MLReadyCandle
        ml_candles = self._dataframe_to_ml_candles(normalized_df, candles)
        
        logger.info(f"✅ {len(ml_candles)} velas normalizadas usando {method.value}")
        return ml_candles, used_params
    
    def _normalize_feature(
        self,
        values: np.ndarray,
        feature_name: str,
        method: NormalizationMethod,
        fit_on_data: bool
    ) -> Tuple[np.ndarray, NormalizationParams]:
        """
        Normaliza uma feature específica
        """
        if fit_on_data:
            # Calcula parâmetros nos dados atuais
            if method == NormalizationMethod.MINMAX:
                params = self._fit_minmax(values, feature_name)
            elif method == NormalizationMethod.ZSCORE:
                params = self._fit_zscore(values, feature_name)
            elif method == NormalizationMethod.ROBUST:
                params = self._fit_robust(values, feature_name)
            else:
                raise ValueError(f"Método de normalização não suportado: {method}")
            
            # Armazena parâmetros para uso futuro
            self.normalization_params[feature_name] = params
        else:
            # Usa parâmetros existentes
            if feature_name not in self.normalization_params:
                raise ValueError(f"Parâmetros não encontrados para feature: {feature_name}")
            params = self.normalization_params[feature_name]
        
        # Aplica normalização
        normalized_values = self._apply_normalization(values, params)
        
        return normalized_values, params
    
    def _fit_minmax(self, values: np.ndarray, feature_name: str) -> NormalizationParams:
        """Calcula parâmetros para MinMax scaling"""
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        
        return NormalizationParams(
            method=NormalizationMethod.MINMAX,
            feature_name=feature_name,
            min_val=min_val,
            max_val=max_val
        )
    
    def _fit_zscore(self, values: np.ndarray, feature_name: str) -> NormalizationParams:
        """Calcula parâmetros para Z-score normalization"""
        mean = float(np.mean(values))
        std = float(np.std(values))
        
        return NormalizationParams(
            method=NormalizationMethod.ZSCORE,
            feature_name=feature_name,
            mean=mean,
            std=std
        )
    
    def _fit_robust(self, values: np.ndarray, feature_name: str) -> NormalizationParams:
        """Calcula parâmetros para Robust scaling"""
        median = float(np.median(values))
        q25 = float(np.percentile(values, 25))
        q75 = float(np.percentile(values, 75))
        
        return NormalizationParams(
            method=NormalizationMethod.ROBUST,
            feature_name=feature_name,
            median=median,
            q25=q25,
            q75=q75
        )
    
    def _apply_normalization(
        self, 
        values: np.ndarray, 
        params: NormalizationParams
    ) -> np.ndarray:
        """Aplica normalização usando parâmetros"""
        
        if params.method == NormalizationMethod.MINMAX:
            # MinMax: (x - min) / (max - min)
            range_val = params.max_val - params.min_val
            if range_val == 0:
                logger.warning(f"Range zero para {params.feature_name}, retornando zeros")
                return np.zeros_like(values)
            return (values - params.min_val) / range_val
        
        elif params.method == NormalizationMethod.ZSCORE:
            # Z-score: (x - mean) / std
            if params.std == 0:
                logger.warning(f"Std zero para {params.feature_name}, retornando zeros")
                return np.zeros_like(values)
            return (values - params.mean) / params.std
        
        elif params.method == NormalizationMethod.ROBUST:
            # Robust: (x - median) / (Q75 - Q25)
            iqr = params.q75 - params.q25
            if iqr == 0:
                logger.warning(f"IQR zero para {params.feature_name}, retornando zeros")
                return np.zeros_like(values)
            return (values - params.median) / iqr
        
        else:
            raise ValueError(f"Método não implementado: {params.method}")
    
    def denormalize_feature(
        self,
        normalized_values: np.ndarray,
        feature_name: str
    ) -> np.ndarray:
        """
        Reverte normalização para valores originais
        """
        if feature_name not in self.normalization_params:
            raise ValueError(f"Parâmetros de normalização não encontrados para: {feature_name}")
        
        params = self.normalization_params[feature_name]
        
        if params.method == NormalizationMethod.MINMAX:
            range_val = params.max_val - params.min_val
            return normalized_values * range_val + params.min_val
        
        elif params.method == NormalizationMethod.ZSCORE:
            return normalized_values * params.std + params.mean
        
        elif params.method == NormalizationMethod.ROBUST:
            iqr = params.q75 - params.q25
            return normalized_values * iqr + params.median
        
        else:
            raise ValueError(f"Método não implementado: {params.method}")
    
    def normalize_single_value(
        self,
        value: float,
        feature_name: str
    ) -> float:
        """
        Normaliza um único valor usando parâmetros existentes
        Útil para dados em tempo real
        """
        if feature_name not in self.normalization_params:
            raise ValueError(f"Parâmetros não encontrados para: {feature_name}")
        
        normalized = self._apply_normalization(
            np.array([value]), 
            self.normalization_params[feature_name]
        )
        
        return float(normalized[0])
    
    def get_normalization_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo dos parâmetros de normalização
        """
        summary = {}
        
        for feature_name, params in self.normalization_params.items():
            summary[feature_name] = {
                "method": params.method.value,
                "created_at": params.created_at.isoformat(),
                "parameters": {}
            }
            
            if params.method == NormalizationMethod.MINMAX:
                summary[feature_name]["parameters"] = {
                    "min": params.min_val,
                    "max": params.max_val,
                    "range": params.max_val - params.min_val
                }
            elif params.method == NormalizationMethod.ZSCORE:
                summary[feature_name]["parameters"] = {
                    "mean": params.mean,
                    "std": params.std
                }
            elif params.method == NormalizationMethod.ROBUST:
                summary[feature_name]["parameters"] = {
                    "median": params.median,
                    "q25": params.q25,
                    "q75": params.q75,
                    "iqr": params.q75 - params.q25
                }
        
        return summary
    
    def save_normalization_params(self, filepath: str) -> None:
        """
        Salva parâmetros de normalização em arquivo
        Para usar em produção/inferência
        """
        import json
        
        # Converte parâmetros para dict serializável
        params_dict = {}
        for feature_name, params in self.normalization_params.items():
            params_dict[feature_name] = {
                "method": params.method.value,
                "feature_name": params.feature_name,
                "mean": params.mean,
                "std": params.std,
                "min_val": params.min_val,
                "max_val": params.max_val,
                "median": params.median,
                "q25": params.q25,
                "q75": params.q75,
                "created_at": params.created_at.isoformat()
            }
        
        with open(filepath, 'w') as f:
            json.dump(params_dict, f, indent=2)
        
        logger.info(f"✅ Parâmetros salvos em: {filepath}")
    
    def load_normalization_params(self, filepath: str) -> None:
        """
        Carrega parâmetros de normalização de arquivo
        """
        import json
        
        with open(filepath, 'r') as f:
            params_dict = json.load(f)
        
        self.normalization_params = {}
        for feature_name, params_data in params_dict.items():
            params = NormalizationParams(
                method=NormalizationMethod(params_data["method"]),
                feature_name=params_data["feature_name"],
                mean=params_data.get("mean"),
                std=params_data.get("std"),
                min_val=params_data.get("min_val"),
                max_val=params_data.get("max_val"),
                median=params_data.get("median"),
                q25=params_data.get("q25"),
                q75=params_data.get("q75"),
                created_at=datetime.fromisoformat(params_data["created_at"])
            )
            self.normalization_params[feature_name] = params
        
        logger.info(f"✅ {len(self.normalization_params)} parâmetros carregados de: {filepath}")
    
    def _candles_to_dataframe(self, candles: List[ValidatedOHLCV]) -> pd.DataFrame:
        """Converte lista de velas para DataFrame"""
        data = []
        for candle in candles:
            data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume,
                'data_source': candle.data_source.value,
                'quality_score': candle.validation.quality_score
            })
        
        return pd.DataFrame(data)
    
    def _dataframe_to_ml_candles(
        self, 
        df: pd.DataFrame, 
        original_candles: List[ValidatedOHLCV]
    ) -> List[MLReadyCandle]:
        """Converte DataFrame normalizado para MLReadyCandle"""
        ml_candles = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            # Extrai valores normalizados
            normalized_features = {}
            for col in df.columns:
                if col.endswith('_normalized'):
                    feature_name = col.replace('_normalized', '')
                    normalized_features[feature_name] = row[col]
            
            # Cria MLReadyCandle
            ml_candle = MLReadyCandle(
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                data_source=original_candles[i].data_source,
                raw_symbol=original_candles[i].raw_symbol,
                validation=original_candles[i].validation,
                is_outlier=original_candles[i].is_outlier,
                is_gap_fill=original_candles[i].is_gap_fill,
                normalized_features=normalized_features,
                technical_indicators={},  # Será preenchido pelo FeatureEngineer
                market_features={}        # Será preenchido pelo MarketFilters
            )
            
            ml_candles.append(ml_candle)
        
        return ml_candles
    
    def validate_normalization(
        self, 
        normalized_values: np.ndarray,
        method: NormalizationMethod
    ) -> Dict[str, Any]:
        """
        Valida se a normalização foi aplicada corretamente
        """
        validation_result = {
            "method": method.value,
            "min": float(np.min(normalized_values)),
            "max": float(np.max(normalized_values)),
            "mean": float(np.mean(normalized_values)),
            "std": float(np.std(normalized_values)),
            "has_nan": bool(np.any(np.isnan(normalized_values))),
            "has_inf": bool(np.any(np.isinf(normalized_values))),
            "is_valid": True
        }
        
        # Validações específicas por método
        if method == NormalizationMethod.MINMAX:
            # MinMax deve estar entre 0 e 1
            expected_min, expected_max = 0.0, 1.0
            tolerance = 1e-6
            
            if not (abs(validation_result["min"] - expected_min) < tolerance and 
                    abs(validation_result["max"] - expected_max) < tolerance):
                validation_result["is_valid"] = False
                validation_result["error"] = f"MinMax fora do range [0,1]: [{validation_result['min']:.6f}, {validation_result['max']:.6f}]"
        
        elif method == NormalizationMethod.ZSCORE:
            # Z-score deve ter média ~0 e std ~1
            tolerance = 1e-1
            
            if not (abs(validation_result["mean"]) < tolerance and 
                    abs(validation_result["std"] - 1.0) < tolerance):
                validation_result["is_valid"] = False
                validation_result["error"] = f"Z-score inválido: mean={validation_result['mean']:.3f}, std={validation_result['std']:.3f}"
        
        # Verifica NaN e Inf
        if validation_result["has_nan"] or validation_result["has_inf"]:
            validation_result["is_valid"] = False
            validation_result["error"] = "Valores NaN ou Inf detectados"
        
        return validation_result


# Factory function
def get_data_normalizer() -> DataNormalizer:
    """Retorna instância de DataNormalizer"""
    return DataNormalizer()