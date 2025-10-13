"""
Feature Importance Analysis Service
Análise de importância de features para modelos de ML e seleção automática
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
from sklearn.feature_selection import (
    mutual_info_regression, 
    SelectKBest, 
    f_regression,
    RFE
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class ImportanceMethod(Enum):
    """Métodos de análise de importância"""
    CORRELATION = "correlation"
    MUTUAL_INFO = "mutual_info"
    RANDOM_FOREST = "random_forest"
    LASSO = "lasso"
    F_STATISTIC = "f_statistic"
    RFE = "recursive_feature_elimination"

@dataclass
class FeatureScore:
    """Score de importância de uma feature"""
    name: str
    score: float
    rank: int
    method: ImportanceMethod
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None

@dataclass
class ImportanceAnalysisResult:
    """Resultado da análise de importância"""
    method: ImportanceMethod
    scores: List[FeatureScore]
    top_features: List[str]
    analysis_params: Dict[str, Any]
    metadata: Dict[str, Any]

class FeatureImportanceAnalyzer:
    """
    Analisador de importância de features
    
    Implementa múltiplos métodos de análise:
    - Correlação (Pearson, Spearman)
    - Mutual Information
    - Random Forest Feature Importance
    - LASSO Regularization
    - F-Statistic
    - Recursive Feature Elimination
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.analysis_cache = {}
        
    def analyze_correlation(self, features: pd.DataFrame, target: pd.Series,
                          method: str = 'pearson') -> ImportanceAnalysisResult:
        """Análise baseada em correlação"""
        scores = []
        
        for i, col in enumerate(features.columns):
            feature_series = features[col].dropna()
            aligned_target = target.loc[feature_series.index]
            
            if len(feature_series) > 0 and len(aligned_target) > 0:
                try:
                    if method == 'pearson':
                        corr, p_value = pearsonr(feature_series, aligned_target)
                    else:  # spearman
                        corr, p_value = spearmanr(feature_series, aligned_target)
                    
                    score = abs(corr) if not np.isnan(corr) else 0.0
                    
                    scores.append(FeatureScore(
                        name=col,
                        score=score,
                        rank=0,  # Será preenchido depois
                        method=ImportanceMethod.CORRELATION,
                        p_value=p_value if not np.isnan(p_value) else None
                    ))
                except:
                    scores.append(FeatureScore(
                        name=col,
                        score=0.0,
                        rank=0,
                        method=ImportanceMethod.CORRELATION
                    ))
        
        # Ordena por score e define ranks
        scores.sort(key=lambda x: x.score, reverse=True)
        for i, score in enumerate(scores):
            score.rank = i + 1
        
        top_features = [s.name for s in scores[:20]]  # Top 20
        
        return ImportanceAnalysisResult(
            method=ImportanceMethod.CORRELATION,
            scores=scores,
            top_features=top_features,
            analysis_params={'correlation_method': method},
            metadata={'total_features': len(features.columns)}
        )
    
    def analyze_mutual_information(self, features: pd.DataFrame, target: pd.Series,
                                 n_neighbors: int = 3) -> ImportanceAnalysisResult:
        """Análise baseada em Mutual Information"""
        # Prepara dados
        X = features.fillna(features.mean())
        y = target.fillna(target.mean())
        
        # Alinha índices
        common_idx = X.index.intersection(y.index)
        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        if len(X_aligned) < 10:
            # Retorna resultado vazio se dados insuficientes
            return ImportanceAnalysisResult(
                method=ImportanceMethod.MUTUAL_INFO,
                scores=[],
                top_features=[],
                analysis_params={'n_neighbors': n_neighbors},
                metadata={'error': 'Insufficient data'}
            )
        
        try:
            # Calcula MI
            mi_scores = mutual_info_regression(
                X_aligned, 
                y_aligned, 
                n_neighbors=n_neighbors,
                random_state=self.random_state
            )
            
            scores = []
            for i, col in enumerate(X_aligned.columns):
                scores.append(FeatureScore(
                    name=col,
                    score=mi_scores[i],
                    rank=0,
                    method=ImportanceMethod.MUTUAL_INFO
                ))
            
            # Ordena e rankeia
            scores.sort(key=lambda x: x.score, reverse=True)
            for i, score in enumerate(scores):
                score.rank = i + 1
            
            top_features = [s.name for s in scores[:20]]
            
            return ImportanceAnalysisResult(
                method=ImportanceMethod.MUTUAL_INFO,
                scores=scores,
                top_features=top_features,
                analysis_params={'n_neighbors': n_neighbors},
                metadata={'total_features': len(X_aligned.columns)}
            )
            
        except Exception as e:
            return ImportanceAnalysisResult(
                method=ImportanceMethod.MUTUAL_INFO,
                scores=[],
                top_features=[],
                analysis_params={'n_neighbors': n_neighbors},
                metadata={'error': str(e)}
            )
    
    def analyze_random_forest(self, features: pd.DataFrame, target: pd.Series,
                            n_estimators: int = 100) -> ImportanceAnalysisResult:
        """Análise usando Random Forest Feature Importance"""
        # Prepara dados
        X = features.fillna(features.mean())
        y = target.fillna(target.mean())
        
        # Alinha índices
        common_idx = X.index.intersection(y.index)
        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        if len(X_aligned) < 10:
            return ImportanceAnalysisResult(
                method=ImportanceMethod.RANDOM_FOREST,
                scores=[],
                top_features=[],
                analysis_params={'n_estimators': n_estimators},
                metadata={'error': 'Insufficient data'}
            )
        
        try:
            # Treina Random Forest
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
            rf.fit(X_aligned, y_aligned)
            
            scores = []
            for i, col in enumerate(X_aligned.columns):
                scores.append(FeatureScore(
                    name=col,
                    score=rf.feature_importances_[i],
                    rank=0,
                    method=ImportanceMethod.RANDOM_FOREST
                ))
            
            # Ordena e rankeia
            scores.sort(key=lambda x: x.score, reverse=True)
            for i, score in enumerate(scores):
                score.rank = i + 1
            
            top_features = [s.name for s in scores[:20]]
            
            return ImportanceAnalysisResult(
                method=ImportanceMethod.RANDOM_FOREST,
                scores=scores,
                top_features=top_features,
                analysis_params={'n_estimators': n_estimators},
                metadata={
                    'total_features': len(X_aligned.columns),
                    'rf_score': rf.score(X_aligned, y_aligned)
                }
            )
            
        except Exception as e:
            return ImportanceAnalysisResult(
                method=ImportanceMethod.RANDOM_FOREST,
                scores=[],
                top_features=[],
                analysis_params={'n_estimators': n_estimators},
                metadata={'error': str(e)}
            )
    
    def analyze_lasso(self, features: pd.DataFrame, target: pd.Series,
                     cv: int = 5) -> ImportanceAnalysisResult:
        """Análise usando LASSO Regularization"""
        # Prepara dados
        X = features.fillna(features.mean())
        y = target.fillna(target.mean())
        
        # Alinha índices
        common_idx = X.index.intersection(y.index)
        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        if len(X_aligned) < 20:  # LASSO precisa de mais dados
            return ImportanceAnalysisResult(
                method=ImportanceMethod.LASSO,
                scores=[],
                top_features=[],
                analysis_params={'cv': cv},
                metadata={'error': 'Insufficient data for LASSO'}
            )
        
        try:
            # Normaliza features
            X_norm = (X_aligned - X_aligned.mean()) / X_aligned.std()
            X_norm = X_norm.fillna(0)
            
            # Treina LASSO com CV
            lasso = LassoCV(cv=cv, random_state=self.random_state, max_iter=2000)
            lasso.fit(X_norm, y_aligned)
            
            scores = []
            for i, col in enumerate(X_aligned.columns):
                scores.append(FeatureScore(
                    name=col,
                    score=abs(lasso.coef_[i]),
                    rank=0,
                    method=ImportanceMethod.LASSO
                ))
            
            # Ordena e rankeia
            scores.sort(key=lambda x: x.score, reverse=True)
            for i, score in enumerate(scores):
                score.rank = i + 1
            
            # Features selecionadas (coef != 0)
            selected_features = [s.name for s in scores if s.score > 1e-6]
            top_features = selected_features[:20] if len(selected_features) >= 20 else selected_features
            
            return ImportanceAnalysisResult(
                method=ImportanceMethod.LASSO,
                scores=scores,
                top_features=top_features,
                analysis_params={'cv': cv, 'alpha': lasso.alpha_},
                metadata={
                    'total_features': len(X_aligned.columns),
                    'selected_features': len(selected_features),
                    'lasso_score': lasso.score(X_norm, y_aligned)
                }
            )
            
        except Exception as e:
            return ImportanceAnalysisResult(
                method=ImportanceMethod.LASSO,
                scores=[],
                top_features=[],
                analysis_params={'cv': cv},
                metadata={'error': str(e)}
            )
    
    def analyze_f_statistic(self, features: pd.DataFrame, target: pd.Series,
                           k: int = 20) -> ImportanceAnalysisResult:
        """Análise usando F-Statistic"""
        # Prepara dados
        X = features.fillna(features.mean())
        y = target.fillna(target.mean())
        
        # Alinha índices
        common_idx = X.index.intersection(y.index)
        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        if len(X_aligned) < 10:
            return ImportanceAnalysisResult(
                method=ImportanceMethod.F_STATISTIC,
                scores=[],
                top_features=[],
                analysis_params={'k': k},
                metadata={'error': 'Insufficient data'}
            )
        
        try:
            # Calcula F-statistics
            selector = SelectKBest(score_func=f_regression, k='all')
            selector.fit(X_aligned, y_aligned)
            
            scores = []
            for i, col in enumerate(X_aligned.columns):
                scores.append(FeatureScore(
                    name=col,
                    score=selector.scores_[i] if not np.isnan(selector.scores_[i]) else 0,
                    rank=0,
                    method=ImportanceMethod.F_STATISTIC,
                    p_value=selector.pvalues_[i] if not np.isnan(selector.pvalues_[i]) else None
                ))
            
            # Ordena e rankeia
            scores.sort(key=lambda x: x.score, reverse=True)
            for i, score in enumerate(scores):
                score.rank = i + 1
            
            top_features = [s.name for s in scores[:k]]
            
            return ImportanceAnalysisResult(
                method=ImportanceMethod.F_STATISTIC,
                scores=scores,
                top_features=top_features,
                analysis_params={'k': k},
                metadata={'total_features': len(X_aligned.columns)}
            )
            
        except Exception as e:
            return ImportanceAnalysisResult(
                method=ImportanceMethod.F_STATISTIC,
                scores=[],
                top_features=[],
                analysis_params={'k': k},
                metadata={'error': str(e)}
            )
    
    def analyze_rfe(self, features: pd.DataFrame, target: pd.Series,
                   n_features: int = 20, step: int = 1) -> ImportanceAnalysisResult:
        """Análise usando Recursive Feature Elimination"""
        # Prepara dados
        X = features.fillna(features.mean())
        y = target.fillna(target.mean())
        
        # Alinha índices
        common_idx = X.index.intersection(y.index)
        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        if len(X_aligned) < 20 or len(X_aligned.columns) < n_features:
            return ImportanceAnalysisResult(
                method=ImportanceMethod.RFE,
                scores=[],
                top_features=[],
                analysis_params={'n_features': n_features, 'step': step},
                metadata={'error': 'Insufficient data or features for RFE'}
            )
        
        try:
            # RFE com RandomForest como estimador
            estimator = RandomForestRegressor(
                n_estimators=50, 
                random_state=self.random_state,
                n_jobs=-1
            )
            
            rfe = RFE(
                estimator=estimator, 
                n_features_to_select=n_features,
                step=step
            )
            rfe.fit(X_aligned, y_aligned)
            
            scores = []
            for i, col in enumerate(X_aligned.columns):
                # Score baseado no ranking (invertido para que menor rank = maior score)
                rank_score = 1.0 / rfe.ranking_[i] if rfe.ranking_[i] > 0 else 0
                
                scores.append(FeatureScore(
                    name=col,
                    score=rank_score,
                    rank=rfe.ranking_[i],
                    method=ImportanceMethod.RFE
                ))
            
            # Ordena por rank (menor rank = melhor)
            scores.sort(key=lambda x: x.rank)
            
            # Features selecionadas
            selected_features = [col for col, selected in zip(X_aligned.columns, rfe.support_) if selected]
            
            return ImportanceAnalysisResult(
                method=ImportanceMethod.RFE,
                scores=scores,
                top_features=selected_features,
                analysis_params={'n_features': n_features, 'step': step},
                metadata={
                    'total_features': len(X_aligned.columns),
                    'selected_features': len(selected_features)
                }
            )
            
        except Exception as e:
            return ImportanceAnalysisResult(
                method=ImportanceMethod.RFE,
                scores=[],
                top_features=[],
                analysis_params={'n_features': n_features, 'step': step},
                metadata={'error': str(e)}
            )
    
    def comprehensive_analysis(self, features: pd.DataFrame, target: pd.Series,
                             methods: Optional[List[ImportanceMethod]] = None) -> Dict[str, ImportanceAnalysisResult]:
        """Executa análise compreensiva com múltiplos métodos"""
        
        if methods is None:
            methods = [
                ImportanceMethod.CORRELATION,
                ImportanceMethod.MUTUAL_INFO,
                ImportanceMethod.RANDOM_FOREST,
                ImportanceMethod.F_STATISTIC
            ]
        
        results = {}
        
        for method in methods:
            print(f"Executando análise: {method.value}")
            
            try:
                if method == ImportanceMethod.CORRELATION:
                    result = self.analyze_correlation(features, target)
                elif method == ImportanceMethod.MUTUAL_INFO:
                    result = self.analyze_mutual_information(features, target)
                elif method == ImportanceMethod.RANDOM_FOREST:
                    result = self.analyze_random_forest(features, target)
                elif method == ImportanceMethod.LASSO:
                    result = self.analyze_lasso(features, target)
                elif method == ImportanceMethod.F_STATISTIC:
                    result = self.analyze_f_statistic(features, target)
                elif method == ImportanceMethod.RFE:
                    result = self.analyze_rfe(features, target)
                
                results[method.value] = result
                
            except Exception as e:
                print(f"Erro na análise {method.value}: {e}")
                results[method.value] = ImportanceAnalysisResult(
                    method=method,
                    scores=[],
                    top_features=[],
                    analysis_params={},
                    metadata={'error': str(e)}
                )
        
        return results
    
    def create_consensus_ranking(self, results: Dict[str, ImportanceAnalysisResult],
                               top_n: int = 15) -> Dict[str, Any]:
        """Cria ranking de consenso baseado em múltiplos métodos"""
        
        # Coleta todas as features
        all_features = set()
        for result in results.values():
            all_features.update([score.name for score in result.scores])
        
        # Calcula score de consenso
        consensus_scores = {}
        
        for feature in all_features:
            scores = []
            ranks = []
            
            for method_name, result in results.items():
                # Encontra score da feature neste método
                feature_score = next((s for s in result.scores if s.name == feature), None)
                
                if feature_score:
                    # Normaliza score (0-1)
                    max_score = max([s.score for s in result.scores]) if result.scores else 1
                    normalized_score = feature_score.score / max_score if max_score > 0 else 0
                    scores.append(normalized_score)
                    
                    # Normaliza rank (0-1, menor rank = maior score)
                    max_rank = len(result.scores)
                    normalized_rank = (max_rank - feature_score.rank + 1) / max_rank if max_rank > 0 else 0
                    ranks.append(normalized_rank)
            
            if scores and ranks:
                # Score de consenso: média de scores normalizados + ranks normalizados
                consensus_scores[feature] = {
                    'consensus_score': (np.mean(scores) + np.mean(ranks)) / 2,
                    'avg_score': np.mean(scores),
                    'avg_rank': np.mean(ranks),
                    'methods_count': len(scores),
                    'score_std': np.std(scores),
                    'rank_std': np.std(ranks)
                }
        
        # Ordena por score de consenso
        sorted_features = sorted(
            consensus_scores.items(),
            key=lambda x: x[1]['consensus_score'],
            reverse=True
        )
        
        # Cria ranking final
        consensus_ranking = []
        for i, (feature, metrics) in enumerate(sorted_features[:top_n]):
            consensus_ranking.append({
                'rank': i + 1,
                'feature': feature,
                'consensus_score': metrics['consensus_score'],
                'avg_score': metrics['avg_score'],
                'avg_rank': metrics['avg_rank'],
                'methods_count': metrics['methods_count'],
                'score_std': metrics['score_std'],
                'rank_std': metrics['rank_std']
            })
        
        return {
            'consensus_ranking': consensus_ranking,
            'total_features_analyzed': len(all_features),
            'methods_used': list(results.keys()),
            'top_features': [item['feature'] for item in consensus_ranking]
        }
    
    def generate_feature_importance_report(self, features: pd.DataFrame, target: pd.Series,
                                         output_file: Optional[str] = None) -> Dict[str, Any]:
        """Gera relatório completo de importância de features"""
        
        print("Iniciando análise compreensiva de importância de features...")
        
        # Executa análise compreensiva
        results = self.comprehensive_analysis(features, target)
        
        # Cria ranking de consenso
        consensus = self.create_consensus_ranking(results)
        
        # Análise estatística das features
        feature_stats = {}
        for col in features.columns:
            series = features[col].dropna()
            if len(series) > 0:
                feature_stats[col] = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'missing_pct': float(features[col].isna().sum() / len(features) * 100),
                    'unique_values': int(series.nunique()),
                    'correlation_with_target': float(series.corr(target.loc[series.index]))
                }
        
        # Compila relatório final
        report = {
            'analysis_summary': {
                'total_features': len(features.columns),
                'target_variable': target.name if hasattr(target, 'name') else 'target',
                'data_points': len(features),
                'methods_executed': len(results),
                'successful_methods': len([r for r in results.values() if not r.metadata.get('error')])
            },
            'consensus_ranking': consensus,
            'method_results': {},
            'feature_statistics': feature_stats,
            'recommendations': self._generate_recommendations(consensus, feature_stats)
        }
        
        # Adiciona resultados por método
        for method_name, result in results.items():
            if not result.metadata.get('error'):
                report['method_results'][method_name] = {
                    'top_features': result.top_features,
                    'total_analyzed': len(result.scores),
                    'analysis_params': result.analysis_params,
                    'metadata': result.metadata
                }
        
        # Salva relatório se especificado
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Relatório salvo em: {output_file}")
        
        return report
    
    def _generate_recommendations(self, consensus: Dict[str, Any], 
                                feature_stats: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas na análise"""
        recommendations = []
        
        top_features = consensus.get('top_features', [])
        
        if len(top_features) >= 10:
            recommendations.append(
                f"✅ Identificadas {len(top_features)} features principais para o modelo ML"
            )
        else:
            recommendations.append(
                "⚠️ Poucas features relevantes identificadas - considere feature engineering adicional"
            )
        
        # Analisa qualidade das features
        high_missing = [name for name, stats in feature_stats.items() 
                       if stats.get('missing_pct', 0) > 50]
        
        if high_missing:
            recommendations.append(
                f"🔧 {len(high_missing)} features com >50% missing values - considere remoção ou imputação"
            )
        
        # Analisa correlação
        high_corr_features = [name for name, stats in feature_stats.items()
                            if abs(stats.get('correlation_with_target', 0)) > 0.7]
        
        if high_corr_features:
            recommendations.append(
                f"🎯 {len(high_corr_features)} features com alta correlação (>0.7) com target"
            )
        
        # Recomendações de uso
        recommendations.extend([
            "📊 Use o consensus ranking para seleção de features",
            "🔍 Considere feature engineering baseado nas top features",
            "⚖️ Teste diferentes combinações das top 10-15 features",
            "📈 Monitore performance do modelo com features selecionadas"
        ])
        
        return recommendations