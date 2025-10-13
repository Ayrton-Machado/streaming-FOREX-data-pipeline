"""
Repository Layer - Data Access Pattern
Etapa 4: CRUD operations otimizado para TimescaleDB
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func, text
import logging

from app.database.models import OHLCVData, TechnicalIndicators, DataQualityLog
from app.domain.schemas import ValidatedOHLCV, DataValidationResult
from app.domain.enums import Granularity, DataSource, DataQuality

logger = logging.getLogger(__name__)


class OHLCVRepository:
    """
    Repository para operações CRUD em dados OHLCV
    Otimizado para consultas de séries temporais no TimescaleDB
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def save_ohlcv_data(self, validated_candles: List[ValidatedOHLCV]) -> List[OHLCVData]:
        """
        Salva dados OHLCV validados no database
        
        Args:
            validated_candles: Lista de velas validadas
            
        Returns:
            Lista de objetos OHLCVData salvos
        """
        saved_records = []
        
        for candle in validated_candles:
            try:
                # Verifica se já existe (evita duplicatas)
                existing = self.get_ohlcv_by_timestamp(
                    timestamp=candle.timestamp,
                    symbol=candle.raw_symbol,
                    granularity=candle.validation.granularity.value
                )
                
                if existing:
                    logger.debug(f"Vela já existe: {candle.timestamp} - atualizando")
                    # Atualiza record existente
                    existing.open = candle.open
                    existing.high = candle.high
                    existing.low = candle.low
                    existing.close = candle.close
                    existing.volume = candle.volume
                    existing.quality_score = candle.validation.quality_score
                    existing.quality_level = candle.validation.quality_level.value
                    existing.is_outlier = candle.is_outlier
                    existing.is_gap_fill = candle.is_gap_fill
                    existing.updated_at = datetime.utcnow()
                    
                    saved_records.append(existing)
                else:
                    # Cria novo record
                    ohlcv_record = OHLCVData(
                        timestamp=candle.timestamp,
                        symbol=candle.raw_symbol,
                        granularity=candle.validation.granularity.value,
                        open=candle.open,
                        high=candle.high,
                        low=candle.low,
                        close=candle.close,
                        volume=candle.volume,
                        data_source=candle.data_source.value,
                        quality_score=candle.validation.quality_score,
                        quality_level=candle.validation.quality_level.value,
                        is_outlier=candle.is_outlier,
                        is_gap_fill=candle.is_gap_fill
                    )
                    
                    self.db.add(ohlcv_record)
                    saved_records.append(ohlcv_record)
                    
            except Exception as e:
                logger.error(f"Erro ao salvar vela {candle.timestamp}: {str(e)}")
                continue
        
        try:
            self.db.commit()
            logger.info(f"✅ {len(saved_records)} velas salvas no database")
            return saved_records
        except Exception as e:
            self.db.rollback()
            logger.error(f"Erro ao commit: {str(e)}")
            raise
    
    def save_single_ohlcv(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
        normalized_features: Dict[str, float] = None,
        technical_indicators: Dict[str, float] = None,
        market_features: Dict[str, float] = None,
        data_source: str = "api",
        is_gap_fill: bool = False
    ) -> Optional[str]:
        """
        Salva um único registro OHLCV com features opcionais
        
        Returns:
            ID do registro salvo ou None se falhou
        """
        try:
            # Verifica se já existe
            existing = self.get_ohlcv_by_timestamp(
                timestamp=timestamp,
                symbol=symbol,
                granularity="1m"  # Default
            )
            
            if existing:
                logger.debug(f"Registro já existe: {timestamp} - pulando")
                return None
            
            # Cria novo registro
            ohlcv_record = OHLCVData(
                symbol=symbol,
                timestamp=timestamp,
                granularity="1m",  # Default
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                data_source=data_source,
                quality_score=95.0,  # Default
                quality_level="good",
                is_outlier=False,
                is_gap_fill=is_gap_fill
            )
            
            # Adiciona features se fornecidas
            if normalized_features:
                ohlcv_record.normalized_features = normalized_features
            if technical_indicators:
                ohlcv_record.technical_indicators = technical_indicators
            if market_features:
                ohlcv_record.market_features = market_features
            
            self.db.add(ohlcv_record)
            self.db.commit()
            
            return str(ohlcv_record.id)
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Erro ao salvar registro único: {str(e)}")
            return None
    
    def get_ohlcv_by_timestamp(
        self, 
        timestamp: datetime,
        symbol: str = "EURUSD=X",
        granularity: str = "1h"
    ) -> Optional[OHLCVData]:
        """
        Busca uma vela específica por timestamp
        """
        return self.db.query(OHLCVData).filter(
            and_(
                OHLCVData.timestamp == timestamp,
                OHLCVData.symbol == symbol,
                OHLCVData.granularity == granularity
            )
        ).first()
    
    def get_ohlcv_range(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol: str = "EURUSD=X",
        granularity: str = "1h",
        limit: Optional[int] = None,
        quality_filter: Optional[DataQuality] = None
    ) -> List[OHLCVData]:
        """
        Busca dados OHLCV em um intervalo de tempo
        Otimizado para TimescaleDB com filtros avançados
        """
        query = self.db.query(OHLCVData).filter(
            and_(
                OHLCVData.timestamp >= start_date,
                OHLCVData.timestamp <= end_date,
                OHLCVData.symbol == symbol,
                OHLCVData.granularity == granularity
            )
        )
        
        # Filtro de qualidade
        if quality_filter:
            query = query.filter(OHLCVData.quality_level == quality_filter.value)
        
        # Ordenação temporal
        query = query.order_by(asc(OHLCVData.timestamp))
        
        # Limite
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def get_latest_ohlcv(
        self,
        symbol: str = "EURUSD=X",
        granularity: str = "1h",
        count: int = 1
    ) -> List[OHLCVData]:
        """
        Busca as últimas N velas de um símbolo
        """
        return self.db.query(OHLCVData).filter(
            and_(
                OHLCVData.symbol == symbol,
                OHLCVData.granularity == granularity
            )
        ).order_by(desc(OHLCVData.timestamp)).limit(count).all()
    
    def get_ohlcv_statistics(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol: str = "EURUSD=X",
        granularity: str = "1h"
    ) -> Dict[str, Any]:
        """
        Calcula estatísticas de dados OHLCV para um período
        Usa funções agregadas do PostgreSQL para performance
        """
        try:
            # Query agregada otimizada
            result = self.db.query(
                func.count(OHLCVData.id).label('total_records'),
                func.min(OHLCVData.low).label('min_price'),
                func.max(OHLCVData.high).label('max_price'),
                func.avg(OHLCVData.close).label('avg_price'),
                func.avg(OHLCVData.quality_score).label('avg_quality'),
                func.sum(func.cast(OHLCVData.is_outlier, int)).label('outlier_count'),
                func.sum(func.cast(OHLCVData.is_gap_fill, int)).label('gap_fill_count')
            ).filter(
                and_(
                    OHLCVData.timestamp >= start_date,
                    OHLCVData.timestamp <= end_date,
                    OHLCVData.symbol == symbol,
                    OHLCVData.granularity == granularity
                )
            ).first()
            
            if not result or result.total_records == 0:
                return {
                    "total_records": 0,
                    "has_data": False
                }
            
            # Distribuição de qualidade
            quality_dist = self.db.query(
                OHLCVData.quality_level,
                func.count(OHLCVData.id).label('count')
            ).filter(
                and_(
                    OHLCVData.timestamp >= start_date,
                    OHLCVData.timestamp <= end_date,
                    OHLCVData.symbol == symbol,
                    OHLCVData.granularity == granularity
                )
            ).group_by(OHLCVData.quality_level).all()
            
            quality_distribution = {item.quality_level: item.count for item in quality_dist}
            
            return {
                "total_records": result.total_records,
                "has_data": True,
                "price_stats": {
                    "min_price": float(result.min_price) if result.min_price else 0,
                    "max_price": float(result.max_price) if result.max_price else 0,
                    "avg_price": float(result.avg_price) if result.avg_price else 0,
                },
                "quality_stats": {
                    "avg_quality_score": float(result.avg_quality) if result.avg_quality else 0,
                    "quality_distribution": quality_distribution,
                    "outlier_count": result.outlier_count or 0,
                    "gap_fill_count": result.gap_fill_count or 0,
                    "outlier_percentage": (result.outlier_count / result.total_records * 100) if result.total_records > 0 else 0,
                },
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "granularity": granularity
                }
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular estatísticas: {str(e)}")
            return {"error": str(e), "total_records": 0, "has_data": False}
    
    def delete_ohlcv_range(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol: str = "EURUSD=X",
        granularity: str = "1h"
    ) -> int:
        """
        Remove dados OHLCV em um intervalo
        Retorna número de registros removidos
        """
        try:
            deleted_count = self.db.query(OHLCVData).filter(
                and_(
                    OHLCVData.timestamp >= start_date,
                    OHLCVData.timestamp <= end_date,
                    OHLCVData.symbol == symbol,
                    OHLCVData.granularity == granularity
                )
            ).delete()
            
            self.db.commit()
            logger.info(f"🗑️ {deleted_count} registros removidos")
            return deleted_count
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Erro ao deletar dados: {str(e)}")
            raise
    
    def get_data_gaps(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol: str = "EURUSD=X",
        granularity: str = "1h"
    ) -> List[Dict[str, Any]]:
        """
        Identifica gaps nos dados usando window functions do PostgreSQL
        Otimizado para TimescaleDB
        """
        try:
            # Query para detectar gaps usando LAG window function
            gap_query = text("""
                SELECT 
                    symbol,
                    granularity,
                    timestamp as gap_start,
                    LAG(timestamp) OVER (ORDER BY timestamp) as previous_timestamp,
                    timestamp - LAG(timestamp) OVER (ORDER BY timestamp) as gap_duration
                FROM ohlcv_data 
                WHERE timestamp >= :start_date 
                    AND timestamp <= :end_date
                    AND symbol = :symbol 
                    AND granularity = :granularity
                ORDER BY timestamp
            """)
            
            result = self.db.execute(gap_query, {
                'start_date': start_date,
                'end_date': end_date,
                'symbol': symbol,
                'granularity': granularity
            })
            
            gaps = []
            expected_interval = self._get_granularity_interval(granularity)
            
            for row in result:
                if row.previous_timestamp and row.gap_duration:
                    # Se o gap é maior que o intervalo esperado
                    if row.gap_duration > expected_interval:
                        gaps.append({
                            "symbol": row.symbol,
                            "granularity": row.granularity,
                            "gap_start": row.previous_timestamp.isoformat(),
                            "gap_end": row.gap_start.isoformat(),
                            "gap_duration_seconds": row.gap_duration.total_seconds(),
                            "expected_interval_seconds": expected_interval.total_seconds(),
                            "missing_periods": int(row.gap_duration.total_seconds() / expected_interval.total_seconds()) - 1
                        })
            
            return gaps
            
        except Exception as e:
            logger.error(f"Erro ao obter gaps: {str(e)}")
            return []
    
    def count_records(
        self,
        symbol: str = None,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> int:
        """
        Conta registros com filtros opcionais
        """
        try:
            query = self.db.query(OHLCVData)
            
            if symbol:
                query = query.filter(OHLCVData.symbol == symbol)
            if start_time:
                query = query.filter(OHLCVData.timestamp >= start_time)
            if end_time:
                query = query.filter(OHLCVData.timestamp <= end_time)
                
            return query.count()
            
        except Exception as e:
            logger.error(f"Erro ao contar registros: {str(e)}")
            return 0
    
    def get_aggregated_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: str,
        limit: int = 1000,
        offset: int = 0
    ) -> List[OHLCVData]:
        """
        Retorna dados agregados (hourly, daily, weekly)
        """
        try:
            # Mapeamento de agregação para intervalo SQL
            interval_map = {
                'hourly': '1 hour',
                'daily': '1 day', 
                'weekly': '1 week'
            }
            
            if aggregation not in interval_map:
                raise ValueError(f"Agregação inválida: {aggregation}")
            
            interval = interval_map[aggregation]
            
            # Query SQL para agregação usando time_bucket do TimescaleDB
            sql = text(f"""
                SELECT 
                    time_bucket('{interval}', timestamp) AS timestamp,
                    symbol,
                    first(open, timestamp) AS open_price,
                    max(high) AS high_price,
                    min(low) AS low_price,
                    last(close, timestamp) AS close_price,
                    sum(volume) AS volume,
                    '{aggregation}' AS granularity,
                    avg(quality_score) AS quality_score,
                    'aggregated' AS data_source,
                    bool_or(is_outlier) AS is_outlier,
                    bool_or(is_gap_fill) AS is_gap_fill
                FROM ohlcv_data 
                WHERE symbol = :symbol 
                    AND timestamp >= :start_time 
                    AND timestamp <= :end_time
                GROUP BY time_bucket('{interval}', timestamp), symbol
                ORDER BY timestamp DESC
                LIMIT :limit OFFSET :offset
            """)
            
            result = self.db.execute(sql, {
                'symbol': symbol,
                'start_time': start_time,
                'end_time': end_time,
                'limit': limit,
                'offset': offset
            })
            
            # Converte resultado para objetos OHLCVData
            aggregated_records = []
            for row in result:
                record = OHLCVData(
                    timestamp=row.timestamp,
                    symbol=row.symbol,
                    granularity=row.granularity,
                    open=row.open_price,
                    high=row.high_price,
                    low=row.low_price,
                    close=row.close_price,
                    volume=row.volume,
                    quality_score=row.quality_score,
                    data_source=row.data_source,
                    is_outlier=row.is_outlier,
                    is_gap_fill=row.is_gap_fill
                )
                aggregated_records.append(record)
            
            return aggregated_records
            
        except Exception as e:
            logger.error(f"Erro na agregação: {str(e)}")
            return []
    
    def get_statistics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retorna estatísticas básicas para um símbolo
        """
        try:
            stats = self.db.query(
                func.count(OHLCVData.id).label('count'),
                func.min(OHLCVData.timestamp).label('start_time'),
                func.max(OHLCVData.timestamp).label('end_time'),
                func.avg(OHLCVData.quality_score).label('avg_quality'),
                func.min(OHLCVData.close).label('min_price'),
                func.max(OHLCVData.close).label('max_price'),
                func.avg(OHLCVData.close).label('avg_price')
            ).filter(OHLCVData.symbol == symbol).first()
            
            if not stats or stats.count == 0:
                return None
            
            return {
                'count': stats.count,
                'start_time': stats.start_time,
                'end_time': stats.end_time,
                'avg_quality': float(stats.avg_quality) if stats.avg_quality else 0.0,
                'price_range': {
                    'min': float(stats.min_price) if stats.min_price else 0.0,
                    'max': float(stats.max_price) if stats.max_price else 0.0,
                    'avg': float(stats.avg_price) if stats.avg_price else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {str(e)}")
            return None
    
    def get_latest_record(self, symbol: str) -> Optional[OHLCVData]:
        """
        Retorna o registro mais recente para um símbolo
        """
        try:
            return self.db.query(OHLCVData).filter(
                OHLCVData.symbol == symbol
            ).order_by(desc(OHLCVData.timestamp)).first()
            
        except Exception as e:
            logger.error(f"Erro ao obter registro mais recente: {str(e)}")
            return None
    
    def delete_old_records(self, symbol: str, cutoff_date: datetime) -> int:
        """
        Remove registros antigos para um símbolo
        """
        try:
            deleted_count = self.db.query(OHLCVData).filter(
                and_(
                    OHLCVData.symbol == symbol,
                    OHLCVData.timestamp < cutoff_date
                )
            ).delete()
            
            self.db.commit()
            return deleted_count
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Erro ao deletar registros antigos: {str(e)}")
            return 0
    
    def _get_granularity_interval(self, granularity: str) -> timedelta:
        """
        Converte granularity string para timedelta
        """
        mapping = {
            "1min": timedelta(minutes=1),
            "5min": timedelta(minutes=5),
            "15min": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1)
        }
        return mapping.get(granularity, timedelta(hours=1))


class TechnicalIndicatorsRepository:
    """
    Repository para indicadores técnicos
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def save_indicators(self, indicators_data: List[Dict[str, Any]]) -> List[TechnicalIndicators]:
        """
        Salva indicadores técnicos calculados
        """
        saved_records = []
        
        for data in indicators_data:
            try:
                indicator = TechnicalIndicators(**data)
                self.db.add(indicator)
                saved_records.append(indicator)
            except Exception as e:
                logger.error(f"Erro ao salvar indicador: {str(e)}")
                continue
        
        try:
            self.db.commit()
            return saved_records
        except Exception as e:
            self.db.rollback()
            raise
    
    def get_indicators_range(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol: str = "EURUSD=X"
    ) -> List[TechnicalIndicators]:
        """
        Busca indicadores em um intervalo de tempo
        """
        return self.db.query(TechnicalIndicators).filter(
            and_(
                TechnicalIndicators.timestamp >= start_date,
                TechnicalIndicators.timestamp <= end_date,
                TechnicalIndicators.symbol == symbol
            )
        ).order_by(asc(TechnicalIndicators.timestamp)).all()


class DataQualityRepository:
    """
    Repository para logs de qualidade de dados
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def log_quality_issue(
        self,
        symbol: str,
        granularity: str,
        issue_type: str,
        severity: str,
        description: str,
        original_data: Optional[Dict] = None,
        corrected_data: Optional[Dict] = None,
        action_taken: Optional[str] = None,
        processing_batch: Optional[str] = None
    ) -> DataQualityLog:
        """
        Registra um problema de qualidade de dados
        """
        quality_log = DataQualityLog(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            granularity=granularity,
            issue_type=issue_type,
            severity=severity,
            description=description,
            original_data=original_data,
            corrected_data=corrected_data,
            action_taken=action_taken,
            processing_batch=processing_batch
        )
        
        self.db.add(quality_log)
        self.db.commit()
        
        return quality_log
    
    def get_quality_summary(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol: str = "EURUSD=X"
    ) -> Dict[str, Any]:
        """
        Retorna resumo de qualidade de dados para um período
        """
        try:
            # Contagem por tipo de issue
            issue_types = self.db.query(
                DataQualityLog.issue_type,
                func.count(DataQualityLog.id).label('count')
            ).filter(
                and_(
                    DataQualityLog.timestamp >= start_date,
                    DataQualityLog.timestamp <= end_date,
                    DataQualityLog.symbol == symbol
                )
            ).group_by(DataQualityLog.issue_type).all()
            
            # Contagem por severidade
            severities = self.db.query(
                DataQualityLog.severity,
                func.count(DataQualityLog.id).label('count')
            ).filter(
                and_(
                    DataQualityLog.timestamp >= start_date,
                    DataQualityLog.timestamp <= end_date,
                    DataQualityLog.symbol == symbol
                )
            ).group_by(DataQualityLog.severity).all()
            
            return {
                "issue_types": {item.issue_type: item.count for item in issue_types},
                "severities": {item.severity: item.count for item in severities},
                "total_issues": sum(item.count for item in issue_types),
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar resumo de qualidade: {str(e)}")
            return {"error": str(e)}