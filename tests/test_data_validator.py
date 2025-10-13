"""
Testes para DataValidator
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.services.data_validator import DataValidator
from app.domain.enums import Granularity, DataSource, DataQuality
from app.domain.schemas import RawOHLCV


class TestDataValidator:
    """Testes para DataValidator"""
    
    @pytest.fixture
    def validator(self):
        """Fixture para DataValidator"""
        return DataValidator()
    
    @pytest.fixture
    def sample_dataframe(self):
        """Fixture com DataFrame de exemplo"""
        dates = pd.date_range('2025-01-01', periods=10, freq='1D', tz='UTC')
        
        df = pd.DataFrame({
            'Open': [1.0850, 1.0860, 1.0870, 1.0865, 1.0855, 1.0845, 1.0850, 1.0860, 1.0870, 1.0875],
            'High': [1.0870, 1.0880, 1.0890, 1.0885, 1.0875, 1.0865, 1.0870, 1.0880, 1.0890, 1.0895],
            'Low':  [1.0840, 1.0850, 1.0860, 1.0855, 1.0845, 1.0835, 1.0840, 1.0850, 1.0860, 1.0865],
            'Close':[1.0860, 1.0870, 1.0880, 1.0875, 1.0865, 1.0855, 1.0860, 1.0870, 1.0880, 1.0885],
            'Volume': [1000, 1100, 1200, 1150, 1050, 950, 1000, 1100, 1200, 1250]
        }, index=dates)
        
        return df
    
    def test_validate_empty_dataframe(self, validator):
        """Testa validação de DataFrame vazio"""
        empty_df = pd.DataFrame()
        
        validated_candles, validation_result = validator.validate_dataframe(
            empty_df, Granularity.ONE_DAY, DataSource.YAHOO_FINANCE
        )
        
        assert len(validated_candles) == 0
        assert validation_result.quality_score == 0.0
        assert validation_result.quality_level == DataQuality.POOR
    
    def test_validate_good_dataframe(self, validator, sample_dataframe):
        """Testa validação de DataFrame de boa qualidade"""
        validated_candles, validation_result = validator.validate_dataframe(
            sample_dataframe, Granularity.ONE_DAY, DataSource.YAHOO_FINANCE
        )
        
        assert len(validated_candles) == 10
        assert validation_result.quality_score > 0.8  # Boa qualidade
        assert not validation_result.has_missing_values
        assert not validation_result.has_duplicates
    
    def test_detect_missing_values(self, validator):
        """Testa detecção de valores faltantes"""
        # DataFrame com NaN
        dates = pd.date_range('2025-01-01', periods=5, freq='1D', tz='UTC')
        df = pd.DataFrame({
            'Open': [1.0850, np.nan, 1.0870, 1.0865, 1.0855],  # Um NaN
            'High': [1.0870, 1.0880, 1.0890, 1.0885, 1.0875],
            'Low':  [1.0840, 1.0850, 1.0860, 1.0855, 1.0845],
            'Close':[1.0860, 1.0870, 1.0880, 1.0875, 1.0865],
            'Volume': [1000, 1100, 1200, 1150, 1050]
        }, index=dates)
        
        missing_info = validator._check_missing_values(df)
        
        assert missing_info["has_missing"] == True
        assert missing_info["count"] == 1
    
    def test_detect_outliers(self, validator, sample_dataframe):
        """Testa detecção de outliers"""
        # Adiciona um outlier óbvio
        outlier_df = sample_dataframe.copy()
        outlier_df.iloc[5, outlier_df.columns.get_loc('Close')] = 2.0000  # Outlier extremo
        
        outliers_info = validator._detect_outliers(outlier_df)
        
        assert outliers_info["has_outliers"] == True
        assert outliers_info["count"] > 0
    
    def test_check_ohlc_consistency(self, validator):
        """Testa verificação de consistência OHLC"""
        # DataFrame com inconsistência
        dates = pd.date_range('2025-01-01', periods=3, freq='1D', tz='UTC')
        df = pd.DataFrame({
            'Open': [1.0850, 1.0860, 1.0870],
            'High': [1.0870, 1.0850, 1.0890],  # High[1] < Open[1] - inconsistente
            'Low':  [1.0840, 1.0850, 1.0860],
            'Close':[1.0860, 1.0870, 1.0880],
            'Volume': [1000, 1100, 1200]
        }, index=dates)
        
        consistency_info = validator._check_ohlc_consistency(df)
        
        assert consistency_info["is_consistent"] == False
        assert len(consistency_info["issues"]) > 0
    
    def test_validate_single_candle(self, validator):
        """Testa validação de vela única"""
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
        
        validated_candle = validator.validate_single_candle(raw_candle)
        
        assert validated_candle.close == 1.0860
        assert validated_candle.validation.quality_score > 0.5
        assert not validated_candle.is_outlier  # Não detecta outlier em vela única
    
    def test_validation_summary(self, validator, sample_dataframe):
        """Testa geração de summary"""
        _, validation_result = validator.validate_dataframe(
            sample_dataframe, Granularity.ONE_DAY, DataSource.YAHOO_FINANCE
        )
        
        summary = validator.get_validation_summary(validation_result)
        
        assert "VALIDAÇÃO DE DADOS" in summary
        assert "Score:" in summary
        assert str(validation_result.quality_score) in summary