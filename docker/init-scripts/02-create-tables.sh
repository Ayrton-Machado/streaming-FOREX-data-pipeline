#!/bin/bash
set -e

# Cria tabelas para aplicação FOREX
echo "📋 Criando tabelas da aplicação FOREX..."

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Tabela principal OHLCV
    CREATE TABLE IF NOT EXISTS ohlcv_data (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        symbol VARCHAR(20) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        granularity VARCHAR(10) NOT NULL DEFAULT '1h',
        open DECIMAL(15,8) NOT NULL,
        high DECIMAL(15,8) NOT NULL,
        low DECIMAL(15,8) NOT NULL,
        close DECIMAL(15,8) NOT NULL,
        volume DECIMAL(20,8) DEFAULT 0,
        
        -- Features normalizadas (JSON)
        normalized_features JSONB,
        
        -- Indicadores técnicos (JSON)
        technical_indicators JSONB,
        
        -- Features de mercado (JSON)
        market_features JSONB,
        
        -- Metadados
        data_source VARCHAR(50) DEFAULT 'yahoo',
        quality_score DECIMAL(5,2) DEFAULT 95.0,
        quality_level VARCHAR(20) DEFAULT 'good',
        
        -- Flags
        is_outlier BOOLEAN DEFAULT FALSE,
        is_gap_fill BOOLEAN DEFAULT FALSE,
        
        -- Timestamps de controle
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Converte para hypertable
    SELECT create_hypertable_if_not_exists('ohlcv_data', 'timestamp', INTERVAL '1 day');
    
    -- Cria índices otimizados
    SELECT create_forex_indexes('ohlcv_data');
    
    -- Constraint para validação OHLC
    ALTER TABLE ohlcv_data ADD CONSTRAINT chk_ohlc_valid 
        CHECK (high >= open AND high >= close AND high >= low AND low <= open AND low <= close);
    
    -- Constraint para qualidade
    ALTER TABLE ohlcv_data ADD CONSTRAINT chk_quality_score 
        CHECK (quality_score >= 0 AND quality_score <= 100);
    
    SELECT 'OHLCV table created successfully' as status;
EOSQL

echo "✅ Tabela ohlcv_data criada"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Tabela para indicadores técnicos históricos
    CREATE TABLE IF NOT EXISTS technical_indicators (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        ohlcv_id UUID REFERENCES ohlcv_data(id) ON DELETE CASCADE,
        symbol VARCHAR(20) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        
        -- Indicadores técnicos
        sma_20 DECIMAL(15,8),
        sma_50 DECIMAL(15,8),
        ema_12 DECIMAL(15,8),
        ema_26 DECIMAL(15,8),
        rsi_14 DECIMAL(8,4),
        macd DECIMAL(15,8),
        macd_signal DECIMAL(15,8),
        macd_histogram DECIMAL(15,8),
        bollinger_upper DECIMAL(15,8),
        bollinger_middle DECIMAL(15,8),
        bollinger_lower DECIMAL(15,8),
        atr_14 DECIMAL(15,8),
        
        -- Features customizadas
        price_momentum DECIMAL(8,4),
        volatility_ratio DECIMAL(8,4),
        volume_sma_ratio DECIMAL(8,4),
        
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Converte para hypertable
    SELECT create_hypertable_if_not_exists('technical_indicators', 'timestamp', INTERVAL '1 day');
    
    -- Índices
    CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_time ON technical_indicators (symbol, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_technical_indicators_ohlcv ON technical_indicators (ohlcv_id);
    
    SELECT 'Technical indicators table created successfully' as status;
EOSQL

echo "✅ Tabela technical_indicators criada"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Tabela para logs de qualidade de dados
    CREATE TABLE IF NOT EXISTS data_quality_log (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        symbol VARCHAR(20) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        
        -- Detalhes do problema
        issue_type VARCHAR(50) NOT NULL,
        severity VARCHAR(20) NOT NULL,
        description TEXT,
        suggested_action TEXT,
        
        -- Valores
        raw_value TEXT,
        corrected_value TEXT,
        
        -- Status
        is_resolved BOOLEAN DEFAULT FALSE,
        resolution_method VARCHAR(50),
        
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Converte para hypertable
    SELECT create_hypertable_if_not_exists('data_quality_log', 'timestamp', INTERVAL '1 day');
    
    -- Índices
    CREATE INDEX IF NOT EXISTS idx_quality_log_symbol_time ON data_quality_log (symbol, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_quality_log_issue_type ON data_quality_log (issue_type);
    CREATE INDEX IF NOT EXISTS idx_quality_log_severity ON data_quality_log (severity);
    CREATE INDEX IF NOT EXISTS idx_quality_log_unresolved ON data_quality_log (timestamp) WHERE is_resolved = FALSE;
    
    SELECT 'Data quality log table created successfully' as status;
EOSQL

echo "✅ Tabela data_quality_log criada"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Tabela para status do mercado
    CREATE TABLE IF NOT EXISTS market_status (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        symbol VARCHAR(20) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        
        -- Status de mercado
        is_market_open BOOLEAN DEFAULT TRUE,
        session_name VARCHAR(20),
        trading_volume DECIMAL(20,8),
        
        -- Métricas de liquidez
        bid_ask_spread DECIMAL(15,8),
        market_impact DECIMAL(8,4),
        liquidity_score DECIMAL(5,2),
        
        -- Eventos de mercado
        has_news_event BOOLEAN DEFAULT FALSE,
        volatility_regime VARCHAR(20) DEFAULT 'normal',
        
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Converte para hypertable
    SELECT create_hypertable_if_not_exists('market_status', 'timestamp', INTERVAL '1 hour');
    
    -- Índices
    CREATE INDEX IF NOT EXISTS idx_market_status_symbol_time ON market_status (symbol, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_market_status_session ON market_status (session_name);
    CREATE INDEX IF NOT EXISTS idx_market_status_open ON market_status (timestamp) WHERE is_market_open = TRUE;
    
    SELECT 'Market status table created successfully' as status;
EOSQL

echo "✅ Tabela market_status criada"

# Configura políticas de retenção e compressão
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Políticas de retenção (90 dias para dados principais)
    SELECT setup_retention_policy('ohlcv_data', INTERVAL '90 days');
    SELECT setup_retention_policy('technical_indicators', INTERVAL '90 days');
    SELECT setup_retention_policy('data_quality_log', INTERVAL '30 days');
    SELECT setup_retention_policy('market_status', INTERVAL '60 days');
    
    -- Políticas de compressão (7 dias)
    SELECT setup_compression_policy('ohlcv_data', INTERVAL '7 days');
    SELECT setup_compression_policy('technical_indicators', INTERVAL '7 days');
    SELECT setup_compression_policy('data_quality_log', INTERVAL '3 days');
    SELECT setup_compression_policy('market_status', INTERVAL '7 days');
    
    SELECT 'Retention and compression policies configured' as status;
EOSQL

echo "✅ Políticas de retenção e compressão configuradas"

# Cria views úteis
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- View para dados OHLCV com indicadores
    CREATE OR REPLACE VIEW forex.ohlcv_with_indicators AS
    SELECT 
        o.*,
        ti.sma_20,
        ti.sma_50,
        ti.ema_12,
        ti.ema_26,
        ti.rsi_14,
        ti.macd,
        ti.bollinger_upper,
        ti.bollinger_lower,
        ti.atr_14
    FROM ohlcv_data o
    LEFT JOIN technical_indicators ti ON o.id = ti.ohlcv_id;
    
    -- View para estatísticas diárias
    CREATE OR REPLACE VIEW forex.daily_stats AS
    SELECT 
        symbol,
        date_trunc('day', timestamp) as date,
        count(*) as records_count,
        first(open, timestamp) as open,
        max(high) as high,
        min(low) as low,
        last(close, timestamp) as close,
        sum(volume) as total_volume,
        avg(quality_score) as avg_quality,
        count(*) FILTER (WHERE is_outlier = true) as outliers_count,
        count(*) FILTER (WHERE is_gap_fill = true) as gaps_filled
    FROM ohlcv_data
    GROUP BY symbol, date_trunc('day', timestamp)
    ORDER BY symbol, date DESC;
    
    -- View para problemas de qualidade recentes
    CREATE OR REPLACE VIEW forex.recent_quality_issues AS
    SELECT 
        symbol,
        issue_type,
        severity,
        count(*) as occurrences,
        max(timestamp) as last_occurrence,
        count(*) FILTER (WHERE is_resolved = false) as unresolved_count
    FROM data_quality_log
    WHERE timestamp >= NOW() - INTERVAL '24 hours'
    GROUP BY symbol, issue_type, severity
    ORDER BY severity DESC, occurrences DESC;
    
    SELECT 'Utility views created' as status;
EOSQL

echo "✅ Views utilitárias criadas"

echo "🎉 Tabelas da aplicação FOREX criadas com sucesso!"
echo "📊 Tabelas: ohlcv_data, technical_indicators, data_quality_log, market_status"
echo "🔍 Views: ohlcv_with_indicators, daily_stats, recent_quality_issues"
echo "⚡ Policies: retention (30-90 days), compression (3-7 days)"