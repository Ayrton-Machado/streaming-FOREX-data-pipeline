#!/bin/bash
set -e

# TimescaleDB Initialization Script
# Executa na primeira inicialização do container

echo "🚀 Inicializando TimescaleDB para FOREX Pipeline..."

# Cria extensão TimescaleDB
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Cria extensão TimescaleDB
    CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
    
    -- Configura timezone
    SET timezone = 'UTC';
    
    -- Configurações de performance para time-series
    ALTER SYSTEM SET shared_preload_libraries = 'timescaledb';
    ALTER SYSTEM SET max_worker_processes = 16;
    ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
    ALTER SYSTEM SET max_parallel_workers = 8;
    
    -- Log de inicialização
    SELECT 'TimescaleDB extension installed successfully' as status;
EOSQL

echo "✅ TimescaleDB extension instalada"

# Cria schema principal
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Schema para dados FOREX
    CREATE SCHEMA IF NOT EXISTS forex;
    
    -- Schema para métricas e logs
    CREATE SCHEMA IF NOT EXISTS metrics;
    
    -- Schema para machine learning features
    CREATE SCHEMA IF NOT EXISTS ml_features;
    
    -- Grants para schemas
    GRANT ALL PRIVILEGES ON SCHEMA forex TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON SCHEMA metrics TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON SCHEMA ml_features TO $POSTGRES_USER;
    
    SELECT 'Schemas created successfully' as status;
EOSQL

echo "✅ Schemas criados"

# Cria função para automatizar hypertables
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Função para criar hypertable automaticamente
    CREATE OR REPLACE FUNCTION create_hypertable_if_not_exists(
        table_name TEXT,
        time_column TEXT,
        chunk_time_interval INTERVAL DEFAULT INTERVAL '1 day'
    ) RETURNS VOID AS \$\$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM _timescaledb_catalog.hypertable 
            WHERE table_name = create_hypertable_if_not_exists.table_name
        ) THEN
            PERFORM create_hypertable(table_name, time_column, chunk_time_interval => chunk_time_interval);
            RAISE NOTICE 'Hypertable % created', table_name;
        ELSE
            RAISE NOTICE 'Hypertable % already exists', table_name;
        END IF;
    END;
    \$\$ LANGUAGE plpgsql;
    
    SELECT 'Hypertable helper function created' as status;
EOSQL

echo "✅ Função helper para hypertables criada"

# Cria índices otimizados para FOREX
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Função para criar índices FOREX otimizados
    CREATE OR REPLACE FUNCTION create_forex_indexes(table_name TEXT) RETURNS VOID AS \$\$
    BEGIN
        -- Índice composto para consultas por símbolo e tempo
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_symbol_time ON %s (symbol, timestamp DESC)', 
                      replace(table_name, '.', '_'), table_name);
        
        -- Índice para granularidade
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_granularity ON %s (granularity)', 
                      replace(table_name, '.', '_'), table_name);
        
        -- Índice para qualidade de dados
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_quality ON %s (quality_level, quality_score)', 
                      replace(table_name, '.', '_'), table_name);
        
        -- Índice para data source
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_source ON %s (data_source)', 
                      replace(table_name, '.', '_'), table_name);
        
        -- Índice parcial para outliers
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_outliers ON %s (timestamp) WHERE is_outlier = true', 
                      replace(table_name, '.', '_'), table_name);
        
        -- Índice parcial para gaps
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_gaps ON %s (timestamp) WHERE is_gap_fill = true', 
                      replace(table_name, '.', '_'), table_name);
        
        RAISE NOTICE 'FOREX indexes created for table %', table_name;
    END;
    \$\$ LANGUAGE plpgsql;
    
    SELECT 'FOREX index helper function created' as status;
EOSQL

echo "✅ Função para índices FOREX criada"

# Cria políticas de retenção
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Função para configurar políticas de retenção
    CREATE OR REPLACE FUNCTION setup_retention_policy(
        table_name TEXT,
        retention_period INTERVAL DEFAULT INTERVAL '90 days'
    ) RETURNS VOID AS \$\$
    BEGIN
        -- Remove política existente se houver
        BEGIN
            PERFORM remove_retention_policy(table_name);
        EXCEPTION WHEN OTHERS THEN
            -- Ignora erro se política não existir
        END;
        
        -- Adiciona nova política
        PERFORM add_retention_policy(table_name, retention_period);
        
        RAISE NOTICE 'Retention policy set for % with period %', table_name, retention_period;
    END;
    \$\$ LANGUAGE plpgsql;
    
    SELECT 'Retention policy helper function created' as status;
EOSQL

echo "✅ Função para políticas de retenção criada"

# Cria função para compressão automática
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Função para configurar compressão automática
    CREATE OR REPLACE FUNCTION setup_compression_policy(
        table_name TEXT,
        compress_after INTERVAL DEFAULT INTERVAL '7 days'
    ) RETURNS VOID AS \$\$
    BEGIN
        -- Habilita compressão na tabela
        BEGIN
            PERFORM add_compression_policy(table_name, compress_after);
            RAISE NOTICE 'Compression policy added for % after %', table_name, compress_after;
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'Compression policy already exists for %', table_name;
        END;
    END;
    \$\$ LANGUAGE plpgsql;
    
    SELECT 'Compression policy helper function created' as status;
EOSQL

echo "✅ Função para compressão criada"

# Configurações de monitoramento
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- View para monitoramento de hypertables
    CREATE OR REPLACE VIEW metrics.hypertable_stats AS
    SELECT 
        ht.table_name,
        ht.num_chunks,
        pg_size_pretty(hypertable_size(format('%I.%I', ht.schema_name, ht.table_name))) as table_size,
        (SELECT count(*) FROM information_schema.tables WHERE table_schema = ht.schema_name AND table_name = ht.table_name) as exists
    FROM _timescaledb_catalog.hypertable ht;
    
    -- View para estatísticas de chunks
    CREATE OR REPLACE VIEW metrics.chunk_stats AS
    SELECT 
        format('%I.%I', schema_name, table_name) as hypertable,
        count(*) as total_chunks,
        count(*) FILTER (WHERE compressed_chunk_id IS NOT NULL) as compressed_chunks,
        min(range_start) as oldest_chunk,
        max(range_end) as newest_chunk
    FROM _timescaledb_catalog.chunk ch
    JOIN _timescaledb_catalog.hypertable ht ON ch.hypertable_id = ht.id
    GROUP BY schema_name, table_name;
    
    SELECT 'Monitoring views created' as status;
EOSQL

echo "✅ Views de monitoramento criadas"

# Configurações finais
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Configurações de performance
    ALTER DATABASE $POSTGRES_DB SET timezone = 'UTC';
    ALTER DATABASE $POSTGRES_DB SET log_statement = 'none';
    ALTER DATABASE $POSTGRES_DB SET log_min_duration_statement = 1000;
    
    -- Estatísticas automáticas
    ALTER DATABASE $POSTGRES_DB SET default_statistics_target = 100;
    
    SELECT 'Database configuration completed' as status;
EOSQL

echo "🎉 TimescaleDB inicialização completa!"
echo "📊 Database: $POSTGRES_DB"
echo "👤 User: $POSTGRES_USER"
echo "🔧 Extensions: timescaledb"
echo "📂 Schemas: forex, metrics, ml_features"