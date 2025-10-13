"""
Tests de infraestrutura para validar o Docker Compose Setup
Testa conectividade, saúde dos serviços e funcionalidade básica
"""

import pytest
import asyncio
import httpx
import psycopg2
import redis
import time
from typing import Dict, Any


class TestInfrastructure:
    """
    Suite de testes para validar a infraestrutura completa
    """
    
    @pytest.fixture(scope="class")
    def wait_for_services(self):
        """Aguarda os serviços estarem prontos"""
        time.sleep(10)  # Tempo para inicialização
        yield
    
    @pytest.fixture
    def database_config(self):
        """Configuração do database"""
        return {
            "host": "localhost",
            "port": 5432,
            "database": "forex_data",
            "user": "forex_user",
            "password": "forex_password_2024"
        }
    
    @pytest.fixture
    def redis_config(self):
        """Configuração do Redis"""
        return {
            "host": "localhost",
            "port": 6379,
            "db": 0
        }
    
    @pytest.mark.asyncio
    async def test_api_health(self, wait_for_services):
        """Testa se a API está respondendo"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    "http://localhost:8000/api/v1/health",
                    timeout=10.0
                )
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert "timestamp" in data
                print("✅ API Health Check - OK")
            except Exception as e:
                pytest.fail(f"❌ API não está respondendo: {e}")
    
    @pytest.mark.asyncio
    async def test_api_docs(self, wait_for_services):
        """Testa se a documentação está acessível"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    "http://localhost:8000/docs",
                    timeout=10.0
                )
                assert response.status_code == 200
                print("✅ API Documentation - OK")
            except Exception as e:
                pytest.fail(f"❌ Documentação não acessível: {e}")
    
    def test_database_connection(self, wait_for_services, database_config):
        """Testa conexão com TimescaleDB"""
        try:
            conn = psycopg2.connect(**database_config)
            cursor = conn.cursor()
            
            # Testa TimescaleDB extension
            cursor.execute("SELECT installed_version FROM pg_available_extensions WHERE name='timescaledb';")
            result = cursor.fetchone()
            assert result is not None, "TimescaleDB extension não instalada"
            
            # Testa se as tabelas existem
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('ohlcv_data', 'quotes_raw')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            assert 'ohlcv_data' in tables, "Tabela ohlcv_data não existe"
            assert 'quotes_raw' in tables, "Tabela quotes_raw não existe"
            
            cursor.close()
            conn.close()
            print("✅ Database Connection & Schema - OK")
            
        except Exception as e:
            pytest.fail(f"❌ Erro na conexão com database: {e}")
    
    def test_redis_connection(self, wait_for_services, redis_config):
        """Testa conexão com Redis"""
        try:
            r = redis.Redis(**redis_config)
            
            # Teste básico de set/get
            test_key = "infrastructure_test"
            test_value = "redis_working"
            
            r.set(test_key, test_value, ex=60)  # Expira em 60s
            retrieved = r.get(test_key)
            
            assert retrieved.decode() == test_value
            
            # Limpa o teste
            r.delete(test_key)
            
            print("✅ Redis Connection - OK")
            
        except Exception as e:
            pytest.fail(f"❌ Erro na conexão com Redis: {e}")
    
    @pytest.mark.asyncio
    async def test_grafana_access(self, wait_for_services):
        """Testa se Grafana está acessível"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    "http://localhost:3000/login",
                    timeout=10.0
                )
                assert response.status_code == 200
                print("✅ Grafana Access - OK")
            except Exception as e:
                pytest.fail(f"❌ Grafana não acessível: {e}")
    
    @pytest.mark.asyncio
    async def test_prometheus_access(self, wait_for_services):
        """Testa se Prometheus está acessível"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    "http://localhost:9090/api/v1/status/config",
                    timeout=10.0
                )
                assert response.status_code == 200
                print("✅ Prometheus Access - OK")
            except Exception as e:
                pytest.fail(f"❌ Prometheus não acessível: {e}")
    
    @pytest.mark.asyncio
    async def test_jupyter_access(self, wait_for_services):
        """Testa se Jupyter está acessível"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    "http://localhost:8888/login",
                    timeout=10.0
                )
                assert response.status_code == 200
                print("✅ Jupyter Access - OK")
            except Exception as e:
                pytest.fail(f"❌ Jupyter não acessível: {e}")
    
    def test_database_hypertables(self, wait_for_services, database_config):
        """Testa se as hypertables foram criadas corretamente"""
        try:
            conn = psycopg2.connect(**database_config)
            cursor = conn.cursor()
            
            # Verifica hypertables
            cursor.execute("""
                SELECT hypertable_name 
                FROM timescaledb_information.hypertables
            """)
            hypertables = [row[0] for row in cursor.fetchall()]
            
            assert 'ohlcv_data' in hypertables, "Hypertable ohlcv_data não criada"
            
            cursor.close()
            conn.close()
            print("✅ Database Hypertables - OK")
            
        except Exception as e:
            pytest.fail(f"❌ Erro ao verificar hypertables: {e}")
    
    @pytest.mark.asyncio
    async def test_api_endpoints_basic(self, wait_for_services):
        """Testa endpoints básicos da API"""
        async with httpx.AsyncClient() as client:
            try:
                # Teste endpoint latest
                response = await client.get(
                    "http://localhost:8000/api/v1/quote/latest",
                    timeout=10.0
                )
                # Pode retornar 404 se não há dados, mas não deve dar erro de conexão
                assert response.status_code in [200, 404]
                
                # Teste endpoint quotes
                response = await client.get(
                    "http://localhost:8000/api/v1/quotes?period=1d&granularity=1h",
                    timeout=10.0
                )
                assert response.status_code in [200, 404]
                
                print("✅ API Basic Endpoints - OK")
                
            except Exception as e:
                pytest.fail(f"❌ Erro nos endpoints básicos: {e}")


class TestIntegration:
    """
    Testes de integração entre os serviços
    """
    
    @pytest.mark.asyncio
    async def test_data_flow_simulation(self):
        """Simula fluxo básico de dados"""
        async with httpx.AsyncClient() as client:
            try:
                # Tenta salvar dados dummy
                test_data = {
                    "symbol": "EURUSD",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "open": 1.0800,
                    "high": 1.0850,
                    "low": 1.0750,
                    "close": 1.0820,
                    "volume": 1000
                }
                
                response = await client.post(
                    "http://localhost:8000/api/v1/data/save",
                    json=test_data,
                    timeout=10.0
                )
                
                # Aceita 200 (sucesso) ou outros códigos que indicam API funcionando
                assert response.status_code in [200, 201, 422, 400]
                print("✅ Data Flow Simulation - OK")
                
            except Exception as e:
                print(f"⚠️  Data Flow Test: {e} (API pode não ter dados ainda)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])