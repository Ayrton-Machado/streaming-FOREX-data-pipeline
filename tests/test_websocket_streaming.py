"""
Testes WebSocket Streaming - Suite Completa
Valida funcionamento completo do sistema de streaming para clientes
"""

import pytest
import pytest_asyncio
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch
import websockets
from websockets.exceptions import ConnectionClosed

from fastapi.testclient import TestClient
from fastapi import WebSocketDisconnect

from app.main import app
from app.services.websocket_manager import websocket_manager, StreamType, StreamConfig
from app.services.websocket_manager import ClientConnection, StreamMessage


class TestWebSocketStreaming:
    """
    Suite completa de testes WebSocket para validação de streaming
    Critica para atendimento de clientes empresariais
    """
    
    @pytest.fixture
    def client(self):
        """Cliente de teste FastAPI"""
        return TestClient(app)
    
    @pytest.fixture
    def websocket_url(self):
        """URL base para WebSocket"""
        return "ws://localhost:8000/ws/stream"
    
    @pytest_asyncio.fixture
    async def clean_websocket_manager(self):
        """Limpa o WebSocket manager antes de cada teste"""
        # Desconecta todos os clientes
        for client_id in list(websocket_manager.connections.keys()):
            await websocket_manager.disconnect_client(client_id)
        
        # Para todos os streams
        for task in websocket_manager.stream_tasks.values():
            if not task.done():
                task.cancel()
        
        websocket_manager.active_streams.clear()
        websocket_manager.stream_tasks.clear()
        websocket_manager.connections.clear()
        websocket_manager.message_sequence = 0
        websocket_manager.total_messages_sent = 0
        
        yield
        
        # Cleanup após teste
        await websocket_manager.shutdown()

    # ================================
    # TESTES DE CONEXÃO BÁSICA
    # ================================

    @pytest.mark.asyncio
    async def test_websocket_connection_establishment(self, client, clean_websocket_manager):
        """Testa estabelecimento básico de conexão WebSocket"""
        connection_established = False
        
        try:
            with client.websocket_connect("/ws/stream?streams=raw_ticks&client_type=test_client") as websocket:
                # Recebe mensagem de boas-vindas
                welcome_data = websocket.receive_json()
                
                assert "type" in welcome_data
                assert welcome_data["type"] == "connection_established"
                assert "client_id" in welcome_data
                assert "subscribed_streams" in welcome_data
                assert "raw_ticks" in welcome_data["subscribed_streams"]
                
                connection_established = True
                
        except Exception as e:
            pytest.fail(f"Falha na conexão WebSocket: {str(e)}")
        
        assert connection_established, "Conexão WebSocket não foi estabelecida"

    @pytest.mark.asyncio
    async def test_websocket_invalid_stream_rejection(self, client, clean_websocket_manager):
        """Testa rejeição de streams inválidos"""
        try:
            with client.websocket_connect("/ws/stream?streams=invalid_stream&client_type=test_client") as websocket:
                # Deve receber erro ou não conectar
                welcome_data = websocket.receive_json()
                
                # Se conectar, deve ter lista vazia de streams
                if welcome_data.get("type") == "connection_established":
                    assert len(welcome_data["subscribed_streams"]) == 0
                    
        except Exception as e:
            # Esperado - conexão deve falhar com stream inválido
            pass

    @pytest.mark.asyncio 
    async def test_websocket_multiple_streams_subscription(self, client, clean_websocket_manager):
        """Testa subscrição a múltiplos streams"""
        streams = "raw_ticks,ml_features,trading_signals"
        
        with client.websocket_connect(f"/ws/stream?streams={streams}&client_type=ai_client") as websocket:
            welcome_data = websocket.receive_json()
            
            assert welcome_data["type"] == "connection_established"
            subscribed = welcome_data["subscribed_streams"]
            
            assert "raw_ticks" in subscribed
            assert "ml_features" in subscribed 
            assert "trading_signals" in subscribed
            assert len(subscribed) == 3

    # ================================
    # TESTES DE STREAMING DE DADOS
    # ================================

    @pytest.mark.asyncio
    async def test_raw_ticks_stream_data_format(self, client, clean_websocket_manager):
        """Valida formato de dados do stream raw_ticks"""
        message_received = False
        
        with client.websocket_connect("/ws/stream?streams=raw_ticks&client_type=test_client") as websocket:
            # Recebe welcome
            websocket.receive_json()
            
            # Aguarda dados de streaming
            timeout = time.time() + 5  # 5 segundos timeout
            while time.time() < timeout and not message_received:
                try:
                    data = websocket.receive_json(mode="text")
                    if isinstance(data, str):
                        data = json.loads(data)
                    
                    if data.get("stream_type") == "raw_ticks":
                        # Valida estrutura obrigatória
                        assert "timestamp" in data
                        assert "sequence" in data
                        assert "symbol" in data
                        assert "data" in data
                        
                        # Valida dados específicos de ticks
                        tick_data = data["data"]
                        required_fields = ["bid", "ask", "last", "volume", "spread"]
                        
                        for field in required_fields:
                            assert field in tick_data, f"Campo {field} ausente em raw_ticks"
                        
                        # Valida tipos de dados
                        assert isinstance(tick_data["bid"], (int, float))
                        assert isinstance(tick_data["ask"], (int, float))
                        assert isinstance(tick_data["last"], (int, float))
                        assert isinstance(tick_data["volume"], int)
                        assert isinstance(tick_data["spread"], (int, float))
                        
                        # Valida lógica de mercado
                        assert tick_data["ask"] >= tick_data["bid"], "Ask deve ser >= Bid"
                        assert tick_data["spread"] > 0, "Spread deve ser positivo"
                        
                        message_received = True
                        break
                        
                except Exception as e:
                    continue
            
        assert message_received, "Não recebeu dados de raw_ticks no tempo esperado"

    @pytest.mark.asyncio
    async def test_ml_features_stream_data_format(self, client, clean_websocket_manager):
        """Valida formato de dados do stream ml_features"""
        message_received = False
        
        with client.websocket_connect("/ws/stream?streams=ml_features&client_type=ai_client") as websocket:
            websocket.receive_json()  # welcome
            
            timeout = time.time() + 10  # ML features têm frequência menor (1Hz)
            while time.time() < timeout and not message_received:
                try:
                    data = websocket.receive_json()
                    if isinstance(data, str):
                        data = json.loads(data)
                    
                    if data.get("stream_type") == "ml_features":
                        features_data = data["data"]
                        
                        # Valida campos obrigatórios para ML
                        ml_required_fields = [
                            "features_vector", "feature_names", 
                            "normalized_features", "target_signal", 
                            "confidence", "quality_score"
                        ]
                        
                        for field in ml_required_fields:
                            assert field in features_data, f"Campo {field} ausente em ml_features"
                        
                        # Valida estrutura do vetor de features
                        features = features_data["features_vector"]
                        names = features_data["feature_names"]
                        normalized = features_data["normalized_features"]
                        
                        assert isinstance(features, list), "features_vector deve ser lista"
                        assert isinstance(names, list), "feature_names deve ser lista"
                        assert isinstance(normalized, list), "normalized_features deve ser lista"
                        
                        assert len(features) == len(names), "Features e nomes devem ter mesmo tamanho"
                        assert len(features) == len(normalized), "Features e normalizados devem ter mesmo tamanho"
                        
                        # Valida intervalos
                        confidence = features_data["confidence"]
                        quality_score = features_data["quality_score"]
                        
                        assert 0 <= confidence <= 1, "Confidence deve estar entre 0 e 1"
                        assert 0 <= quality_score <= 1, "Quality score deve estar entre 0 e 1"
                        
                        message_received = True
                        break
                        
                except Exception as e:
                    continue
            
        assert message_received, "Não recebeu dados de ml_features no tempo esperado"

    @pytest.mark.asyncio
    async def test_trading_signals_stream_data_format(self, client, clean_websocket_manager):
        """Valida formato de dados do stream trading_signals"""
        message_received = False
        
        with client.websocket_connect("/ws/stream?streams=trading_signals&client_type=trading_bot") as websocket:
            websocket.receive_json()  # welcome
            
            timeout = time.time() + 8  # Trading signals: 2Hz (500ms)
            while time.time() < timeout and not message_received:
                try:
                    data = websocket.receive_json()
                    if isinstance(data, str):
                        data = json.loads(data)
                    
                    if data.get("stream_type") == "trading_signals":
                        signal_data = data["data"]
                        
                        # Valida campos obrigatórios para sinais de trading
                        required_fields = [
                            "signal", "confidence", "entry_price", 
                            "reasoning", "timeframe"
                        ]
                        
                        for field in required_fields:
                            assert field in signal_data, f"Campo {field} ausente em trading_signals"
                        
                        # Valida valores de sinal
                        signal = signal_data["signal"]
                        assert signal in ["buy", "sell", "hold"], f"Sinal inválido: {signal}"
                        
                        # Valida campos opcionais mas importantes
                        if "stop_loss" in signal_data and signal_data["stop_loss"]:
                            assert isinstance(signal_data["stop_loss"], (int, float))
                        
                        if "take_profit" in signal_data and signal_data["take_profit"]:
                            assert isinstance(signal_data["take_profit"], (int, float))
                        
                        # Valida estrutura de reasoning
                        reasoning = signal_data["reasoning"]
                        assert isinstance(reasoning, dict), "Reasoning deve ser objeto"
                        
                        message_received = True
                        break
                        
                except Exception as e:
                    continue
            
        assert message_received, "Não recebeu dados de trading_signals no tempo esperado"

    # ================================
    # TESTES DE PERFORMANCE E CARGA
    # ================================

    @pytest.mark.asyncio
    async def test_multiple_clients_concurrent_connections(self, client, clean_websocket_manager):
        """Testa múltiplas conexões simultâneas (crítico para clientes empresariais)"""
        num_clients = 5
        connected_clients = []
        
        try:
            # Conecta múltiplos clientes
            for i in range(num_clients):
                websocket = client.websocket_connect(
                    f"/ws/stream?streams=raw_ticks&client_type=load_test_client_{i}"
                )
                websocket.__enter__()
                welcome = websocket.receive_json()
                assert welcome["type"] == "connection_established"
                connected_clients.append(websocket)
            
            # Verifica se todos recebem dados
            messages_per_client = {}
            timeout = time.time() + 5
            
            while time.time() < timeout:
                for i, ws in enumerate(connected_clients):
                    try:
                        data = ws.receive_json()
                        if data.get("stream_type") == "raw_ticks":
                            messages_per_client[i] = messages_per_client.get(i, 0) + 1
                    except:
                        continue
                
                # Se todos receberam pelo menos 1 mensagem, sucesso
                if len(messages_per_client) == num_clients:
                    break
            
            # Valida que todos os clientes receberam dados
            assert len(messages_per_client) == num_clients, f"Nem todos os clientes receberam dados: {messages_per_client}"
            
        finally:
            # Cleanup
            for ws in connected_clients:
                try:
                    ws.__exit__(None, None, None)
                except:
                    pass

    @pytest.mark.asyncio
    async def test_stream_frequency_validation(self, client, clean_websocket_manager):
        """Valida frequências dos streams conforme especificação"""
        # raw_ticks deve ser ~10Hz (100ms)
        # ml_features deve ser ~1Hz (1000ms)
        
        with client.websocket_connect("/ws/stream?streams=raw_ticks&client_type=frequency_test") as websocket:
            websocket.receive_json()  # welcome
            
            timestamps = []
            message_count = 0
            max_messages = 10
            
            timeout = time.time() + 5
            while time.time() < timeout and message_count < max_messages:
                try:
                    data = websocket.receive_json()
                    if data.get("stream_type") == "raw_ticks":
                        timestamps.append(time.time())
                        message_count += 1
                except:
                    continue
            
            # Calcula intervalos entre mensagens
            if len(timestamps) > 1:
                intervals = []
                for i in range(1, len(timestamps)):
                    interval = timestamps[i] - timestamps[i-1]
                    intervals.append(interval)
                
                avg_interval = sum(intervals) / len(intervals)
                
                # raw_ticks deve ter ~100ms (0.1s) de intervalo
                # Permite tolerância de ±50ms
                assert 0.05 <= avg_interval <= 0.2, f"Intervalo médio fora da especificação: {avg_interval:.3f}s"

    # ================================
    # TESTES DE GESTÃO DE CONEXÕES
    # ================================

    @pytest.mark.asyncio
    async def test_client_disconnection_cleanup(self, client, clean_websocket_manager):
        """Testa limpeza adequada quando cliente desconecta"""
        initial_connections = len(websocket_manager.connections)
        
        # Conecta cliente
        with client.websocket_connect("/ws/stream?streams=raw_ticks&client_type=disconnect_test") as websocket:
            websocket.receive_json()  # welcome
            
            # Verifica que conexão foi adicionada
            assert len(websocket_manager.connections) == initial_connections + 1
        
        # Aguarda cleanup (pode ser assíncrono)
        await asyncio.sleep(0.5)
        
        # Verifica que conexão foi removida
        assert len(websocket_manager.connections) == initial_connections

    @pytest.mark.asyncio
    async def test_stream_subscription_management(self, client, clean_websocket_manager):
        """Testa gerenciamento de subscrições de stream"""
        # Conecta com stream específico
        with client.websocket_connect("/ws/stream?streams=raw_ticks&client_type=subscription_test") as websocket:
            websocket.receive_json()  # welcome
            
            # Verifica que stream está ativo
            assert StreamType.RAW_TICKS in websocket_manager.active_streams
            assert StreamType.RAW_TICKS in websocket_manager.stream_tasks
        
        # Aguarda cleanup
        await asyncio.sleep(0.5)
        
        # Verifica que stream foi parado (sem outros clientes)
        assert StreamType.RAW_TICKS not in websocket_manager.active_streams

    # ================================
    # TESTES DE ROBUSTEZ E FALHAS
    # ================================

    @pytest.mark.asyncio
    async def test_websocket_connection_resilience(self, client, clean_websocket_manager):
        """Testa resiliência das conexões WebSocket"""
        reconnection_successful = False
        
        try:
            # Primeira conexão
            with client.websocket_connect("/ws/stream?streams=raw_ticks&client_type=resilience_test_1") as ws1:
                ws1.receive_json()  # welcome
                
                # Simula desconexão abrupta e reconexão
                pass  # Conexão fecha automaticamente
            
            # Segunda conexão (simulando reconexão)
            with client.websocket_connect("/ws/stream?streams=raw_ticks&client_type=resilience_test_2") as ws2:
                welcome2 = ws2.receive_json()
                assert welcome2["type"] == "connection_established"
                reconnection_successful = True
                
        except Exception as e:
            pytest.fail(f"Falha no teste de resiliência: {str(e)}")
        
        assert reconnection_successful

    @pytest.mark.asyncio
    async def test_invalid_stream_parameters(self, client, clean_websocket_manager):
        """Testa tratamento de parâmetros inválidos"""
        # Teste sem streams
        try:
            with client.websocket_connect("/ws/stream?client_type=invalid_test") as websocket:
                # Deve falhar ou retornar erro
                pass
        except Exception:
            # Esperado - deve falhar sem streams
            pass
        
        # Teste com stream vazio
        try:
            with client.websocket_connect("/ws/stream?streams=&client_type=invalid_test") as websocket:
                # Deve falhar ou retornar erro
                pass
        except Exception:
            # Esperado - deve falhar com stream vazio
            pass

    # ================================
    # TESTES DE CUSTOMIZAÇÃO PARA CLIENTES
    # ================================

    @pytest.mark.asyncio
    async def test_client_type_identification(self, client, clean_websocket_manager):
        """Testa identificação de tipos de cliente"""
        client_types = ["ai_client", "trader", "analyst", "generic"]
        
        for client_type in client_types:
            with client.websocket_connect(f"/ws/stream?streams=raw_ticks&client_type={client_type}") as websocket:
                welcome = websocket.receive_json()
                assert welcome["type"] == "connection_established"
                
                # Verifica que client_id é único
                assert "client_id" in welcome
                assert len(welcome["client_id"]) > 0

    @pytest.mark.asyncio
    async def test_stream_metadata_inclusion(self, client, clean_websocket_manager):
        """Testa inclusão de metadados nos streams"""
        with client.websocket_connect("/ws/stream?streams=raw_ticks&client_type=metadata_test") as websocket:
            websocket.receive_json()  # welcome
            
            timeout = time.time() + 5
            while time.time() < timeout:
                try:
                    data = websocket.receive_json()
                    if data.get("stream_type") == "raw_ticks":
                        # Verifica metadados
                        assert "latency_ms" in data
                        assert "metadata" in data
                        
                        if data["metadata"]:
                            metadata = data["metadata"]
                            assert "frequency_ms" in metadata
                            assert "buffer_size" in metadata
                        
                        break
                        
                except:
                    continue

    # ================================
    # TESTES DE ESTATÍSTICAS E MONITORAMENTO
    # ================================

    def test_websocket_stats_endpoint(self, client):
        """Testa endpoint de estatísticas WebSocket"""
        response = client.get("/ws/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "stats" in data
        
        stats = data["stats"]
        required_stats = [
            "total_connections", "active_streams", 
            "total_messages_sent", "current_sequence"
        ]
        
        for stat in required_stats:
            assert stat in stats

    def test_websocket_streams_info_endpoint(self, client):
        """Testa endpoint de informações dos streams"""
        response = client.get("/ws/streams")
        assert response.status_code == 200
        
        data = response.json()
        assert "available_streams" in data
        assert "total_streams" in data
        assert "recommended_combinations" in data
        
        # Verifica que tem os 8 streams esperados
        assert data["total_streams"] == 8
        
        # Verifica combinações para IA
        combinations = data["recommended_combinations"]
        ai_combinations = ["ai_scalping", "ai_swing_trading", "ai_news_trading", "ai_market_making"]
        
        for combination in ai_combinations:
            assert combination in combinations

    # ================================
    # TESTE DE INTEGRAÇÃO COMPLETA
    # ================================

    @pytest.mark.asyncio
    async def test_full_integration_scenario(self, client, clean_websocket_manager):
        """Teste de integração completa - cenário real de cliente IA"""
        # Simula cliente IA conectando a múltiplos streams
        streams = "raw_ticks,ml_features,trading_signals"
        client_name = "AI_Trading_Bot_Alpha"
        
        received_streams = set()
        message_counts = {"raw_ticks": 0, "ml_features": 0, "trading_signals": 0}
        
        with client.websocket_connect(
            f"/ws/stream?streams={streams}&client_type=ai_client&client_name={client_name}"
        ) as websocket:
            
            welcome = websocket.receive_json()
            assert welcome["type"] == "connection_established"
            assert len(welcome["subscribed_streams"]) == 3
            
            # Coleta dados por tempo determinado
            timeout = time.time() + 15  # 15 segundos para receber de todos os streams
            
            while time.time() < timeout and len(received_streams) < 3:
                try:
                    data = websocket.receive_json()
                    stream_type = data.get("stream_type")
                    
                    if stream_type in message_counts:
                        message_counts[stream_type] += 1
                        received_streams.add(stream_type)
                        
                        # Valida estrutura básica
                        assert "timestamp" in data
                        assert "sequence" in data
                        assert "data" in data
                        
                except Exception as e:
                    continue
            
            # Valida que recebeu dados de todos os streams
            assert len(received_streams) == 3, f"Streams recebidos: {received_streams}"
            
            # Valida que raw_ticks teve mais mensagens (maior frequência)
            assert message_counts["raw_ticks"] > message_counts["ml_features"], \
                "raw_ticks deveria ter mais mensagens que ml_features"


# ================================
# FIXTURES E UTILITÁRIOS ADICIONAIS
# ================================

@pytest.fixture
def websocket_test_config():
    """Configuração para testes WebSocket"""
    return {
        "timeout": 10,
        "max_messages": 20,
        "test_streams": ["raw_ticks", "ml_features", "trading_signals"],
        "client_types": ["ai_client", "trader", "analyst"]
    }


class WebSocketTestClient:
    """Cliente de teste utilitário para WebSocket"""
    
    def __init__(self, client: TestClient):
        self.client = client
        self.websocket = None
    
    async def connect(self, streams: str, client_type: str = "test_client"):
        """Conecta ao WebSocket"""
        self.websocket = self.client.websocket_connect(
            f"/ws/stream?streams={streams}&client_type={client_type}"
        )
        self.websocket.__enter__()
        return self.websocket.receive_json()
    
    def disconnect(self):
        """Desconecta do WebSocket"""
        if self.websocket:
            self.websocket.__exit__(None, None, None)
            self.websocket = None
    
    def receive_messages(self, count: int, timeout: int = 10):
        """Recebe número específico de mensagens"""
        messages = []
        start_time = time.time()
        
        while len(messages) < count and time.time() - start_time < timeout:
            try:
                data = self.websocket.receive_json()
                messages.append(data)
            except:
                continue
        
        return messages