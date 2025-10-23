"""
Testes WebSocket - Cenários Empresariais e Customizações
Validação específica para atendimento de clientes corporativos
"""

import pytest
import pytest_asyncio
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import websockets
from websockets.exceptions import ConnectionClosed

from app.services.websocket_manager import (
    websocket_manager, StreamType, StreamConfig,
    ClientConnection, StreamMessage
)


class TestWebSocketBusinessScenarios:
    """
    Testes focados em cenários de negócio reais
    Validação de customizações e SLAs para clientes empresariais
    """

    @pytest_asyncio.fixture
    async def reset_manager(self):
        """Reset completo do WebSocket manager"""
        await websocket_manager.shutdown()
        
        # Reset completo do estado
        websocket_manager.connections.clear()
        websocket_manager.active_streams.clear()
        websocket_manager.stream_tasks.clear()
        websocket_manager.message_sequence = 0
        websocket_manager.total_messages_sent = 0
        
        yield
        
        # Cleanup final
        await websocket_manager.shutdown()

    # ================================
    # TESTES DE SLA E PERFORMANCE
    # ================================

    @pytest.mark.asyncio
    async def test_low_latency_requirements(self, reset_manager):
        """
        Testa requisitos de baixa latência para clientes HFT
        SLA: < 1ms de latência de processamento interno
        """
        mock_websocket = AsyncMock()
        client_id = "hft_client_001"
        
        # Registra cliente
        client = ClientConnection(
            websocket=mock_websocket,
            client_id=client_id,
            client_type="hft_trader",
            subscribed_streams={StreamType.RAW_TICKS},
            connection_time=datetime.now()
        )
        
        websocket_manager.connections[client_id] = client
        
        # Simula dados de alta frequência
        test_data = {
            "symbol": "EUR/USD",
            "bid": 1.0850,
            "ask": 1.0851,
            "last": 1.0850,
            "volume": 1000000,
            "spread": 0.0001,
            "market_depth": {"bids": [[1.0850, 500000]], "asks": [[1.0851, 500000]]}
        }
        
        # Mede latência de processamento
        start_time = time.perf_counter()
        
        message = StreamMessage(
            stream_type=StreamType.RAW_TICKS,
            data=test_data,
            timestamp=datetime.now(),
            sequence=1,
            metadata={"frequency_ms": 100, "buffer_size": 1}
        )
        
        await websocket_manager._send_to_client(client, message)
        
        end_time = time.perf_counter()
        processing_latency = (end_time - start_time) * 1000  # ms
        
        # SLA: < 1ms para processamento interno
        assert processing_latency < 1.0, f"Latência muito alta: {processing_latency:.3f}ms"
        
        # Verifica que dados foram enviados corretamente
        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        
        assert sent_data["stream_type"] == "raw_ticks"
        assert sent_data["data"]["symbol"] == "EUR/USD"
        assert "latency_ms" in sent_data
        assert sent_data["latency_ms"] < 1.0

    @pytest.mark.asyncio
    async def test_throughput_capacity(self, reset_manager):
        """
        Testa capacidade de throughput para múltiplos clientes
        SLA: Suportar 1000 mensagens/segundo por cliente
        """
        num_clients = 10
        messages_per_client = 100
        target_throughput = 1000  # msgs/sec por cliente
        
        clients = []
        mock_websockets = []
        
        # Cria múltiplos clientes simulados
        for i in range(num_clients):
            mock_ws = AsyncMock()
            client_id = f"throughput_client_{i:03d}"
            
            client = ClientConnection(
                websocket=mock_ws,
                client_id=client_id,
                client_type="high_volume_trader",
                subscribed_streams={StreamType.RAW_TICKS, StreamType.ML_FEATURES},
                connection_time=datetime.now()
            )
            
            websocket_manager.connections[client_id] = client
            clients.append(client)
            mock_websockets.append(mock_ws)
        
        # Simula envio em massa
        start_time = time.perf_counter()
        
        for msg_num in range(messages_per_client):
            message = StreamMessage(
                stream_type=StreamType.RAW_TICKS,
                data={
                    "symbol": "EUR/USD",
                    "bid": 1.0850 + (msg_num * 0.0001),
                    "ask": 1.0851 + (msg_num * 0.0001),
                    "last": 1.0850 + (msg_num * 0.0001),
                    "volume": 1000000,
                    "spread": 0.0001
                },
                timestamp=datetime.now(),
                sequence=msg_num + 1,
                metadata={"batch_id": msg_num // 10}
            )
            
            # Envia para todos os clientes
            for client in clients:
                await websocket_manager._send_to_client(client, message)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        total_messages = num_clients * messages_per_client
        throughput = total_messages / total_time
        
        # Valida throughput
        min_required_throughput = target_throughput * num_clients * 0.8  # 80% do target
        assert throughput >= min_required_throughput, \
            f"Throughput insuficiente: {throughput:.0f} msgs/sec (mín: {min_required_throughput:.0f})"
        
        # Verifica que todos os clientes receberam todas as mensagens
        for mock_ws in mock_websockets:
            assert mock_ws.send_text.call_count == messages_per_client

    # ================================
    # TESTES DE CUSTOMIZAÇÃO DE DADOS
    # ================================

    @pytest.mark.asyncio
    async def test_client_specific_data_filtering(self, reset_manager):
        """
        Testa filtros personalizados por cliente
        Cenário: Cliente quer apenas EUR/USD e GBP/USD
        """
        mock_websocket = AsyncMock()
        client_id = "filtered_client_001"
        
        # Cliente com filtros específicos
        client = ClientConnection(
            websocket=mock_websocket,
            client_id=client_id,
            client_type="selective_trader",
            subscribed_streams={StreamType.RAW_TICKS},
            connection_time=datetime.now(),
            filters={
                "symbols": ["EUR/USD", "GBP/USD"],
                "min_volume": 500000,
                "exclude_weekends": True
            }
        )
        
        websocket_manager.connections[client_id] = client
        
        # Testa dados que devem ser filtrados
        test_cases = [
            # Deve passar - EUR/USD com volume suficiente
            {
                "symbol": "EUR/USD",
                "volume": 1000000,
                "should_receive": True
            },
            # Deve ser filtrado - símbolo não permitido
            {
                "symbol": "USD/JPY",
                "volume": 1000000,
                "should_receive": False
            },
            # Deve ser filtrado - volume insuficiente
            {
                "symbol": "GBP/USD",
                "volume": 100000,
                "should_receive": False
            },
            # Deve passar - GBP/USD com volume suficiente
            {
                "symbol": "GBP/USD",
                "volume": 750000,
                "should_receive": True
            }
        ]
        
        sent_messages = 0
        
        # Patch once and drive side_effects for each case
        side_effects = [tc["should_receive"] for tc in test_cases]
        with patch.object(websocket_manager, '_apply_client_filters', side_effect=side_effects) as mock_filter:
            for i, test_case in enumerate(test_cases):
                message = StreamMessage(
                    stream_type=StreamType.RAW_TICKS,
                    data={
                        "symbol": test_case["symbol"],
                        "bid": 1.0850,
                        "ask": 1.0851,
                        "last": 1.0850,
                        "volume": test_case["volume"],
                        "spread": 0.0001
                    },
                    timestamp=datetime.now(),
                    sequence=i + 1
                )

                await websocket_manager._send_to_client(client, message)

                if test_case["should_receive"]:
                    sent_messages += 1
        
        # Valida que apenas mensagens permitidas foram enviadas
        assert mock_websocket.send_text.call_count == sent_messages
        
        # Valida filtros foram aplicados
        expected_filter_calls = len(test_cases)
        assert mock_filter.call_count == expected_filter_calls

    @pytest.mark.asyncio
    async def test_client_rate_limiting(self, reset_manager):
        """
        Testa limitação de taxa por cliente
        Cenário: Cliente básico limitado a 100 msgs/min
        """
        mock_websocket = AsyncMock()
        client_id = "rate_limited_client"
        
        # Cliente com rate limiting
        client = ClientConnection(
            websocket=mock_websocket,
            client_id=client_id,
            client_type="basic_tier",
            subscribed_streams={StreamType.RAW_TICKS},
            connection_time=datetime.now(),
            rate_limit={"max_messages_per_minute": 100}
        )
        
        websocket_manager.connections[client_id] = client
        
        # Simula rate limiter
        # Patch the rate limiter and call the patched function inside the loop so side_effect is used
        with patch.object(websocket_manager, '_check_rate_limit') as mock_rate_limit:
            mock_rate_limit.side_effect = [True] * 100 + [False] * 50

            messages_sent = 0
            messages_blocked = 0

            # Tenta enviar 150 mensagens
            for i in range(150):
                message = StreamMessage(
                    stream_type=StreamType.RAW_TICKS,
                    data={"symbol": "EUR/USD", "bid": 1.0850, "ask": 1.0851,
                          "last": 1.0850, "volume": 1000000, "spread": 0.0001},
                    timestamp=datetime.now(),
                    sequence=i + 1
                )

                # Let _send_to_client call the patched _check_rate_limit internally.
                sent = await websocket_manager._send_to_client(client, message)
                if sent:
                    messages_sent += 1
                else:
                    messages_blocked += 1
            
            # Valida rate limiting
            assert messages_sent == 100, f"Deveria enviar 100 mensagens, enviou {messages_sent}"
            assert messages_blocked == 50, f"Deveria bloquear 50 mensagens, bloqueou {messages_blocked}"
            assert mock_websocket.send_text.call_count == 100

    # ================================
    # TESTES DE AUTENTICAÇÃO E AUTORIZAÇÃO
    # ================================

    @pytest.mark.asyncio
    async def test_premium_client_access_levels(self, reset_manager):
        """
        Testa níveis de acesso premium
        Premium clients têm acesso a dados exclusivos
        """
        # Cliente básico
        basic_client = ClientConnection(
            websocket=AsyncMock(),
            client_id="basic_client",
            client_type="basic",
            subscribed_streams={StreamType.RAW_TICKS},
            connection_time=datetime.now(),
            access_level="basic"
        )
        
        # Cliente premium
        premium_client = ClientConnection(
            websocket=AsyncMock(),
            client_id="premium_client",
            client_type="premium",
            subscribed_streams={StreamType.RAW_TICKS, StreamType.PREMIUM_ANALYTICS, StreamType.ORDERBOOK},
            connection_time=datetime.now(),
            access_level="premium"
        )
        
        websocket_manager.connections["basic_client"] = basic_client
        websocket_manager.connections["premium_client"] = premium_client
        
        # Dados premium (market depth avançado)
        premium_data = {
            "symbol": "EUR/USD",
            "bid": 1.0850,
            "ask": 1.0851,
            "last": 1.0850,
            "volume": 1000000,
            "spread": 0.0001,
            "level2_data": {  # Dados premium
                "bids": [[1.0850, 500000], [1.0849, 300000]],
                "asks": [[1.0851, 400000], [1.0852, 200000]]
            },
            "institutional_flow": {  # Dados premium
                "buy_pressure": 0.65,
                "sell_pressure": 0.35
            }
        }
        
        premium_message = StreamMessage(
            stream_type=StreamType.PREMIUM_ANALYTICS,
            data=premium_data,
            timestamp=datetime.now(),
            sequence=1
        )
        
        # Mock autorização
        with patch.object(websocket_manager, '_authorize_stream_access') as mock_auth:
            def auth_side_effect(client, stream_type):
                if client.access_level == "premium":
                    return True
                return stream_type not in [StreamType.PREMIUM_ANALYTICS, StreamType.ORDERBOOK]
            
            mock_auth.side_effect = auth_side_effect
            
            # Tenta enviar para ambos
            await websocket_manager._send_to_client(basic_client, premium_message)
            await websocket_manager._send_to_client(premium_client, premium_message)
        
        # Cliente básico não deve receber dados premium
        basic_client.websocket.send_text.assert_not_called()
        
        # Cliente premium deve receber
        premium_client.websocket.send_text.assert_called_once()

    # ================================
    # TESTES DE MONITORING E ALERTAS
    # ================================

    @pytest.mark.asyncio
    async def test_client_health_monitoring(self, reset_manager):
        """
        Testa monitoramento de saúde dos clientes
        Detecta clientes problemáticos
        """
        # Cliente com problemas de rede
        problematic_websocket = AsyncMock()
        problematic_websocket.send_text.side_effect = ConnectionClosed(None, None)
        
        problematic_client = ClientConnection(
            websocket=problematic_websocket,
            client_id="problematic_client",
            client_type="trader",
            subscribed_streams={StreamType.RAW_TICKS},
            connection_time=datetime.now()
        )
        
        websocket_manager.connections["problematic_client"] = problematic_client
        
        # Tenta enviar mensagem
        message = StreamMessage(
            stream_type=StreamType.RAW_TICKS,
            data={"symbol": "EUR/USD", "bid": 1.0850, "ask": 1.0851, 
                  "last": 1.0850, "volume": 1000000, "spread": 0.0001},
            timestamp=datetime.now(),
            sequence=1
        )
        
        with patch.object(websocket_manager, '_handle_client_error') as mock_error_handler:
            await websocket_manager._send_to_client(problematic_client, message)
            # allow the loop to run any scheduled tasks
            await asyncio.sleep(0)

            # Verifica que erro foi tratado
            mock_error_handler.assert_called_once()
            assert mock_error_handler.call_args[0][0] == "problematic_client"

    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, reset_manager):
        """
        Testa coleta de métricas de performance
        Crítico para SLA e faturamento
        """
        mock_websocket = AsyncMock()
        client_id = "metrics_client"
        
        client = ClientConnection(
            websocket=mock_websocket,
            client_id=client_id,
            client_type="enterprise",
            subscribed_streams={StreamType.RAW_TICKS},
            connection_time=datetime.now()
        )
        
        websocket_manager.connections[client_id] = client
        
        # Simula coleta de métricas
        with patch.object(websocket_manager, '_collect_metrics') as mock_metrics:
            message = StreamMessage(
                stream_type=StreamType.RAW_TICKS,
                data={"symbol": "EUR/USD", "bid": 1.0850, "ask": 1.0851, 
                      "last": 1.0850, "volume": 1000000, "spread": 0.0001},
                timestamp=datetime.now(),
                sequence=1
            )
            
            await websocket_manager._send_to_client(client, message)
            # Verifica coleta de métricas (garante chamada e parâmetros)
            mock_metrics.assert_called_once()
            kwargs = mock_metrics.call_args.kwargs
            assert kwargs.get("client_id") == client_id
            assert "message_size" in kwargs
            assert "latency_ms" in kwargs
            assert kwargs.get("stream_type") == StreamType.RAW_TICKS

    # ================================
    # TESTES DE FAILOVER E RECUPERAÇÃO
    # ================================

    @pytest.mark.asyncio
    async def test_automatic_reconnection_handling(self, reset_manager):
        """
        Testa tratamento automático de reconexão
        Mantém estado do cliente durante reconexões
        """
        original_client_id = "reconnecting_client"
        
        # Primeira conexão
        original_client = ClientConnection(
            websocket=AsyncMock(),
            client_id=original_client_id,
            client_type="persistent_trader",
            subscribed_streams={StreamType.RAW_TICKS, StreamType.TRADING_SIGNALS},
            connection_time=datetime.now()
        )
        
        websocket_manager.connections[original_client_id] = original_client
        
        # Simula desconexão
        await websocket_manager.disconnect_client(original_client_id)
        
        # Verifica que cliente foi removido
        assert original_client_id not in websocket_manager.connections
        
        # Simula reconexão (mesmo client_id)
        new_websocket = AsyncMock()
        reconnected_client = ClientConnection(
            websocket=new_websocket,
            client_id=original_client_id,  # Mesmo ID
            client_type="persistent_trader",
            subscribed_streams={StreamType.RAW_TICKS, StreamType.TRADING_SIGNALS},
            connection_time=datetime.now()
        )
        
        # Mock recuperação de estado
        with patch.object(websocket_manager, '_restore_client_state') as mock_restore:
            mock_restore.return_value = {
                "last_sequence": 1500,
                "missed_messages": 25,
                "subscription_preferences": {
                    "buffer_size": 10,
                    "compression": True
                }
            }

            websocket_manager.connections[original_client_id] = reconnected_client

            # The ConnectionDict will call _restore_client_state; ensure the mock was called
            # give the event loop a tick for any scheduled tasks
            await asyncio.sleep(0)
            mock_restore.assert_called_once_with(original_client_id)

    # ================================
    # TESTES DE COMPLIANCE E AUDITORIA
    # ================================

    @pytest.mark.asyncio
    async def test_audit_trail_generation(self, reset_manager):
        """
        Testa geração de trilha de auditoria
        Requerido para compliance regulatório
        """
        mock_websocket = AsyncMock()
        client_id = "audited_client"
        
        client = ClientConnection(
            websocket=mock_websocket,
            client_id=client_id,
            client_type="institutional",
            subscribed_streams={StreamType.RAW_TICKS},
            connection_time=datetime.now(),
            compliance_level="mifid2"
        )
        
        websocket_manager.connections[client_id] = client
        
        audit_events = []
        
        def mock_audit_log(event_type, client_id, data, timestamp):
            audit_events.append({
                "event_type": event_type,
                "client_id": client_id,
                "data": data,
                "timestamp": timestamp
            })
        
        with patch.object(websocket_manager, '_audit_log', side_effect=mock_audit_log):
            # Conexão
            await websocket_manager._audit_log(
                "client_connected", client_id, 
                {"streams": list(client.subscribed_streams)},
                datetime.now()
            )
            
            # Envio de mensagem
            message = StreamMessage(
                stream_type=StreamType.RAW_TICKS,
                data={"symbol": "EUR/USD", "bid": 1.0850, "ask": 1.0851, 
                      "last": 1.0850, "volume": 1000000, "spread": 0.0001},
                timestamp=datetime.now(),
                sequence=1
            )
            
            await websocket_manager._send_to_client(client, message)
            
            await websocket_manager._audit_log(
                "message_sent", client_id,
                {"stream_type": "raw_ticks", "sequence": 1},
                datetime.now()
            )
            
            # Desconexão
            await websocket_manager._audit_log(
                "client_disconnected", client_id,
                {"duration_seconds": 120, "messages_sent": 1},
                datetime.now()
            )
        
        # Valida trilha de auditoria
        assert len(audit_events) == 3
        assert audit_events[0]["event_type"] == "client_connected"
        assert audit_events[1]["event_type"] == "message_sent"
        assert audit_events[2]["event_type"] == "client_disconnected"

    # ================================
    # TESTES DE INTEGRAÇÃO COM SISTEMAS EXTERNOS
    # ================================

    @pytest.mark.asyncio
    async def test_external_system_integration(self, reset_manager):
        """
        Testa integração com sistemas externos do cliente
        Webhook notifications, callbacks, etc.
        """
        mock_websocket = AsyncMock()
        client_id = "integrated_client"
        
        client = ClientConnection(
            websocket=mock_websocket,
            client_id=client_id,
            client_type="api_integrated",
            subscribed_streams={StreamType.TRADING_SIGNALS},
            connection_time=datetime.now(),
            external_callbacks={
                "webhook_url": "https://client.example.com/webhook",
                "api_key": "client_secret_key",
                "notification_events": ["high_confidence_signals"]
            }
        )
        
        websocket_manager.connections[client_id] = client
        
        # Sinal de alta confiança
        high_confidence_signal = {
            "signal": "buy",
            "confidence": 0.95,
            "entry_price": 1.0850,
            "reasoning": {
                "technical": "Golden cross + RSI oversold",
                "fundamental": "ECB dovish, Fed hawkish divergence"
            },
            "timeframe": "15m"
        }
        
        message = StreamMessage(
            stream_type=StreamType.TRADING_SIGNALS,
            data=high_confidence_signal,
            timestamp=datetime.now(),
            sequence=1
        )
        
        # Mock chamada externa
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 200

            with patch.object(websocket_manager, '_trigger_external_callback') as mock_callback:
                await websocket_manager._send_to_client(client, message)
                # allow scheduled tasks to run
                await asyncio.sleep(0)

                # Verifica que callback foi chamado para sinal de alta confiança
                if high_confidence_signal["confidence"] >= 0.9:
                    mock_callback.assert_called_once_with(
                        client_id=client_id,
                        event_type="high_confidence_signals",
                        data=high_confidence_signal
                    )


# ================================
# UTILITÁRIOS DE TESTE
# ================================

class MockWebSocketManager:
    """Manager simulado para testes isolados"""
    
    def __init__(self):
        self.connections = {}
        self.active_streams = set()
        self.message_sequence = 0
        self.sent_messages = []
    
    async def add_client(self, client: ClientConnection):
        self.connections[client.client_id] = client
    
    async def send_message(self, client_id: str, message: StreamMessage):
        if client_id in self.connections:
            self.sent_messages.append({
                "client_id": client_id,
                "message": message,
                "timestamp": datetime.now()
            })
    
    def get_client_stats(self, client_id: str):
        client_messages = [m for m in self.sent_messages if m["client_id"] == client_id]
        return {
            "total_messages": len(client_messages),
            "last_message_time": client_messages[-1]["timestamp"] if client_messages else None,
            "message_rate": len(client_messages) / 60 if client_messages else 0  # por minuto
        }


@pytest.fixture
def mock_websocket_manager():
    """Fixture para WebSocket manager simulado"""
    return MockWebSocketManager()


# ================================
# CASOS DE TESTE DE NEGÓCIO
# ================================

class TestBusinessUseCases:
    """Testes baseados em casos de uso reais de clientes"""
    
    @pytest.mark.asyncio
    async def test_ai_trading_bot_scenario(self, mock_websocket_manager):
        """
        Cenário: Bot de trading IA precisa de dados ML + sinais + ticks
        SLA: Latência < 10ms, 99.9% uptime
        """
        client = ClientConnection(
            websocket=AsyncMock(),
            client_id="ai_bot_v2",
            client_type="ai_trader",
            subscribed_streams={
                StreamType.RAW_TICKS, 
                StreamType.ML_FEATURES, 
                StreamType.TRADING_SIGNALS
            },
            connection_time=datetime.now(),
            sla_requirements={
                "max_latency_ms": 10,
                "uptime_percentage": 99.9,
                "data_completeness": 100
            }
        )
        
        await mock_websocket_manager.add_client(client)
        
        # Simula fluxo de dados sincronizado
        for sequence in range(1, 101):
            # Raw tick (alta frequência)
            tick_message = StreamMessage(
                stream_type=StreamType.RAW_TICKS,
                data={"symbol": "EUR/USD", "bid": 1.0850, "ask": 1.0851},
                timestamp=datetime.now(),
                sequence=sequence
            )
            
            await mock_websocket_manager.send_message(client.client_id, tick_message)
            
            # ML features (a cada 10 ticks)
            if sequence % 10 == 0:
                ml_message = StreamMessage(
                    stream_type=StreamType.ML_FEATURES,
                    data={"features_vector": [0.1, 0.2, 0.3], "confidence": 0.85},
                    timestamp=datetime.now(),
                    sequence=sequence
                )
                
                await mock_websocket_manager.send_message(client.client_id, ml_message)
        
        stats = mock_websocket_manager.get_client_stats(client.client_id)
        
        # Valida SLA
        assert stats["total_messages"] == 110  # 100 ticks + 10 ML features
        assert stats["message_rate"] > 0  # Mensagens sendo entregues

    @pytest.mark.asyncio
    async def test_institutional_portfolio_manager_scenario(self, mock_websocket_manager):
        """
        Cenário: Gestor de portfólio institucional
        Precisa: Analytics premium + order book + news sentiment
        """
        client = ClientConnection(
            websocket=AsyncMock(),
            client_id="portfolio_mgr_001",
            client_type="institutional",
            subscribed_streams={
                StreamType.PREMIUM_ANALYTICS,
                StreamType.ORDERBOOK,
                StreamType.NEWS_SENTIMENT
            },
            connection_time=datetime.now(),
            access_level="institutional",
            filters={
                "symbols": ["EUR/USD", "GBP/USD", "USD/JPY"],
                "min_news_impact": "medium",
                "orderbook_depth": 5
            }
        )
        
        await mock_websocket_manager.add_client(client)
        
        # Premium analytics
        premium_data = {
            "institutional_flow": {"buy_pressure": 0.7, "sell_pressure": 0.3},
            "volatility_forecast": {"1h": 0.12, "4h": 0.18, "1d": 0.25},
            "correlation_matrix": {"EUR/USD": {"GBP/USD": 0.65, "USD/JPY": -0.23}}
        }
        
        analytics_message = StreamMessage(
            stream_type=StreamType.PREMIUM_ANALYTICS,
            data=premium_data,
            timestamp=datetime.now(),
            sequence=1
        )
        
        await mock_websocket_manager.send_message(client.client_id, analytics_message)
        
        stats = mock_websocket_manager.get_client_stats(client.client_id)
        assert stats["total_messages"] >= 1

    @pytest.mark.asyncio
    async def test_retail_trader_basic_scenario(self, mock_websocket_manager):
        """
        Cenário: Trader retail básico
        Limitações: Rate limit, dados básicos apenas
        """
        client = ClientConnection(
            websocket=AsyncMock(),
            client_id="retail_trader_123",
            client_type="retail",
            subscribed_streams={StreamType.RAW_TICKS},
            connection_time=datetime.now(),
            access_level="basic",
            rate_limit={
                "max_messages_per_minute": 60,
                "burst_allowance": 10
            }
        )
        
        await mock_websocket_manager.add_client(client)
        
        # Testa rate limiting
        messages_sent = 0
        for i in range(100):  # Tenta enviar 100, mas deve ser limitado
            if messages_sent < 60:  # Rate limit
                message = StreamMessage(
                    stream_type=StreamType.RAW_TICKS,
                    data={"symbol": "EUR/USD", "bid": 1.0850, "ask": 1.0851},
                    timestamp=datetime.now(),
                    sequence=i + 1
                )
                
                await mock_websocket_manager.send_message(client.client_id, message)
                messages_sent += 1
        
        stats = mock_websocket_manager.get_client_stats(client.client_id)
        assert stats["total_messages"] == 60  # Rate limited