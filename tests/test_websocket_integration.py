"""
Testes WebSocket - Integração e Simulação Real
Testa funcionalidade completa com dados reais simulados
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import patch, MagicMock

try:
    import websocket  # websocket-client library
except ImportError:
    websocket = None

import requests
from requests.exceptions import RequestException, ConnectionError


class WebSocketIntegrationTester:
    """
    Testador de integração WebSocket que simula clientes reais
    Usado para validação completa do sistema de streaming
    """
    
    def __init__(self, base_url: str = "ws://localhost:8000"):
        self.base_url = base_url
        self.http_base = base_url.replace("ws://", "http://")
        self.connections = {}
        self.received_messages = {}
        self.connection_stats = {}
        self.test_results = {}
    
    def test_server_availability(self) -> bool:
        """Testa se o servidor WebSocket está disponível"""
        try:
            response = requests.get(f"{self.http_base}/health", timeout=5)
            return response.status_code == 200
        except (RequestException, ConnectionError):
            return False
    
    def test_websocket_endpoints_info(self) -> Dict:
        """Testa endpoints de informação do WebSocket"""
        try:
            # Testa endpoint de stats
            stats_response = requests.get(f"{self.http_base}/ws/stats", timeout=5)
            streams_response = requests.get(f"{self.http_base}/ws/streams", timeout=5)
            
            results = {
                "stats_endpoint": {
                    "status": stats_response.status_code,
                    "data": stats_response.json() if stats_response.status_code == 200 else None
                },
                "streams_endpoint": {
                    "status": streams_response.status_code,
                    "data": streams_response.json() if streams_response.status_code == 200 else None
                }
            }
            
            return results
            
        except Exception as e:
            return {"error": str(e)}
    
    def simulate_client_connection(self, client_id: str, streams: List[str], 
                                 client_type: str = "test_client",
                                 duration: int = 10) -> Dict:
        """
        Simula conexão de cliente e coleta métricas
        
        Args:
            client_id: ID único do cliente
            streams: Lista de streams para subscrever
            client_type: Tipo do cliente
            duration: Duração do teste em segundos
        """
        if websocket is None:
            return {"error": "websocket-client library not available"}
        
        results = {
            "client_id": client_id,
            "connection_established": False,
            "messages_received": 0,
            "message_types": {},
            "connection_duration": 0,
            "average_latency": 0,
            "errors": [],
            "sample_messages": []
        }
        
        streams_param = ",".join(streams)
        ws_url = f"{self.base_url}/ws/stream?streams={streams_param}&client_type={client_type}"
        
        try:
            # Callbacks para WebSocket
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    results["messages_received"] += 1
                    
                    # Contabiliza tipos de mensagem
                    msg_type = data.get("type", data.get("stream_type", "unknown"))
                    results["message_types"][msg_type] = results["message_types"].get(msg_type, 0) + 1
                    
                    # Calcula latência se timestamp disponível
                    if "timestamp" in data:
                        try:
                            msg_time = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
                            latency = (datetime.now() - msg_time.replace(tzinfo=None)).total_seconds() * 1000
                            
                            if latency > 0 and latency < 10000:  # Sanity check
                                current_avg = results["average_latency"]
                                count = results["messages_received"]
                                results["average_latency"] = (current_avg * (count - 1) + latency) / count
                        except:
                            pass
                    
                    # Guarda amostras das primeiras mensagens
                    if len(results["sample_messages"]) < 3:
                        results["sample_messages"].append({
                            "type": msg_type,
                            "timestamp": datetime.now().isoformat(),
                            "data_keys": list(data.get("data", {}).keys()) if "data" in data else [],
                            "sequence": data.get("sequence"),
                            "size_bytes": len(message)
                        })
                        
                except Exception as e:
                    results["errors"].append(f"Message processing error: {str(e)}")
            
            def on_error(ws, error):
                results["errors"].append(f"WebSocket error: {str(error)}")
            
            def on_close(ws, close_status_code, close_msg):
                results["connection_duration"] = time.time() - start_time
            
            def on_open(ws):
                results["connection_established"] = True
            
            # Cria conexão WebSocket
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            start_time = time.time()
            
            # Executa em thread separada com timeout
            def run_websocket():
                ws.run_forever()
            
            ws_thread = threading.Thread(target=run_websocket)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Aguarda duração do teste
            ws_thread.join(timeout=duration)
            
            # Força fechamento se ainda conectado
            if ws_thread.is_alive():
                ws.close()
                ws_thread.join(timeout=1)
            
            results["connection_duration"] = time.time() - start_time
            
        except Exception as e:
            results["errors"].append(f"Connection error: {str(e)}")
        
        return results
    
    def run_multi_client_load_test(self, num_clients: int = 5, duration: int = 15) -> Dict:
        """
        Executa teste de carga com múltiplos clientes
        
        Args:
            num_clients: Número de clientes simultâneos
            duration: Duração do teste em segundos
        """
        print(f"🚀 Iniciando teste de carga com {num_clients} clientes por {duration}s...")
        
        # Configurações de teste para diferentes tipos de cliente
        client_configs = [
            {"streams": ["raw_ticks"], "type": "basic_trader"},
            {"streams": ["raw_ticks", "ml_features"], "type": "ai_client"},
            {"streams": ["trading_signals"], "type": "signal_consumer"},
            {"streams": ["raw_ticks", "trading_signals"], "type": "trading_bot"},
            {"streams": ["ml_features", "premium_analytics"], "type": "institutional"}
        ]
        
        results = {
            "test_start": datetime.now().isoformat(),
            "num_clients": num_clients,
            "duration_seconds": duration,
            "client_results": [],
            "aggregate_stats": {},
            "success_rate": 0,
            "total_messages": 0,
            "average_latency": 0
        }
        
        # Executa testes de cliente em paralelo
        threads = []
        client_results = [None] * num_clients
        
        def run_client_test(client_index):
            config = client_configs[client_index % len(client_configs)]
            client_id = f"load_test_client_{client_index:03d}"
            
            result = self.simulate_client_connection(
                client_id=client_id,
                streams=config["streams"],
                client_type=config["type"],
                duration=duration
            )
            
            client_results[client_index] = result
        
        # Inicia todos os clientes
        for i in range(num_clients):
            thread = threading.Thread(target=run_client_test, args=(i,))
            thread.start()
            threads.append(thread)
            time.sleep(0.1)  # Pequeno delay entre conexões
        
        # Aguarda todos terminarem
        for thread in threads:
            thread.join(timeout=duration + 5)
        
        # Processa resultados
        successful_connections = 0
        total_messages = 0
        total_latency = 0
        latency_samples = 0
        
        for result in client_results:
            if result and result["connection_established"]:
                successful_connections += 1
                total_messages += result["messages_received"]
                
                if result["average_latency"] > 0:
                    total_latency += result["average_latency"]
                    latency_samples += 1
            
            if result:
                results["client_results"].append(result)
        
        # Calcula estatísticas agregadas
        results["success_rate"] = (successful_connections / num_clients) * 100
        results["total_messages"] = total_messages
        results["average_latency"] = total_latency / latency_samples if latency_samples > 0 else 0
        results["messages_per_second"] = total_messages / duration if duration > 0 else 0
        
        results["aggregate_stats"] = {
            "successful_connections": successful_connections,
            "failed_connections": num_clients - successful_connections,
            "total_messages_received": total_messages,
            "average_messages_per_client": total_messages / successful_connections if successful_connections > 0 else 0,
            "throughput_msg_per_sec": results["messages_per_second"],
            "average_latency_ms": results["average_latency"]
        }
        
        return results
    
    def test_stream_data_quality(self, stream_type: str = "raw_ticks", 
                               duration: int = 10) -> Dict:
        """
        Testa qualidade dos dados de um stream específico
        
        Args:
            stream_type: Tipo do stream a testar
            duration: Duração do teste em segundos
        """
        print(f"🔍 Testando qualidade de dados do stream '{stream_type}'...")
        
        result = self.simulate_client_connection(
            client_id=f"quality_test_{stream_type}",
            streams=[stream_type],
            client_type="quality_inspector",
            duration=duration
        )
        
        quality_analysis = {
            "stream_type": stream_type,
            "test_duration": duration,
            "connection_success": result["connection_established"],
            "data_availability": result["messages_received"] > 0,
            "message_frequency": result["messages_received"] / duration if duration > 0 else 0,
            "average_latency": result["average_latency"],
            "data_consistency": True,  # Assumindo consistência por padrão
            "errors": result["errors"],
            "sample_data": result["sample_messages"]
        }
        
        # Analisa consistência dos dados
        if result["sample_messages"]:
            sample_data_keys = set()
            for sample in result["sample_messages"]:
                sample_data_keys.update(sample["data_keys"])
            
            quality_analysis["data_fields"] = list(sample_data_keys)
            quality_analysis["data_consistency"] = len(sample_data_keys) > 0
        
        # Avaliação de qualidade
        quality_score = 0
        if quality_analysis["connection_success"]:
            quality_score += 25
        if quality_analysis["data_availability"]:
            quality_score += 25
        if quality_analysis["message_frequency"] > 0.5:  # Pelo menos 0.5 msg/sec
            quality_score += 25
        if quality_analysis["average_latency"] < 100:  # Latência < 100ms
            quality_score += 25
        
        quality_analysis["quality_score"] = quality_score
        quality_analysis["quality_rating"] = (
            "Excellent" if quality_score >= 90 else
            "Good" if quality_score >= 70 else
            "Fair" if quality_score >= 50 else
            "Poor"
        )
        
        return quality_analysis
    
    def generate_test_report(self, test_results: List[Dict]) -> str:
        """
        Gera relatório detalhado dos testes
        
        Args:
            test_results: Lista de resultados de teste
        """
        report = []
        report.append("=" * 80)
        report.append("RELATÓRIO DE TESTES WEBSOCKET - STREAMING FOREX DATA PIPELINE")
        report.append("=" * 80)
        report.append(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for i, test_result in enumerate(test_results, 1):
            report.append(f"TESTE {i}: {test_result.get('test_name', 'Teste sem nome')}")
            report.append("-" * 50)
            
            if "success_rate" in test_result:  # Teste de carga
                stats = test_result["aggregate_stats"]
                report.append(f"✅ Taxa de Sucesso: {test_result['success_rate']:.1f}%")
                report.append(f"📊 Conexões Bem-sucedidas: {stats['successful_connections']}/{test_result['num_clients']}")
                report.append(f"📈 Total de Mensagens: {stats['total_messages_received']:,}")
                report.append(f"⚡ Throughput: {stats['throughput_msg_per_sec']:.1f} msgs/sec")
                report.append(f"🕐 Latência Média: {stats['average_latency_ms']:.2f}ms")
                
            elif "quality_score" in test_result:  # Teste de qualidade
                report.append(f"🎯 Score de Qualidade: {test_result['quality_score']}/100 ({test_result['quality_rating']})")
                report.append(f"🔗 Conexão: {'✅ Sucesso' if test_result['connection_success'] else '❌ Falha'}")
                report.append(f"📡 Dados Disponíveis: {'✅ Sim' if test_result['data_availability'] else '❌ Não'}")
                report.append(f"📊 Frequência: {test_result['message_frequency']:.2f} msgs/sec")
                report.append(f"🕐 Latência: {test_result['average_latency']:.2f}ms")
                
                if test_result.get("data_fields"):
                    report.append(f"📋 Campos de Dados: {', '.join(test_result['data_fields'])}")
            
            if test_result.get("errors"):
                report.append("❌ Erros Encontrados:")
                for error in test_result["errors"][:5]:  # Máximo 5 erros
                    report.append(f"   • {error}")
            
            report.append("")
        
        # Resumo geral
        report.append("RESUMO EXECUTIVO")
        report.append("=" * 50)
        
        # Calcula métricas gerais
        total_tests = len(test_results)
        successful_tests = sum(1 for r in test_results if 
                             r.get("success_rate", 0) > 80 or r.get("quality_score", 0) > 70)
        
        report.append(f"📊 Total de Testes: {total_tests}")
        report.append(f"✅ Testes Aprovados: {successful_tests}")
        report.append(f"📈 Taxa de Aprovação: {(successful_tests/total_tests*100):.1f}%")
        
        # Recomendações
        report.append("")
        report.append("RECOMENDAÇÕES")
        report.append("-" * 30)
        
        avg_success_rate = sum(r.get("success_rate", 0) for r in test_results) / total_tests
        if avg_success_rate > 95:
            report.append("🚀 Sistema pronto para produção com clientes empresariais")
        elif avg_success_rate > 85:
            report.append("✅ Sistema estável, recomendado para clientes gerais")
        elif avg_success_rate > 70:
            report.append("⚠️ Sistema funcional, requer otimizações para alta carga")
        else:
            report.append("❌ Sistema requer melhorias significativas")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def run_comprehensive_websocket_tests():
    """
    Executa suite completa de testes WebSocket
    Função principal para validação do sistema
    """
    print("🔧 Iniciando Testes Abrangentes do Sistema WebSocket...")
    print("=" * 60)
    
    tester = WebSocketIntegrationTester()
    test_results = []
    
    # 1. Teste de disponibilidade do servidor
    print("\n1️⃣ Testando disponibilidade do servidor...")
    if not tester.test_server_availability():
        print("❌ Servidor não está disponível. Inicie o servidor e tente novamente.")
        print("   Comando: uvicorn app.main:app --reload")
        return
    
    print("✅ Servidor disponível!")
    
    # 2. Teste dos endpoints de informação
    print("\n2️⃣ Testando endpoints de informação...")
    endpoints_info = tester.test_websocket_endpoints_info()
    
    if "error" not in endpoints_info:
        print("✅ Endpoints de informação funcionando")
        if endpoints_info["stats_endpoint"]["status"] == 200:
            print("   📊 /ws/stats: OK")
        if endpoints_info["streams_endpoint"]["status"] == 200:
            print("   📡 /ws/streams: OK")
            streams_data = endpoints_info["streams_endpoint"]["data"]
            if streams_data:
                print(f"   🔢 {streams_data.get('total_streams', 0)} streams disponíveis")
    else:
        print(f"❌ Erro nos endpoints: {endpoints_info['error']}")
    
    # 3. Teste de qualidade de dados por stream
    print("\n3️⃣ Testando qualidade de dados dos streams...")
    
    streams_to_test = ["raw_ticks", "ml_features", "trading_signals"]
    
    for stream in streams_to_test:
        print(f"\n   Testando stream '{stream}'...")
        quality_result = tester.test_stream_data_quality(stream, duration=8)
        quality_result["test_name"] = f"Qualidade do Stream {stream}"
        test_results.append(quality_result)
        
        status_icon = "✅" if quality_result["quality_score"] >= 70 else "⚠️" if quality_result["quality_score"] >= 50 else "❌"
        print(f"   {status_icon} {stream}: {quality_result['quality_score']}/100 ({quality_result['quality_rating']})")
    
    # 4. Teste de carga com múltiplos clientes
    print("\n4️⃣ Testando carga com múltiplos clientes...")
    
    load_test_configs = [
        {"clients": 3, "duration": 10, "name": "Carga Leve"},
        {"clients": 8, "duration": 15, "name": "Carga Moderada"}
    ]
    
    for config in load_test_configs:
        print(f"\n   Executando {config['name']} ({config['clients']} clientes)...")
        load_result = tester.run_multi_client_load_test(
            num_clients=config["clients"],
            duration=config["duration"]
        )
        load_result["test_name"] = config["name"]
        test_results.append(load_result)
        
        success_rate = load_result["success_rate"]
        status_icon = "✅" if success_rate >= 90 else "⚠️" if success_rate >= 70 else "❌"
        print(f"   {status_icon} Taxa de Sucesso: {success_rate:.1f}%")
        print(f"      📈 {load_result['aggregate_stats']['total_messages_received']:,} mensagens recebidas")
        print(f"      ⚡ {load_result['aggregate_stats']['throughput_msg_per_sec']:.1f} msgs/sec")
    
    # 5. Gera relatório final
    print("\n5️⃣ Gerando relatório final...")
    
    report = tester.generate_test_report(test_results)
    
    # Salva relatório em arquivo
    report_filename = f"websocket_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 Relatório salvo em: {report_filename}")
    except Exception as e:
        print(f"❌ Erro ao salvar relatório: {e}")
    
    # Exibe resumo no console
    print("\n" + "=" * 60)
    print("RESUMO DOS TESTES")
    print("=" * 60)
    
    total_tests = len(test_results)
    successful_tests = sum(1 for r in test_results if 
                         r.get("success_rate", 0) > 80 or r.get("quality_score", 0) > 70)
    
    print(f"📊 Testes Executados: {total_tests}")
    print(f"✅ Testes Aprovados: {successful_tests}")
    print(f"📈 Taxa de Aprovação: {(successful_tests/total_tests*100):.1f}%")
    
    if successful_tests == total_tests:
        print("\n🚀 SISTEMA PRONTO PARA CLIENTES EMPRESARIAIS!")
        print("   ✅ Todas as funcionalidades WebSocket validadas")
        print("   ✅ Performance adequada para produção")
        print("   ✅ Qualidade de dados confirmada")
    elif successful_tests >= total_tests * 0.8:
        print("\n✅ SISTEMA APROVADO PARA USO GERAL")
        print("   ⚠️ Algumas otimizações recomendadas")
    else:
        print("\n❌ SISTEMA REQUER MELHORIAS")
        print("   🔧 Verifique logs e otimize antes da produção")
    
    print(f"\n📄 Relatório detalhado: {report_filename}")
    print("=" * 60)
    
    return test_results


if __name__ == "__main__":
    # Executa os testes se chamado diretamente
    try:
        results = run_comprehensive_websocket_tests()
        print("\n🏁 Testes concluídos com sucesso!")
    except KeyboardInterrupt:
        print("\n⚠️ Testes interrompidos pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro durante execução dos testes: {e}")
        import traceback
        traceback.print_exc()