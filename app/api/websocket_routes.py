"""
WebSocket API Endpoints - Real-time Streaming Interface
Multiple channels for AI trading system integration
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse
from typing import List, Optional, Dict, Any
import json
import asyncio
import logging

from app.services.websocket_manager import websocket_manager, StreamType
from app.core.config import settings

router = APIRouter(prefix="/ws", tags=["WebSocket Streaming"])
logger = logging.getLogger(__name__)


@router.websocket("/stream")
async def websocket_stream_endpoint(
    websocket: WebSocket,
    streams: str = Query(..., description="Comma-separated list of streams: raw_ticks,ml_features,trading_signals"),
    client_type: str = Query("generic", description="Client type: ai_client, trader, analyst"),
    symbol: str = Query("EURUSD", description="Trading symbol"),
    client_name: Optional[str] = Query(None, description="Optional client identifier")
):
    """
    **Real-time WebSocket streaming endpoint**
    
    **Available Streams:**
    - `raw_ticks`: High-frequency tick data (10Hz)
    - `ml_features`: ML-ready feature vectors (1Hz)
    - `trading_signals`: Trading signals with entry/exit (2Hz)
    - `pattern_alerts`: Chart pattern detection alerts
    - `technical_analysis`: Technical indicators and analysis
    - `order_book`: Level 2 order book data (5Hz)
    - `microstructure`: Market microstructure metrics
    - `economic_events`: Economic calendar events
    
    **Usage Example:**
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/ws/stream?streams=raw_ticks,ml_features&client_type=ai_client');
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(`${data.stream_type}:`, data);
    };
    ```
    """
    
    # Parse stream types
    stream_list = [s.strip() for s in streams.split(",")]
    
    # Validate streams
    valid_streams = []
    for stream_str in stream_list:
        try:
            StreamType(stream_str)
            valid_streams.append(stream_str)
        except ValueError:
            logger.warning(f"Invalid stream type: {stream_str}")
    
    if not valid_streams:
        await websocket.close(code=1003, reason="No valid streams specified")
        return
    
    # Client information
    client_info = {
        "client_type": client_type,
        "symbol": symbol,
        "client_name": client_name,
        "is_ai_client": client_type == "ai_client"
    }
    
    try:
        # Connect client
        client_id = await websocket_manager.connect_client(
            websocket=websocket,
            stream_types=valid_streams,
            client_info=client_info
        )
        
        logger.info(f"WebSocket client {client_id} connected: {client_info}")
        
        # Send welcome message
        welcome_message = {
            "type": "connection_established",
            "client_id": client_id,
            "subscribed_streams": valid_streams,
            "server_time": asyncio.get_event_loop().time(),
            "message": f"Connected to {len(valid_streams)} streams"
        }
        await websocket.send_text(json.dumps(welcome_message))
        
        # Keep connection alive and handle ping/pong
        try:
            while True:
                # Wait for client messages (ping, subscription changes, etc.)
                try:
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    await handle_client_message(client_id, message)
                except asyncio.TimeoutError:
                    # Send ping to check if client is alive
                    ping_message = {
                        "type": "ping",
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    await websocket.send_text(json.dumps(ping_message))
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket client {client_id} disconnected normally")
        except Exception as e:
            logger.error(f"WebSocket error for client {client_id}: {e}")
        
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        try:
            await websocket.close(code=1011, reason=f"Server error: {str(e)}")
        except:
            pass
    
    finally:
        # Cleanup
        try:
            await websocket_manager.disconnect_client(client_id)
        except:
            pass


async def handle_client_message(client_id: str, message: str):
    """Handle incoming client messages"""
    try:
        data = json.loads(message)
        msg_type = data.get("type")
        
        if msg_type == "pong":
            # Update client ping time
            if client_id in websocket_manager.connections:
                websocket_manager.connections[client_id].last_ping = asyncio.get_event_loop().time()
        
        elif msg_type == "subscribe":
            # Handle subscription changes (future feature)
            logger.info(f"Client {client_id} subscription change request: {data}")
        
        elif msg_type == "custom_request":
            # Handle custom data requests
            response = await handle_custom_request(client_id, data)
            if response:
                await websocket_manager.send_custom_message(client_id, response)
        
        else:
            logger.debug(f"Unknown message type from client {client_id}: {msg_type}")
            
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON from client {client_id}: {message}")
    except Exception as e:
        logger.error(f"Error handling message from client {client_id}: {e}")


async def handle_custom_request(client_id: str, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle custom data requests"""
    request_type = request.get("request")
    
    if request_type == "historical_data":
        # Request for historical data
        symbol = request.get("symbol", "EURUSD")
        limit = min(request.get("limit", 100), 1000)  # Max 1000 points
        
        # Get historical tick data
        try:
            from app.services.premium_data_provider import PremiumDataProvider
            provider = PremiumDataProvider()
            ticks = await provider.get_tick_data(symbol, limit=limit)
            
            return {
                "type": "historical_data_response",
                "request_id": request.get("request_id"),
                "symbol": symbol,
                "data": [
                    {
                        "timestamp": tick.timestamp.isoformat(),
                        "bid": tick.bid,
                        "ask": tick.ask,
                        "last": tick.last,
                        "volume": tick.volume
                    }
                    for tick in ticks
                ],
                "count": len(ticks)
            }
        except Exception as e:
            return {
                "type": "error_response",
                "request_id": request.get("request_id"),
                "error": str(e)
            }
    
    elif request_type == "stream_stats":
        # Request for streaming statistics
        stats = await websocket_manager.get_stream_stats()
        return {
            "type": "stream_stats_response",
            "request_id": request.get("request_id"),
            **stats
        }
    
    return None


@router.get("/stats", summary="Get streaming statistics")
async def get_streaming_stats():
    """
    **Get real-time streaming statistics**
    
    Returns detailed information about:
    - Active connections and their streams
    - Message throughput and latency
    - Stream buffer status
    - AI client connections
    """
    try:
        stats = await websocket_manager.get_stream_stats()
        return {
            "status": "success",
            "stats": stats,
            "server_info": {
                "supported_streams": [stream.value for stream in StreamType],
                "max_connections": getattr(settings, 'MAX_WEBSOCKET_CONNECTIONS', 100),
                "version": "1.0.0"
            }
        }
    except Exception as e:
        logger.error(f"Error getting streaming stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/streams", summary="List available streams")
async def list_available_streams():
    """
    **List all available streaming channels**
    
    Returns information about each stream type including:
    - Stream purpose and data format
    - Update frequency
    - Recommended use cases
    """
    stream_info = {
        "raw_ticks": {
            "description": "High-frequency tick data with bid/ask/last prices",
            "frequency": "100ms (10Hz)",
            "use_case": "Scalping algorithms, latency arbitrage",
            "data_size": "~50KB/hour",
            "ai_suitability": "High - Real-time execution"
        },
        "ml_features": {
            "description": "Pre-calculated ML feature vectors and normalized data",
            "frequency": "1000ms (1Hz)",
            "use_case": "Machine learning model input, backtesting",
            "data_size": "~20KB/hour",
            "ai_suitability": "Very High - Ready for model inference"
        },
        "trading_signals": {
            "description": "Trading signals with entry/exit points and risk management",
            "frequency": "500ms (2Hz)",
            "use_case": "Automated trading systems, signal following",
            "data_size": "~15KB/hour",
            "ai_suitability": "High - Decision support"
        },
        "pattern_alerts": {
            "description": "Chart pattern recognition and technical alerts",
            "frequency": "2000ms (0.5Hz)",
            "use_case": "Pattern-based strategies, alert systems",
            "data_size": "~5KB/hour",
            "ai_suitability": "Medium - Pattern analysis"
        },
        "technical_analysis": {
            "description": "Technical indicators, support/resistance levels",
            "frequency": "1000ms (1Hz)",
            "use_case": "Technical analysis, trend following",
            "data_size": "~10KB/hour",
            "ai_suitability": "High - Technical signals"
        },
        "order_book": {
            "description": "Level 2 order book data with bid/ask depth",
            "frequency": "200ms (5Hz)",
            "use_case": "Market making, liquidity analysis",
            "data_size": "~100KB/hour",
            "ai_suitability": "Very High - Market microstructure"
        },
        "microstructure": {
            "description": "Market microstructure analysis and liquidity metrics",
            "frequency": "5000ms (0.2Hz)",
            "use_case": "Advanced trading algorithms, market analysis",
            "data_size": "~8KB/hour",
            "ai_suitability": "High - Deep market insights"
        },
        "economic_events": {
            "description": "Economic calendar events and impact analysis",
            "frequency": "10000ms (0.1Hz)",
            "use_case": "News trading, fundamental analysis",
            "data_size": "~3KB/hour",
            "ai_suitability": "Medium - Event-driven trading"
        }
    }
    
    return {
        "available_streams": stream_info,
        "total_streams": len(stream_info),
        "recommended_combinations": {
            "ai_scalping": ["raw_ticks", "ml_features", "order_book"],
            "ai_swing_trading": ["ml_features", "trading_signals", "technical_analysis"],
            "ai_news_trading": ["economic_events", "pattern_alerts", "trading_signals"],
            "ai_market_making": ["order_book", "microstructure", "raw_ticks"]
        }
    }


@router.get("/test-client", response_class=HTMLResponse, summary="WebSocket test client")
async def websocket_test_client():
    """
    **Interactive WebSocket test client**
    
    HTML page for testing WebSocket connections and viewing real-time data streams.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FOREX Data Pipeline - WebSocket Test Client</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
            .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
            .control-group { background: #ecf0f1; padding: 15px; border-radius: 5px; }
            .control-group label { display: block; margin-bottom: 5px; font-weight: bold; color: #34495e; }
            .control-group input, .control-group select { width: 100%; padding: 8px; border: 1px solid #bdc3c7; border-radius: 4px; }
            .button { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
            .button:hover { background: #2980b9; }
            .button.disconnect { background: #e74c3c; }
            .button.disconnect:hover { background: #c0392b; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; font-weight: bold; }
            .status.connected { background: #d5f4e6; color: #27ae60; }
            .status.disconnected { background: #fadbd8; color: #e74c3c; }
            .messages { border: 1px solid #bdc3c7; height: 400px; overflow-y: auto; padding: 10px; background: #fafafa; font-family: monospace; font-size: 12px; }
            .stream-data { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-top: 20px; }
            .stream-box { background: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }
            .stream-box h4 { margin: 0 0 10px 0; color: #2c3e50; }
            .stream-box .data { font-family: monospace; font-size: 11px; background: white; padding: 8px; border-radius: 3px; max-height: 150px; overflow-y: auto; }
            .stats { background: #e8f5e8; padding: 15px; border-radius: 5px; margin-top: 20px; }
            .checkbox-group { display: flex; flex-wrap: wrap; gap: 10px; }
            .checkbox-group label { display: flex; align-items: center; gap: 5px; background: white; padding: 5px 10px; border-radius: 3px; border: 1px solid #bdc3c7; cursor: pointer; }
            .checkbox-group input[type="checkbox"] { margin: 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Data Provider Elite - WebSocket Test Client</h1>
                <p>Real-time streaming interface for AI trading systems</p>
            </div>
            
            <div class="controls">
                <div class="control-group">
                    <label>WebSocket URL:</label>
                    <input type="text" id="wsUrl" value="ws://localhost:8000/ws/stream">
                    
                    <label>Client Type:</label>
                    <select id="clientType">
                        <option value="ai_client">AI Trading Client</option>
                        <option value="trader">Human Trader</option>
                        <option value="analyst">Market Analyst</option>
                        <option value="generic">Generic Client</option>
                    </select>
                    
                    <label>Symbol:</label>
                    <input type="text" id="symbol" value="EURUSD">
                    
                    <label>Client Name:</label>
                    <input type="text" id="clientName" placeholder="Optional identifier">
                </div>
                
                <div class="control-group">
                    <label>Select Streams:</label>
                    <div class="checkbox-group">
                        <label><input type="checkbox" value="raw_ticks" checked> Raw Ticks</label>
                        <label><input type="checkbox" value="ml_features" checked> ML Features</label>
                        <label><input type="checkbox" value="trading_signals"> Trading Signals</label>
                        <label><input type="checkbox" value="pattern_alerts"> Pattern Alerts</label>
                        <label><input type="checkbox" value="technical_analysis"> Technical Analysis</label>
                        <label><input type="checkbox" value="order_book"> Order Book</label>
                        <label><input type="checkbox" value="microstructure"> Microstructure</label>
                        <label><input type="checkbox" value="economic_events"> Economic Events</label>
                    </div>
                </div>
            </div>
            
            <div>
                <button class="button" onclick="connect()">Connect</button>
                <button class="button disconnect" onclick="disconnect()">Disconnect</button>
                <button class="button" onclick="clearMessages()">Clear Messages</button>
                <button class="button" onclick="requestStats()">Get Stats</button>
            </div>
            
            <div id="status" class="status disconnected">Disconnected</div>
            
            <div id="stats" class="stats" style="display: none;">
                <h3>Connection Statistics</h3>
                <div id="statsContent"></div>
            </div>
            
            <div class="stream-data" id="streamData"></div>
            
            <div>
                <h3>Raw Messages</h3>
                <div id="messages" class="messages"></div>
            </div>
        </div>
        
        <script>
            let ws = null;
            let messageCount = 0;
            let streamData = {};
            let connectionTime = null;
            
            function getSelectedStreams() {
                const checkboxes = document.querySelectorAll('.checkbox-group input[type="checkbox"]:checked');
                return Array.from(checkboxes).map(cb => cb.value);
            }
            
            function connect() {
                if (ws) {
                    disconnect();
                }
                
                const selectedStreams = getSelectedStreams();
                if (selectedStreams.length === 0) {
                    alert('Please select at least one stream');
                    return;
                }
                
                const wsUrl = document.getElementById('wsUrl').value;
                const clientType = document.getElementById('clientType').value;
                const symbol = document.getElementById('symbol').value;
                const clientName = document.getElementById('clientName').value;
                
                const params = new URLSearchParams({
                    streams: selectedStreams.join(','),
                    client_type: clientType,
                    symbol: symbol
                });
                
                if (clientName) {
                    params.append('client_name', clientName);
                }
                
                const fullUrl = `${wsUrl}?${params.toString()}`;
                
                try {
                    ws = new WebSocket(fullUrl);
                    connectionTime = Date.now();
                    
                    ws.onopen = function(event) {
                        document.getElementById('status').className = 'status connected';
                        document.getElementById('status').textContent = `Connected to ${selectedStreams.length} streams`;
                        addMessage('Connected to WebSocket', 'info');
                    };
                    
                    ws.onmessage = function(event) {
                        messageCount++;
                        const data = JSON.parse(event.data);
                        handleMessage(data);
                    };
                    
                    ws.onclose = function(event) {
                        document.getElementById('status').className = 'status disconnected';
                        document.getElementById('status').textContent = `Disconnected (${event.code}: ${event.reason})`;
                        addMessage(`WebSocket closed: ${event.code} - ${event.reason}`, 'error');
                        ws = null;
                    };
                    
                    ws.onerror = function(error) {
                        addMessage(`WebSocket error: ${error}`, 'error');
                    };
                    
                } catch (error) {
                    addMessage(`Connection error: ${error}`, 'error');
                }
            }
            
            function disconnect() {
                if (ws) {
                    ws.close();
                    ws = null;
                }
            }
            
            function handleMessage(data) {
                addMessage(JSON.stringify(data, null, 2), 'data');
                
                if (data.stream_type) {
                    // Update stream-specific data display
                    streamData[data.stream_type] = data;
                    updateStreamDisplay();
                } else if (data.type === 'stream_stats_response') {
                    updateStatsDisplay(data);
                }
            }
            
            function updateStreamDisplay() {
                const container = document.getElementById('streamData');
                container.innerHTML = '';
                
                for (const [streamType, data] of Object.entries(streamData)) {
                    const box = document.createElement('div');
                    box.className = 'stream-box';
                    
                    const title = document.createElement('h4');
                    title.textContent = `${streamType.replace('_', ' ').toUpperCase()} (Seq: ${data.sequence})`;
                    
                    const content = document.createElement('div');
                    content.className = 'data';
                    content.textContent = JSON.stringify(data.data, null, 2);
                    
                    box.appendChild(title);
                    box.appendChild(content);
                    container.appendChild(box);
                }
            }
            
            function addMessage(message, type) {
                const messages = document.getElementById('messages');
                const timestamp = new Date().toLocaleTimeString();
                const div = document.createElement('div');
                div.style.marginBottom = '5px';
                div.style.color = type === 'error' ? '#e74c3c' : type === 'info' ? '#3498db' : '#2c3e50';
                div.innerHTML = `<strong>[${timestamp}]</strong> ${message}`;
                messages.appendChild(div);
                messages.scrollTop = messages.scrollHeight;
            }
            
            function clearMessages() {
                document.getElementById('messages').innerHTML = '';
                document.getElementById('streamData').innerHTML = '';
                streamData = {};
                messageCount = 0;
            }
            
            function requestStats() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    const request = {
                        type: 'custom_request',
                        request: 'stream_stats',
                        request_id: Date.now()
                    };
                    ws.send(JSON.stringify(request));
                } else {
                    alert('WebSocket not connected');
                }
            }
            
            function updateStatsDisplay(statsData) {
                const statsDiv = document.getElementById('stats');
                const contentDiv = document.getElementById('statsContent');
                
                contentDiv.innerHTML = `
                    <p><strong>Total Connections:</strong> ${statsData.total_connections}</p>
                    <p><strong>Active Streams:</strong> ${statsData.active_streams.join(', ')}</p>
                    <p><strong>Total Messages:</strong> ${statsData.total_messages_sent}</p>
                    <p><strong>AI Clients:</strong> ${statsData.ai_clients}</p>
                    <p><strong>Your Messages:</strong> ${messageCount}</p>
                    <p><strong>Connection Duration:</strong> ${connectionTime ? Math.round((Date.now() - connectionTime) / 1000) : 0}s</p>
                `;
                
                statsDiv.style.display = 'block';
            }
            
            // Auto-scroll messages
            setInterval(() => {
                const messages = document.getElementById('messages');
                if (messages.scrollHeight - messages.scrollTop > messages.clientHeight + 100) {
                    // Only auto-scroll if user is near the bottom
                    return;
                }
                messages.scrollTop = messages.scrollHeight;
            }, 1000);
        </script>
    </body>
    </html>
    """
    return html_content


@router.post("/broadcast", summary="Broadcast custom message")
async def broadcast_custom_message(
    stream_type: str,
    message: Dict[str, Any],
    client_filter: Optional[str] = None
):
    """
    **Broadcast custom message to connected clients**
    
    Useful for:
    - Emergency notifications
    - System maintenance alerts
    - Custom trading alerts
    - Market status updates
    """
    try:
        # Validate stream type
        try:
            stream_enum = StreamType(stream_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid stream type: {stream_type}")
        
        # Create broadcast message
        broadcast_msg = {
            "type": "custom_broadcast",
            "stream_type": stream_type,
            "timestamp": asyncio.get_event_loop().time(),
            "message": message,
            "source": "api_broadcast"
        }
        
        # Count affected clients
        affected_clients = 0
        for client_id, connection in websocket_manager.connections.items():
            if stream_enum in connection.subscribed_streams:
                if not client_filter or client_filter in str(connection.client_info):
                    success = await websocket_manager.send_custom_message(client_id, broadcast_msg)
                    if success:
                        affected_clients += 1
        
        return {
            "status": "success",
            "message": "Broadcast sent",
            "affected_clients": affected_clients,
            "stream_type": stream_type
        }
        
    except Exception as e:
        logger.error(f"Error broadcasting message: {e}")
        raise HTTPException(status_code=500, detail=str(e))