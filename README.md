# 📈 Data Provider Elite - FOREX Streaming Pipeline

![Python](https://img.shields.io/badge/python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/fastapi-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![TimescaleDB](https://img.shields.io/badge/timescaledb-FDB515?style=for-the-badge&logo=timescale&logoColor=white)
![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?style=for-the-badge&logo=redis&logoColor=white)
![WebSocket](https://img.shields.io/badge/websocket-010101?style=for-the-badge&logo=socketdotio&logoColor=white)

> **Provedor de dados institucional para sistemas de trading IA** com streaming WebSocket de alta frequência, fontes premium (Alpha Vantage, Polygon.io, FXCM) e 50+ indicadores técnicos. 
> Suporta **8 canais especializados, microestrutura de mercado, ML-ready features e análise de order flow em tempo real**.

## 🚀 Características Principais

### 💎 **Dados Premium Institucionais**
- **Alpha Vantage**: Dados profissionais e indicadores econômicos
- **Polygon.io**: Tick data institucional e order book L2
- **FXCM**: Feeds de alta qualidade para trading algorítmico
- **Fallback Inteligente**: Yahoo Finance como backup automático

### ⚡ **WebSocket Streaming de Alta Performance**
- **8 Canais Especializados** com frequências otimizadas (0.1Hz - 10Hz)
- **Latência Ultra-Baixa** para scalping e arbitragem
- **ML-Ready Features** com vetores normalizados
- **Gestão Automática** de conexões e cleanup

### 🧠 **Machine Learning Ready**
- **50+ Indicadores Técnicos** pré-calculados
- **Feature Engineering** automatizado
- **Detecção de Padrões** chartistas em tempo real
- **Backtesting Engine** integrado

### 🏗️ **Arquitetura Robusta**
- **FastAPI** com documentação automática
- **TimescaleDB** otimizado para séries temporais
- **Redis** para cache de alta performance
- **Docker** com orchestração completa

## � Canais de Streaming

| Canal | Frequência | Uso Recomendado | Tamanho/Hora |
|-------|------------|-----------------|--------------|
| `raw_ticks` | 100ms (10Hz) | Scalping IA, Arbitragem | ~50KB |
| `ml_features` | 1000ms (1Hz) | Inferência ML, Backtesting | ~20KB |
| `trading_signals` | 500ms (2Hz) | Trading Automatizado | ~15KB |
| `pattern_alerts` | 2000ms (0.5Hz) | Estratégias Padrões | ~5KB |
| `technical_analysis` | 1000ms (1Hz) | Análise Técnica | ~10KB |
| `order_book` | 200ms (5Hz) | Market Making | ~100KB |
| `microstructure` | 5000ms (0.2Hz) | Análise Avançada | ~8KB |
| `economic_events` | 10000ms (0.1Hz) | News Trading | ~3KB |

## 🤖 Combinações para IA

### **AI Scalping**
```javascript
streams: ["raw_ticks", "ml_features", "order_book"]
```

### **AI Swing Trading**
```javascript
streams: ["ml_features", "trading_signals", "technical_analysis"]
```

### **AI News Trading**
```javascript
streams: ["economic_events", "pattern_alerts", "trading_signals"]
```

### **AI Market Making**
```javascript
streams: ["order_book", "microstructure", "raw_ticks"]
```

## 🔧 Desenvolvimento e Roadmap

### ✅ **Etapas Concluídas**

- [x] **ETAPA 1**: Estrutura Base - FastAPI + Docker + Configurações
- [x] **ETAPA 2**: Coleta de Dados - yfinance com validação de qualidade
- [x] **ETAPA 3**: API REST - Endpoints para dados FOREX processados
- [x] **ETAPA 4**: Banco de Dados - TimescaleDB para séries temporais
- [x] **ETAPA 5**: Feature Engineering - 50+ indicadores técnicos + backtesting
- [x] **ETAPA 6**: Dados Premium - Alpha Vantage + Polygon.io + FXCM
- [x] **ETAPA 8**: WebSocket Streaming - 8 canais especializados para IA

### 🚧 **Próximas Atualizações**

- [ ] **ETAPA 7**: Monitoramento - Health checks, métricas e observabilidade
- [ ] **ETAPA 9**: Machine Learning - Modelos de predição e detecção de padrões
- [ ] **ETAPA 10**: Deploy Produção - CI/CD e infraestrutura cloud

## � Pré-requisitos

Antes de começar, verifique se você tem:

- [Docker](https://www.docker.com/) e [Docker Compose](https://docs.docker.com/compose/)
- [Python 3.11+](https://www.python.org/) (para desenvolvimento local)
- Chaves API (opcionais):
  - [Alpha Vantage API Key](https://www.alphavantage.co/support/#api-key)
  - [Polygon.io API Key](https://polygon.io/)
  - [FXCM API Credentials](https://www.fxcm.com/uk/api/)

## 🚀 Instalação e Execução

### **Método 1: Docker (Recomendado)**

1. **Clone o repositório:**
```bash
git clone https://github.com/Ayrton-Machado/streaming-FOREX-data-pipeline
cd streaming-FOREX-data-pipeline
```

2. **Configure as variáveis de ambiente:**
```bash
cp .env.example .env
# Edite .env com suas chaves API (opcional)
```

3. **Execute com Docker:**
```bash
docker-compose up --build -d
```

4. **Verifique os serviços:**
```bash
docker-compose ps
```

### **Método 2: Desenvolvimento Local**

1. **Clone e prepare o ambiente:**
```bash
git clone https://github.com/Ayrton-Machado/streaming-FOREX-data-pipeline
cd streaming-FOREX-data-pipeline
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. **Instale dependências:**
```bash
pip install -r requirements.txt
```

3. **Configure banco de dados:**
```bash
# Execute apenas TimescaleDB e Redis via Docker
docker-compose up timescaledb redis -d
```

4. **Execute a aplicação:**
```bash
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## ☕ Como Usar

### **1. Acesse a Documentação Interativa**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### **2. Teste os Endpoints REST**

**Cotação Atual:**
```bash
curl http://localhost:8000/api/v1/quote/latest
```

**Dados Premium:**
```bash
curl http://localhost:8000/api/v1/premium/health
curl http://localhost:8000/api/v1/premium/tick-data
```

**Microestrutura de Mercado:**
```bash
curl http://localhost:8000/api/v1/premium/microstructure
```

### **3. Cliente WebSocket Interativo**
Acesse: http://localhost:8000/ws/test-client

### **4. Conexão WebSocket Programática**

**JavaScript:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream?streams=raw_ticks,ml_features&client_type=ai_client');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(`${data.stream_type}:`, data.data);
};
```

**Python:**
```python
import asyncio
import websockets
import json

async def connect_to_stream():
    uri = "ws://localhost:8000/ws/stream?streams=ml_features,trading_signals&client_type=ai_client"
    
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Stream: {data['stream_type']}, Data: {data['data']}")

asyncio.run(connect_to_stream())
```

**Node.js:**
```javascript
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:8000/ws/stream?streams=order_book,microstructure&client_type=ai_client');

ws.on('message', (data) => {
    const message = JSON.parse(data);
    console.log(`Received: ${message.stream_type}`, message.data);
});
```
```

## 📊 Endpoints Principais

### **REST API**
| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/v1/quote/latest` | GET | Cotação mais recente |
| `/api/v1/quotes` | GET | Dados históricos |
| `/api/v1/premium/health` | GET | Status fontes premium |
| `/api/v1/premium/tick-data` | GET | Dados tick institucional |
| `/api/v1/premium/order-book` | GET | Order book L2 |
| `/api/v1/premium/microstructure` | GET | Análise microestrutura |

### **WebSocket**
| Endpoint | Descrição |
|----------|-----------|
| `/ws/stream` | Conexão streaming principal |
| `/ws/stats` | Estatísticas em tempo real |
| `/ws/streams` | Canais disponíveis |
| `/ws/test-client` | Cliente de teste interativo |

## 🔧 Configuração Avançada

### **Variáveis de Ambiente (.env)**
```env
# API Keys (Opcional - usa fallback se não definidas)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
POLYGON_API_KEY=your_polygon_key
FXCM_API_KEY=your_fxcm_key

# Database
POSTGRES_DB=forex_data
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres123

# Redis
REDIS_URL=redis://localhost:6379

# Application
FOREX_PAIR=EURUSD
API_BASE_URL=http://localhost:8000
```

### **Configuração Docker**
```yaml
# docker-compose.override.yml para desenvolvimento
services:
  forex-api:
    volumes:
      - ./app:/app
    environment:
      - DEBUG=true
    ports:
      - "8001:8000"  # Port alternativo
```

## 🧪 Testes e Desenvolvimento

### **Executar Testes**
```bash
# Com pytest
pytest tests/ -v

# Com coverage
pytest tests/ --cov=app --cov-report=html
```

### **Linting e Formatação**
```bash
# Black formatter
black app/

# Ruff linter
ruff check app/
```

### **Monitoramento em Desenvolvimento**
```bash
# Logs em tempo real
docker-compose logs -f forex-api

# Métricas do sistema
curl http://localhost:8000/api/v1/health
```

## 🌟 Casos de Uso

### **1. Sistema de Trading IA**
```python
# Conecta aos feeds de alta frequência
streams = ["raw_ticks", "ml_features", "order_book"]
# Processo ML em tempo real
# Execução de trades baseada em sinais
```

### **2. Análise de Mercado**
```python
# Conecta aos feeds analíticos
streams = ["technical_analysis", "pattern_alerts", "microstructure"]
# Dashboard em tempo real
# Alertas personalizados
```

### **3. Backtesting Avançado**
```python
# Usa dados históricos + features ML
# Simula estratégias com dados reais
# Análise de performance detalhada
```

### **4. Market Making**
```python
# Conecta ao order book L2
streams = ["order_book", "microstructure", "raw_ticks"]
# Análise de liquidez em tempo real
# Gestão de spread dinâmico
```

## 🏗️ Arquitetura

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   FastAPI App   │    │   Databases     │
│                 │    │                 │    │                 │
│ • Alpha Vantage │───▶│ • REST API      │───▶│ • TimescaleDB   │
│ • Polygon.io    │    │ • WebSocket     │    │ • Redis Cache   │
│ • FXCM          │    │ • Streaming     │    │                 │
│ • Yahoo Finance │    │ • ML Features   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   AI Clients    │
                       │                 │
                       │ • Trading Bots  │
                       │ • ML Models     │
                       │ • Analytics     │
                       │ • Dashboards    │
                       └─────────────────┘
```

## 🤝 Contribuindo

1. **Fork** o projeto
2. **Crie** uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. **Push** para a branch (`git push origin feature/AmazingFeature`)
5. **Abra** um Pull Request

## 📞 Suporte

- **Issues**: [GitHub Issues](https://github.com/Ayrton-Machado/streaming-FOREX-data-pipeline/issues)
- **Documentação**: http://localhost:8000/docs
- **WebSocket Test**: http://localhost:8000/ws/test-client

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

**⚡ Desenvolvido para sistemas de trading IA de alta performance**

*Data Provider Elite - Transformando dados FOREX em oportunidades de trading inteligente.*

# Logs em tempo real
docker-compose -f docker-compose.simple.yml logs -f

# Logs de serviço específico
docker-compose -f docker-compose.simple.yml logs -f forex-api
docker-compose -f docker-compose.simple.yml logs -f timescaledb
docker-compose -f docker-compose.simple.yml logs -f redis

# Estatísticas de recursos
docker stats --no-stream

# Conectar ao database
docker-compose -f docker-compose.simple.yml exec timescaledb psql -U forex_user -d forex_data

# Conectar ao Redis
docker-compose -f docker-compose.simple.yml exec redis redis-cli
```

### 🔄 **Reinicialização e Restart**

```powershell
# Reiniciar todos os serviços
docker-compose -f docker-compose.simple.yml restart

# Reiniciar serviço específico
docker-compose -f docker-compose.simple.yml restart forex-api
docker-compose -f docker-compose.simple.yml restart timescaledb

# Rebuild específico (após mudanças no código)
docker-compose -f docker-compose.simple.yml up -d --build forex-api

# Rebuild completo
docker-compose -f docker-compose.simple.yml up -d --build --force-recreate
```

### 🛑 **Desativação e Parada**

```powershell
# Parar todos os serviços (mantém dados)
docker-compose -f docker-compose.simple.yml stop

# Parar serviço específico
docker-compose -f docker-compose.simple.yml stop forex-api

# Parar e remover containers (mantém volumes/dados)
docker-compose -f docker-compose.simple.yml down

# Parar serviços essenciais apenas
docker-compose -f docker-compose.simple.yml stop timescaledb redis forex-api
```

### 🧹 **Limpeza e Reset**

```powershell
# ⚠️ CUIDADO: Remove containers e volumes (APAGA TODOS OS DADOS)
docker-compose -f docker-compose.simple.yml down -v

# Limpeza completa do sistema Docker
docker system prune -f

# Limpeza de volumes não utilizados
docker volume prune -f

# Limpeza de imagens não utilizadas
docker image prune -f

# Reset completo (remove tudo)
docker-compose -f docker-compose.simple.yml down -v
docker system prune -a -f --volumes

# Rebuild do zero (após reset)
docker-compose -f docker-compose.simple.yml up -d --build
```

### 💾 **Backup e Restore**

```powershell
# Criar backup do database
mkdir -p backups
docker-compose -f docker-compose.simple.yml exec -T timescaledb pg_dump -U forex_user forex_data > backups/forex_backup_$(Get-Date -Format "yyyyMMdd_HHmmss").sql

# Restaurar backup
docker-compose -f docker-compose.simple.yml exec -T timescaledb psql -U forex_user -d forex_data < backups/forex_backup_YYYYMMDD_HHMMSS.sql

# Backup usando Makefile (se disponível)
make backup
make restore
```

### 🔍 **Diagnóstico e Troubleshooting**

```powershell
# Verificar portas em uso
netstat -ano | findstr :8000
netstat -ano | findstr :5432

# Matar processo específico (substitua PID)
taskkill /PID <PID> /F

# Verificar espaço em disco do Docker
docker system df

# Inspecionar container específico
docker inspect forex-api

# Executar comando dentro do container
docker-compose -f docker-compose.simple.yml exec forex-api /bin/bash

# Verificar configuração do Docker Compose
docker-compose -f docker-compose.simple.yml config

# Verificar se TimescaleDB está pronto
docker-compose -f docker-compose.simple.yml exec timescaledb pg_isready -U forex_user
```

### ⚡ **Comandos Úteis Rápidos**

```powershell
# Aliases úteis (adicionar ao seu perfil PowerShell)
function forex-status { docker-compose -f docker-compose.simple.yml ps }
function forex-logs { docker-compose -f docker-compose.simple.yml logs -f $args }
function forex-restart { docker-compose -f docker-compose.simple.yml restart $args }
function forex-stop { docker-compose -f docker-compose.simple.yml stop }
function forex-start { docker-compose -f docker-compose.simple.yml up -d timescaledb redis forex-api }
function forex-clean { docker-compose -f docker-compose.simple.yml down -v; docker system prune -f }

# Usar os aliases
forex-status
forex-logs forex-api
forex-restart forex-api
```

## 🎯 **Principais Endpoints**

### API REST
```bash
# Health check
GET /api/v1/health

# Cotação mais recente
GET /api/v1/quote/latest

# Dados históricos
GET /api/v1/quotes?period=7d&granularity=1h

# Salvar dados com preprocessamento
POST /api/v1/data/save

# Consultar dados com filtros
POST /api/v1/data/query

# Estatísticas
GET /api/v1/data/stats/EURUSD

# Análise detalhada
GET /api/v1/data/analyze/EURUSD

# === PREMIUM DATA ENDPOINTS (ETAPA 6) ===
# Health check de fontes premium
GET /api/v1/premium/health

# Dados tick-level institucionais
GET /api/v1/premium/tick-data?symbol=EURUSD&limit=1000

# Order book (Level II)
GET /api/v1/premium/order-book?symbol=EURUSD&depth=10

# Análise de microestrutura
GET /api/v1/premium/microstructure?symbol=EURUSD

# Cotação premium
GET /api/v1/premium/quote/latest?symbol=EURUSD

# Estatísticas premium
GET /api/v1/premium/stats?symbol=EURUSD&hours=24

# Fontes disponíveis
GET /api/v1/premium/sources
```

## ✅ **Confirmação de Sucesso**

### Como verificar se o projeto está funcionando 100%:

```powershell
# 1. Status dos containers (3 essenciais)
docker-compose -f docker-compose.simple.yml ps
# Esperado: forex-api, forex-redis, forex-timescaledb (todos "Up" e "healthy")

# 2. Health check da API
Invoke-WebRequest -Uri "http://localhost:8000/api/v1/health" -UseBasicParsing
# Esperado: StatusCode 200 e JSON com "status":"healthy"

# 3. Documentação Swagger
# Abrir navegador: http://localhost:8000/docs
# Esperado: Interface Swagger carregando

# 4. Teste de cotação
Invoke-WebRequest -Uri "http://localhost:8000/api/v1/quote/latest" -UseBasicParsing
# Esperado: Dados do EUR/USD em JSON

# 5. Conectividade do database
docker-compose -f docker-compose.simple.yml exec timescaledb pg_isready -U forex_user
# Esperado: "accepting connections"
```

### ✅ **Checklist de Ativação Completa**

- [ ] ✅ **3 containers rodando** (timescaledb, redis, forex-api)
- [ ] ✅ **API responde** em http://localhost:8000/api/v1/health  
- [ ] ✅ **Swagger docs** abrem em http://localhost:8000/docs
- [ ] ✅ **Database conecta** via psql
- [ ] ✅ **Redis funciona** via redis-cli
- [ ] ✅ **Endpoints respondem** (quote/latest, health)

### 🎉 **Sucesso Confirmado!**

Quando todos os itens acima estiverem funcionando, seu projeto está **100% ativo** e pronto para:

- **Desenvolvimento** das próximas etapas
- **Testes** de integração
- **Feature Engineering** (Etapa 5)
- **ML-Ready Data** (Etapa 6)
- **WebSocket Streaming** (Etapa 8)

---

**⚡ Tempo total de ativação: 5-8 minutos**  
**🔄 Próximas execuções: 1-2 minutos**  
**🚀 Pipeline FOREX totalmente funcional!**

## 📁 Estrutura do Projeto

```
streaming-forex-data-pipeline/
├── app/
│   ├── core/              # Configurações centrais
│   │   ├── config.py      # Settings (Pydantic)
│   │   └── constants.py   # Constantes
│   ├── services/          # Lógica de negócio
│   │   └── data_fetcher.py
│   ├── domain/            # Schemas Pydantic (futuro)
│   ├── api/               # Endpoints REST/WebSocket (futuro)
│   └── data/              # Acesso a dados (futuro)
├── scripts/
│   └── test_fetch.py      # Script de teste Etapa 1
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 Quick Start

### 1. Instalação

```bash
# Clone o repositório
git clone https://github.com/Ayrton-Machado/streaming-FOREX-data-pipeline.git
cd streaming-FOREX-data-pipeline

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows

# Instale dependências
pip install -r requirements.txt
```

### 2. Configuração

```bash
# Copie o template de variáveis de ambiente
cp .env.example .env

# Edite .env com suas configurações (opcional para Etapa 1)
```

### 3. Teste a Etapa 1

```bash
# Execute o script de teste
python scripts/test_fetch.py
```

Você deverá ver:
- ✅ Teste 1: Dados diários EUR/USD (últimos 7 dias)
- ✅ Teste 2: Dados horários EUR/USD (últimas 48h)
- ✅ Teste 3: Cotação mais recente
- ✅ Teste 4: Validação de inputs inválidos

## 📖 Uso Atual (Etapa 1)

### Buscar Dados Históricos

```python
from datetime import datetime, timedelta
from app.services.data_fetcher import DataFetcher

# Inicializa o fetcher
fetcher = DataFetcher()

# Define período
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

# Busca dados
df = fetcher.get_historical_data(
    start_date=start_date,
    end_date=end_date,
    granularity="1h"  # Opções: 1min, 5min, 15min, 1h, 1d
)

print(df.head())
```

### Buscar Cotação Mais Recente

```python
# Cotação atual
quote = fetcher.get_latest_quote()
print(f"EUR/USD: ${quote['close']:.5f}")
```

## 🎯 Endpoints Futuros

### REST API (Etapa 3+)

```
GET /api/v1/eurusd/quotes
    ?limit=100&granularity=1d

GET /api/v1/eurusd/ml-ready
    ?granularity=1h&start=2024-01-01&end=2025-01-01
    &normalization=minmax&include_indicators=true
```

### WebSocket (Etapa 8)

```javascript
ws://api/v1/eurusd/stream
```

## 🧪 Testes

```bash
# Executar testes
pytest

# Com coverage
pytest --cov=app tests/
```

## 📝 Configurações Principais

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `FOREX_PAIR` | `EURUSD=X` | Ticker Yahoo Finance |
| `DEFAULT_GRANULARITY` | `1h` | Granularidade padrão |
| `MAX_HISTORICAL_DAYS` | `1825` | Máx. dias históricos (~5 anos) |
| `REMOVE_OUTLIERS` | `true` | Remove outliers automaticamente |
| `DEFAULT_NORMALIZATION` | `minmax` | Método de normalização |

## 🤝 Contribuindo

Projeto em desenvolvimento incremental (filosofia SLC).  
Cada etapa é independente e validável.

## 📄 Licença

---

**⚡ Desenvolvido para sistemas de trading IA de alta performance**

*Data Provider Elite - Transformando dados FOREX em oportunidades de trading inteligente.*