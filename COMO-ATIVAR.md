# 🚀 Como Ativar o Projeto FOREX Pipeline Completamente

## 📋 Pré-requisitos

### ✅ **Verificar Docker Desktop**
```powershell
# Verifique se o Docker Desktop está rodando
docker --version
docker-compose --version

# Se não estiver instalado, baixe em:
# https://www.docker.com/products/docker-desktop/
```

### ✅ **RAM Disponível**
- **Mínimo**: 4GB RAM livres
- **Recomendado**: 8GB RAM livres

## 🎯 **Método 1: Ativação Automática (RECOMENDADO)**

### **Windows (PowerShell)**
```powershell
# 1. Navegue até o projeto
cd "C:\Users\k8s\Documents\AAAAAAAAAA\PROJETOS\streaming-FOREX-data-pipeline"

# 2. Execute o setup automático
.\scripts\setup-dev.ps1

# 3. Aguarde o setup completo (pode demorar 5-10 minutos na primeira vez)
```

### **Linux/MacOS**
```bash
# 1. Navegue até o projeto
cd "/path/to/streaming-FOREX-data-pipeline"

# 2. Execute o setup automático
chmod +x scripts/setup-dev.sh
./scripts/setup-dev.sh
```

### **Usando Makefile (Alternativa)**
```bash
# Setup completo
make dev

# Ou etapa por etapa
make build    # Build das imagens
make up       # Subir serviços
make health   # Verificar saúde
```

## 🎯 **Método 2: Ativação Manual (Passo a Passo)**

### **1. Limpar Ambiente (se necessário)**
```powershell
# Parar todos os containers
docker-compose down

# Limpar containers orfãos
docker container prune -f

# Limpar volumes (CUIDADO: apaga dados)
docker volume prune -f
```

### **2. Build das Imagens**
```powershell
# Build das imagens Docker
docker-compose build --no-cache

# Ou build individual
docker-compose build forex-api
docker-compose build jupyter
```

### **3. Iniciar Serviços**
```powershell
# Subir todos os serviços
docker-compose up -d

# Verificar status
docker-compose ps
```

### **4. Verificar Logs**
```powershell
# Logs de todos os serviços
docker-compose logs -f

# Logs de serviço específico
docker-compose logs -f forex-api
docker-compose logs -f timescaledb
```

## 🔍 **Verificação de Status**

### **✅ Verificar se todos os serviços estão rodando**
```powershell
# Status dos containers
docker-compose ps

# Esperado: 7 serviços em "Up" status
# - forex-api
# - timescaledb  
# - redis
# - nginx
# - grafana
# - prometheus
# - jupyter
```

### **✅ Testar Conectividade**
```powershell
# Teste da API
curl http://localhost:8000/api/v1/health

# Ou via navegador:
# http://localhost:8000/docs
```

### **✅ Health Checks**
```powershell
# Health check automático
make health

# Ou manual
docker-compose exec forex-api curl http://localhost:8000/api/v1/health
```

## 🌐 **URLs dos Serviços**

Após a ativação completa, os seguintes serviços estarão disponíveis:

| Serviço | URL | Credenciais |
|---------|-----|-------------|
| 📊 **API FastAPI** | http://localhost:8000 | - |
| 📚 **Documentação Swagger** | http://localhost:8000/docs | - |
| 📈 **Grafana** | http://localhost:3000 | admin/admin123 |
| 📊 **Prometheus** | http://localhost:9090 | - |
| 🪐 **Jupyter Lab** | http://localhost:8888 | token: forex123 |

### **🗄️ Database Connection**
```
Host: localhost
Port: 5432
Database: forex_data
User: forex_user
Password: forex_password_2024
```

## 🧪 **Testes de Funcionalidade**

### **1. Teste da API**
```powershell
# Health check
curl http://localhost:8000/api/v1/health

# Buscar cotação atual
curl http://localhost:8000/api/v1/quote/latest

# Buscar histórico
curl "http://localhost:8000/api/v1/quotes?period=7d&granularity=1h"
```

### **2. Teste do Database**
```powershell
# Conectar ao banco
docker-compose exec timescaledb psql -U forex_user -d forex_data

# Verificar tabelas
\dt

# Sair
\q
```

### **3. Teste do Redis**
```powershell
# Conectar ao Redis
docker-compose exec redis redis-cli

# Testar
ping
# Esperado: PONG

# Sair
exit
```

## 🐛 **Solução de Problemas Comuns**

### **❌ Problema: Port já em uso**
```powershell
# Verificar processos usando as portas
netstat -ano | findstr :8000
netstat -ano | findstr :5432
netstat -ano | findstr :3000

# Matar processo específico
taskkill /PID <PID> /F
```

### **❌ Problema: Falta de memória**
```powershell
# Parar containers desnecessários
docker stop $(docker ps -q)

# Limpar sistema Docker
docker system prune -f
```

### **❌ Problema: Build falha**
```powershell
# Build sem cache
docker-compose build --no-cache

# Ou remover imagens e rebuild
docker rmi streaming-forex-data-pipeline-forex-api
docker rmi streaming-forex-data-pipeline-jupyter
docker-compose up -d --build
```

### **❌ Problema: Database não conecta**
```powershell
# Verificar se TimescaleDB está healthy
docker-compose exec timescaledb pg_isready -U forex_user

# Reiniciar apenas o database
docker-compose restart timescaledb
```

## ⚡ **Comandos Úteis**

```powershell
# Status completo
make ps

# Logs em tempo real
make logs

# Reiniciar serviços
make restart

# Backup do database
make backup

# Parar tudo
make down

# Limpar tudo
make clean
```

## 🎉 **Confirmação de Sucesso**

O projeto está **completamente ativado** quando:

✅ Todos os 7 containers estão com status "Up"  
✅ API responde em http://localhost:8000/docs  
✅ Grafana abre em http://localhost:3000  
✅ Jupyter funciona em http://localhost:8888  
✅ Database aceita conexões  
✅ Health checks passam  

## 📞 **Se Precisar de Ajuda**

1. **Logs detalhados**: `docker-compose logs -f [service_name]`
2. **Status dos containers**: `docker-compose ps`
3. **Reiniciar serviços**: `docker-compose restart [service_name]`
4. **Limpar e recomeçar**: `make clean && make dev`

---

**⚡ Tempo estimado de ativação completa: 5-10 minutos na primeira vez**