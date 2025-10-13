# 🚀 GUIA COMPLETO: Como Ativar o Projeto FOREX Pipeline

## 📋 **RESUMO EXECUTIVO**

Este projeto possui **3 métodos de ativação**. Recomendo começar pelo **Método 1** (mais simples).

---

## 🎯 **MÉTODO 1: Ativação Simplificada (RECOMENDADO)**

### **✅ Verificar Pré-requisitos**
```powershell
# 1. Docker Desktop rodando
docker --version

# 2. Estar no diretório correto
cd "C:\Users\k8s\Documents\AAAAAAAAAA\PROJETOS\streaming-FOREX-data-pipeline"
```

### **🚀 Comando Único de Ativação**
```powershell
# Ativar projeto simplificado (sem Jupyter)
docker-compose -f docker-compose.simple.yml up -d --build
```

### **⏱️ Aguardar Build (5-10 minutos na primeira vez)**
- O comando vai baixar imagens Docker
- Fazer build da aplicação FastAPI
- Subir 6 serviços essenciais

### **✅ Verificar Status**
```powershell
# Ver status dos containers
docker-compose -f docker-compose.simple.yml ps

# Esperado: 6 containers "Up"
# ✅ forex-timescaledb
# ✅ forex-redis  
# ✅ forex-api
# ✅ forex-nginx
# ✅ forex-prometheus
# ✅ forex-grafana
```

### **🌐 Testar Acesso**
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000 (admin/admin123)

---

## 🎯 **MÉTODO 2: Ativação Completa (COM JUPYTER)**

Se você quiser o Jupyter Lab incluído:

```powershell
# 1. Corrigir primeiro o Dockerfile do Jupyter
# (já foi corrigido nos arquivos anteriores)

# 2. Ativar projeto completo
docker-compose up -d --build

# 3. Aguardar build completo (pode demorar mais)
```

Serviços adicionais:
- **Jupyter Lab**: http://localhost:8888 (token: forex123)

---

## 🎯 **MÉTODO 3: Desenvolvimento Local (SEM DOCKER)**

Para desenvolvimento direto em Python:

### **1. Criar Ambiente Virtual**
```powershell
# Criar venv
python -m venv venv

# Ativar (Windows)
.\venv\Scripts\Activate.ps1

# Instalar dependências
pip install -r requirements.minimal.txt
```

### **2. Configurar Database Local**
```powershell
# Subir apenas o TimescaleDB
docker run -d --name timescaledb -p 5432:5432 \
  -e POSTGRES_DB=forex_data \
  -e POSTGRES_USER=forex_user \
  -e POSTGRES_PASSWORD=forex_password_2024 \
  timescale/timescaledb:latest-pg16
```

### **3. Executar API Local**
```powershell
# Criar .env
echo "DATABASE_URL=postgresql://forex_user:forex_password_2024@localhost:5432/forex_data" > .env

# Executar API
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## ✅ **COMO VERIFICAR SE ESTÁ FUNCIONANDO**

### **1. Health Check da API**
```powershell
curl http://localhost:8000/api/v1/health
# Esperado: {"status": "healthy", ...}
```

### **2. Buscar Cotação Atual**
```powershell
curl http://localhost:8000/api/v1/quote/latest
# Esperado: dados do EUR/USD
```

### **3. Testar Database**
```powershell
# Conectar ao banco
docker-compose -f docker-compose.simple.yml exec timescaledb psql -U forex_user -d forex_data

# Verificar tabelas
\dt

# Sair
\q
```

### **4. Verificar Logs**
```powershell
# Logs de todos os serviços
docker-compose -f docker-compose.simple.yml logs -f

# Logs da API apenas
docker-compose -f docker-compose.simple.yml logs -f forex-api
```

---

## 🐛 **SOLUÇÃO DE PROBLEMAS**

### **❌ Port já em uso**
```powershell
# Verificar o que está usando a porta 8000
netstat -ano | findstr :8000

# Matar processo (substitua PID)
taskkill /PID <PID> /F

# Ou mudar porta no docker-compose.simple.yml
```

### **❌ Docker out of space**
```powershell
# Limpar Docker
docker system prune -f
docker volume prune -f
```

### **❌ Container não sobe**
```powershell
# Parar tudo
docker-compose -f docker-compose.simple.yml down

# Rebuild forçado
docker-compose -f docker-compose.simple.yml up -d --build --force-recreate
```

### **❌ Database não conecta**
```powershell
# Verificar se TimescaleDB está healthy
docker-compose -f docker-compose.simple.yml exec timescaledb pg_isready -U forex_user

# Reiniciar apenas o database
docker-compose -f docker-compose.simple.yml restart timescaledb
```

---

## 📊 **URLS DOS SERVIÇOS ATIVOS**

Após ativação bem-sucedida:

| Serviço | URL | Credenciais |
|---------|-----|-------------|
| 📊 **API FastAPI** | http://localhost:8000 | - |
| 📚 **Documentação Swagger** | http://localhost:8000/docs | - |
| 📈 **Grafana** | http://localhost:3000 | admin/admin123 |
| 📊 **Prometheus** | http://localhost:9090 | - |

**Database:**
- **Host**: localhost:5432
- **DB**: forex_data
- **User**: forex_user
- **Password**: forex_password_2024

---

## ⚡ **COMANDOS ÚTEIS**

```powershell
# Status
docker-compose -f docker-compose.simple.yml ps

# Logs
docker-compose -f docker-compose.simple.yml logs -f

# Parar
docker-compose -f docker-compose.simple.yml down

# Reiniciar
docker-compose -f docker-compose.simple.yml restart

# Limpar tudo
docker-compose -f docker-compose.simple.yml down -v
docker system prune -f
```

---

## 🎉 **CONFIRMAÇÃO DE SUCESSO**

O projeto está **100% ativo** quando:

✅ **6 containers** estão com status "Up"  
✅ **API responde** em http://localhost:8000/docs  
✅ **Grafana abre** em http://localhost:3000  
✅ **Database aceita** conexões  
✅ **Health check passa**: `curl http://localhost:8000/api/v1/health`

---

**⚡ Tempo de ativação: 5-10 minutos na primeira vez**  
**🔄 Próximas vezes: 1-2 minutos**