# Multi-stage Dockerfile para FOREX Pipeline
# Etapa 4: Container otimizado para desenvolvimento e produção

# Stage 1: Base Python com dependências do sistema
FROM python:3.11-slim as python-base

# Variáveis de ambiente Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Cria usuário não-root
RUN groupadd -r forex && useradd -r -g forex forex

# Stage 2: Dependências Python
FROM python-base as python-deps

# Copia requirements
COPY requirements.minimal.txt /tmp/requirements.txt

# Instala dependências Python
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Stage 3: Development
FROM python-deps as development

# Instala dependências de desenvolvimento
RUN pip install \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy \
    jupyter \
    ipython

# Diretório de trabalho
WORKDIR /app

# Copia código da aplicação
COPY --chown=forex:forex . /app/

# Cria diretórios necessários
RUN mkdir -p /app/logs /app/data && \
    chown -R forex:forex /app/logs /app/data

# Muda para usuário não-root
USER forex

# Porta da aplicação
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Comando padrão para desenvolvimento
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 4: Production
FROM python-deps as production

# Instala apenas dependências de produção
RUN pip install gunicorn

# Diretório de trabalho
WORKDIR /app

# Copia apenas arquivos necessários
COPY --chown=forex:forex app/ /app/app/
COPY --chown=forex:forex requirements.txt /app/

# Cria diretórios
RUN mkdir -p /app/logs /app/data && \
    chown -R forex:forex /app/logs /app/data

# Remove arquivos desnecessários
RUN find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -exec rm -rf {} + || true

# Muda para usuário não-root
USER forex

# Porta da aplicação
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Comando para produção
CMD ["gunicorn", "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "60", \
     "--keep-alive", "5", \
     "--max-requests", "1000", \
     "--preload"]