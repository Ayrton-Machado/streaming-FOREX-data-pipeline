#!/bin/bash

# FOREX Pipeline - Development Setup Script
# Etapa 4: Script para inicialização completa do ambiente de desenvolvimento

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para log
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Verifica se Docker está instalado
check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker não está instalado"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose não está instalado"
        exit 1
    fi

    log "Docker e Docker Compose estão instalados ✓"
}

# Cria diretórios necessários
create_directories() {
    log "Criando diretórios necessários..."
    
    mkdir -p data/timescaledb
    mkdir -p data/redis
    mkdir -p data/grafana
    mkdir -p data/prometheus
    mkdir -p logs
    mkdir -p notebooks
    mkdir -p docker/ssl
    
    log "Diretórios criados ✓"
}

# Copia arquivo de environment
setup_environment() {
    log "Configurando environment..."
    
    if [[ ! -f .env ]]; then
        if [[ -f .env.docker ]]; then
            cp .env.docker .env
            log "Arquivo .env criado a partir de .env.docker ✓"
        else
            warn "Arquivo .env.docker não encontrado"
            warn "Criando .env mínimo..."
            cat > .env << EOF
POSTGRES_DB=forex_data
POSTGRES_USER=forex_user
POSTGRES_PASSWORD=forex_password_2024
ENVIRONMENT=development
DEBUG=true
EOF
        fi
    else
        info ".env já existe, mantendo configuração atual"
    fi
}

# Build das imagens
build_images() {
    log "Building Docker images..."
    
    docker-compose build --no-cache
    
    log "Images built successfully ✓"
}

# Inicia os serviços
start_services() {
    log "Iniciando serviços..."
    
    # Inicia apenas o banco primeiro
    info "Iniciando TimescaleDB..."
    docker-compose up -d timescaledb
    
    # Aguarda TimescaleDB ficar pronto
    info "Aguardando TimescaleDB ficar pronto..."
    timeout=60
    counter=0
    
    while ! docker-compose exec -T timescaledb pg_isready -U forex_user -d forex_data > /dev/null 2>&1; do
        if [ $counter -eq $timeout ]; then
            error "TimescaleDB não ficou pronto em ${timeout}s"
            exit 1
        fi
        sleep 1
        counter=$((counter+1))
        echo -n "."
    done
    echo ""
    
    log "TimescaleDB está pronto ✓"
    
    # Inicia Redis
    info "Iniciando Redis..."
    docker-compose up -d redis
    
    # Aguarda Redis
    sleep 5
    
    # Inicia API
    info "Iniciando FOREX API..."
    docker-compose up -d forex-api
    
    # Inicia serviços de monitoramento
    info "Iniciando serviços de monitoramento..."
    docker-compose up -d grafana prometheus
    
    # Inicia Nginx
    info "Iniciando Nginx..."
    docker-compose up -d nginx
    
    # Inicia Jupyter (opcional)
    if [[ "${SKIP_JUPYTER:-false}" != "true" ]]; then
        info "Iniciando Jupyter Lab..."
        docker-compose up -d jupyter
    fi
    
    log "Todos os serviços iniciados ✓"
}

# Verifica saúde dos serviços
check_health() {
    log "Verificando saúde dos serviços..."
    
    # Lista de serviços para verificar
    services=("timescaledb" "redis" "forex-api")
    
    for service in "${services[@]}"; do
        info "Verificando $service..."
        
        # Aguarda até 30s para o serviço ficar healthy
        timeout=30
        counter=0
        
        while [[ "$(docker-compose ps -q $service | xargs docker inspect --format='{{.State.Health.Status}}')" != "healthy" ]]; do
            if [ $counter -eq $timeout ]; then
                warn "$service não ficou healthy em ${timeout}s"
                break
            fi
            sleep 1
            counter=$((counter+1))
        done
        
        status=$(docker-compose ps -q $service | xargs docker inspect --format='{{.State.Health.Status}}')
        if [[ "$status" == "healthy" ]]; then
            log "$service está healthy ✓"
        else
            warn "$service status: $status"
        fi
    done
}

# Testa conectividade
test_connectivity() {
    log "Testando conectividade..."
    
    # Testa API
    if curl -s http://localhost:8000/api/v1/health > /dev/null; then
        log "API está respondendo ✓"
    else
        warn "API não está respondendo"
    fi
    
    # Testa banco
    if docker-compose exec -T timescaledb psql -U forex_user -d forex_data -c "SELECT 1;" > /dev/null 2>&1; then
        log "Database está acessível ✓"
    else
        warn "Database não está acessível"
    fi
}

# Mostra informações finais
show_info() {
    log "🎉 Setup completo!"
    echo ""
    echo "📊 Serviços disponíveis:"
    echo "  • API:        http://localhost:8000"
    echo "  • Docs:       http://localhost:8000/docs"
    echo "  • Grafana:    http://localhost:3000 (admin/admin123)"
    echo "  • Prometheus: http://localhost:9090"
    
    if [[ "${SKIP_JUPYTER:-false}" != "true" ]]; then
        echo "  • Jupyter:    http://localhost:8888 (token: forex123)"
    fi
    
    echo ""
    echo "🗄️ Database:"
    echo "  • Host: localhost:5432"
    echo "  • DB: forex_data"
    echo "  • User: forex_user"
    echo ""
    echo "📝 Logs:"
    echo "  docker-compose logs -f [service_name]"
    echo ""
    echo "🛑 Para parar:"
    echo "  docker-compose down"
}

# Função principal
main() {
    log "🚀 Iniciando setup do FOREX Pipeline..."
    
    check_docker
    create_directories
    setup_environment
    build_images
    start_services
    
    # Aguarda um pouco para os serviços terminarem de subir
    sleep 10
    
    check_health
    test_connectivity
    show_info
}

# Executa se chamado diretamente
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi