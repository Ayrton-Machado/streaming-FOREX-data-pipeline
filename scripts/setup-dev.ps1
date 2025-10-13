# FOREX Pipeline - Development Setup Script (PowerShell)
# Etapa 4: Script para inicialização completa do ambiente de desenvolvimento no Windows

param(
    [switch]$SkipJupyter = $false,
    [switch]$Verbose = $false
)

# Função para log colorido
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    switch ($Level) {
        "ERROR" { Write-Host "[$timestamp] ERROR: $Message" -ForegroundColor Red }
        "WARN"  { Write-Host "[$timestamp] WARN: $Message" -ForegroundColor Yellow }
        "INFO"  { Write-Host "[$timestamp] INFO: $Message" -ForegroundColor Cyan }
        default { Write-Host "[$timestamp] $Message" -ForegroundColor Green }
    }
}

# Verifica se Docker está instalado
function Test-Docker {
    Write-Log "Verificando Docker..."
    
    try {
        $dockerVersion = docker --version
        $composeVersion = docker-compose --version
        Write-Log "Docker instalado: $dockerVersion"
        Write-Log "Docker Compose instalado: $composeVersion"
        return $true
    }
    catch {
        Write-Log "Docker ou Docker Compose não está instalado" -Level "ERROR"
        return $false
    }
}

# Cria diretórios necessários
function New-Directories {
    Write-Log "Criando diretórios necessários..."
    
    $directories = @(
        "data\timescaledb",
        "data\redis", 
        "data\grafana",
        "data\prometheus",
        "logs",
        "notebooks",
        "docker\ssl"
    )
    
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            if ($Verbose) { Write-Log "Criado: $dir" -Level "INFO" }
        }
    }
    
    Write-Log "Diretórios criados ✓"
}

# Configura environment
function Set-Environment {
    Write-Log "Configurando environment..."
    
    if (!(Test-Path ".env")) {
        if (Test-Path ".env.docker") {
            Copy-Item ".env.docker" ".env"
            Write-Log "Arquivo .env criado a partir de .env.docker ✓"
        }
        else {
            Write-Log "Arquivo .env.docker não encontrado" -Level "WARN"
            Write-Log "Criando .env mínimo..." -Level "WARN"
            
            $envContent = @"
POSTGRES_DB=forex_data
POSTGRES_USER=forex_user
POSTGRES_PASSWORD=forex_password_2024
ENVIRONMENT=development
DEBUG=true
"@
            $envContent | Out-File -FilePath ".env" -Encoding UTF8
        }
    } else {
        Write-Log ".env ja existe, mantendo configuracao atual" -Level "INFO"
    }
}

# Build das imagens
function Build-Images {
    Write-Log "Building Docker images..."
    
    try {
        & docker-compose build --no-cache
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Images built successfully ✓"
        }
        else {
            throw "Build failed"
        }
    }
    catch {
        Write-Log "Erro no build das images: $_" -Level "ERROR"
        return $false
    }
    return $true
}

# Inicia serviços
function Start-Services {
    Write-Log "Iniciando serviços..."
    
    # Inicia TimescaleDB primeiro
    Write-Log "Iniciando TimescaleDB..." -Level "INFO"
    & docker-compose up -d timescaledb
    
    # Aguarda TimescaleDB ficar pronto
    Write-Log "Aguardando TimescaleDB ficar pronto..." -Level "INFO"
    $timeout = 60
    $counter = 0
    
    do {
        Start-Sleep 1
        $counter++
        Write-Host "." -NoNewline
        
        if ($counter -eq $timeout) {
            Write-Host ""
            Write-Log "TimescaleDB não ficou pronto em ${timeout}s" -Level "ERROR"
            return $false
        }
        
        $ready = & docker-compose exec -T timescaledb pg_isready -U forex_user -d forex_data 2>$null
    } while ($LASTEXITCODE -ne 0)
    
    Write-Host ""
    Write-Log "TimescaleDB está pronto ✓"
    
    # Inicia Redis
    Write-Log "Iniciando Redis..." -Level "INFO"
    & docker-compose up -d redis
    Start-Sleep 5
    
    # Inicia API
    Write-Log "Iniciando FOREX API..." -Level "INFO"
    & docker-compose up -d forex-api
    
    # Inicia monitoramento
    Write-Log "Iniciando serviços de monitoramento..." -Level "INFO"
    & docker-compose up -d grafana prometheus
    
    # Inicia Nginx
    Write-Log "Iniciando Nginx..." -Level "INFO"
    & docker-compose up -d nginx
    
    # Inicia Jupyter (opcional)
    if (!$SkipJupyter) {
        Write-Log "Iniciando Jupyter Lab..." -Level "INFO"
        & docker-compose up -d jupyter
    }
    
    Write-Log "Todos os serviços iniciados ✓"
    return $true
}

# Verifica saúde dos serviços
function Test-Health {
    Write-Log "Verificando saúde dos serviços..."
    
    $services = @("timescaledb", "redis", "forex-api")
    
    foreach ($service in $services) {
        Write-Log "Verificando $service..." -Level "INFO"
        
        $timeout = 30
        $counter = 0
        
        do {
            Start-Sleep 1
            $counter++
            
            if ($counter -eq $timeout) {
                Write-Log "$service não ficou healthy em ${timeout}s" -Level "WARN"
                break
            }
            
            $containerId = & docker-compose ps -q $service
            if ($containerId) {
                $status = & docker inspect --format='{{.State.Health.Status}}' $containerId 2>$null
            }
            else {
                $status = "unknown"
            }
        } while ($status -ne "healthy")
        
        if ($status -eq "healthy") {
            Write-Log "$service esta healthy"
        } else {
            Write-Log "$service status: $status" -Level "WARN"
        }
    }
}

# Testa conectividade
function Test-Connectivity {
    Write-Log "Testando conectividade..."
    
    # Testa API
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/health" -UseBasicParsing -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Log "API está respondendo ✓"
        }
    }
    catch {
        Write-Log "API não está respondendo" -Level "WARN"
    }
    
    # Testa banco
    try {
        $result = & docker-compose exec -T timescaledb psql -U forex_user -d forex_data -c "SELECT 1;" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Database esta acessivel"
        }
    } catch {
        Write-Log "Database nao esta acessivel" -Level "WARN"
    }
}

# Mostra informações finais
function Show-Info {
    Write-Log "🎉 Setup completo!"
    Write-Host ""
    Write-Host "📊 Servicos disponiveis:" -ForegroundColor Cyan
    Write-Host "  • API:        http://localhost:8000"
    Write-Host "  • Docs:       http://localhost:8000/docs"
    Write-Host "  • Grafana:    http://localhost:3000 (admin/admin123)"
    Write-Host "  • Prometheus: http://localhost:9090"
    
    if (!$SkipJupyter) {
        Write-Host "  • Jupyter:    http://localhost:8888 (token: forex123)"
    }
    
    Write-Host ""
    Write-Host "🗄️ Database:" -ForegroundColor Cyan
    Write-Host "  • Host: localhost:5432"
    Write-Host "  • DB: forex_data"
    Write-Host "  • User: forex_user"
    Write-Host ""
    Write-Host "📝 Logs:" -ForegroundColor Cyan
    Write-Host "  docker-compose logs -f [service_name]"
    Write-Host ""
    Write-Host "  Para parar: docker-compose down"
}

# Função principal
function Main {
    Write-Log "🚀 Iniciando setup do FOREX Pipeline..."
    
    if (!(Test-Docker)) {
        exit 1
    }
    
    New-Directories
    Set-Environment
    
    if (!(Build-Images)) {
        exit 1
    }
    
    if (!(Start-Services)) {
        exit 1
    }
    
    # Aguarda serviços terminarem de subir
    Start-Sleep 10
    
    Test-Health
    Test-Connectivity
    Show-Info
}

# Executa se chamado diretamente
if ($MyInvocation.InvocationName -ne ".") {
    Main
}