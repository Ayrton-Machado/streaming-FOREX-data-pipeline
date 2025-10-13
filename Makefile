# FOREX Pipeline - Makefile
# Etapa 4: Comandos para gerenciamento do ambiente Docker

.PHONY: help dev prod build up down logs clean test backup restore

# Default target
help:
	@echo "🚀 FOREX Pipeline - Docker Commands"
	@echo ""
	@echo "📋 Available commands:"
	@echo "  make dev        - Start development environment"
	@echo "  make prod       - Start production environment"
	@echo "  make build      - Build all Docker images"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make logs       - Show logs for all services"
	@echo "  make clean      - Clean up Docker resources"
	@echo "  make test       - Run tests"
	@echo "  make backup     - Backup database"
	@echo "  make restore    - Restore database from backup"
	@echo ""
	@echo "📊 Service specific:"
	@echo "  make logs-api   - Show API logs"
	@echo "  make logs-db    - Show database logs"
	@echo "  make shell-api  - Shell into API container"
	@echo "  make shell-db   - Shell into database container"

# Development environment
dev:
	@echo "🔧 Starting development environment..."
	@cp .env.docker .env || true
	@docker-compose up -d
	@echo "✅ Development environment started!"
	@echo "📊 Services available at:"
	@echo "  • API: http://localhost:8000"
	@echo "  • Docs: http://localhost:8000/docs" 
	@echo "  • Grafana: http://localhost:3000"
	@echo "  • Jupyter: http://localhost:8888"

# Production environment
prod:
	@echo "🚀 Starting production environment..."
	@docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "✅ Production environment started!"

# Build all images
build:
	@echo "🔨 Building Docker images..."
	@docker-compose build --no-cache
	@echo "✅ Images built successfully!"

# Start all services
up:
	@echo "⬆️ Starting all services..."
	@docker-compose up -d

# Stop all services
down:
	@echo "⬇️ Stopping all services..."
	@docker-compose down

# Show logs
logs:
	@docker-compose logs -f

# Service specific logs
logs-api:
	@docker-compose logs -f forex-api

logs-db:
	@docker-compose logs -f timescaledb

logs-redis:
	@docker-compose logs -f redis

logs-nginx:
	@docker-compose logs -f nginx

# Shell access
shell-api:
	@docker-compose exec forex-api /bin/bash

shell-db:
	@docker-compose exec timescaledb psql -U forex_user -d forex_data

shell-redis:
	@docker-compose exec redis redis-cli

# Restart specific services
restart-api:
	@docker-compose restart forex-api

restart-db:
	@docker-compose restart timescaledb

restart-nginx:
	@docker-compose restart nginx

# Health checks
health:
	@echo "🏥 Checking service health..."
	@docker-compose ps
	@echo ""
	@echo "🔍 API Health:"
	@curl -s http://localhost:8000/api/v1/health | python -m json.tool || echo "API not responding"
	@echo ""
	@echo "🗄️ Database Health:"
	@docker-compose exec -T timescaledb pg_isready -U forex_user -d forex_data || echo "Database not ready"

# Clean up
clean:
	@echo "🧹 Cleaning up Docker resources..."
	@docker-compose down -v
	@docker system prune -f
	@docker volume prune -f
	@echo "✅ Cleanup completed!"

# Deep clean (removes everything)
clean-all:
	@echo "🗑️ Deep cleaning (REMOVES ALL DATA)..."
	@read -p "Are you sure? This will delete all data! (y/N): " confirm && [ "$$confirm" = "y" ]
	@docker-compose down -v --rmi all
	@docker system prune -af
	@docker volume prune -f
	@rm -rf data/
	@echo "✅ Deep cleanup completed!"

# Database operations
backup:
	@echo "💾 Creating database backup..."
	@mkdir -p backups
	@docker-compose exec -T timescaledb pg_dump -U forex_user forex_data > backups/forex_backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "✅ Backup created in backups/ directory"

restore:
	@echo "📂 Available backups:"
	@ls -la backups/*.sql || echo "No backups found"
	@read -p "Enter backup filename: " backup && \
	docker-compose exec -T timescaledb psql -U forex_user -d forex_data < backups/$$backup
	@echo "✅ Database restored!"

# Testing
test-install:
	@echo "📦 Installing test dependencies..."
	pip install -r requirements-test.txt
	@echo "✅ Test dependencies installed!"

test: test-install
	@echo "🧪 Running infrastructure tests..."
	@if [ ! "$$(docker ps -q -f name=forex-api)" ]; then \
		echo "❌ Services not running. Starting services first..."; \
		$(MAKE) up; \
		sleep 15; \
	fi
	python -m pytest tests/test_infrastructure.py -v --tb=short
	@echo "✅ Infrastructure tests completed!"

test-quick:
	@echo "🧪 Running quick tests (assuming services are running)..."
	python -m pytest tests/test_infrastructure.py -v --tb=short

test-api:
	@echo "🔌 Testing API endpoints..."
	@docker-compose exec forex-api python -m pytest tests/test_api.py -v

test-db:
	@echo "🗄️ Testing database..."
	@docker-compose exec forex-api python -m pytest tests/test_database.py -v

# Monitoring
stats:
	@echo "📊 Docker stats:"
	@docker stats --no-stream

ps:
	@echo "📋 Running containers:"
	@docker-compose ps

top:
	@echo "⚡ Container processes:"
	@docker-compose top

# Development helpers
format:
	@echo "🎨 Formatting code..."
	@docker-compose exec forex-api black app/
	@docker-compose exec forex-api flake8 app/

lint:
	@echo "🔍 Linting code..."
	@docker-compose exec forex-api flake8 app/
	@docker-compose exec forex-api mypy app/

# Update dependencies
update-deps:
	@echo "📦 Updating dependencies..."
	@docker-compose exec forex-api pip install --upgrade -r requirements.txt

# Quick development workflow
dev-rebuild:
	@echo "🔄 Rebuilding for development..."
	@docker-compose down
	@docker-compose build --no-cache forex-api
	@docker-compose up -d
	@echo "✅ Development environment rebuilt!"

# Production deployment
deploy:
	@echo "🚀 Deploying to production..."
	@docker-compose -f docker-compose.yml -f docker-compose.prod.yml down
	@docker-compose -f docker-compose.yml -f docker-compose.prod.yml build --no-cache
	@docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "✅ Production deployment completed!"

# Show environment info
info:
	@echo "ℹ️ Environment Information:"
	@echo "Docker version: $$(docker --version)"
	@echo "Docker Compose version: $$(docker-compose --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Available services:"
	@docker-compose config --services