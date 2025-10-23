Local setup and testing guide

This document explains how to run the full development stack locally using Docker Compose
and how to run the test suite and validate metrics and backups.

Prerequisites
- Docker Desktop (with Compose v2)
- Python 3.11
- Git

1. Start the stack

```powershell
# from project root
docker compose up -d
```

Wait until `timescaledb`, `redis`, `minio`, and `forex-api` are healthy (use `docker compose ps`)

2. Install dev dependencies

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

3. Run tests

```powershell
# run all tests (integration tests require services up)
python -m pytest tests/ -v --junitxml=reports/junit.xml --cov=app --cov-report=html:reports/coverage
```

4. Run quick smoke tests

```powershell
# run quick unit-only tests
python -m pytest tests/ -q -k "not integration"
```

5. Backups (MinIO local testing)

- MinIO console is available at http://localhost:9001 (default credentials in docker-compose)
- Use AWS CLI or MinIO client to upload/download backups to `minio` for testing

6. Metrics

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)


If you want, I can now run a quick unit-only pytest locally to validate the dev setup.