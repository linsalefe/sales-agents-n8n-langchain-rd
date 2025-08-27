#!/usr/bin/env bash
set -euo pipefail

export COMPOSE_DOCKER_CLI_BUILD=1
export DOCKER_BUILDKIT=1

echo "[+] Subindo containers (api + n8n)..."
docker compose up -d --build

echo "[✓] Pronto!"
echo "API:    http://localhost:${API_PORT:-8000}/docs"
echo "n8n:    http://localhost:${N8N_PORT:-5678}/"
