#!/usr/bin/env bash
set -euo pipefail

echo "[+] Derrubando containers..."
docker compose down -v

echo "[✓] Containers removidos e volumes limpos."
