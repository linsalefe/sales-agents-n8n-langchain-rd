#!/usr/bin/env bash
set -euo pipefail

BASE="http://localhost:8000"
TZ_ENV="America/Fortaleza"

echo "→ /health"
curl -fsS "$BASE/health" | jq .

echo "→ /mega-api-status"
curl -fsS "$BASE/mega-api-status" | jq .

echo "→ /whatsapp-reply"
REPLY=$(curl -fsS -X POST "$BASE/whatsapp-reply" \
  -H "Content-Type: application/json" \
  -d "{
    \"lead_name\": \"Maria\",
    \"course_name\": \"Psicologia Clínica\",
    \"lead_phone\": \"558388046720\",
    \"lead_email\": \"maria@example.com\",
    \"last_user_message\": \"Posso amanhã às 09:00?\",
    \"timezone\": \"${TZ_ENV}\",
    \"history\": [
      { \"role\": \"assistant\", \"content\": \"Oi, Maria! Sou a Ana do CENAT 😊 Posso te ligar para explicar rapidinho?\" }
    ]
  }" | jq -r '.reply')

echo "REPLY =========="
echo "$REPLY"
echo "================"

# Verifica se veio o marcador na 1ª linha (literal '|')
echo "$REPLY" | grep -q '^#AGENDAR[|]' || echo "⚠️  Sem #AGENDAR na primeira linha (ok se vier na 2ª)."

echo "→ /whatsapp-parse"
jq -n --arg reply "$REPLY" --arg tz "$TZ_ENV" \
  '{reply:$reply, timezone:$tz}' \
| curl -fsS -X POST "$BASE/whatsapp-parse" \
  -H "Content-Type: application/json" \
  -d @- | tee /tmp/parse.json | jq .

echo "→ /calendar-ics"
jq -n --arg reply "$REPLY" --arg tz "$TZ_ENV" \
  '{reply:$reply, timezone:$tz}' \
| curl -fsS -X POST "$BASE/calendar-ics" \
  -H "Content-Type: application/json" \
  -o /tmp/agendar-evento.ics -d @-

echo "→ Preview do ICS"
head -n 12 /tmp/agendar-evento.ics
echo "OK ✅"
