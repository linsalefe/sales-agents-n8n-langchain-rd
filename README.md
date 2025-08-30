WhatsApp AI Agent v2.1 — Guia Atualizado

Agente de IA para WhatsApp usando FastAPI + OpenAI com RAG simples (arquivos .txt em data/). Integra com a Mega API (webhook e envio) e está preparado para deploy em Ubuntu (EC2) com Nginx + systemd. Nesta versão:

Envio correto via /rest/sendMessage/{INSTANCE}/text

Anti-loop/anti-eco + deduplicação + lock por contato

RAG auto-reload (watcher) + endpoints /reload-context e /context/preview

Modelo padrão configurável via MODEL_NAME (ex.: gpt-4o ou gpt-4o-mini)

📦 Stack

Python 3.10+/3.11

FastAPI, Uvicorn, Gunicorn

OpenAI SDK (Chat Completions)

httpx, loguru, python-dotenv

Nginx (proxy)

Mega API para WhatsApp

📁 Estrutura recomendada
/opt/whatsapp-agent/
├─ sales-agents-n8n-langchain-rd/
│  └─ api/
│     └─ main.py            # API (webhook, IA, envio Mega, RAG watcher)
├─ data/                     # <- arquivos .txt do RAG (subpastas livres)
└─ venv/                     # virtualenv


Observação: o serviço systemd usa WorkingDirectory=/opt/whatsapp-agent/sales-agents-n8n-langchain-rd.

⚙️ Variáveis de Ambiente (ex.: /etc/whatsapp-agent.env)
# OpenAI
OPENAI_API_KEY=COLOQUE_SUA_CHAVE_AQUI
MODEL_NAME=gpt-4o           # ou gpt-4o-mini
AI_DRY_RUN=0                # 1 para respostas mock, 0 para IA real

# API
API_HOST=0.0.0.0
API_PORT=8000

# Mega API
MEGA_API_BASE_URL=https://apistart01.megaapi.com.br
MEGA_API_TOKEN=SEU_TOKEN
MEGA_INSTANCE_ID=megastart-XXXXXX

# Webhook / anti-loop
IGNORE_FROM_ME=1
DEDUP_TTL=12

# RAG
RAG_DIR=/opt/whatsapp-agent/data
RAG_AUTO_RELOAD=1
RAG_WATCH_INTERVAL=3


Não comitar .env. Rotacione chaves expostas.

🚀 Setup Local (sem requirements.txt)
# ativar venv
python3 -m venv venv
source venv/bin/activate

# instalar pacotes
pip install --upgrade pip
pip install fastapi httpx loguru openai python-dotenv uvicorn gunicorn

# rodar local
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload


Endpoints úteis

GET /health

POST /webhook

POST /send-message

GET /mega-status

POST /reload-context (reler arquivos do RAG)

GET /context/preview?n=1000

🔌 Mega API — envio correto

Envio de texto (usado pelo código):

POST {MEGA_API_BASE_URL}/rest/sendMessage/{MEGA_INSTANCE_ID}/text
Body:
{
  "messageData": {
    "to": "55DDDNNNNNNN@s.whatsapp.net",
    "text": "sua mensagem"
  }
}


Se usar o endpoint antigo sem /text, receberá 404.

🔄 RAG (arquivos .txt)

Coloque .txt dentro de RAG_DIR (ex.: /opt/whatsapp-agent/data/...). Subpastas são livres.

O watcher detecta mudanças e recarrega automaticamente.

Para recarregar manualmente: POST /reload-context.

Para conferir o que está carregado: GET /context/preview.

Dicas de conteúdo (.txt):

=== produtos/marketing_digital.txt ===
Preço: R$ 497
Carga horária: 40h
Inclui: 6 meses de mentoria
FAQs:
- Certificado? Sim (digital).

🛠 Deploy em Ubuntu (EC2) — Resumo do que já está em produção

Pacotes: git nginx python3-venv python3-pip (e afins)

Projeto: /opt/whatsapp-agent (com sales-agents-n8n-langchain-rd dentro)

Virtualenv: /opt/whatsapp-agent/venv

Env File: /etc/whatsapp-agent.env

systemd: /etc/systemd/system/whatsapp-agent.service

[Unit]
Description=WhatsApp AI Agent (FastAPI)
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/whatsapp-agent/sales-agents-n8n-langchain-rd
EnvironmentFile=/etc/whatsapp-agent.env
ExecStart=/opt/whatsapp-agent/venv/bin/gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 127.0.0.1:8000 api.main:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target


Comandos:

sudo systemctl daemon-reload
sudo systemctl enable --now whatsapp-agent
sudo systemctl status whatsapp-agent --no-pager


Nginx (HTTP): /etc/nginx/sites-available/whatsapp-agent

server {
    listen 80;
    server_name _;

    client_max_body_size 10m;

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 300;
    }
}


Ativar:

sudo ln -sf /etc/nginx/sites-available/whatsapp-agent /etc/nginx/sites-enabled/whatsapp-agent
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx


Webhook Mega (produção):

URL: http://SEU_IP/webhook

Testes de fumaça:

curl -s http://127.0.0.1:8000/health
curl -s http://SEU_IP/health

# simular webhook
curl -s -X POST http://SEU_IP/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "messageType":"notification",
    "key":{"fromMe":false,"remoteJid":"5583XXXXXXXX@s.whatsapp.net"},
    "pushName":"Teste",
    "message":{"conversation":"Qual o preço do curso?"}
  }'

🧩 Como funciona internamente

Webhook tolerante: aceita formatos diferentes (mensagens simples, efêmeras, etc.).

Anti-eco/loop: ignora fromMe=True e filtra eco do último texto enviado por DEDUP_TTL s.

Deduplicação: evita processar (telefone+texto) repetidos na janela TTL.

Lock por contato: impede envios simultâneos conflitantes por número.

Logs: loguru com mensagens claras de fluxo e erros.

🐛 Troubleshooting

ModuleNotFoundError: No module named 'api'

Verifique WorkingDirectory do systemd apontando para a pasta que contém api/main.py.

Nginx 404/502

curl http://127.0.0.1:8000/health deve funcionar.

Se local OK e proxy falha: sudo nginx -t && sudo systemctl reload nginx.

MEGA não envia

Cheque MEGA_API_TOKEN, MEGA_INSTANCE_ID e o endpoint /text.

RAG não atualiza

Confira RAG_DIR, permissões e logs do watcher.

Forçar: POST /reload-context.

Loop de mensagens

IGNORE_FROM_ME=1 no env + aguardar janela DEDUP_TTL.

🔒 Boas práticas

Usar Secrets Manager/SSM (ou arquivos root-only) para chaves.

Rotacionar OPENAI_API_KEY e MEGA_API_TOKEN se vazar.

Logs e auditoria: exportar para CloudWatch/ELK se necessário.

📜 Licença

Uso interno. Ajuste conforme sua política.