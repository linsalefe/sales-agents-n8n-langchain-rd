# WhatsApp AI Agent v2.0

**Agente de IA simples para responder mensagens no WhatsApp**

## 🎯 O que faz

1. **Recebe** mensagens do WhatsApp via webhook (MEGA API)
2. **Processa** com OpenAI + contexto dos seus documentos
3. **Responde** automaticamente (máximo 3 linhas)

## 🚀 Setup Rápido

```bash
# 1. Clone e entre na pasta
cd sales-agents-n8n-langchain-rd

# 2. Crie ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instale dependências
pip install -r api/requirements.txt

# 4. Configure ambiente
cp .env.example .env
# Edite .env com suas chaves

# 5. Execute
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📁 Estrutura Simplificada

```
├── api/
│   ├── main.py           # API completa (tudo em um arquivo)
│   └── requirements.txt  # dependências mínimas
├── data/                 # documentos para contexto IA
│   ├── produtos/
│   │   └── *.txt        # coloque arquivos .txt aqui
│   └── empresas/
└── .env                 # configurações
```

## 🔧 Configuração

### Variáveis Obrigatórias

```bash
# OpenAI (obrigatório para produção)
OPENAI_API_KEY="sk-..."

# MEGA API (obrigatório para WhatsApp)
MEGA_API_TOKEN="seu-token"
MEGA_INSTANCE_ID="megastart-XXXX"
```

### Configurações Opcionais

```bash
AI_DRY_RUN=1              # 1 = respostas mock, 0 = IA real
MODEL_NAME="gpt-4o-mini"  # modelo OpenAI
OPENAI_TIMEOUT=25         # timeout em segundos
```

## 📱 Endpoints

### Core
- `GET /health` - status do sistema
- `POST /webhook` - recebe mensagens WhatsApp
- `POST /send-message` - envia mensagem manual
- `GET /mega-status` - status da instância MEGA

## 📝 Como Funciona

1. **Webhook**: MEGA API envia mensagens recebidas para `/webhook`
2. **IA**: Sistema processa com OpenAI + contexto dos arquivos em `data/`
3. **Resposta**: Envia resposta automática via MEGA API
4. **Logs**: Tudo logado com Loguru

## 🧪 Testes Rápidos

```bash
# 1. Servidor funcionando?
curl http://localhost:8000/health

# 2. MEGA API conectada?
curl http://localhost:8000/mega-status

# 3. Envio manual
curl -X POST http://localhost:8000/send-message \
  -H "Content-Type: application/json" \
  -d '{"phone":"5511999999999","message":"Teste"}'
```

## 📊 Status do Projeto

### ✅ Implementado
- API FastAPI simples
- Webhook WhatsApp (MEGA API)
- Resposta automática com IA
- RAG simples (arquivos .txt)
- Logs básicos
- DRY_RUN mode

### 🎯 Próximos Passos
1. Testar integração MEGA API real
2. Adicionar mais tipos de arquivo (PDF, etc.)
3. Melhorar prompts da IA
4. Adicionar métricas básicas
5. Implementar rate limiting

## 🐛 Troubleshooting

### Erro: ModuleNotFoundError
```bash
# Ative o venv primeiro
source venv/bin/activate
pip install -r api/requirements.txt
```

### Erro 500 na IA
- Verifique `OPENAI_API_KEY` no .env
- Use `AI_DRY_RUN=1` para testar sem OpenAI

### MEGA API não funciona
- Verifique `MEGA_API_TOKEN` e `MEGA_INSTANCE_ID`
- Teste com `curl http://localhost:8000/mega-status`

## 📞 Suporte

Este é um projeto em evolução! Para dúvidas:
1. Verifique os logs com Loguru
2. Teste os endpoints individualmente
3. Use DRY_RUN para debug sem custos