# api/main.py - Versão Simplificada v2.0
import os
import asyncio
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import httpx
import openai
from loguru import logger

# Configurações simples
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
AI_DRY_RUN = os.getenv("AI_DRY_RUN", "1") == "1"
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "25"))

# MEGA API Config
MEGA_API_BASE_URL = os.getenv("MEGA_API_BASE_URL", "https://apistart01.megaapi.com.br")
MEGA_API_TOKEN = os.getenv("MEGA_API_TOKEN", "")
MEGA_INSTANCE_ID = os.getenv("MEGA_INSTANCE_ID", "")

app = FastAPI(title="WhatsApp AI Agent", version="2.0", description="Agente de IA simples para WhatsApp")

# Modelos Pydantic simples
class WhatsAppMessage(BaseModel):
    messageType: str
    key: Dict[str, Any]
    pushName: Optional[str] = None
    message: Dict[str, Any]

class SendMessageRequest(BaseModel):
    phone: str
    message: str

# ==================== RAG SIMPLES ====================
def load_simple_rag() -> str:
    """Carrega documentos de data/ e retorna como string simples"""
    context_data = []
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        return "Nenhum documento encontrado para contexto."
    
    # Procura arquivos .txt recursivamente
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            context_data.append(f"--- {file} ---\n{content}\n")
                except Exception as e:
                    logger.warning(f"Erro ao ler {file_path}: {e}")
    
    return "\n".join(context_data) if context_data else "Nenhum conteúdo válido encontrado."

# Carrega contexto uma vez na inicialização
RAG_CONTEXT = load_simple_rag()

# ==================== IA AGENT ====================
async def generate_ai_response(user_message: str, user_name: str = "") -> str:
    """Gera resposta usando OpenAI + RAG simples"""
    
    if AI_DRY_RUN:
        return f"[DRY RUN] Olá {user_name}! Recebi sua mensagem: '{user_message[:50]}...'. Como posso ajudar?"
    
    if not OPENAI_API_KEY:
        return "Desculpe, estou com problemas técnicos no momento. Tente novamente mais tarde."
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT)
        
        system_prompt = f"""Você é um assistente de atendimento via WhatsApp.

CONTEXTO DOS NOSSOS PRODUTOS/SERVIÇOS:
{RAG_CONTEXT}

REGRAS:
- Seja cordial, prestativo e direto
- Responda com no máximo 3 linhas
- Use o contexto acima para responder sobre nossos produtos/serviços
- Se não souber algo, seja honesto e ofereça ajuda para conectar com humano
- Mantenha tom profissional mas amigável
- Nome do usuário: {user_name or 'Cliente'}
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content or "Desculpe, não consegui processar sua mensagem."
        
    except Exception as e:
        logger.error(f"Erro na OpenAI: {e}")
        return "Desculpe, estou com dificuldades técnicas. Um humano entrará em contato em breve."

# ==================== MEGA API ====================
async def send_whatsapp_message(phone: str, message: str) -> Dict[str, Any]:
    """Envia mensagem via MEGA API"""
    
    if not all([MEGA_API_TOKEN, MEGA_INSTANCE_ID]):
        logger.warning("MEGA API não configurada")
        return {"success": False, "error": "MEGA API não configurada"}
    
    url = f"{MEGA_API_BASE_URL}/rest/sendMessage/{MEGA_INSTANCE_ID}"
    headers = {
        "Authorization": f"Bearer {MEGA_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "number": phone,
        "text": message
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return {"success": True, "data": response.json()}
    except Exception as e:
        logger.error(f"Erro ao enviar WhatsApp: {e}")
        return {"success": False, "error": str(e)}

# ==================== ENDPOINTS ====================

@app.get("/health")
async def health():
    """Healthcheck simples"""
    return {
        "status": "ok", 
        "version": "2.0",
        "ai_dry_run": AI_DRY_RUN,
        "rag_loaded": len(RAG_CONTEXT) > 50
    }

@app.post("/webhook")
async def webhook(data: WhatsAppMessage, background_tasks: BackgroundTasks):
    """Recebe mensagens do WhatsApp e responde com IA"""
    
    try:
        # Extrai informações básicas
        remote_jid = data.key.get("remoteJid", "")
        from_me = data.key.get("fromMe", True)
        user_name = data.pushName or "Cliente"
        
        # Ignora mensagens próprias
        if from_me:
            return {"status": "ignored", "reason": "message_from_me"}
        
        # Extrai texto da mensagem
        message_text = ""
        if "conversation" in data.message:
            message_text = data.message["conversation"]
        elif "extendedTextMessage" in data.message:
            message_text = data.message["extendedTextMessage"].get("text", "")
        
        if not message_text:
            return {"status": "ignored", "reason": "no_text_content"}
        
        # Extrai número do telefone (remove @s.whatsapp.net)
        phone = remote_jid.replace("@s.whatsapp.net", "").replace("@c.us", "")
        
        logger.info(f"Mensagem recebida de {user_name} ({phone}): {message_text[:100]}")
        
        # Processa em background para não bloquear webhook
        background_tasks.add_task(process_and_reply, phone, message_text, user_name)
        
        return {"status": "received", "processing": "background"}
        
    except Exception as e:
        logger.error(f"Erro no webhook: {e}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")

async def process_and_reply(phone: str, message: str, user_name: str):
    """Processa mensagem e envia resposta (executa em background)"""
    try:
        # Gera resposta da IA
        ai_response = await generate_ai_response(message, user_name)
        
        # Envia resposta
        result = await send_whatsapp_message(phone, ai_response)
        
        if result["success"]:
            logger.info(f"Resposta enviada para {user_name} ({phone}): {ai_response[:50]}")
        else:
            logger.error(f"Falha ao enviar resposta: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")

@app.post("/send-message")
async def send_message(request: SendMessageRequest):
    """Endpoint para enviar mensagem manualmente"""
    result = await send_whatsapp_message(request.phone, request.message)
    
    if result["success"]:
        return {"status": "sent", "data": result["data"]}
    else:
        raise HTTPException(status_code=500, detail=result["error"])

@app.get("/mega-status")
async def mega_status():
    """Verifica status da instância MEGA"""
    
    if not all([MEGA_API_TOKEN, MEGA_INSTANCE_ID]):
        raise HTTPException(status_code=400, detail="MEGA API não configurada")
    
    url = f"{MEGA_API_BASE_URL}/rest/instance/{MEGA_INSTANCE_ID}"
    headers = {"Authorization": f"Bearer {MEGA_API_TOKEN}"}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Erro ao verificar status MEGA: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== STARTUP ====================
@app.on_event("startup")
async def startup_event():
    """Configurações na inicialização"""
    logger.info(f"🚀 WhatsApp AI Agent v2.0 iniciado")
    logger.info(f"📝 RAG carregado: {len(RAG_CONTEXT)} caracteres")
    logger.info(f"🤖 AI_DRY_RUN: {AI_DRY_RUN}")
    logger.info(f"📱 MEGA configurado: {bool(MEGA_API_TOKEN and MEGA_INSTANCE_ID)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)