# api/main.py - WhatsApp AI Agent v2.1 OTIMIZADO
# - Mega API /text
# - Webhook tolerante
# - Anti-loop/eco + dedupe + lock
# - RAG auto-reload (watcher de arquivos em data/)
# - Modelo padrão: gpt-4o (configurável via .env)
# - OTIMIZAÇÕES: Prompt inteligente, detecção de intenção, RAG priorizado

import os
import asyncio
from time import monotonic
from collections import defaultdict
from typing import Dict, Any, Optional
import hashlib

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
import httpx
from loguru import logger

# OpenAI SDK v1+
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Carregar .env (opcional)
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("✅ Arquivo .env carregado")
except Exception:
    logger.warning("⚠️ python-dotenv não instalado. Usando variáveis do sistema.")

# ======================
# Configurações
# ======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")  # ← padrão 4o
AI_DRY_RUN = os.getenv("AI_DRY_RUN", "1") == "1"

# produção: ignorar sempre mensagens "fromMe" (ecos do próprio número)
IGNORE_FROM_ME = os.getenv("IGNORE_FROM_ME", "1") == "1"
# janela para anti-eco/duplicatas (segundos)
DEDUP_TTL = float(os.getenv("DEDUP_TTL", "12"))

# Watcher do RAG
RAG_DIR = os.getenv("RAG_DIR", "data")
RAG_AUTO_RELOAD = os.getenv("RAG_AUTO_RELOAD", "1") == "1"
RAG_WATCH_INTERVAL = float(os.getenv("RAG_WATCH_INTERVAL", "3"))

# MEGA API
MEGA_API_BASE_URL = os.getenv("MEGA_API_BASE_URL", "https://apistart01.megaapi.com.br")
MEGA_API_TOKEN = os.getenv("MEGA_API_TOKEN", "")
MEGA_INSTANCE_ID = os.getenv("MEGA_INSTANCE_ID", "")

app = FastAPI(title="WhatsApp AI Agent", version="2.1")

# ======================
# Modelos (usado em /send-message)
# ======================
class WhatsAppMessage(BaseModel):
    messageType: str
    key: Dict[str, Any]
    pushName: Optional[str] = None
    message: Dict[str, Any]

class SendMessageRequest(BaseModel):
    phone: str
    message: str

# ======================
# Estados (anti-eco / dedupe / locks)
# ======================
# último texto que o bot enviou para cada phone
LAST_SENT: Dict[str, tuple[str, float]] = {}
# dedupe por (phone+texto)
DEDUP: Dict[str, float] = {}
# locks por contato para evitar corrida
LOCKS: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

# ======================
# RAG otimizado com priorização
# ======================
def load_context() -> str:
    context = []
    data_dir = RAG_DIR
    if not os.path.exists(data_dir):
        return "Sem contexto disponível."
    
    # Arquivos prioritários que devem aparecer primeiro
    priority_keywords = [
        'resumo_executivo', '00_resumo', 'faq', 'principais',
        'pos_graduacao', 'congresso', 'eventos', 'inscricoes_pagamento',
        'cenat_institucional', 'comunidade'
    ]
    
    priority_files = []
    regular_files = []
    
    for root, _, files in os.walk(data_dir):
        for file in sorted(files):
            if file.endswith(".txt"):
                fp = os.path.join(root, file)
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        content_text = f.read().strip()
                        if content_text:
                            rel = os.path.relpath(fp, data_dir)
                            formatted_content = f"=== {rel} ===\n{content_text}"
                            
                            # Verificar se é arquivo prioritário
                            is_priority = any(keyword in file.lower() for keyword in priority_keywords)
                            if is_priority:
                                priority_files.append(formatted_content)
                            else:
                                regular_files.append(formatted_content)
                                
                except Exception as e:
                    logger.warning(f"Erro ao ler {fp}: {e}")
    
    # Montar contexto priorizando arquivos importantes
    all_content = priority_files + regular_files
    full_context = "\n\n".join(all_content) if all_content else "Nenhum documento encontrado."
    
    # Limitar tamanho total para não estourar contexto
    if len(full_context) > 12000:  # ~8k tokens
        # Manter sempre os prioritários + o que couber dos regulares
        priority_context = "\n\n".join(priority_files)
        if len(priority_context) > 12000:
            return priority_context[:12000] + "\n\n[CONTEXTO TRUNCADO - MANTIDOS ARQUIVOS PRIORITÁRIOS]"
        
        remaining_space = 12000 - len(priority_context) - 100
        regular_context = ""
        for content in regular_files:
            if len(regular_context + "\n\n" + content) < remaining_space:
                regular_context += "\n\n" + content if regular_context else content
            else:
                break
        
        final_context = priority_context + ("\n\n" + regular_context if regular_context else "")
        return final_context
    
    return full_context

def data_signature() -> str:
    """Assinatura do estado do diretório RAG (para detectar mudanças)."""
    h = hashlib.sha1()
    base = RAG_DIR
    if not os.path.exists(base):
        return ""
    for root, _, files in os.walk(base):
        for file in sorted(f for f in files if f.endswith(".txt")):
            fp = os.path.join(root, file)
            try:
                st = os.stat(fp)
                rel = os.path.relpath(fp, base)
                h.update(rel.encode("utf-8"))
                h.update(str(st.st_mtime_ns).encode("utf-8"))
                h.update(str(st.st_size).encode("utf-8"))
            except Exception:
                continue
    return h.hexdigest()

RAG_CONTEXT = load_context()
_RAG_SIG = data_signature()

# ======================
# Detecção de intenção otimizada
# ======================
def detect_user_intent(message: str) -> str:
    """Detecta a intenção do usuário para personalizar resposta."""
    message_lower = message.lower().strip()
    
    # Saudações e início de conversa
    if any(word in message_lower for word in ['oi', 'olá', 'bom dia', 'boa tarde', 'boa noite', 'eae', 'e ai']):
        return 'greeting'
    
    # Intenções de alta conversão (leads quentes)
    if any(word in message_lower for word in ['preço', 'valor', 'quanto custa', 'investimento', 'pagar', 'custo']):
        return 'pricing'
    
    if any(word in message_lower for word in ['inscrição', 'inscrever', 'matricula', 'vaga', 'me inscrever', 'quero me inscrever']):
        return 'enrollment'
    
    if any(word in message_lower for word in ['link', 'site', 'página', 'url', 'endereço']):
        return 'link_request'
    
    # Produtos/serviços específicos
    if any(word in message_lower for word in ['congresso', 'evento', 'palestras', 'seminário']):
        return 'events'
    
    if any(word in message_lower for word in ['pós', 'pos-graduacao', 'especialização', 'mestrado', 'pós-graduação']):
        return 'postgrad'
    
    if any(word in message_lower for word in ['curso', 'cursos', 'formação', 'capacitação', 'comunidade', 'online']):
        return 'courses'
    
    # Informações específicas
    if any(word in message_lower for word in ['quando', 'data', 'cronograma', 'calendario', 'prazo']):
        return 'schedule'
    
    if any(word in message_lower for word in ['onde', 'local', 'endereço', 'cidade', 'lugar']):
        return 'location'
    
    if any(word in message_lower for word in ['como', 'processo', 'funciona', 'etapas', 'procedimento']):
        return 'process'
    
    # Certificação e reconhecimento
    if any(word in message_lower for word in ['certificado', 'certificação', 'mec', 'reconhecido', 'válido']):
        return 'certification'
    
    # Interesse geral
    if any(word in message_lower for word in ['informação', 'informações', 'gostaria de saber', 'quero saber', 'me fala']):
        return 'info_request'
    
    # Respostas de confirmação/interesse
    if any(word in message_lower for word in ['sim', 'ok', 'certo', 'beleza', 'pode', 'quero', 'tenho interesse']):
        return 'positive_response'
    
    # Cidades específicas (interesse em congressos)
    cities = ['maceió', 'belém', 'florianópolis', 'floripa', 'vitória', 'online']
    if any(city in message_lower for city in cities):
        return 'city_specific'
    
    return 'general'

# ======================
# Utilidades
# ======================
def _digits_only(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())

def to_wa_jid(phone: str) -> str:
    digits = _digits_only(phone)
    if not digits:
        return phone
    return f"{digits}@s.whatsapp.net"

def _unwrap_ephemeral(msg: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(msg, dict) and "ephemeralMessage" in msg:
        return msg["ephemeralMessage"].get("message", {}) or {}
    return msg or {}

def _extract_text(msg: Dict[str, Any]) -> str:
    """Extrai texto de diferentes tipos de mensagem do WA/Mega."""
    msg = _unwrap_ephemeral(msg)
    return (
        (msg or {}).get("conversation")
        or (msg.get("extendedTextMessage") or {}).get("text")
        or (msg.get("imageMessage") or {}).get("caption")
        or (msg.get("documentMessage") or {}).get("caption")
        or (msg.get("videoMessage") or {}).get("caption")
        or ""
    ).strip()

# ======================
# IA Agent OTIMIZADO
# ======================
async def generate_response(user_message: str, user_name: str = "") -> str:
    """Gera resposta inteligente com IA + contexto RAG + detecção de intenção."""
    if AI_DRY_RUN:
        return f"[TESTE] Olá {user_name or 'Cliente'}! Vi sua mensagem '{user_message[:30]}...'. Como posso ajudar?"

    if not OPENAI_API_KEY or OpenAI is None:
        return "Desculpe, estou com problemas técnicos. Tente mais tarde."

    try:
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=25)
        
        # Detectar intenção para personalizar resposta
        intent = detect_user_intent(user_message)
        logger.info(f"🎯 Intenção detectada: {intent} para '{user_message[:50]}...'")

        system_prompt = f"""Você é LINA, assistente especializada do CENAT (Centro de Estudos em Saúde Mental).

🎯 MISSÃO: Ser consultiva, identificar necessidades e conectar com soluções CENAT de forma inteligente.

📋 CONTEXTO CENAT (fonte única da verdade):
{RAG_CONTEXT}

🔍 INTENÇÃO DETECTADA: {intent}

🗣️ ESTILO DE COMUNICAÇÃO:
- Tom: consultivo, acolhedor, especialista em saúde mental
- Tamanho: 3-4 linhas (máx ~400 caracteres)
- Use o nome: {user_name or 'Cliente'}
- Máximo 1 emoji quando relevante
- SEMPRE termine com pergunta direta ou CTA claro

🧠 FLUXO INTELIGENTE por INTENÇÃO:

**GREETING/GERAL** → Apresente 3 opções principais:
"Oi {user_name}! Sou a Lina do CENAT 😊
Posso ajudar com:
1. Pós-graduação Saúde Mental
2. Congressos 2025
3. Cursos online
Qual te interessa mais?"

**EVENTS/CONGRESSOS** → Liste próximos com datas:
"Temos congressos confirmados:
• Maceió: 05-06/set
• Belém: 09-10/set  
• Floripa: 21-22/out
• Vitória: 24-25/out
Qual cidade te interessa? Envio o link! 🎯"

**POSTGRAD/PÓS** → Qualifique primeiro:
"Nossa pós em Saúde Mental é reconhecida pelo MEC! 
Você já concluiu sua graduação? 
Em qual área atua/pretende atuar?
Posso explicar o processo seletivo por telefone!"

**PRICING/PREÇOS** → Dê valores + desconto:
"Pós: ~R$ 300/mês | Congressos: varia por lote
Temos desconto para estudantes e grupos!
Quer saber sobre parcelamento e condições?"

**ENROLLMENT/INSCRIÇÃO** → Direcione ação:
"Para pós: processo seletivo (formulário + entrevista)
Para congressos: link direto da cidade
Qual você quer se inscrever? Te passo o caminho!"

**CITY_SPECIFIC** → Link direto:
[Se mencionar cidade específica, ofereça link do congresso daquela cidade]

**LINK_REQUEST** → Confirme e envie:
"Qual link precisa? 
• Site geral: cenatsaudemental.com
• Congresso específico?
• Processo seletivo pós?
Me fala qual!"

**INFO_REQUEST/GERAL** → Ofereça opções específicas:
"Posso detalhar:
1. Processo seletivo pós (graduação obrigatória)
2. Programação dos congressos
3. Valores e descontos
Sobre o que quer saber primeiro?"

**POSITIVE_RESPONSE** → Avance no funil:
[Continue a conversa anterior com próximo passo específico]

🚨 LEADS QUENTES - PRIORIZE:
- Pergunta preços = Dar valores + CTA parcelamento
- Menciona graduação = Qualificar para pós
- Cita cidade = Link congresso específico
- Quer inscrição = Processo específico

⛔ REGRAS CRÍTICAS:
- NUNCA inventar datas, preços ou informações
- SE não souber: "Não tenho essa info específica. Posso conectar você com nossa consultora?"
- SEMPRE oferecer alternativa relacionada do contexto
- Não deixar conversa "morrer" - sempre próximo passo
- Máximo 1 emoji por resposta

EXEMPLO DE RESPOSTA OTIMIZADA:
"Oi João! Temos 3 congressos até outubro:
• Maceió: 05-06/set
• Belém: 09-10/set  
• Floripa: 21-22/out

Qual região te interessa? Posso enviar o link direto! 🎯"

Nome do cliente: {user_name or 'Cliente'}"""

        # Parâmetros otimizados da OpenAI
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=280,  # Aumentado para respostas mais completas
            temperature=0.3,  # Reduzido para mais consistência
            presence_penalty=0.1,  # Evita repetições
            frequency_penalty=0.2,  # Incentiva variedade
            top_p=0.9  # Mais focado nas respostas relevantes
        )
        
        response = resp.choices[0].message.content or "Desculpe, não consegui processar sua mensagem."
        
        # Log da resposta gerada
        logger.info(f"💬 Resposta gerada para {user_name} (intenção: {intent}): {response[:100]}...")
        
        return response
        
    except Exception as e:
        logger.error(f"Erro OpenAI: {e}")
        return f"Olá {user_name or 'Cliente'}! Estou com dificuldades técnicas agora. Pode tentar novamente em alguns minutos? Ou me chama no (47) 99242-8886 que nossa equipe te atende! 🙏"

# ======================
# MEGA API - envio de mensagem (registra LAST_SENT)
# ======================
async def send_whatsapp(phone: str, message: str) -> bool:
    """Envia mensagem de texto via MEGA API (endpoint /text)."""
    if not MEGA_API_TOKEN or not MEGA_INSTANCE_ID:
        logger.warning("MEGA API não configurada")
        return False

    norm_phone = _digits_only(phone)
    to = f"{norm_phone}@s.whatsapp.net" if norm_phone else phone

    url = f"{MEGA_API_BASE_URL}/rest/sendMessage/{MEGA_INSTANCE_ID}/text"
    headers = {
        "Authorization": f"Bearer {MEGA_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"messageData": {"to": to, "text": message}}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code >= 400:
                logger.error(f"MEGA send failed {resp.status_code}: {resp.text}")
                resp.raise_for_status()
            # registra último texto enviado para anti-eco
            LAST_SENT[norm_phone] = (message.strip(), monotonic())
            logger.info(f"Mensagem enviada para {to}: {message[:120]}")
            return True
    except Exception as e:
        logger.error(f"Erro envio WhatsApp: {e}")
        return False

# ======================
# ENDPOINTS
# ======================
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.1-OTIMIZADO",
        "ai_mode": "DRY_RUN" if AI_DRY_RUN else "REAL",
        "context_loaded": len(RAG_CONTEXT) > 10,
        "mega_configured": bool(MEGA_API_TOKEN and MEGA_INSTANCE_ID),
        "optimizations": {
            "intelligent_prompt": True,
            "intent_detection": True,
            "prioritized_rag": True,
            "improved_parameters": True
        },
        "debug": {
            "ai_dry_run_env": os.getenv("AI_DRY_RUN"),
            "mega_token_present": bool(MEGA_API_TOKEN),
            "openai_key_present": bool(OPENAI_API_KEY),
            "ignore_from_me": IGNORE_FROM_ME,
            "dedup_ttl": DEDUP_TTL,
            "rag_auto_reload": RAG_AUTO_RELOAD,
            "rag_watch_interval": RAG_WATCH_INTERVAL,
            "rag_dir": RAG_DIR,
            "context_length": len(RAG_CONTEXT),
        },
    }

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    """Recebe mensagens do WhatsApp via MEGA API (tolerante + anti-eco/loop)."""
    try:
        payload = await request.json()
    except Exception:
        logger.warning("Webhook: corpo não-JSON; ignorando.")
        return {"status": "ignored", "reason": "invalid_json"}

    # Formatos possíveis
    key = (payload.get("key") or {}) if isinstance(payload, dict) else {}
    msg = (payload.get("message") or {}) if isinstance(payload, dict) else {}
    push_name = payload.get("pushName") or "Cliente"

    if (not key or not msg) and isinstance(payload.get("messages"), list) and payload["messages"]:
        first = payload["messages"][0] or {}
        key = first.get("key") or {}
        msg = first.get("message") or {}
        push_name = first.get("pushName") or push_name

    remote_jid = key.get("remoteJid") or ""
    phone = _digits_only(remote_jid) or _digits_only(payload.get("phone") or "")
    from_me = bool(key.get("fromMe"))
    text = _extract_text(msg)

    caller_ip = request.client.host if request.client else "unknown"
    logger.info(f"🌐 Webhook de {caller_ip} | fromMe={from_me} | jid={remote_jid} | texto='{text[:80]}'")

    # 1) Ignora ecos do próprio número (produção)
    if from_me and IGNORE_FROM_ME:
        return {"status": "ignored", "reason": "own_message"}

    # 2) Campos mínimos
    if not phone or not text:
        return {"status": "ignored", "reason": "no_phone_or_text"}

    # 3) Anti-eco: se texto == último que o bot enviou p/ esse phone recentemente → ignora
    sent = LAST_SENT.get(phone)
    if sent:
        last_text, t0 = sent
        if text == last_text and (monotonic() - t0) < DEDUP_TTL:
            logger.info("🔁 Ignorado: eco do próprio envio recente.")
            return {"status": "ignored", "reason": "echo_recent_outbound"}

    # 4) Dedup simples por (phone+texto) na janela TTL (contra re-entregas)
    dedup_key = f"{phone}:{hash(text)}"
    t_last = DEDUP.get(dedup_key)
    now = monotonic()
    if t_last and (now - t_last) < DEDUP_TTL:
        logger.info("⏱️ Ignorado: duplicata recente.")
        return {"status": "ignored", "reason": "duplicate"}
    DEDUP[dedup_key] = now

    # 5) Processar em background com lock do contato
    background_tasks.add_task(process_and_reply, phone, text, push_name)
    return {"status": "processing"}

@app.post("/send-message")
async def send_message_manual(request: SendMessageRequest):
    """Envio manual de mensagem via nossa API."""
    success = await send_whatsapp(request.phone, request.message)
    if success:
        return {"status": "sent", "phone": request.phone, "message": request.message[:60]}
    raise HTTPException(status_code=500, detail="Falha ao enviar mensagem")

@app.get("/mega-status")
async def mega_status():
    """Consulta status da instância na Mega (QR, conectado, etc.)."""
    if not MEGA_API_TOKEN or not MEGA_INSTANCE_ID:
        raise HTTPException(status_code=400, detail="MEGA API não configurada")

    url = f"{MEGA_API_BASE_URL}/rest/instance/{MEGA_INSTANCE_ID}"
    headers = {"Authorization": f"Bearer {MEGA_API_TOKEN}"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"Erro status MEGA: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ====== Endpoints para gerenciar o RAG ======

@app.post("/reload-context")
async def reload_context():
    """Recarrega os arquivos .txt de data/ sem reiniciar o servidor."""
    global RAG_CONTEXT, _RAG_SIG
    RAG_CONTEXT = load_context()
    _RAG_SIG = data_signature()
    logger.info(f"🔄 RAG recarregado: {len(RAG_CONTEXT)} caracteres")
    return {"status": "ok", "context_len": len(RAG_CONTEXT), "optimized": True}

@app.get("/context/preview")
async def context_preview(n: int = 1500):
    """Mostra uma amostra do contexto carregado (para verificação)."""
    n = max(0, min(n, 8000))
    return {
        "preview": RAG_CONTEXT[:n], 
        "len": len(RAG_CONTEXT),
        "optimizations": "Priorização de arquivos ativada"
    }

# Novo endpoint para testar detecção de intenção
@app.post("/test-intent")
async def test_intent(message: str):
    """Testa a detecção de intenção para uma mensagem."""
    intent = detect_user_intent(message)
    return {
        "message": message,
        "intent": intent,
        "timestamp": monotonic()
    }

# ======================
# Worker com lock por contato
# ======================
async def process_and_reply(phone: str, message: str, user_name: str):
    try:
        async with LOCKS[phone]:
            response = await generate_response(message, user_name)
            logger.info(f"🤖 IA gerou resposta: {response}")
            ok = await send_whatsapp(phone, response)
            if ok:
                logger.info(f"✅ Resposta enviada para {user_name}: {response[:120]}...")
            else:
                logger.error(f"❌ Falha ao enviar para {user_name}")
                logger.info(f"📝 Resposta que seria enviada: {response}")
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")

# ======================
# Watcher do RAG (assíncrono)
# ======================
async def rag_watcher():
    global _RAG_SIG, RAG_CONTEXT
    logger.info(f"👀 RAG watcher OTIMIZADO ativo em '{RAG_DIR}' a cada {RAG_WATCH_INTERVAL}s")
    while True:
        try:
            sig = data_signature()
            if sig != _RAG_SIG:
                logger.info("🪄 Mudanças detectadas em data/: recarregando RAG...")
                RAG_CONTEXT = load_context()
                _RAG_SIG = sig
                logger.info(f"🔄 RAG recarregado automaticamente: {len(RAG_CONTEXT)} caracteres")
        except Exception as e:
            logger.warning(f"Watcher RAG: {e}")
        await asyncio.sleep(RAG_WATCH_INTERVAL)

# ======================
# Startup
# ======================
@app.on_event("startup")
async def startup():
    logger.info("🚀 WhatsApp AI Agent v2.1 OTIMIZADO iniciado")
    logger.info("⚡ OTIMIZAÇÕES ATIVAS:")
    logger.info("   • Prompt inteligente com detecção de intenção")
    logger.info("   • RAG com priorização de arquivos importantes")
    logger.info("   • Parâmetros IA otimizados para conversão")
    logger.info("   • Fluxo consultivo por tipo de interesse")
    logger.info(f"📄 Contexto RAG: {len(RAG_CONTEXT)} caracteres")
    logger.info(f"🤖 Modo IA: {'DRY_RUN (teste)' if AI_DRY_RUN else 'REAL (OpenAI)'}")
    logger.info(f"📱 MEGA API: {'configurada' if (MEGA_API_TOKEN and MEGA_INSTANCE_ID) else 'NÃO CONFIGURADA'}")
    logger.info(f"🧰 Flags: IGNORE_FROM_ME={IGNORE_FROM_ME}, DEDUP_TTL={DEDUP_TTL}s")
    logger.info(f"📂 RAG_DIR='{RAG_DIR}', AUTO_RELOAD={RAG_AUTO_RELOAD}, INTERVALO={RAG_WATCH_INTERVAL}s")
    if RAG_AUTO_RELOAD:
        asyncio.create_task(rag_watcher())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "8000")))