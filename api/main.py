# api/main.py - WhatsApp AI Agent v2.1
# - Mega API /text
# - Webhook tolerante
# - Anti-loop/eco + dedupe + lock
# - RAG auto-reload (watcher de arquivos em data/)
# - Modelo padrão: gpt-4o (configurável via .env)

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
# RAG simples (TXT em data/)
# ======================
def load_context() -> str:
    context = []
    data_dir = RAG_DIR
    if not os.path.exists(data_dir):
        return "Sem contexto disponível."
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".txt"):
                fp = os.path.join(root, file)
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            rel = os.path.relpath(fp, data_dir)
                            context.append(f"=== {rel} ===\n{content}")
                except Exception as e:
                    logger.warning(f"Erro ao ler {fp}: {e}")
    return "\n\n".join(context) if context else "Nenhum documento encontrado."

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
# IA Agent
# ======================
async def generate_response(user_message: str, user_name: str = "") -> str:
    """Gera resposta com IA + contexto RAG."""
    if AI_DRY_RUN:
        return f"[TESTE] Olá {user_name or 'Cliente'}! Vi sua mensagem '{user_message[:30]}...'. Como posso ajudar?"

    if not OPENAI_API_KEY or OpenAI is None:
        return "Desculpe, estou com problemas técnicos. Tente mais tarde."

    try:
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=20)

        system_prompt = f"""Você é um assistente de atendimento via WhatsApp.

CONTEXTO DOS NOSSOS PRODUTOS/SERVIÇOS:
{RAG_CONTEXT}

REGRAS:
- Seja cordial, prestativo e direto
- Responda com no máximo 3 linhas
- Use o contexto acima para responder sobre nossos produtos/serviços
- Se não souber algo, seja honesto e ofereça ajuda humana
- Mantenha tom profissional mas amigável
- Nome do cliente: {user_name or 'Cliente'}"""

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        return resp.choices[0].message.content or "Desculpe, não consegui processar sua mensagem."
    except Exception as e:
        logger.error(f"Erro OpenAI: {e}")
        return "Estou com dificuldades técnicas. Um humano entrará em contato em breve."

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
        "version": "2.1",
        "ai_mode": "DRY_RUN" if AI_DRY_RUN else "REAL",
        "context_loaded": len(RAG_CONTEXT) > 10,
        "mega_configured": bool(MEGA_API_TOKEN and MEGA_INSTANCE_ID),
        "debug": {
            "ai_dry_run_env": os.getenv("AI_DRY_RUN"),
            "mega_token_present": bool(MEGA_API_TOKEN),
            "openai_key_present": bool(OPENAI_API_KEY),
            "ignore_from_me": IGNORE_FROM_ME,
            "dedup_ttl": DEDUP_TTL,
            "rag_auto_reload": RAG_AUTO_RELOAD,
            "rag_watch_interval": RAG_WATCH_INTERVAL,
            "rag_dir": RAG_DIR,
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
    return {"status": "ok", "context_len": len(RAG_CONTEXT)}

@app.get("/context/preview")
async def context_preview(n: int = 800):
    """Mostra uma amostra do contexto carregado (para verificação)."""
    n = max(0, min(n, 5000))
    return {"preview": RAG_CONTEXT[:n], "len": len(RAG_CONTEXT)}

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
    logger.info(f"👀 RAG watcher ativo em '{RAG_DIR}' a cada {RAG_WATCH_INTERVAL}s")
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
    logger.info("🚀 WhatsApp AI Agent v2.1 iniciado")
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
