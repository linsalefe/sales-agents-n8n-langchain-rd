# api/main.py - WhatsApp AI Agent v2.1 NATURAL (histórico curto + anti-reset)

import os
import asyncio
from time import monotonic
from collections import defaultdict, deque
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
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
AI_DRY_RUN = os.getenv("AI_DRY_RUN", "1") == "1"

IGNORE_FROM_ME = os.getenv("IGNORE_FROM_ME", "1") == "1"
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
# Modelos
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
# Estados
# ======================
LAST_SENT: Dict[str, tuple[str, float]] = {}
DEDUP: Dict[str, float] = {}
LOCKS: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
CONVERSATION_HISTORY: Dict[str, int] = defaultdict(int)
# Histórico curto por contato (memória de contexto)
CHAT_HISTORY: Dict[str, deque] = defaultdict(lambda: deque(maxlen=12))

# ======================
# RAG otimizado
# ======================
def load_context() -> str:
    context = []
    data_dir = RAG_DIR
    if not os.path.exists(data_dir):
        return "Sem contexto disponível."
    priority_keywords = [
        'resumo_executivo', '00_resumo', 'faq', 'principais',
        'pos_graduacao', 'congresso', 'eventos', 'inscricoes_pagamento',
        'cenat_institucional', 'comunidade'
    ]
    priority_files, regular_files = [], []
    for root, _, files in os.walk(data_dir):
        for file in sorted(files):
            if not file.endswith(".txt"):
                continue
            fp = os.path.join(root, file)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    content_text = f.read().strip()
                if not content_text:
                    continue
                rel = os.path.relpath(fp, data_dir)
                formatted = f"=== {rel} ===\n{content_text}"
                (priority_files if any(k in file.lower() for k in priority_keywords) else regular_files).append(formatted)
            except Exception as e:
                logger.warning(f"Erro ao ler {fp}: {e}")
    all_content = priority_files + regular_files
    full = "\n\n".join(all_content) if all_content else "Nenhum documento encontrado."
    if len(full) > 12000:
        pr = "\n\n".join(priority_files)
        if len(pr) > 12000:
            return pr[:12000] + "\n\n[CONTEXTO TRUNCADO - PRIORITÁRIOS]"
        remaining = 12000 - len(pr) - 100
        reg = ""
        for content in regular_files:
            if len(reg + ("\n\n" if reg else "") + content) < remaining:
                reg += ("\n\n" if reg else "") + content
            else:
                break
        return pr + (("\n\n" + reg) if reg else "")
    return full

def data_signature() -> str:
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
# Intenções
# ======================
def detect_user_intent(message: str) -> str:
    m = message.lower().strip()
    if any(w in m for w in ['oi', 'olá', 'bom dia', 'boa tarde', 'boa noite', 'eae', 'e ai']):
        return 'greeting'
    if any(w in m for w in ['preço', 'valor', 'quanto custa', 'investimento', 'pagar', 'custo']):
        return 'pricing'
    if any(w in m for w in ['inscrição', 'inscrever', 'matricula', 'vaga', 'me inscrever', 'quero me inscrever']):
        return 'enrollment'
    if any(w in m for w in ['link', 'site', 'página', 'url', 'endereço']):
        return 'link_request'
    if any(w in m for w in ['congresso', 'evento', 'palestras', 'seminário']):
        return 'events'
    if any(w in m for w in ['pós', 'pos-graduacao', 'especialização', 'mestrado', 'pós-graduação']):
        return 'postgrad'
    if any(w in m for w in ['curso', 'cursos', 'formação', 'capacitação', 'comunidade', 'online']):
        return 'courses'
    if any(w in m for w in ['quando', 'data', 'cronograma', 'calendario', 'prazo', 'programação', 'programacao']):
        return 'schedule'
    if any(w in m for w in ['onde', 'local', 'endereço', 'cidade', 'lugar']):
        return 'location'
    if any(w in m for w in ['como', 'processo', 'funciona', 'etapas', 'procedimento']):
        return 'process'
    if any(w in m for w in ['certificado', 'certificação', 'mec', 'reconhecido', 'válido']):
        return 'certification'
    if any(w in m for w in ['informação', 'informações', 'gostaria de saber', 'quero saber', 'me fala']):
        return 'info_request'
    if any(w in m for w in ['sim', 'ok', 'certo', 'beleza', 'pode', 'quero', 'tenho interesse']):
        return 'positive_response'
    if any(w in m for w in ['não', 'nao', 'talvez', 'depois', 'mais tarde']):
        return 'negative_response'
    if any(city in m for city in ['maceió', 'belém', 'florianópolis', 'floripa', 'vitória', 'vitoria', 'online']):
        return 'city_specific'
    return 'general'

def should_use_name(phone: str) -> bool:
    c = CONVERSATION_HISTORY[phone]
    return c == 0 or c % 5 == 0

# ======================
# Utilidades
# ======================
def _digits_only(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())

def to_wa_jid(phone: str) -> str:
    digits = _digits_only(phone)
    return f"{digits}@s.whatsapp.net" if digits else phone

def _unwrap_ephemeral(msg: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(msg, dict) and "ephemeralMessage" in msg:
        return msg["ephemeralMessage"].get("message", {}) or {}
    return msg or {}

def _extract_text(msg: Dict[str, Any]) -> str:
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
# IA Agent NATURAL (histórico curto + anti-reset)
# ======================
async def generate_response(user_message: str, user_name: str = "", phone: str = "") -> str:
    if AI_DRY_RUN:
        return f"[TESTE] Olá! Vi sua mensagem sobre '{user_message[:30]}...'. Como posso ajudar?"
    if not OPENAI_API_KEY or OpenAI is None:
        return "Desculpe, estou com problemas técnicos. Tente mais tarde."

    try:
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=25)

        intent = detect_user_intent(user_message)
        CONVERSATION_HISTORY[phone] += 1
        conversation_count = CONVERSATION_HISTORY[phone]
        use_name = should_use_name(phone) if user_name else False
        logger.info(f"🎯 Conversa #{conversation_count}, intenção: {intent}, usar nome: {use_name}")

        # Últimos turnos ajudam a interpretar “Sim/Ok”
        history = list(CHAT_HISTORY.get(phone, deque()))
        last_assistant = next((m["content"] for m in reversed(history) if m.get("role") == "assistant"), "")
        last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")

        system_prompt = f"""Você é Nat, atendente do CENAT (Centro de Estudos em Saúde Mental).
PERSONALIDADE: simpática, natural e prestativa. Responda como pessoa real no WhatsApp.

CONTEXTO CENAT:
{RAG_CONTEXT}

REGRAS DE FLUXO (ANTI-RESET):
• NÃO volte para perguntas genéricas se o histórico já estreitou o tema.
• Interprete “Sim/Ok/Beleza” como resposta à ÚLTIMA PERGUNTA feita pela Nat.
• Se a cidade já foi escolhida (ex.: Vitória) e o usuário pedir “programação” ou “inscrição”, mantenha essa cidade.
• Seja breve (2–3 linhas), ofereça o próximo passo, evite repetição e formalidade.
• Se faltar dado exato, ofereça link ou conexão com a equipe. Máx. 1 emoji opcional.

ÚLTIMO TURNO (para manter o fio):
• NAT: {last_assistant[-320:]}
• USUÁRIO: {last_user[-320:]}
"""

        # Monta mensagens com histórico + msg atual
        messages = [{"role": "system", "content": system_prompt}] + history + [
            {"role": "user", "content": user_message}
        ]

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=200,
            temperature=0.4,
            presence_penalty=0.2,
            frequency_penalty=0.3,
            top_p=0.9,
        )
        response = resp.choices[0].message.content or "Desculpe, não consegui processar sua mensagem."
        logger.info(f"💬 Resposta #{conversation_count} (intenção: {intent}): {response[:80]}...")
        return response.strip()

    except Exception as e:
        logger.error(f"Erro OpenAI: {e}")
        return "Ops, tive um problema técnico. Pode tentar de novo? Ou me chama em (47) 99242-8886!"

# ======================
# MEGA API - envio
# ======================
async def send_whatsapp(phone: str, message: str) -> bool:
    if not MEGA_API_TOKEN or not MEGA_INSTANCE_ID:
        logger.warning("MEGA API não configurada")
        return False
    norm_phone = _digits_only(phone)
    to = f"{norm_phone}@s.whatsapp.net" if norm_phone else phone
    url = f"{MEGA_API_BASE_URL}/rest/sendMessage/{MEGA_INSTANCE_ID}/text"
    headers = {"Authorization": f"Bearer {MEGA_API_TOKEN}", "Content-Type": "application/json"}
    payload = {"messageData": {"to": to, "text": message}}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code >= 400:
                logger.error(f"MEGA send failed {resp.status_code}: {resp.text}")
                resp.raise_for_status()
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
        "version": "2.1-NATURAL",
        "ai_mode": "DRY_RUN" if AI_DRY_RUN else "REAL",
        "context_loaded": len(RAG_CONTEXT) > 10,
        "mega_configured": bool(MEGA_API_TOKEN and MEGA_INSTANCE_ID),
        "optimizations": {
            "natural_conversation": True,
            "context_aware": True,
            "name_usage_control": True,
            "intent_detection": True,
            "chat_history_window": 12
        },
        "conversation_stats": {
            "active_conversations": len(CONVERSATION_HISTORY),
            "total_messages": sum(CONVERSATION_HISTORY.values())
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
    try:
        payload = await request.json()
    except Exception:
        logger.warning("Webhook: corpo não-JSON; ignorando.")
        return {"status": "ignored", "reason": "invalid_json"}

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

    if from_me and IGNORE_FROM_ME:
        return {"status": "ignored", "reason": "own_message"}
    if not phone or not text:
        return {"status": "ignored", "reason": "no_phone_or_text"}

    sent = LAST_SENT.get(phone)
    if sent:
        last_text, t0 = sent
        if text == last_text and (monotonic() - t0) < DEDUP_TTL:
            logger.info("🔁 Ignorado: eco do próprio envio recente.")
            return {"status": "ignored", "reason": "echo_recent_outbound"}

    dedup_key = f"{phone}:{hash(text)}"
    t_last = DEDUP.get(dedup_key)
    now = monotonic()
    if t_last and (now - t_last) < DEDUP_TTL:
        logger.info("⏱️ Ignorado: duplicata recente.")
        return {"status": "ignored", "reason": "duplicate"}
    DEDUP[dedup_key] = now

    background_tasks.add_task(process_and_reply, phone, text, push_name)
    return {"status": "processing"}

@app.post("/send-message")
async def send_message_manual(request: SendMessageRequest):
    success = await send_whatsapp(request.phone, request.message)
    if success:
        return {"status": "sent", "phone": request.phone, "message": request.message[:60]}
    raise HTTPException(status_code=500, detail="Falha ao enviar mensagem")

@app.get("/mega-status")
async def mega_status():
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

# ====== RAG ======
@app.post("/reload-context")
async def reload_context():
    global RAG_CONTEXT, _RAG_SIG
    RAG_CONTEXT = load_context()
    _RAG_SIG = data_signature()
    logger.info(f"🔄 RAG recarregado: {len(RAG_CONTEXT)} caracteres")
    return {"status": "ok", "context_len": len(RAG_CONTEXT), "version": "natural"}

@app.get("/context/preview")
async def context_preview(n: int = 1500):
    n = max(0, min(n, 8000))
    return {"preview": RAG_CONTEXT[:n], "len": len(RAG_CONTEXT), "version": "natural-conversation"}

# ====== Conversas ======
@app.post("/reset-conversation/{phone}")
async def reset_conversation(phone: str):
    clean = _digits_only(phone)
    CONVERSATION_HISTORY.pop(clean, None)
    CHAT_HISTORY.pop(clean, None)
    return {"status": "reset", "phone": clean}

@app.get("/conversation-stats")
async def conversation_stats():
    return {
        "active_conversations": len(CONVERSATION_HISTORY),
        "total_messages": sum(CONVERSATION_HISTORY.values()),
        "conversations": {p: c for p, c in CONVERSATION_HISTORY.items()},
        "history_window": 12,
    }

# ======================
# Worker
# ======================
async def process_and_reply(phone: str, message: str, user_name: str):
    try:
        async with LOCKS[phone]:
            # salva turno do usuário
            CHAT_HISTORY[phone].append({"role": "user", "content": message})
            response = await generate_response(message, user_name, phone)
            logger.info(f"🤖 IA gerou resposta natural: {response}")
            # salva turno da Nat
            CHAT_HISTORY[phone].append({"role": "assistant", "content": response})
            ok = await send_whatsapp(phone, response)
            if ok:
                logger.info(f"✅ Resposta enviada para {user_name}: {response[:100]}...")
            else:
                logger.error(f"❌ Falha ao enviar para {user_name}")
                logger.info(f"📝 Resposta que seria enviada: {response}")
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")

# ======================
# RAG watcher
# ======================
async def rag_watcher():
    global _RAG_SIG, RAG_CONTEXT
    logger.info(f"👀 RAG watcher NATURAL ativo em '{RAG_DIR}' a cada {RAG_WATCH_INTERVAL}s")
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
    logger.info("🚀 WhatsApp AI Agent v2.1 CONVERSAÇÃO NATURAL iniciado")
    logger.info("💬 Melhorias: histórico curto (12), anti-reset de assunto, uso inteligente do nome")
    logger.info(f"📄 Contexto RAG: {len(RAG_CONTEXT)} chars | 🤖 Modo IA: {'DRY_RUN' if AI_DRY_RUN else 'REAL'}")
    logger.info(f"📱 MEGA API: {'configurada' if (MEGA_API_TOKEN and MEGA_INSTANCE_ID) else 'NÃO CONFIGURADA'}")
    logger.info(f"🧰 Flags: IGNORE_FROM_ME={IGNORE_FROM_ME}, DEDUP_TTL={DEDUP_TTL}s")
    logger.info(f"📂 RAG_DIR='{RAG_DIR}', AUTO_RELOAD={RAG_AUTO_RELOAD}, INTERVALO={RAG_WATCH_INTERVAL}s")
    if RAG_AUTO_RELOAD:
        asyncio.create_task(rag_watcher())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "8000")))
