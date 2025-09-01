# api/main.py - WhatsApp AI Agent v2.2 NATURAL (catálogo dinâmico + histórico curto + guard-rails)

import os
import json
import asyncio
from time import monotonic
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import hashlib
import unicodedata

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

app = FastAPI(title="WhatsApp AI Agent", version="2.2")

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
# Estados e memória curta
# ======================
LAST_SENT: Dict[str, Tuple[str, float]] = {}
DEDUP: Dict[str, float] = {}
LOCKS: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
CONVERSATION_HISTORY: Dict[str, int] = defaultdict(int)
CHAT_HISTORY: Dict[str, deque] = defaultdict(lambda: deque(maxlen=12))

@dataclass
class SessionState:
    product_slug: Optional[str] = None  # slug do produto selecionado
    last_question: Optional[str] = None

SESSION: Dict[str, SessionState] = defaultdict(SessionState)

# ======================
# Catálogo dinâmico (JSON em data/catalog.json)
# ======================
CATALOG_PATH = None
CATALOG: Dict[str, Any] = {}          # {"products":[...]}
PRODUCT_BY_SLUG: Dict[str, Dict[str, Any]] = {}
ALIAS_INDEX: Dict[str, str] = {}       # alias_normalizado -> slug

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower().strip()

def _catalog_path() -> str:
    global CATALOG_PATH
    if CATALOG_PATH:
        return CATALOG_PATH
    CATALOG_PATH = os.path.join(RAG_DIR, "catalog.json")
    return CATALOG_PATH

def load_catalog() -> None:
    """Carrega o catálogo JSON (produtos, aliases, links) de data/catalog.json."""
    global CATALOG, PRODUCT_BY_SLUG, ALIAS_INDEX
    PRODUCT_BY_SLUG = {}
    ALIAS_INDEX = {}
    path = _catalog_path()
    if not os.path.exists(path):
        logger.warning(f"📦 Catálogo não encontrado em {path} (seguindo sem catálogo).")
        CATALOG = {"products": []}
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            CATALOG = json.load(f)
    except Exception as e:
        logger.error(f"Erro ao ler catálogo: {e}")
        CATALOG = {"products": []}
        return

    for p in CATALOG.get("products", []):
        slug = p.get("slug")
        if not slug:
            continue
        PRODUCT_BY_SLUG[slug] = p
        aliases = list(p.get("aliases", []))
        title = p.get("title") or ""
        if title:
            aliases.append(title)
        for a in aliases:
            a_norm = _strip_accents(a)
            if not a_norm:
                continue
            ALIAS_INDEX.setdefault(a_norm, slug)

def find_product_in_text(text: str) -> Optional[str]:
    """Procura por algum alias do catálogo no texto e retorna o slug encontrado."""
    if not ALIAS_INDEX:
        return None
    t = _strip_accents(text)
    for alias_norm, slug in ALIAS_INDEX.items():
        if alias_norm and alias_norm in t:
            return slug
    return None

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
            if not (file.endswith(".txt")):
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
    """Hash do estado de data/ para detectar mudanças (.txt e .json)."""
    h = hashlib.sha1()
    base = RAG_DIR
    if not os.path.exists(base):
        return ""
    for root, _, files in os.walk(base):
        for file in sorted(files):
            if not (file.endswith(".txt") or file.endswith(".json")):
                continue
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
load_catalog()
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
    if find_product_in_text(message):
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

def update_session_with_text(phone: str, text: str):
    """Atualiza o estado do contato a partir do texto (seleção de produto)."""
    slug = find_product_in_text(text)
    if slug:
        SESSION[phone].product_slug = slug

# ======================
# IA Agent NATURAL (histórico curto + catálogo)
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
        count = CONVERSATION_HISTORY[phone]
        use_name = should_use_name(phone) if user_name else False

        # histórico curto
        history = list(CHAT_HISTORY.get(phone, deque()))
        last_assistant = next((m["content"] for m in reversed(history) if m.get("role") == "assistant"), "")
        last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")

        # estado corrente
        st = SESSION.get(phone, SessionState())
        prod = PRODUCT_BY_SLUG.get(st.product_slug or "", {})
        prod_title = prod.get("title", "")
        prod_type = prod.get("type", "")
        enroll_url = prod.get("enroll_url", "")
        program_url = prod.get("program_url", "")
        dates = prod.get("dates", "")
        location = prod.get("location", "")

        logger.info(f"🎯 Conversa #{count} | intenção={intent} | produto={st.product_slug or 'indefinido'}")

        system_prompt = f"""Você é Nat, atendente do CENAT (Centro de Estudos em Saúde Mental).
PERSONALIDADE: simpática, natural e prestativa. Responda como pessoa real no WhatsApp.

CONTEXTO CENAT:
{RAG_CONTEXT}

ESTADO ATUAL (use para manter o fio da conversa):
• produto_selecionado: {prod_title or 'indefinido'} (tipo: {prod_type or '—'})
• datas: {dates or '—'}
• local: {location or '—'}
• link_inscrição: {enroll_url or '—'}
• link_programação: {program_url or '—'}

REGRAS IMPORTANTES:
• NÃO volte ao menu genérico se já houver produto_selecionado.
• Se o usuário pedir "inscrição" e houver link_inscrição, responda direto com esse link (sem citar outros).
• Se pedir "programação" e houver link_programação, responda com esse link do produto selecionado.
• Interprete "Sim/Ok/Beleza" como resposta à ÚLTIMA pergunta feita pela Nat.
• 2–3 linhas, próximas do natural. Se faltar dado, ofereça o link ou encaminhamento. Máx. 1 emoji opcional.

ÚLTIMO TURNO (para manter o fio):
• NAT: {last_assistant[-320:]}
• USUÁRIO: {last_user[-320:]}
"""

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
        "version": "2.2-NATURAL",
        "ai_mode": "DRY_RUN" if AI_DRY_RUN else "REAL",
        "context_loaded": len(RAG_CONTEXT) > 10,
        "mega_configured": bool(MEGA_API_TOKEN and MEGA_INSTANCE_ID),
        "optimizations": {
            "natural_conversation": True,
            "context_aware": True,
            "name_usage_control": True,
            "intent_detection": True,
            "chat_history_window": 12,
            "catalog_driven": True
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
            "catalog_products": len(CATALOG.get("products", [])),
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

    # Atualiza estado pelo texto (produto)
    update_session_with_text(phone, text)

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

# ====== RAG / Catálogo ======
@app.post("/reload-context")
async def reload_context():
    global RAG_CONTEXT, _RAG_SIG
    RAG_CONTEXT = load_context()
    load_catalog()
    _RAG_SIG = data_signature()
    logger.info(f"🔄 RAG+Catálogo recarregados: ctx={len(RAG_CONTEXT)} chars | produtos={len(CATALOG.get('products', []))}")
    return {"status": "ok", "context_len": len(RAG_CONTEXT), "products": len(CATALOG.get("products", [])), "version": "natural"}

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
    SESSION.pop(clean, None)
    return {"status": "reset", "phone": clean}

@app.get("/conversation-stats")
async def conversation_stats():
    return {
        "active_conversations": len(CONVERSATION_HISTORY),
        "total_messages": sum(CONVERSATION_HISTORY.values()),
        "history_window": 12,
        "sessions": {p: {"product_slug": SESSION[p].product_slug} for p in SESSION},
        "catalog_products": len(CATALOG.get("products", [])),
    }

# ======================
# Worker (guard-rails determinísticos)
# ======================
async def process_and_reply(phone: str, message: str, user_name: str):
    try:
        async with LOCKS[phone]:
            CHAT_HISTORY[phone].append({"role": "user", "content": message})

            intent_now = detect_user_intent(message)

            # Resposta IA (usa histórico + estado + catálogo)
            response = await generate_response(message, user_name, phone)

            # Guard-rails: se já temos produto e a intenção é direta, responda de forma determinística
            st = SESSION.get(phone, SessionState())
            prod = PRODUCT_BY_SLUG.get(st.product_slug or "", {})

            if st.product_slug and prod:
                if intent_now == "enrollment" and prod.get("enroll_url"):
                    response = f"Perfeito! Para se inscrever em **{prod.get('title','')}**, acesse {prod['enroll_url']} e siga as instruções. Qualquer dúvida, estou aqui. 🙂"
                elif intent_now == "schedule" and prod.get("program_url"):
                    response = f"A programação de **{prod.get('title','')}** está em {prod['program_url']}. Quer que eu destaque os principais horários?"

            CHAT_HISTORY[phone].append({"role": "assistant", "content": response})
            ok = await send_whatsapp(phone, response)
            if ok:
                logger.info(f"✅ Resposta enviada para {user_name}: {response[:120]}...")
            else:
                logger.error(f"❌ Falha ao enviar para {user_name}")
                logger.info(f"📝 Resposta que seria enviada: {response}")
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")

# ======================
# Watcher (RAG + Catálogo)
# ======================
async def rag_watcher():
    global _RAG_SIG, RAG_CONTEXT
    logger.info(f"👀 RAG watcher NATURAL ativo em '{RAG_DIR}' a cada {RAG_WATCH_INTERVAL}s")
    while True:
        try:
            sig = data_signature()
            if sig != _RAG_SIG:
                logger.info("🪄 Mudanças detectadas em data/: recarregando RAG + Catálogo...")
                RAG_CONTEXT = load_context()
                load_catalog()
                _RAG_SIG = sig
                logger.info(f"🔄 Recarregado: ctx={len(RAG_CONTEXT)} chars | produtos={len(CATALOG.get('products', []))}")
        except Exception as e:
            logger.warning(f"Watcher RAG: {e}")
        await asyncio.sleep(RAG_WATCH_INTERVAL)

# ======================
# Startup
# ======================
@app.on_event("startup")
async def startup():
    logger.info("🚀 WhatsApp AI Agent v2.2 NATURAL iniciado (catálogo dinâmico)")
    logger.info(f"📄 Contexto RAG: {len(RAG_CONTEXT)} chars | Produtos no catálogo: {len(CATALOG.get('products', []))}")
    logger.info(f"🤖 Modo IA: {'DRY_RUN (teste)' if AI_DRY_RUN else 'REAL (OpenAI)'}")
    logger.info(f"📱 MEGA API: {'configurada' if (MEGA_API_TOKEN and MEGA_INSTANCE_ID) else 'NÃO CONFIGURADA'}")
    logger.info(f"🧰 Flags: IGNORE_FROM_ME={IGNORE_FROM_ME}, DEDUP_TTL={DEDUP_TTL}s")
    logger.info(f"📂 RAG_DIR='{RAG_DIR}', AUTO_RELOAD={RAG_AUTO_RELOAD}, INTERVALO={RAG_WATCH_INTERVAL}s")
    if RAG_AUTO_RELOAD:
        asyncio.create_task(rag_watcher())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "8000")))
