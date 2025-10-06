# api/main.py - WhatsApp AI Agent v3.5 CENAT
# Respeita quando o cliente quer encerrar

import os
import asyncio
import re
from time import monotonic
from collections import defaultdict
from typing import Dict, Any, Optional, List
import hashlib

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
import httpx
from loguru import logger

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

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
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
AI_DRY_RUN = os.getenv("AI_DRY_RUN", "0") == "1"

IGNORE_FROM_ME = os.getenv("IGNORE_FROM_ME", "1") == "1"
DEDUP_TTL = float(os.getenv("DEDUP_TTL", "12"))

RAG_DIR = os.getenv("RAG_DIR", "data")
RAG_AUTO_RELOAD = os.getenv("RAG_AUTO_RELOAD", "1") == "1"
RAG_WATCH_INTERVAL = float(os.getenv("RAG_WATCH_INTERVAL", "3"))

MEGA_API_BASE_URL = os.getenv("MEGA_API_BASE_URL", "https://apistart01.megaapi.com.br")
MEGA_API_TOKEN = os.getenv("MEGA_API_TOKEN", "")
MEGA_INSTANCE_ID = os.getenv("MEGA_INSTANCE_ID", "")

WHATSAPP_ATENDIMENTO = "+55 47 99242-8886"

app = FastAPI(title="WhatsApp AI Agent CENAT", version="3.5")

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

CONVERSATION_HISTORY: Dict[str, List[Dict[str, str]]] = defaultdict(list)
LAST_CONGRESS: Dict[str, str] = {}

# ======================
# Detecção de Encerramento
# ======================
SINAIS_ENCERRAMENTO = [
    "não", "nao", "não obrigado", "nao obrigado", "só isso", "so isso",
    "era só isso", "era so isso", "tranquilo", "valeu", "obrigado",
    "obrigada", "tudo bem", "tá bom", "ta bom", "ok obrigado",
    "agora não", "agora nao", "depois", "mais tarde"
]

def detectar_encerramento(mensagem: str) -> bool:
    """Detecta se o cliente quer encerrar a conversa."""
    msg_lower = mensagem.lower().strip()
    
    # Mensagens curtas que são claramente encerramento
    if msg_lower in ["não", "nao", "não obrigado", "nao obrigado", "só isso", "so isso", "valeu", "tranquilo", "ok"]:
        return True
    
    # Frases de encerramento
    for sinal in SINAIS_ENCERRAMENTO:
        if sinal in msg_lower:
            return True
    
    return False

# ======================
# Mapa de Congressos
# ======================
CONGRESSOS_MAP = {
    "maceio": {
        "nome": "Maceió/AL",
        "data": "05 e 06/09",
        "url": "https://cenatsaudemental.com/boas-praticas-em-saude-mental-maceio-2025",
        "palavras": ["maceió", "maceio", "alagoas", "al"]
    },
    "belem": {
        "nome": "Belém/PA",
        "data": "09 e 10/09",
        "url": "https://cenatsaudemental.com/v-congresso-internacional-bpsm-belem-2025",
        "palavras": ["belém", "belem", "pará", "para", "pa"]
    },
    "floripa": {
        "nome": "Florianópolis/SC",
        "data": "21 e 22/10",
        "url": "https://cenatsaudemental.com/boas-praticas-em-saude-mental-floripa-2025",
        "palavras": ["florianópolis", "florianopolis", "floripa", "santa catarina", "sc"]
    },
    "vitoria": {
        "nome": "Vitória/ES",
        "data": "24 e 25/10",
        "url": "https://cenatsaudemental.com/novas-abordagens-sm-vitoria-2025",
        "palavras": ["vitória", "vitoria", "espírito santo", "espirito santo", "es"]
    },
    "ouvidores": {
        "nome": "Ouvidores de Vozes (Online)",
        "data": "05 e 06/12",
        "url": "https://cenatsaudemental.com/congresso-online-ouvidores-2025",
        "palavras": ["ouvidores", "ouvidor", "vozes", "online"]
    }
}

def detectar_congresso_especifico(mensagem: str) -> Optional[str]:
    msg_lower = mensagem.lower()
    for key, info in CONGRESSOS_MAP.items():
        if any(palavra in msg_lower for palavra in info["palavras"]):
            return key
    return None

# ======================
# RAG simples
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
# Sistema de Intenções
# ======================
INTENCOES = {
    "saudacao": ["oi", "olá", "ola", "bom dia", "boa tarde", "boa noite"],
    "congressos_lista": ["congressos", "eventos", "próximos eventos", "quais eventos"],
    "congresso_programacao": ["programação", "programacao", "grade", "horários", "cronograma"],
    "congresso_link": ["link", "site"],
    "certificados": ["certificado", "certificação", "diploma"],
    "reembolso": ["reembolso", "devolução", "cancelar", "estornar"],
    "trabalho_submeter": ["submeter trabalho", "enviar trabalho", "submissão"],
    "trabalho_status": ["não recebi devolutiva", "sem retorno trabalho", "status trabalho"],
    "trabalho_apresentacao": ["apresentação trabalho", "apresentar trabalho", "pôster", "poster", "banner"],
    "gravacao": ["gravação", "gravado", "assistir depois", "replay"],
    "transmissao_problema": ["transmissão não funciona", "não abre transmissão", "travando"],
    "ingresso": ["não recebi ingresso", "link evento", "acesso evento"],
    "comunidade": ["comunidade", "cursos gravados", "assinatura"],
    "comunidade_cursos": ["lista de cursos", "cursos disponíveis", "ementas"],
    "intercambio": ["intercâmbio", "intercambio", "viagem", "exterior"],
    "pos": ["pós", "pos graduação", "especialização"],
    "desconto": ["desconto", "promoção", "cupom", "valor", "preço"],
    "pagamento_erro": ["cartão não passou", "erro pagamento"],
    "boleto": ["novo boleto", "segunda via boleto", "perdi prazo"],
    "empenho": ["empenho", "nota de empenho"],
    "materiais_gratuitos": ["materiais gratuitos", "material grátis", "conteúdo gratuito"],
}

def detectar_intencao(mensagem: str) -> str:
    msg_lower = mensagem.lower()
    
    if len(msg_lower.split()) <= 2:
        for palavra in INTENCOES["saudacao"]:
            if palavra == msg_lower.strip():
                return "saudacao"
    
    if any(p in msg_lower for p in ["programação", "programacao", "grade", "horários"]):
        return "congresso_programacao"
    
    for intencao, palavras in INTENCOES.items():
        for palavra in palavras:
            if palavra in msg_lower:
                return intencao
    
    return "geral"

# ======================
# Respostas Padronizadas
# ======================
RESPOSTAS_PADRAO = {
    "encerramento": """Tudo bem! Qualquer dúvida, é só chamar. Tenha um ótimo dia!""",

    "saudacao": """Olá! Seja bem-vindo(a) ao CENAT. Como posso ajudar?

Posso te auxiliar com congressos, certificados, pós-graduação, comunidade online ou intercâmbios.""",

    "congressos_lista": """Temos esses congressos confirmados para 2025:

- Maceió/AL – 05 e 06/09
- Belém/PA – 09 e 10/09  
- Florianópolis/SC – 21 e 22/10
- Vitória/ES – 24 e 25/10
- Ouvidores de Vozes (Online) – 05 e 06/12

Sobre qual você gostaria de saber mais?""",

    "certificados_congressos": """Para certificados de congressos/eventos/seminários, acesse:

https://doity.com.br/area-do-participante/certificado

Preencha com o e-mail usado na inscrição. Se tiver dificuldades, me avise!""",

    "certificados_cursos": """#HUMANO

Para certificados de cursos/comunidade:

WhatsApp: +55 47 99242-8886
E-mail: atendimento@cenatcursos.com.br""",

    "reembolso": """#HUMANO

Para solicitar reembolso:

WhatsApp: +55 47 99242-8886
E-mail: atendimento@cenatcursos.com.br""",

    "trabalho_submeter": """#HUMANO

Para submeter trabalhos:

WhatsApp: +55 47 99242-8886
E-mail: atendimento@cenatcursos.com.br""",

    "trabalho_status": """#HUMANO

Para verificar status:

WhatsApp: +55 47 99242-8886
E-mail: atendimento@cenatcursos.com.br""",

    "trabalho_apresentacao": """#HUMANO

Para orientações sobre apresentação:

WhatsApp: +55 47 99242-8886
E-mail: atendimento@cenatcursos.com.br""",

    "gravacao": """#HUMANO

Para acesso a gravações:

WhatsApp: +55 47 99242-8886
E-mail: atendimento@cenatcursos.com.br""",

    "transmissao_problema": """#HUMANO

Para problemas técnicos:

WhatsApp: +55 47 99242-8886
E-mail: atendimento@cenatcursos.com.br""",

    "ingresso": """#HUMANO

Verifique seu e-mail (inclusive Spam).

Se não localizar:
WhatsApp: +55 47 99242-8886""",

    "comunidade": """A Comunidade Novas Abordagens em Saúde Mental oferece:

- Mais de 30 cursos gravados (~250h)
- Encontros mensais ao vivo
- Certificados de 20-80h

Inscreva-se:
https://cenatsaudemental.com/comunidademsaudemental

Posso te indicar cursos de alguma área específica?""",

    "comunidade_cursos": """Lista completa:
https://cenatsaudemental.com/comunidademsaudemental#section-17454852

Quer saber sobre alguma área específica?""",

    "intercambio": """Nossos intercâmbios incluem visitas, workshops, palestras e certificado.

Destinos: Dinamarca, Trieste, Portugal, Inglaterra

https://cenatsaudemental.com/cenat-intercambios

Algum destino te interessa?""",

    "pos": """Para pós-graduação:

E-mail: secretaria@cenatcursos.com.br""",

    "desconto": """Sim, temos descontos:

- Estudantes com carteira
- Por lote
- Grupos: 10-20 pessoas (10%) / +20 (15%)

Qual evento te interessa?""",

    "pagamento_erro": """#HUMANO

Para problemas com pagamento:

WhatsApp: +55 47 99242-8886
E-mail: atendimento@cenatcursos.com.br""",

    "boleto": """#HUMANO

Para nova via de boleto:

WhatsApp: +55 47 99242-8886
E-mail: atendimento@cenatcursos.com.br""",

    "empenho": """#HUMANO

Para pagamento via empenho:

WhatsApp: +55 47 99242-8886
E-mail: atendimento@cenatcursos.com.br""",

    "materiais_gratuitos": """Materiais gratuitos:

https://cenatsaudemental.com/""",
}

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
# IA Agent (RESPEITA ENCERRAMENTO)
# ======================
async def generate_response(user_message: str, user_name: str = "", phone: str = "") -> str:
    """Gera resposta respeitando quando o cliente quer encerrar."""
    
    if AI_DRY_RUN:
        return f"Olá {user_name or 'Cliente'}! Como posso ajudar?"

    # 1. PRIORIDADE MÁXIMA: Detectar encerramento
    if detectar_encerramento(user_message):
        logger.info("🛑 Cliente quer encerrar - respeitando")
        return RESPOSTAS_PADRAO["encerramento"]

    # 2. Detectar congresso específico
    congresso_atual = detectar_congresso_especifico(user_message)
    if congresso_atual:
        LAST_CONGRESS[phone] = congresso_atual
        logger.info(f"📍 Congresso: {congresso_atual}")

    # 3. Detectar intenção
    intencao = detectar_intencao(user_message)
    logger.info(f"🎯 Intenção: {intencao}")

    # 4. PROGRAMAÇÃO
    if intencao == "congresso_programacao":
        ultimo = LAST_CONGRESS.get(phone)
        
        if ultimo and ultimo in CONGRESSOS_MAP:
            info = CONGRESSOS_MAP[ultimo]
            return f"""A programação completa está aqui:

{info['url']}

Qualquer dúvida, me avise!"""
        else:
            return """Qual congresso?

- Maceió/AL
- Belém/PA
- Florianópolis/SC
- Vitória/ES
- Ouvidores de Vozes"""

    # 5. LINK
    if intencao == "congresso_link":
        ultimo = LAST_CONGRESS.get(phone)
        
        if ultimo and ultimo in CONGRESSOS_MAP:
            info = CONGRESSOS_MAP[ultimo]
            return f"""{info['url']}

Se precisar de mais informações, estou à disposição!"""
        else:
            return """Qual congresso?

- Maceió/AL
- Belém/PA
- Florianópolis/SC
- Vitória/ES
- Ouvidores de Vozes"""

    # 6. Certificados
    if intencao == "certificados":
        msg_lower = user_message.lower()
        if any(p in msg_lower for p in ["congresso", "evento", "seminário"]):
            return RESPOSTAS_PADRAO["certificados_congressos"]
        elif any(p in msg_lower for p in ["curso", "comunidade"]):
            return RESPOSTAS_PADRAO["certificados_cursos"]
        else:
            return """Certificado de congresso ou curso?

- Congresso/Evento: https://doity.com.br/area-do-participante/certificado
- Curso/Comunidade: WhatsApp +55 47 99242-8886"""

    # 7. Respostas diretas
    if intencao in RESPOSTAS_PADRAO:
        return RESPOSTAS_PADRAO[intencao]

    # 8. IA (só casos não mapeados)
    if not OPENAI_API_KEY or OpenAI is None:
        return f"#HUMANO\n\nWhatsApp: {WHATSAPP_ATENDIMENTO}"

    try:
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=20)
        
        history = CONVERSATION_HISTORY.get(phone, [])[-8:]
        
        congresso_context = ""
        if phone in LAST_CONGRESS and LAST_CONGRESS[phone] in CONGRESSOS_MAP:
            info = CONGRESSOS_MAP[LAST_CONGRESS[phone]]
            congresso_context = f"\n\nÚLTIMO CONGRESSO: {info['nome']}\nLink: {info['url']}"
        
        system_prompt = f"""Você é atendente CENAT. Tom profissional, educado.

CRÍTICO:
- NUNCA invente informações
- Se não souber: #HUMANO
- 3-4 linhas máximo
- Seja cordial mas conciso{congresso_context}

CONTEXTO:
{RAG_CONTEXT[:3000]}

CONGRESSOS:
- Maceió/AL – 05 e 06/09 - https://cenatsaudemental.com/boas-praticas-em-saude-mental-maceio-2025
- Belém/PA – 09 e 10/09 - https://cenatsaudemental.com/v-congresso-internacional-bpsm-belem-2025
- Florianópolis/SC – 21 e 22/10 - https://cenatsaudemental.com/boas-praticas-em-saude-mental-floripa-2025
- Vitória/ES – 24 e 25/10 - https://cenatsaudemental.com/novas-abordagens-sm-vitoria-2025
- Ouvidores (Online) – 05 e 06/12 - https://cenatsaudemental.com/congresso-online-ouvidores-2025

CONTATOS:
- WhatsApp: {WHATSAPP_ATENDIMENTO}
- Email: atendimento@cenatcursos.com.br"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=150,
            temperature=0.4,
        )
        
        resposta = resp.choices[0].message.content or "Não consegui processar."
        
        CONVERSATION_HISTORY[phone].append({"role": "user", "content": user_message})
        CONVERSATION_HISTORY[phone].append({"role": "assistant", "content": resposta})
        
        if len(CONVERSATION_HISTORY[phone]) > 16:
            CONVERSATION_HISTORY[phone] = CONVERSATION_HISTORY[phone][-16:]
        
        return resposta
        
    except Exception as e:
        logger.error(f"Erro: {e}")
        return f"#HUMANO\n\nWhatsApp: {WHATSAPP_ATENDIMENTO}"

# ======================
# MEGA API
# ======================
async def send_whatsapp(phone: str, message: str) -> bool:
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
                logger.error(f"MEGA fail {resp.status_code}: {resp.text}")
                resp.raise_for_status()
            LAST_SENT[norm_phone] = (message.strip(), monotonic())
            logger.info(f"✅ Enviado: {message[:60]}")
            return True
    except Exception as e:
        logger.error(f"Erro envio: {e}")
        return False

# ======================
# ENDPOINTS
# ======================
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "3.5-RESPEITA-NAO",
        "temperature": 0.4,
        "ai_mode": "DRY_RUN" if AI_DRY_RUN else "REAL",
        "model": MODEL_NAME,
    }

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        payload = await request.json()
    except Exception:
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
    phone = _digits_only(remote_jid)
    from_me = bool(key.get("fromMe"))
    text = _extract_text(msg)

    logger.info(f"📩 {remote_jid} | '{text[:50]}'")

    if from_me and IGNORE_FROM_ME:
        return {"status": "ignored", "reason": "own_message"}

    if not phone or not text:
        return {"status": "ignored", "reason": "no_phone_or_text"}

    sent = LAST_SENT.get(phone)
    if sent:
        last_text, t0 = sent
        if text == last_text and (monotonic() - t0) < DEDUP_TTL:
            return {"status": "ignored", "reason": "echo"}

    dedup_key = f"{phone}:{hash(text)}"
    t_last = DEDUP.get(dedup_key)
    now = monotonic()
    if t_last and (now - t_last) < DEDUP_TTL:
        return {"status": "ignored", "reason": "duplicate"}
    DEDUP[dedup_key] = now

    background_tasks.add_task(process_and_reply, phone, text, push_name)
    return {"status": "processing"}

@app.post("/send-message")
async def send_message_manual(request: SendMessageRequest):
    success = await send_whatsapp(request.phone, request.message)
    if success:
        return {"status": "sent"}
    raise HTTPException(status_code=500, detail="Falha")

@app.get("/mega-status")
async def mega_status():
    if not MEGA_API_TOKEN or not MEGA_INSTANCE_ID:
        raise HTTPException(status_code=400, detail="MEGA não configurada")

    url = f"{MEGA_API_BASE_URL}/rest/instance/{MEGA_INSTANCE_ID}"
    headers = {"Authorization": f"Bearer {MEGA_API_TOKEN}"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-context")
async def reload_context():
    global RAG_CONTEXT, _RAG_SIG
    RAG_CONTEXT = load_context()
    _RAG_SIG = data_signature()
    logger.info(f"🔄 RAG: {len(RAG_CONTEXT)} chars")
    return {"status": "ok", "context_len": len(RAG_CONTEXT)}

@app.get("/context/preview")
async def context_preview(n: int = 800):
    n = max(0, min(n, 5000))
    return {"preview": RAG_CONTEXT[:n], "len": len(RAG_CONTEXT)}

# ======================
# Worker
# ======================
async def process_and_reply(phone: str, message: str, user_name: str):
    try:
        async with LOCKS[phone]:
            response = await generate_response(message, user_name, phone)
            
            if "#HUMANO" in response:
                response = response.replace("#HUMANO", "").strip()
                if not response:
                    response = f"Vou te transferir.\n\nWhatsApp: {WHATSAPP_ATENDIMENTO}"
            
            await send_whatsapp(phone, response)
            
    except Exception as e:
        logger.error(f"Erro: {e}")

# ======================
# Watcher
# ======================
async def rag_watcher():
    global _RAG_SIG, RAG_CONTEXT
    logger.info("👀 Watcher ativo")
    while True:
        try:
            sig = data_signature()
            if sig != _RAG_SIG:
                RAG_CONTEXT = load_context()
                _RAG_SIG = sig
                logger.info(f"🔄 RAG reload")
        except Exception as e:
            logger.warning(f"Watcher: {e}")
        await asyncio.sleep(RAG_WATCH_INTERVAL)

# ======================
# Startup
# ======================
@app.on_event("startup")
async def startup():
    logger.info("🚀 v3.5 RESPEITA NÃO")
    logger.info(f"📄 RAG: {len(RAG_CONTEXT)} chars")
    logger.info(f"🤖 {'DRY_RUN' if AI_DRY_RUN else f'REAL ({MODEL_NAME})'}")
    if RAG_AUTO_RELOAD:
        asyncio.create_task(rag_watcher())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "8000")))
    