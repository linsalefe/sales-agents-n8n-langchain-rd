# api/main.py - WhatsApp AI Agent v3.0 CENAT
# Atualizado com Guia de Respostas Padronizadas + Contexto Aguçado

import os
import asyncio
import re
from time import monotonic
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple
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

# Número do atendimento humano
WHATSAPP_ATENDIMENTO = "+55 47 99242-8886"

app = FastAPI(title="WhatsApp AI Agent CENAT", version="3.0")

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

# Histórico de conversas por usuário (para contexto aguçado)
CONVERSATION_HISTORY: Dict[str, List[Dict[str, str]]] = defaultdict(list)

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
# Sistema de Intenções (baseado no PDF + ajustes)
# ======================
INTENCOES = {
    "saudacao": ["oi", "olá", "ola", "bom dia", "boa tarde", "boa noite", "oie", "opa"],
    # removido "seminário" daqui para não conflitar com gravação
    "congressos": ["congresso", "eventos", "próximos eventos", "onde tem evento", "palestras"],
    "certificados": ["certificado", "certificação", "diploma"],
    "reembolso": ["reembolso", "devolução", "devolver dinheiro", "cancelar", "estornar"],
    "trabalho": ["trabalho", "submeter", "submissão", "apresentação", "pôster", "poster", "banner", "artigo"],
    # nova intenção priorizada para status/devolutiva de trabalho
    "trabalho_nao_recebi": [
        "não recebi", "nao recebi", "sem retorno", "sem devolutiva", "devolutiva",
        "status do trabalho", "status da submissão", "status da submissao",
        "foi aceito", "nao chegou email", "não chegou e-mail"
    ],
    "gravacao": ["gravação", "gravado", "assistir depois", "replay", "reprise", "seminário", "seminario"],
    "transmissao": ["transmissão", "não consigo assistir", "não abre", "não funciona", "travando"],
    "publicacao": ["publicação", "anais", "publicado"],
    "programacao": ["programação", "horários", "grade", "agenda", "cronograma"],
    "ingresso": ["ingresso", "receber ingresso", "link evento", "acesso", "ingresso por e-mail", "ingresso por email"],
    "empenho": ["empenho", "nota de empenho", "órgão público", "prefeitura"],
    "comunidade": ["comunidade", "cursos gravados", "assinatura", "plataforma"],
    # nova intenção: cliente perguntando a lista de cursos
    "comunidade_cursos": [
        "quais são os cursos", "quais sao os cursos", "lista de cursos", "cursos disponíveis",
        "cursos disponiveis", "ementas", "grade", "catálogo", "catalogo"
    ],
    "intercambio": ["intercâmbio", "intercambio", "viagem", "lisboa", "buenos aires", "exterior"],
    "pos": ["pós", "pos graduação", "pos-graduação", "especialização", "turmas", "mestrado"],
    "desconto": ["desconto", "promoção", "cupom", "valor", "preço", "barato"],
    "pagamento": ["pagamento", "cartão não passou", "erro pagamento", "boleto", "pix"],
}

def detectar_intencao(mensagem: str) -> str:
    """Detecta a intenção da mensagem com base em palavras-chave."""
    msg_lower = mensagem.lower()

    # Prioridade para saudação se for muito curta
    if len(msg_lower.split()) <= 2:
        for palavra in INTENCOES["saudacao"]:
            if palavra in msg_lower:
                return "saudacao"

    # PRIORIDADES ESPECIAIS
    # 1) Dúvidas sobre gravação (antes de congressos etc.)
    for palavra in INTENCOES.get("gravacao", []):
        if palavra in msg_lower:
            return "gravacao"

    # 2) Status/devolutiva de trabalho
    for palavra in INTENCOES.get("trabalho_nao_recebi", []):
        if palavra in msg_lower:
            return "trabalho_nao_recebi"

    # 3) Comunidade - pedido de lista de cursos
    for palavra in INTENCOES.get("comunidade_cursos", []):
        if palavra in msg_lower:
            return "comunidade_cursos"

    # Demais intenções
    for intencao, palavras in INTENCOES.items():
        if intencao in ("gravacao", "trabalho_nao_recebi", "comunidade_cursos"):
            continue
        for palavra in palavras:
            if palavra in msg_lower:
                return intencao

    return "geral"

# ======================
# Respostas Padronizadas
# ======================
RESPOSTAS_PADRAO = {
    "saudacao": """Olá, tudo bem? Seja bem-vindo(a)! Me conta: Como posso te ajudar? 

Posso te auxiliar com:
1. Congressos e eventos
2. Certificados
3. Pós-graduação
4. Comunidade online
5. Intercâmbios

Sobre o que você deseja saber?""",

    "congressos": """Perfeito! Esses são os próximos congressos confirmados:

1. MACEIÓ/AL – 05 e 06/09
2. BELÉM/PA – 09 e 10/09
3. FLORIANÓPOLIS/SC – 21 e 22/10
4. VITÓRIA/ES – 24 e 25/10
5. OUVIDORES DE VOZES (Online) – 05 e 06/12

Gostaria de saber mais detalhes sobre algum deles?""",

    "certificados_info": """Todos os nossos congressos e cursos emitem certificado. A qual evento ou curso você se refere? Me informe o título completo para que possamos localizar!

⚠️ Aviso: Certificados de participação têm prazo de até 7 dias úteis. Os de apresentação, até 15 dias úteis após o evento.""",

    # Reembolso: encaminha direto ao atendimento humano (sem pedir dados antes)
    "reembolso": """#HUMANO

Recomendo entrar em contato com o nosso atendimento humano para tratar do seu reembolso.

Envie um e-mail para atendimento@cenatcursos.com.br com:
- Seu nome completo
- Nome do evento
- Forma de pagamento e data da compra
- Motivo do reembolso

Se preferir, você também pode falar pelo WhatsApp: +55 47 99242-8886.

Nossa equipe verifica o caso e orienta os próximos passos.""",

    "trabalho_info": """Qual sua dúvida sobre submissão? Você quer informações sobre confecção de banners e formatação?

Geralmente enviamos essas orientações no e-mail do autor principal quando o trabalho é aceito.

📋 FORMATO DO PÔSTER:
- Vertical: 90cm (largura) x 120cm (altura)
- Com corda para pendurar
- Conteúdo: título, autores, instituição, eixo temático, referências

Se precisar de mais ajuda específica, me avise!""",

    # Quando o cliente diz que submeteu e não recebeu retorno → atendimento humano
    "trabalho_nao_recebi": """#HUMANO

Vou te transferir para nossa equipe para verificar sua submissão.

WhatsApp: +55 47 99242-8886
E-mail: atendimento@cenatcursos.com.br""",

    # Gravação: texto solicitado
    "gravacao": """Sim! Todos os nossos eventos online ficam disponíveis para acesso posterior, porém você precisa estar inscrito para ter acesso às gravações, ok? 

Se não conseguir assistir de forma síncrona, podemos liberar o acesso após a finalização completa do evento.

Para solicitar, envie um e-mail para atendimento@cenatcursos.com.br com:
- Seu nome completo
- Nome do evento
- (se houver) detalhes sobre sua apresentação/trabalho

Se preferir, fale no WhatsApp: +55 47 99242-8886.""",

    # Ingresso: primeiro checar e-mail/Spam, depois atendimento humano
    "ingresso": """#HUMANO

Antes de tudo, recomendo verificar novamente sua caixa de e-mail (inclusive Spam/Lixo eletrônico), pois o ingresso costuma chegar por lá.

Se não localizar, nosso atendimento humano pode te ajudar rapidamente a reenviar o link de acesso:
WhatsApp: +55 47 99242-8886
E-mail: atendimento@cenatcursos.com.br""",

    # Comunidade (texto baseado no site, sem preço)
    "comunidade": """A Comunidade Novas Abordagens em Saúde Mental é nossa plataforma com estudos online.

Você terá:
• +30 cursos gravados (~250h) para ver no seu tempo (acesso por 1 ano);
• Encontros AO VIVO mensais para discussão de casos e plantão de dúvidas no Zoom;
• Área exclusiva para assinantes, materiais complementares, documentários e séries;
• Certificados em cada curso (20–80h) com código de verificação.

Quer que eu te envie o link de inscrição ou as ementas?""",

    # Comunidade - lista de cursos
    "comunidade_cursos": """Temos mais de 30 cursos gravados (~250h) com acesso por 1 ano, encontros ao vivo mensais e certificados (20–80h).

A lista completa e sempre atualizada está aqui:
https://cenatsaudemental.com/comunidademsaudemental#section-17454852

Posso te enviar as ementas dos cursos que mais combinam com seu interesse. Prefere alguma área (ex.: clínica, infantojuvenil, trabalho/SST, SUS, dependência química)?""",

    "intercambio": """Olá! Nossos intercâmbios incluem visitas a instituições, workshops, palestras, transportes locais, kit, manual e certificado.

Temos destinos como Dinamarca, Trieste, Portugal e Inglaterra, com grupos acompanhados por facilitador local.

Link: https://cenatsaudemental.com/cenat-intercambios

Algum desses destinos te interessa mais?""",

    "pos": """Para dúvidas sobre pós-graduação, encaminhe para:

✉️ secretaria@cenatcursos.com.br

Nossa equipe especializada te atende com todos os detalhes sobre turmas, valores e processo seletivo.""",

    "desconto": """Sim! Oferecemos:
✓ Desconto para estudantes com carteira válida
✓ Desconto progressivo por lote
✓ Desconto para grupos (10-20 pessoas: 10% | +20: 15%)
✓ Combos especiais

Qual evento você tem interesse?""",

    "encerramento": """Disponha! Foi um prazer te atender! Sigo à disposição sempre que precisar.

Ah! Em breve você receberá uma pesquisa sobre o atendimento. Sua opinião é muito importante!

Um abraço. 😊"""
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

def limitar_linhas(texto: str, max_linhas: int = 10) -> str:
    """Garante que a resposta tenha no máximo N linhas."""
    linhas = texto.strip().split('\n')
    if len(linhas) <= max_linhas:
        return texto
    return '\n'.join(linhas[:max_linhas])

# ======================
# IA Agent com Guia CENAT + Contexto Aguçado
# ======================
async def generate_response(user_message: str, user_name: str = "", phone: str = "") -> str:
    """Gera resposta seguindo o Guia de Respostas Padronizadas CENAT com contexto aguçado."""
    
    if AI_DRY_RUN:
        return f"[TESTE] Olá {user_name or 'Cliente'}! Vi sua mensagem '{user_message[:30]}...'"

    # 1. Detectar intenção
    intencao = detectar_intencao(user_message)
    logger.info(f"Intenção detectada: {intencao} | Mensagem: {user_message[:50]}")

    # 2. Resposta direta para intenções mapeadas
    if intencao in [
        "saudacao", "congressos", "certificados", "reembolso", "gravacao",
        "comunidade", "comunidade_cursos", "intercambio", "pos", "desconto",
        "ingresso", "trabalho_nao_recebi", "trabalho"
    ]:
        if intencao == "saudacao":
            return RESPOSTAS_PADRAO["saudacao"]
        elif intencao == "congressos":
            return RESPOSTAS_PADRAO["congressos"]
        elif intencao == "certificados":
            return RESPOSTAS_PADRAO["certificados_info"]
        elif intencao == "reembolso":
            return RESPOSTAS_PADRAO["reembolso"]
        elif intencao == "gravacao":
            return RESPOSTAS_PADRAO["gravacao"]
        elif intencao == "comunidade":
            return RESPOSTAS_PADRAO["comunidade"]
        elif intencao == "comunidade_cursos":
            return RESPOSTAS_PADRAO["comunidade_cursos"]
        elif intencao == "intercambio":
            return RESPOSTAS_PADRAO["intercambio"]
        elif intencao == "pos":
            return RESPOSTAS_PADRAO["pos"]
        elif intencao == "desconto":
            return RESPOSTAS_PADRAO["desconto"]
        elif intencao == "ingresso":
            return RESPOSTAS_PADRAO["ingresso"]
        elif intencao == "trabalho_nao_recebi":
            return RESPOSTAS_PADRAO["trabalho_nao_recebi"]
        elif intencao == "trabalho":
            return RESPOSTAS_PADRAO["trabalho_info"]

    # 3. Para casos que precisam contexto específico ou IA
    if not OPENAI_API_KEY or OpenAI is None:
        return f"#HUMANO\n\nVou te transferir para nosso atendimento humano.\n\nWhatsApp: {WHATSAPP_ATENDIMENTO}\n\nAguarde o contato!"

    try:
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=20)

        # Histórico da conversa (últimas 8 trocas = 16 mensagens)
        history = CONVERSATION_HISTORY.get(phone, [])[-16:]
        
        system_prompt = f"""Você é um atendente do CENAT via WhatsApp. Seu nome pode ser Ana ou você pode se apresentar como "equipe CENAT".

PERSONALIDADE:
- Tom NATURAL e CONVERSACIONAL (como um humano real)
- Amigável mas profissional
- Empático e atencioso
- Use expressões naturais: "entendi", "perfeito", "claro", "sem problema"
- Pode usar "a gente" em vez de "nós" para soar mais próximo
- NUNCA seja robótico ou engessado

CONTEXTO AGUÇADO - MUITO IMPORTANTE:
- Você TEM MEMÓRIA da conversa anterior (veja o histórico)
- Se o cliente mencionou algo antes, LEMBRE e CONECTE
- Se ele perguntou sobre X e depois Y, você sabe que já falaram de X
- Se ele voltar em um assunto anterior, retome naturalmente
- Cliente confuso/mal formulado? Você ENTENDE pela intenção

EXEMPLOS DE CONTEXTO:
Cliente: "Quero saber sobre Maceió"
Você: [explica Maceió]
Cliente: "E Floripa?"
Você: "Opa! Florianópolis também é show. Esse é em outubro, diferente de Maceió que é setembro..."

Cliente: "Quanto custa?"
Você: [vê no histórico que ele perguntou sobre comunidade] "A comunidade? São R$ 387 por ano..."

REGRAS DE RESPOSTA:
1. Máximo 6-8 linhas (seja conciso mas completo)
2. Use o nome: {user_name or 'Cliente'}
3. Se precisar de dados específicos (nome completo, evento exato, CPF), escreva: #HUMANO
4. NUNCA invente datas, preços ou eventos que não estão no contexto
5. Se não souber COM CERTEZA, seja honesto

QUANDO USAR #HUMANO:
- Solicitar dados pessoais específicos (nome completo para certificado, CPF, etc)
- Problemas técnicos que você não resolve
- Verificações internas (status de pagamento, inscrição)
- Qualquer dúvida que você NÃO tem certeza

CONTEXTO DOS NOSSOS PRODUTOS/SERVIÇOS:
{RAG_CONTEXT}

CONGRESSOS 2025:
1. MACEIÓ/AL – 05 e 06/09
2. BELÉM/PA – 09 e 10/09  
3. FLORIANÓPOLIS/SC – 21 e 22/10
4. VITÓRIA/ES – 24 e 25/10
5. OUVIDORES DE VOZES (Online) – 05 e 06/12

LINKS IMPORTANTES:
- Site: https://cenatsaudemental.com/
- Comunidade: https://cenatsaudemental.com/comunidademsaudemental
- Intercâmbios: https://cenatsaudemental.com/cenat-intercambios
- Email: atendimento@cenatcursos.com.br
- Pós: secretaria@cenatcursos.com.br
- WhatsApp atendimento: {WHATSAPP_ATENDIMENTO}"""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Adiciona histórico (contexto aguçado)
        for item in history:
            messages.append(item)
        
        # Mensagem atual
        messages.append({"role": "user", "content": user_message})

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=300,
            temperature=0.4,
        )
        
        resposta = resp.choices[0].message.content or "Desculpe, não consegui processar sua mensagem."
        resposta = limitar_linhas(resposta, max_linhas=10)
        
        # Atualiza histórico (mantém contexto)
        CONVERSATION_HISTORY[phone].append({"role": "user", "content": user_message})
        CONVERSATION_HISTORY[phone].append({"role": "assistant", "content": resposta})
        
        # Limita histórico a 30 mensagens (15 trocas)
        if len(CONVERSATION_HISTORY[phone]) > 30:
            CONVERSATION_HISTORY[phone] = CONVERSATION_HISTORY[phone][-30:]
        
        return resposta
        
    except Exception as e:
        logger.error(f"Erro OpenAI: {e}")
        return f"#HUMANO\n\nEstou com dificuldades técnicas.\n\nPor favor, entre em contato pelo WhatsApp: {WHATSAPP_ATENDIMENTO}"

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
        "version": "3.0-CENAT-CONTEXTO",
        "ai_mode": "DRY_RUN" if AI_DRY_RUN else "REAL",
        "context_loaded": len(RAG_CONTEXT) > 10,
        "mega_configured": bool(MEGA_API_TOKEN and MEGA_INSTANCE_ID),
        "intencoes_ativas": len(INTENCOES),
        "whatsapp_atendimento": WHATSAPP_ATENDIMENTO,
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

@app.post("/reload-context")
async def reload_context():
    global RAG_CONTEXT, _RAG_SIG
    RAG_CONTEXT = load_context()
    _RAG_SIG = data_signature()
    logger.info(f"🔄 RAG recarregado: {len(RAG_CONTEXT)} caracteres")
    return {"status": "ok", "context_len": len(RAG_CONTEXT)}

@app.get("/context/preview")
async def context_preview(n: int = 800):
    n = max(0, min(n, 5000))
    return {"preview": RAG_CONTEXT[:n], "len": len(RAG_CONTEXT)}

@app.get("/stats")
async def stats():
    """Estatísticas do sistema."""
    return {
        "conversas_ativas": len(CONVERSATION_HISTORY),
        "intencoes_disponiveis": list(INTENCOES.keys()),
        "rag_size": len(RAG_CONTEXT),
    }

# ======================
# Worker
# ======================
async def process_and_reply(phone: str, message: str, user_name: str):
    try:
        async with LOCKS[phone]:
            response = await generate_response(message, user_name, phone)
            logger.info(f"🤖 IA gerou resposta: {response[:200]}")
            
            # Verifica se precisa de humano
            if "#HUMANO" in response:
                logger.warning(f"⚠️ Marcador #HUMANO detectado para {user_name}")
                # Remove o marcador antes de enviar
                response = response.replace("#HUMANO", "").strip()
                if not response:
                    response = f"Vou te transferir para nosso atendimento humano.\n\nWhatsApp: {WHATSAPP_ATENDIMENTO}\n\nAguarde o contato!"
            
            ok = await send_whatsapp(phone, response)
            if ok:
                logger.info(f"✅ Resposta enviada para {user_name}")
            else:
                logger.error(f"❌ Falha ao enviar para {user_name}")
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")

# ======================
# Watcher do RAG
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
    logger.info("🚀 WhatsApp AI Agent v3.0 CENAT CONTEXTO AGUÇADO iniciado")
    logger.info(f"📄 Contexto RAG: {len(RAG_CONTEXT)} caracteres")
    logger.info(f"🤖 Modo IA: {'DRY_RUN (teste)' if AI_DRY_RUN else 'REAL (OpenAI)'}")
    logger.info(f"📱 MEGA API: {'configurada' if (MEGA_API_TOKEN and MEGA_INSTANCE_ID) else 'NÃO CONFIGURADA'}")
    logger.info(f"🎯 Intenções mapeadas: {len(INTENCOES)}")
    logger.info(f"📞 WhatsApp atendimento: {WHATSAPP_ATENDIMENTO}")
    if RAG_AUTO_RELOAD:
        asyncio.create_task(rag_watcher())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "8000")))
