# api/main.py
from __future__ import annotations

# --- .env loader (sem dependências externas) ---
import os
from pathlib import Path

def _load_env_file():
    """
    Carrega variáveis do arquivo .env na raiz do repo e
    **SOBRESCREVE** o ambiente atual (prioridade para .env).
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            # sobrescreve SEM usar setdefault
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

_load_env_file()
# --- fim do loader ---

from fastapi import FastAPI
from loguru import logger

from agents.conversation_agent import process_lead_conversation
from agents.sdr_whatsapp import SdrWhatsappAgent, LeadContext
from api.models import (
    ProcessLeadRequest,
    ProcessLeadInput,
    WhatsAppMessage,
    WhatsAppReply,
)

app = FastAPI(title="Sales Agent Simple", version="1.1.2")


@app.get("/health")
def health():
    return {"status": "ok"}


# Entrada "flat" tradicional
@app.post("/process-lead")
def process_lead(payload: ProcessLeadRequest):
    try:
        logger.info(f"Processing lead (flat): {payload.name} - interest={payload.interest}")
        result = process_lead_conversation(payload.model_dump())
        return result
    except Exception as e:
        logger.exception(f"/process-lead failed: {e}")
        return {
            "status": "error",
            "message": f"Falha no /process-lead: {str(e)}"
        }


# Entrada estruturada (lead + context)
@app.post("/process-lead-structured")
def process_lead_structured(payload: ProcessLeadInput):
    lead = payload.lead
    ctx = payload.context

    flat = {
        "name": lead.name,
        "email": lead.email,
        "phone": lead.phone,
        "interest": lead.interest or (ctx.product if ctx else None),
        "source": lead.source,
        "utm_source": lead.utm_source,
        "utm_medium": lead.utm_medium,
        "utm_campaign": lead.utm_campaign,
    }

    logger.info(f"Processing lead (structured): {lead.name} - interest={flat.get('interest')}")
    result = process_lead_conversation(flat)
    return result


# WhatsApp SDR (respostas curtas + #AGENDAR)
@app.post("/whatsapp-reply", response_model=WhatsAppReply)
def whatsapp_reply(payload: WhatsAppMessage) -> WhatsAppReply:
    lead = LeadContext(
        lead_name=payload.lead_name,
        course_name=payload.course_name,
        lead_phone=payload.lead_phone,
        lead_email=payload.lead_email,
        timezone=payload.timezone or "America/Fortaleza",
    )
    history = (
        [{"role": h.role, "content": h.content} for h in payload.history]
        if payload.history else None
    )

    agent = SdrWhatsappAgent()
    reply_text = agent.reply(
        lead_ctx=lead,
        last_user_message=payload.last_user_message,
        history=history,
    )
    return WhatsAppReply(reply=reply_text)
