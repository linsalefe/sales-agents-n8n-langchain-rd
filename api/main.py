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
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

_load_env_file()
# --- fim do loader ---

from fastapi import FastAPI
from loguru import logger
import requests
import threading

from agents.conversation_agent import process_lead_conversation
from agents.sdr_whatsapp import SdrWhatsappAgent, LeadContext
from api.models import (
    ProcessLeadRequest,
    ProcessLeadInput,
    WhatsAppMessage,
    WhatsAppReply,
)

app = FastAPI(title="Sales Agent Simple", version="1.1.4")

# =========================
# Configurações MEGA API
# =========================
MEGA_API_BASE_URL = (os.getenv("MEGA_API_BASE_URL") or "").rstrip("/")
MEGA_API_TOKEN = os.getenv("MEGA_API_TOKEN") or ""
MEGA_INSTANCE_ID = os.getenv("MEGA_INSTANCE_ID") or ""


def _mega_headers() -> dict:
    return {
        "Authorization": f"Bearer {MEGA_API_TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _format_wa_number(phone_number: str) -> str:
    """Normaliza destino do WhatsApp para JID, se necessário."""
    if "@s.whatsapp.net" in phone_number or "@g.us" in phone_number:
        return phone_number
    return f"{phone_number}@s.whatsapp.net"


def send_whatsapp_message(phone_number: str, message: str) -> bool:
    """Envia mensagem via MEGA API."""
    try:
        if not (MEGA_API_BASE_URL and MEGA_API_TOKEN and MEGA_INSTANCE_ID):
            logger.warning("MEGA API não configurada completamente")
            return False

        url = f"{MEGA_API_BASE_URL}/rest/sendMessage/{MEGA_INSTANCE_ID}/text"
        payload = {
            "messageData": {
                "to": _format_wa_number(phone_number),
                "text": message
            }
        }

        logger.info(f"Enviando WhatsApp para {payload['messageData']['to']}")
        resp = requests.post(url, json=payload, headers=_mega_headers(), timeout=15)
        resp.raise_for_status()
        logger.info("WhatsApp enviado com sucesso")
        return True

    except Exception as e:
        logger.error(f"Erro ao enviar WhatsApp: {e}")
        return False


@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# Endpoints principais
# =========================
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


# =========================
# SDR WhatsApp (curto + #AGENDAR)
# =========================
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


# =========================
# Processar lead e enviar via WhatsApp
# =========================
@app.post("/send-whatsapp")
def send_whatsapp(payload: ProcessLeadRequest):
    """Processa lead com IA e envia mensagem via WhatsApp."""
    try:
        logger.info(f"Processing and sending WhatsApp: {payload.name} - {payload.phone}")

        # 1) Processar lead com IA
        result = process_lead_conversation(payload.model_dump())

        # 2) Enviar WhatsApp se houver telefone e mensagem
        whatsapp_sent = False
        if payload.phone and result.get("message"):
            whatsapp_sent = send_whatsapp_message(payload.phone, result["message"])
        elif not payload.phone:
            logger.warning(f"Lead {payload.name} sem telefone - WhatsApp não enviado")

        # 3) Anexar status
        result["whatsapp_sent"] = whatsapp_sent
        return result

    except Exception as e:
        logger.exception(f"/send-whatsapp failed: {e}")
        return {
            "status": "error",
            "message": f"Falha no /send-whatsapp: {str(e)}",
            "whatsapp_sent": False
        }


# =========================
# Webhook MEGA API
# =========================
@app.post("/webhook")
def whatsapp_webhook(data: dict):
    """Recebe webhooks da MEGA API e responde automaticamente."""
    try:
        logger.info(f"Webhook recebido: {data}")

        if not _is_valid_message(data):
            logger.info("Webhook ignorado - não é mensagem válida")
            return {"status": "ignored"}

        # Processar em thread separada para não travar o webhook
        thread = threading.Thread(target=_process_webhook_async, args=(data,))
        thread.daemon = True
        thread.start()

        return {"status": "received", "message": "Processando mensagem"}

    except Exception as e:
        logger.error(f"Erro no webhook: {e}")
        return {"status": "error", "message": str(e)}


def _is_valid_message(data: dict) -> bool:
    """Verifica se webhook é uma mensagem válida para processar."""
    try:
        # Deve ser mensagem de texto
        if data.get("messageType") != "conversation":
            return False

        # Não deve ser mensagem própria
        if data.get("key", {}).get("fromMe", False):
            return False

        # Deve ter conteúdo
        message_content = data.get("message", {}).get("conversation")
        if not message_content:
            return False

        # Deve ter remetente
        sender_jid = data.get("key", {}).get("remoteJid")
        if not sender_jid:
            return False

        return True
    except Exception:
        return False


def _process_webhook_async(data: dict):
    """Processa webhook de forma assíncrona."""
    try:
        sender_jid = data["key"]["remoteJid"]
        message_text = data["message"]["conversation"]
        sender_name = data.get("pushName", "Usuário")

        logger.info(f"Processando mensagem de {sender_name}: {message_text}")

        lead_payload = {
            "name": sender_name,
            "email": f"{sender_jid.split('@')[0]}@whatsapp.user",
            "phone": sender_jid.split("@")[0],
            "interest": message_text,
            "utm_source": "whatsapp",
        }

        # Gerar resposta com IA
        result = process_lead_conversation(lead_payload)
        ai_response = result.get(
            "message",
            "Olá! Recebemos sua mensagem e entraremos em contato em breve."
        )

        # Enviar resposta
        success = send_whatsapp_message(sender_jid, ai_response)
        if success:
            logger.info(f"Resposta enviada com sucesso para {sender_name}")
        else:
            logger.error(f"Falha ao enviar resposta para {sender_name}")

    except Exception as e:
        logger.error(f"Erro no processamento assíncrono: {e}")


# =========================
# Teste de conectividade MEGA API
# =========================
@app.get("/mega-api-status")
def mega_api_status():
    """Verifica status da conexão com a MEGA API (sem /status)."""
    try:
        if not (MEGA_API_BASE_URL and MEGA_API_TOKEN and MEGA_INSTANCE_ID):
            return {
                "status": "error",
                "message": "Variáveis MEGA API não configuradas",
                "configured": {
                    "MEGA_API_BASE_URL": bool(MEGA_API_BASE_URL),
                    "MEGA_API_TOKEN": bool(MEGA_API_TOKEN),
                    "MEGA_INSTANCE_ID": bool(MEGA_INSTANCE_ID),
                },
            }

        url = f"{MEGA_API_BASE_URL}/rest/instance/{MEGA_INSTANCE_ID}"
        try:
            resp = requests.get(url, headers=_mega_headers(), timeout=10)
            text = resp.text
            data = None
            try:
                data = resp.json()
            except Exception:
                pass
        except Exception as e:
            return {"status": "error", "message": str(e), "url_used": url}

        if resp.status_code == 200 and isinstance(data, dict) and not data.get("error", False):
            inst = data.get("instance", {})
            return {
                "status": "connected",
                "http_status": resp.status_code,
                "instance_id": inst.get("key"),
                "instance_status": inst.get("status"),
                "wa_user": (inst.get("user") or {}).get("id"),
                "url_used": url,
            }

        return {
            "status": "error",
            "http_status": resp.status_code,
            "instance_id": MEGA_INSTANCE_ID,
            "response": text,
            "url_used": url,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# =========================
# Parser do marcador #AGENDAR
# =========================
import re
from typing import Optional, Dict
from pydantic import BaseModel
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

class ParseAgendarRequest(BaseModel):
    reply: str                      # texto completo que contém a linha #AGENDAR|...
    timezone: Optional[str] = "America/Fortaleza"

def _parse_agendar_line(text: str) -> Dict[str, str]:
    """
    Encontra a linha que começa com '#AGENDAR|' e transforma em dict {k: v}.
    Ex.: #AGENDAR|data=2025-08-29|hora=09:00|duracao=20|lead=Maria|...
    """
    m = re.search(r"^#AGENDAR\|(.+)$", text, flags=re.MULTILINE)
    if not m:
        raise ValueError("Marcador #AGENDAR não encontrado no texto")
    payload = m.group(1)

    result: Dict[str, str] = {}
    for part in payload.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip()] = v.strip()
    if not result:
        raise ValueError("Payload do #AGENDAR está vazio ou inválido")
    return result

@app.post("/whatsapp-parse")
def whatsapp_parse(req: ParseAgendarRequest):
    try:
        data = _parse_agendar_line(req.reply)

        # Campos principais
        lead = data.get("lead", "Lead")
        curso = data.get("curso", "Contato com lead")
        contato = data.get("contato", "")
        email = data.get("email", "")
        try:
            duration_minutes = int(data.get("duracao", "20"))
        except Exception:
            duration_minutes = 20

        # Monta ISO local (se houver data e hora)
        start_iso = None
        end_iso = None
        if data.get("data") and data.get("hora"):
            try:
                tz = ZoneInfo(req.timezone or "America/Fortaleza")
                start_dt = datetime.strptime(
                    f"{data['data']} {data['hora']}", "%Y-%m-%d %H:%M"
                ).replace(tzinfo=tz)
                end_dt = start_dt + timedelta(minutes=duration_minutes)
                start_iso = start_dt.isoformat()
                end_iso = end_dt.isoformat()
            except Exception:
                start_iso = None
                end_iso = None

        summary = f"{curso} — {lead}"
        description_lines = [
            f"Lead: {lead}",
            f"Curso: {curso}",
            f"Contato: {contato}" if contato else "",
            f"E-mail: {email}" if email else "",
        ]
        description = "\n".join([l for l in description_lines if l])

        return {
            "status": "ok",
            "timezone": req.timezone,
            "data": data,
            "start_iso": start_iso,
            "end_iso": end_iso,
            "summary": summary,
            "description": description,
            "duration_minutes": duration_minutes,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# =========================
# Geração de convite (.ics) a partir do #AGENDAR
# =========================
from fastapi import Body
from typing import Optional
from uuid import uuid4

@app.post("/calendar-ics")
def calendar_ics(
    reply: str = Body(..., embed=True, description="Texto que contém a linha #AGENDAR|..."),
    timezone: Optional[str] = Body("America/Fortaleza", embed=True)
):
    """
    Converte o marcador #AGENDAR em um arquivo ICS (iCalendar) para importação no calendário.
    Retorna o conteúdo .ics (text/calendar).
    """
    try:
        data = _parse_agendar_line(reply)

        # Campos básicos
        lead = data.get("lead", "Lead")
        curso = data.get("curso", "Contato com lead")
        contato = data.get("contato", "")
        email = data.get("email", "")
        dur_min = 20
        try:
            dur_min = int(data.get("duracao", "20"))
        except Exception:
            pass

        # Monta datas (converte para UTC para o ICS)
        from datetime import datetime, timedelta, timezone as tzmod
        from zoneinfo import ZoneInfo

        if not (data.get("data") and data.get("hora")):
            return {"status": "error", "message": "Faltam campos 'data' e/ou 'hora' no #AGENDAR"}

        tz = ZoneInfo(timezone or "America/Fortaleza")
        start_local = datetime.strptime(
            f"{data['data']} {data['hora']}", "%Y-%m-%d %H:%M"
        ).replace(tzinfo=tz)
        end_local = start_local + timedelta(minutes=dur_min)

        start_utc = start_local.astimezone(tzmod.utc)
        end_utc = end_local.astimezone(tzmod.utc)
        now_utc = datetime.now(tzmod.utc)

        def fmt(dt: datetime) -> str:
            # Formato iCal UTC: YYYYMMDDTHHMMSSZ
            return dt.strftime("%Y%m%dT%H%M%SZ")

        uid = f"{uuid4()}@sales-agent-simple"
        summary = f"{curso} — {lead}"
        description_lines = [
            f"Lead: {lead}",
            f"Curso: {curso}",
            f"Contato: {contato}" if contato else "",
            f"E-mail: {email}" if email else "",
        ]
        description = "\\n".join([l for l in description_lines if l])

        ics_lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//Sales Agent Simple//BR",
            "CALSCALE:GREGORIAN",
            "METHOD:PUBLISH",
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{fmt(now_utc)}",
            f"DTSTART:{fmt(start_utc)}",
            f"DTEND:{fmt(end_utc)}",
            f"SUMMARY:{summary}",
            "LOCATION:WhatsApp",
            f"DESCRIPTION:{description}",
            "END:VEVENT",
            "END:VCALENDAR",
        ]
        ics = "\r\n".join(ics_lines) + "\r\n"

        from fastapi.responses import PlainTextResponse
        headers = {"Content-Disposition": 'attachment; filename="agendar-evento.ics"'}
        return PlainTextResponse(content=ics, media_type="text/calendar", headers=headers)

    except Exception as e:
        return {"status": "error", "message": str(e)}
# =========================
# Parse + ICS (retorno em Base64)
# =========================
import base64
from fastapi import Body
from typing import Optional
from uuid import uuid4
from datetime import datetime, timedelta, timezone as tzmod
from zoneinfo import ZoneInfo

@app.post("/whatsapp-parse-ics")
def whatsapp_parse_ics(
    reply: str = Body(..., embed=True, description="Texto que contém #AGENDAR|..."),
    timezone: Optional[str] = Body("America/Fortaleza", embed=True)
):
    try:
        data = _parse_agendar_line(reply)

        # Campos
        lead = data.get("lead", "Lead")
        curso = data.get("curso", "Contato com lead")
        contato = data.get("contato", "")
        email = data.get("email", "")
        try:
            dur_min = int(data.get("duracao", "20"))
        except Exception:
            dur_min = 20

        if not (data.get("data") and data.get("hora")):
            return {"status": "error", "message": "Faltam 'data' e/ou 'hora' no #AGENDAR"}

        tz = ZoneInfo(timezone or "America/Fortaleza")
        start_local = datetime.strptime(
            f"{data['data']} {data['hora']}", "%Y-%m-%d %H:%M"
        ).replace(tzinfo=tz)
        end_local = start_local + timedelta(minutes=dur_min)

        start_utc = start_local.astimezone(tzmod.utc)
        end_utc = end_local.astimezone(tzmod.utc)
        now_utc = datetime.now(tzmod.utc)

        def fmt(dt: datetime) -> str:
            return dt.strftime("%Y%m%dT%H%M%SZ")

        uid = f"{uuid4()}@sales-agent-simple"
        summary = f"{curso} — {lead}"
        description_lines = [
            f"Lead: {lead}",
            f"Curso: {curso}",
            f"Contato: {contato}" if contato else "",
            f"E-mail: {email}" if email else "",
        ]
        description = "\\n".join([l for l in description_lines if l])

        ics_lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//Sales Agent Simple//BR",
            "CALSCALE:GREGORIAN",
            "METHOD:PUBLISH",
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{fmt(now_utc)}",
            f"DTSTART:{fmt(start_utc)}",
            f"DTEND:{fmt(end_utc)}",
            f"SUMMARY:{summary}",
            "LOCATION:WhatsApp",
            f"DESCRIPTION:{description}",
            "END:VEVENT",
            "END:VCALENDAR",
        ]
        ics_str = "\r\n".join(ics_lines) + "\r\n"
        ics_b64 = base64.b64encode(ics_str.encode("utf-8")).decode("ascii")

        return {
            "status": "ok",
            "timezone": timezone,
            "data": data,
            "start_iso": start_local.isoformat(),
            "end_iso": end_local.isoformat(),
            "summary": summary,
            "description": description.replace("\\n", "\n"),
            "duration_minutes": dur_min,
            "ics_filename": "agendar-evento.ics",
            "ics_base64": ics_b64
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
