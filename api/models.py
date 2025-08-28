# api/models.py
from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# =========================
# MODELOS PARA /process-lead
# =========================

class Lead(BaseModel):
    """Perfil do lead (uso em entrada estruturada)."""
    name: str
    email: str
    phone: Optional[str] = None
    profession: Optional[str] = None
    # Campo usado pelo agente de conversão:
    interest: Optional[str] = None
    # Metadados/UTMs:
    source: Optional[str] = None
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None


class ICPProfile(BaseModel):
    segment: Optional[str] = None
    company_size: Optional[str] = None
    role: Optional[str] = None


class Context(BaseModel):
    product: str
    region: str = "BR"
    language: str = "pt-BR"
    icp_profile: Optional[ICPProfile] = None


class ProcessLeadInput(BaseModel):
    """
    Entrada estruturada (opcional) para /process-lead.
    Mantida para futuros cenários em que você queira enviar lead+context juntos.
    """
    lead: Lead
    context: Context


# ---- Compatibilidade: entrada "flat" usada hoje pelo endpoint /process-lead ----
class ProcessLeadRequest(BaseModel):
    """
    Entrada simples/flat usada atualmente pelo /process-lead.
    """
    name: str
    email: str
    phone: Optional[str] = None
    interest: Optional[str] = None
    source: Optional[str] = None
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None


# ---------- Saídas do /process-lead ----------
class CRMTask(BaseModel):
    title: str
    due_date: str  # ISO 8601 em string (ex.: "2025-09-05")
    owner: str


class MessagingEmail(BaseModel):
    subject: str
    body: str


class Messaging(BaseModel):
    email: MessagingEmail
    whatsapp: str
    call_script: str


class CRMActionBlock(BaseModel):
    stage: str
    tags: List[str] = Field(default_factory=list)
    tasks: List[CRMTask] = Field(default_factory=list)
    note: str
    next_step_date: str  # ISO 8601 em string (ex.: "2025-09-06")


class ProcessLeadOutput(BaseModel):
    status: str
    lead_score: int
    reasoning: str
    crm_actions: CRMActionBlock
    messaging: Messaging


# =========================
# MODELOS PARA /whatsapp-reply
# =========================

class WhatsAppHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class WhatsAppMessage(BaseModel):
    lead_name: str
    course_name: str
    lead_phone: str
    lead_email: str
    last_user_message: str
    timezone: Optional[str] = "America/Fortaleza"
    history: Optional[List[WhatsAppHistoryItem]] = None


class WhatsAppReply(BaseModel):
    reply: str
