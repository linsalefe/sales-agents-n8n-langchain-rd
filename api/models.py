from pydantic import BaseModel, Field
from typing import List, Optional

class Lead(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    profession: Optional[str] = None
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
    lead: Lead
    context: Context

class CRMTask(BaseModel):
    title: str
    due_date: str
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
    next_step_date: str

class ProcessLeadOutput(BaseModel):
    status: str
    lead_score: int
    reasoning: str
    crm_actions: CRMActionBlock
    messaging: Messaging
