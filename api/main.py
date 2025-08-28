from fastapi import FastAPI
from pydantic import BaseModel
from agents.conversation_agent import process_lead_conversation
from loguru import logger

app = FastAPI(title="Sales Agent Simple", version="1.0.0")

class LeadInput(BaseModel):
    name: str
    email: str
    phone: str = None
    interest: str = None
    utm_source: str = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/process-lead")
def process_lead(lead: LeadInput):
    logger.info(f"Processing lead: {lead.name}")
    result = process_lead_conversation(lead.dict())
    return result
