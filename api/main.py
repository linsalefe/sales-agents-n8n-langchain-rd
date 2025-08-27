# api/main.py
from fastapi import FastAPI
from loguru import logger

from .models import ProcessLeadInput, ProcessLeadOutput
from agents.graph import run_graph

app = FastAPI(title="Sales Agents API", version="0.2.0")


@app.get("/health")
def health():
    logger.debug("Healthcheck OK")
    return {"status": "ok"}


@app.post("/process-lead", response_model=ProcessLeadOutput)
def process_lead(payload: ProcessLeadInput):
    """
    Executa o LangGraph (context_builder -> sales_agent -> crm_agent)
    e retorna no contrato esperado pela API.
    """
    result = run_graph(payload.model_dump())
    logger.info("process-lead (graph) -> {}", result)
    return ProcessLeadOutput(**result)
