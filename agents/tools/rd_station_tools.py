# agents/tools/rd_station_tools.py
from typing import Dict, Any
from loguru import logger

class RDStationClient:
    """
    Wrapper seguro (mockável) da API do RD Station.
    Por enquanto apenas loga e retorna sucesso. Depois trocamos por chamadas reais.
    """

    def __init__(self, base_url: str, access_token: str):
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    async def update_stage(self, lead_email: str, stage: str) -> Dict[str, Any]:
        logger.info(f"[RD] update_stage email={lead_email} -> {stage}")
        return {"ok": True, "stage": stage}

    async def add_tags(self, lead_email: str, tags: list[str]) -> Dict[str, Any]:
        logger.info(f"[RD] add_tags email={lead_email} -> {tags}")
        return {"ok": True, "tags": tags}

    async def create_task(self, title: str, due_date: str, owner: str) -> Dict[str, Any]:
        logger.info(f"[RD] create_task title={title} due={due_date} owner={owner}")
        return {"ok": True, "id": "task_123"}

    async def add_note(self, lead_email: str, note: str) -> Dict[str, Any]:
        logger.info(f"[RD] add_note email={lead_email} note='{note[:60]}...'")
        return {"ok": True}

    async def send_email(self, to_email: str, subject: str, body: str) -> Dict[str, Any]:
        logger.info(f"[RD] send_email to={to_email} subject='{subject}'")
        return {"ok": True}
