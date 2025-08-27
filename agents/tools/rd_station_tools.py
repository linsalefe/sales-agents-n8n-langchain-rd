# agents/tools/rd_station_tools.py
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15.0


class RDStationError(Exception):
    """Erro de integração com RD Station."""


class RDStationClient:
    """
    Wrapper seguro para RD Station.
    - Autenticação via Bearer token (RD_ACCESS_TOKEN)
    - Retries exponenciais para 5xx/timeout
    - Logs estruturados
    """

    def __init__(
        self,
        base_url: str,
        access_token: str,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not base_url or not access_token:
            raise ValueError("base_url e access_token são obrigatórios")

        self.base_url = base_url.rstrip("/")
        self.access_token = access_token
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=self.timeout,
        )

    def close(self) -> None:
        self._client.close()

    # ---------- Core HTTP com retry ----------

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type((httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError, httpx.NetworkError)),
    )
    def _request(self, method: str, url: str, *, json_body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
        try:
            logger.info(
                "rd_request",
                extra={
                    "method": method,
                    "url": url,
                    "params": params or {},
                    "body": json_body or {},
                },
            )
            resp = self._client.request(method, url, json=json_body, params=params)
            resp.raise_for_status()
            data = resp.json() if resp.content else {}
            logger.info("rd_response", extra={"status_code": resp.status_code, "data": data})
            return data
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else "no_status"
            text = e.response.text if e.response is not None else str(e)
            logger.error("rd_http_error", extra={"status": status, "text": text})
            # 4xx não deve ficar em retry infinito; raise para caller decidir
            raise RDStationError(f"HTTP {status}: {text}") from e
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            logger.error("rd_network_error", extra={"err": str(e)})
            raise

    # ============= Operações de CRM =============

    def update_stage(self, deal_id: str, stage: str) -> dict:
        """
        Atualiza o estágio do negócio (exemplo de endpoint; ajuste conforme conta).
        """
        if not deal_id or not stage:
            raise ValueError("deal_id e stage são obrigatórios")
        url = "/crm/deals/{deal_id}/stage".format(deal_id=deal_id)
        body = {"stage": stage}
        return self._request("PUT", url, json_body=body)

    def add_tags(self, lead_email: str, tags: List[str]) -> dict:
        """
        Adiciona tags a um contato identificado por email.
        """
        if not lead_email or not tags:
            raise ValueError("lead_email e tags são obrigatórios")
        url = "/platform/contacts/tags"
        body = {"email": lead_email, "tags": tags}
        return self._request("POST", url, json_body=body)

    def create_task(self, deal_id: str, title: str, due_date: str, owner: Optional[str] = None) -> dict:
        """
        Cria uma tarefa vinculada a um negócio.
        """
        if not deal_id or not title or not due_date:
            raise ValueError("deal_id, title e due_date são obrigatórios")
        url = "/crm/deals/{deal_id}/tasks".format(deal_id=deal_id)
        body: Dict[str, Optional[str]] = {"title": title, "due_date": due_date}
        if owner:
            body["owner"] = owner
        return self._request("POST", url, json_body=body)

    def add_note(self, deal_id: str, note: str) -> dict:
        """
        Adiciona uma nota a um negócio.
        """
        if not deal_id or not note:
            raise ValueError("deal_id e note são obrigatórios")
        url = "/crm/deals/{deal_id}/notes".format(deal_id=deal_id)
        body = {"note": note}
        return self._request("POST", url, json_body=body)

    def send_email(self, to_email: str, subject: str, body: str, sender: Optional[str] = None) -> dict:
        """
        Envia email via RD (ajuste endpoint conforme plano/conta).
        """
        if not to_email or not subject or not body:
            raise ValueError("to_email, subject e body são obrigatórios")
        url = "/marketing/emails/send"
        payload: Dict[str, Optional[str]] = {
            "to": to_email,
            "subject": subject,
            "body": body,
        }
        if sender:
            payload["from"] = sender
        return self._request("POST", url, json_body=payload)
