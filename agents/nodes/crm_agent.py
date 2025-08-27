# agents/nodes/crm_agent.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from agents.tools.rd_station_tools import RDStationClient, RDStationError


def _bool_env(name: str, default: bool = True) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    return val if val not in (None, "", "null", "None") else default


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _summarize_result(ok: bool, data: Any = None, err: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": ok}
    if data is not None:
        out["data"] = data
    if err is not None:
        out["err"] = err
    return out


def _build_email_payload(messaging: Dict[str, Any], lead_email: str) -> Optional[Tuple[str, str, str, Optional[str]]]:
    email = messaging.get("email") if isinstance(messaging, dict) else None
    if not email or not isinstance(email, dict):
        return None
    subject = email.get("subject")
    body = email.get("body")
    if not (lead_email and subject and body):
        return None
    # remetente opcional via env
    rd_from = _get_env("RD_EMAIL_FROM", None)
    return lead_email, subject, body, rd_from


def _coalesce_tags(crm_actions: Dict[str, Any]) -> List[str]:
    tags = crm_actions.get("tags") if isinstance(crm_actions, dict) else []
    return [t for t in tags or [] if isinstance(t, str) and t.strip()]


def _deal_id_from_input(graph_input: Dict[str, Any]) -> Optional[str]:
    # Preferência: context.deal_id (se você já tiver o negócio criado no RD)
    return _safe_get(graph_input, ["context", "deal_id"])


def _lead_email_from_input(graph_input: Dict[str, Any]) -> Optional[str]:
    return _safe_get(graph_input, ["lead", "email"])


def _stage_from_actions(crm_actions: Dict[str, Any]) -> Optional[str]:
    stage = crm_actions.get("stage") if isinstance(crm_actions, dict) else None
    return stage if isinstance(stage, str) and stage.strip() else None


def _task_from_actions(crm_actions: Dict[str, Any]) -> Optional[Dict[str, str]]:
    tasks = crm_actions.get("tasks") if isinstance(crm_actions, dict) else None
    if not tasks or not isinstance(tasks, list):
        return None
    task = tasks[0] if tasks else None
    if not isinstance(task, dict):
        return None
    title = task.get("title")
    due_date = task.get("due_date")
    owner = task.get("owner")
    if not (title and due_date):
        return None
    return {"title": title, "due_date": due_date, "owner": owner}


def _note_from_actions(crm_actions: Dict[str, Any]) -> Optional[str]:
    note = crm_actions.get("note") if isinstance(crm_actions, dict) else None
    return note if isinstance(note, str) and note.strip() else None


def crm_agent_node(graph_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node do CRM Agent:
    - Lê o output do Sales Agent (crm_actions, messaging) e executa no RD (ou DRY-RUN).
    - Mantém idempotência e tolerância a dados faltantes.
    - Retorna o payload original + "rd_results" (sumário das operações).
    """
    # ----- Config flags -----
    RD_DRY_RUN = _bool_env("RD_DRY_RUN", True)
    RD_BASE_URL = _get_env("RD_BASE_URL", "https://api.rd.services")
    RD_ACCESS_TOKEN = _get_env("RD_ACCESS_TOKEN", "")

    crm_actions: Dict[str, Any] = graph_input.get("crm_actions", {}) or {}
    messaging: Dict[str, Any] = graph_input.get("messaging", {}) or {}

    deal_id = _deal_id_from_input(graph_input)  # opcional
    lead_email = _lead_email_from_input(graph_input)

    stage = _stage_from_actions(crm_actions)
    tags = _coalesce_tags(crm_actions)
    task = _task_from_actions(crm_actions)
    note = _note_from_actions(crm_actions)

    email_payload = _build_email_payload(messaging, lead_email)  # (to, subj, body, from?) ou None

    rd_results: Dict[str, Any] = {
        "dry_run": RD_DRY_RUN,
        "stage": _summarize_result(ok=False),
        "tags": _summarize_result(ok=False),
        "task": _summarize_result(ok=False),
        "note": _summarize_result(ok=False),
        "email": _summarize_result(ok=False),
    }

    # DRY-RUN: apenas loga e devolve
    if RD_DRY_RUN:
        logger.info(
            "crm_agent_dry_run",
            extra={
                "stage": stage,
                "tags": tags,
                "task": task,
                "note": note,
                "email_to": email_payload[0] if email_payload else None,
            },
        )
        # emula sucesso para facilitar e2e
        if stage:
            rd_results["stage"] = _summarize_result(True, {"stage_set": stage})
        if tags and lead_email:
            rd_results["tags"] = _summarize_result(True, {"email": lead_email, "tags_applied": tags})
        if task and deal_id:
            rd_results["task"] = _summarize_result(True, {"deal_id": deal_id, "task": task})
        if note and deal_id:
            rd_results["note"] = _summarize_result(True, {"deal_id": deal_id})
        if email_payload:
            to, subj, body, sender = email_payload
            rd_results["email"] = _summarize_result(True, {"to": to, "subject": subj})
        return {**graph_input, "rd_results": rd_results}

    # EXECUÇÃO REAL
    if not RD_ACCESS_TOKEN:
        logger.warning("RD_ACCESS_TOKEN ausente. Habilite RD_DRY_RUN ou configure a chave.")
        return {**graph_input, "rd_results": rd_results}

    client = RDStationClient(base_url=RD_BASE_URL or "", access_token=RD_ACCESS_TOKEN or "")

    try:
        # 1) Tags no contato (usa email; independe de deal_id)
        if tags and lead_email:
            try:
                resp = client.add_tags(lead_email, tags)
                rd_results["tags"] = _summarize_result(True, resp)
            except RDStationError as e:
                rd_results["tags"] = _summarize_result(False, err=str(e))

        # As operações abaixo dependem de deal_id (negócio existente)
        if deal_id:
            # 2) Estágio
            if stage:
                try:
                    resp = client.update_stage(deal_id, stage)
                    rd_results["stage"] = _summarize_result(True, resp)
                except RDStationError as e:
                    rd_results["stage"] = _summarize_result(False, err=str(e))

            # 3) Task
            if task:
                try:
                    resp = client.create_task(
                        deal_id,
                        title=task["title"],
                        due_date=task["due_date"],
                        owner=task.get("owner"),
                    )
                    rd_results["task"] = _summarize_result(True, resp)
                except RDStationError as e:
                    rd_results["task"] = _summarize_result(False, err=str(e))

            # 4) Note
            if note:
                try:
                    resp = client.add_note(deal_id, note=note)
                    rd_results["note"] = _summarize_result(True, resp)
                except RDStationError as e:
                    rd_results["note"] = _summarize_result(False, err=str(e))
        else:
            logger.info("crm_agent_skip_deal_ops", extra={"reason": "deal_id ausente; pulando stage/task/note"})

        # 5) Email (independe de deal_id)
        if email_payload:
            to, subj, body, sender = email_payload
            try:
                resp = client.send_email(to, subj, body, sender=sender)
                rd_results["email"] = _summarize_result(True, resp)
            except RDStationError as e:
                rd_results["email"] = _summarize_result(False, err=str(e))

    finally:
        try:
            client.close()
        except Exception:
            pass

    return {**graph_input, "rd_results": rd_results}

crm_agent = crm_agent_node
__all__ = ["crm_agent_node", "crm_agent"]
