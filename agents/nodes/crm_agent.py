# agents/nodes/crm_agent.py
from typing import Dict, Any

def crm_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    lead = state["lead"]
    status = state["status"]
    score = state["lead_score"]
    ctx = state["context"]
    tags = state.get("tags", [])

    stage = "Qualificação" if status == "qualificado" else "Novo Lead"

    crm_actions = {
        "stage": stage,
        "tags": tags,
        "tasks": [
            {
                "title": f"Ligar para {lead.get('name')}",
                "due_date": "2025-08-28",
                "owner": "SDR-Ana",
            }
        ],
        "note": f"Lead {status}. Score {score}. Origem: {lead.get('source') or 'desconhecida'}.",
        "next_step_date": "2025-08-29",
    }

    return {**state, "crm_actions": crm_actions}
