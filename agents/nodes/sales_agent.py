# agents/nodes/sales_agent.py
from typing import Dict, Any
from agents.tools import rag

def sales_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    lead = state["lead"]
    ctx = state["context"]

    score, reasons = rag.icp_fit_score(lead, ctx)
    status = "qualificado" if score >= 70 else "avaliar"

    messaging = {
        "email": rag.render_email(lead, ctx),
        "whatsapp": rag.render_whatsapp(lead, ctx),
        "call_script": rag.render_call_script(ctx),
    }

    reasoning = "; ".join(reasons) if reasons else "Heurística padrão aplicada."
    tags = rag.default_tags(ctx, score)

    return {
        **state,
        "lead_score": score,
        "status": status,
        "messaging": messaging,
        "reasoning": reasoning,
        "tags": tags,
    }
