# agents/nodes/context_builder.py
from typing import Dict, Any

def context_builder(state: Dict[str, Any]) -> Dict[str, Any]:
    lead = state.get("lead", {}) or {}
    ctx = state.get("context", {}) or {}

    # Defaults seguros
    ctx.setdefault("region", "BR")
    ctx.setdefault("language", "pt-BR")
    ctx.setdefault("icp_profile", {})

    return {
        **state,
        "lead": lead,
        "context": ctx,
    }
