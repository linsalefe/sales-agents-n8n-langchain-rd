# agents/graph.py
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from agents.nodes.context_builder import context_builder
from agents.nodes.sales_agent import sales_agent
from agents.nodes.crm_agent import crm_agent


class State(TypedDict, total=False):
    lead: Dict[str, Any]
    context: Dict[str, Any]
    lead_score: int
    status: str
    messaging: Dict[str, Any]
    crm_actions: Dict[str, Any]
    reasoning: str
    tags: list[str]


def build_graph():
    graph = StateGraph(State)

    graph.add_node("context_builder", context_builder)
    graph.add_node("sales_agent", sales_agent)
    graph.add_node("crm_agent", crm_agent)

    graph.set_entry_point("context_builder")
    graph.add_edge("context_builder", "sales_agent")
    graph.add_edge("sales_agent", "crm_agent")
    graph.add_edge("crm_agent", END)

    return graph.compile()


compiled = build_graph()


def run_graph(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executa o grafo e retorna já no contrato esperado pelo /process-lead.
    """
    state_in: State = {
        "lead": payload["lead"],
        "context": payload["context"],
    }

    out: State = compiled.invoke(state_in)

    return {
        "status": out["status"],
        "lead_score": out["lead_score"],
        "reasoning": out.get("reasoning", ""),
        "crm_actions": out["crm_actions"],
        "messaging": out["messaging"],
    }
