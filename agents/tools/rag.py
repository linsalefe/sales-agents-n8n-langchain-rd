# agents/tools/rag.py
from typing import Dict, Tuple, List

def slugify(text: str) -> str:
    return (
        text.lower()
        .replace("ç", "c")
        .replace("ã", "a")
        .replace("á", "a")
        .replace("à", "a")
        .replace("é", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ô", "o")
        .replace("ú", "u")
        .replace("/", "-")
        .replace(" ", "-")
    )

# Base mínima de ICP/Playbooks (pode evoluir para Chroma/PGVector)
ICP_KEYWORDS: Dict[str, List[str]] = {
    "Educação / Saúde": ["psicolog", "psico", "clinica", "saude mental", "coordenador", "coordenadora"],
    "Educação": ["professor", "gestor", "coordenador"],
    "Saúde": ["psicolog", "enferm", "terapeuta", "psiqu"],
}

def icp_fit_score(lead: Dict, ctx: Dict) -> Tuple[int, List[str]]:
    base = 50
    reasons: List[str] = []

    seg = (ctx.get("icp_profile") or {}).get("segment", "")
    prof = (lead or {}).get("profession", "") or ""
    phone = (lead or {}).get("phone")
    utm_medium = (lead or {}).get("utm_medium", "")
    utm_source = (lead or {}).get("utm_source", "")

    # Segmento
    if seg in ICP_KEYWORDS:
        base += 10
        reasons.append(f"Segmento {seg} reconhecido")

    # Profissão
    prof_l = prof.lower()
    for _seg, kws in ICP_KEYWORDS.items():
        if any(k in prof_l for k in kws):
            base += 10
            reasons.append(f"Profissão com match ({prof})")
            break

    # UTM e canais
    if utm_medium and utm_medium.lower() == "cpc":
        base += 5
        reasons.append("Tráfego pago (cpc)")
    if utm_source and utm_source.lower() in {"facebook", "meta", "instagram"}:
        base += 5
        reasons.append(f"Fonte social ({utm_source})")

    # Dados de contato
    if phone:
        base += 10
        reasons.append("Telefone presente")

    score = max(0, min(100, base))
    return score, reasons

def render_email(lead: Dict, ctx: Dict) -> Dict[str, str]:
    name = lead.get("name", "")
    product = ctx.get("product", "seu interesse")
    subj = f"[CENAT] Próximo passo sobre {product}"
    body = (
        f"Olá {name},\n\n"
        f"Obrigado pelo interesse em {product}. "
        "Podemos agendar uma conversa rápida para entender seu momento e orientar o melhor caminho?\n\n"
        "Agenda sugerida: hoje às 16h ou amanhã às 10h.\n\n"
        "Abraços,\nEquipe CENAT"
    )
    return {"subject": subj, "body": body}

def render_whatsapp(lead: Dict, ctx: Dict) -> str:
    name = lead.get("name", "")
    product = ctx.get("product", "o curso")
    return (
        f"Oi {name}! Vi seu interesse em {product}. "
        "Podemos falar rapidinho hoje ou amanhã para te passar os próximos passos?"
    )

def render_call_script(ctx: Dict) -> str:
    product = ctx.get("product", "o curso")
    return f"Abertura → Diagnóstico → Oferta ({product}) → CTA"

def default_tags(ctx: Dict, score: int) -> List[str]:
    tags = []
    product = ctx.get("product")
    if product:
        tags.append(slugify(product))
    if score >= 75:
        tags.append("alto-fit")
    elif score < 60:
        tags.append("avaliar-fit")
    return tags
