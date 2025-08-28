# agents/conversation_agent.py
from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, Optional

from loguru import logger
from openai import OpenAI

from agents.simple_rag import get_rag


class ConversationAgent:
    """
    Gera uma estratégia de conversão usando:
      - Contexto do RAG (data/produtos + data/empresas)
      - Saída STRICT JSON (response_format=json_object)
    Retorno padrão:
      {
        "status": "processed",
        "lead_name": "...",
        "message": "...",
        "next_action": "whatsapp|call|email",
        "priority": "baixa|média|alta",
        "product_suggestion": "..."
      }
    """

    def __init__(self) -> None:
        # Flags/timeout
        self.dry_run = os.getenv("AI_DRY_RUN", "0") == "1"
        self.timeout = float(os.getenv("OPENAI_TIMEOUT", "25"))  # segundos
        logger.warning(f"ConversationAgent init: DRY_RUN={self.dry_run}, timeout={self.timeout}s")

        # NÃO carrega RAG aqui (lazy para evitar travar em DRY_RUN)
        self.rag: Optional[Any] = None

        # Só inicializa OpenAI se NÃO estiver em DRY_RUN
        api_key = os.getenv("OPENAI_API_KEY")
        if self.dry_run:
            self.client = None
            logger.warning("AI_DRY_RUN=1 -> OpenAI desabilitada neste agente")
        else:
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY não configurada.")
            self.client = OpenAI(api_key=api_key)

    # ------------- Público -------------
    def process_lead(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Processa o lead e retorna uma resposta JSON estruturada."""
        try:
            name = (lead_data.get("name") or "").strip() or "Lead"
            interest = (lead_data.get("interest") or "serviços gerais").strip()
            utm_source = (lead_data.get("utm_source") or "site").strip()

            # DRY-RUN: resposta imediata (sem RAG/OpenAI)
            if self.dry_run:
                logger.warning("AI_DRY_RUN=1 -> devolvendo resposta mock sem RAG/OpenAI")
                return {
                    "status": "processed",
                    "lead_name": name,
                    "message": (f"Olá {name}! Vi seu interesse em {interest}. "
                                "Podemos falar por WhatsApp para alinhar sua inscrição?"),
                    "next_action": "whatsapp",
                    "priority": "média",
                    "product_suggestion": interest or "Consulta inicial",
                }

            # 1) Contexto do RAG (carrega lazy)
            ctx = self._build_context(query=f"{interest} {utm_source}", top_k=4)

            # 2) Mensagens + restrições
            system_msg = (
                "Você é um agente de Vendas e CRM do CENAT. "
                "Escreva de forma acolhedora, técnica e direta, com linguagem acessível. "
                "Responda SEMPRE apenas um JSON válido conforme o schema fornecido."
            )

            schema_hint = {
                "message": "mensagem personalizada e curta (máx. 500 caracteres) para enviar ao lead",
                "next_action": "whatsapp | call | email",
                "priority": "baixa | média | alta",
                "product_suggestion": "nome do produto/serviço mais adequado"
            }

            user_msg = (
                f"LEAD:\n"
                f"- Nome: {name}\n"
                f"- Interesse: {interest}\n"
                f"- Origem: {utm_source}\n"
                f"- Email: {lead_data.get('email', '')}\n"
                f"- Telefone: {lead_data.get('phone', '')}\n\n"
                f"CONTEXTOS DO RAG (use apenas o que for útil):\n{ctx}\n\n"
                "INSTRUÇÕES DE SAÍDA (OBRIGATÓRIO):\n"
                "Retorne SOMENTE um objeto JSON, sem texto antes/depois, seguindo exatamente este schema:\n"
                f"{json.dumps(schema_hint, ensure_ascii=False)}\n"
                "REGRAS:\n"
                "- Se o interesse combinar com congresso/evento, priorize 'whatsapp' como next_action.\n"
                "- Se for lead muito quente (congresso em breve, urgência, datas próximas), prioridade 'alta'.\n"
                "- Mantenha a mensagem até 500 caracteres, com CTA claro (ex.: link/whatsapp).\n"
                "- NÃO invente preços/datas. Use apenas o que consta no contexto.\n"
            )

            # 3) Chamada ao modelo (saída forçada em JSON)
            try:
                logger.info("Chamando OpenAI (gpt-4o-mini)")
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=300,
                    temperature=0.2,
                    timeout=self.timeout,
                )
                content = resp.choices[0].message.content.strip()
                payload = self._parse_json(content)
                return {
                    "status": "processed",
                    "lead_name": name,
                    "message": payload.get("message", f"Olá {name}, em breve entraremos em contato."),
                    "next_action": payload.get("next_action", "whatsapp"),
                    "priority": payload.get("priority", "média"),
                    "product_suggestion": payload.get("product_suggestion", "Consulta inicial"),
                }
            except Exception as e:
                # Fallback pragmático se a chamada à OpenAI falhar
                logger.error(f"Falha na chamada OpenAI: {e}")
                return {
                    "status": "processed",
                    "lead_name": name,
                    "message": (f"Olá {name}! Recebemos seu interesse em {interest}. "
                                "Vamos te chamar no WhatsApp para avançarmos."),
                    "next_action": "whatsapp",
                    "priority": "média",
                    "product_suggestion": interest or "Consulta inicial",
                    "error": str(e),
                }

        except Exception as e:
            logger.error(f"Erro no conversation agent: {e}")
            return {
                "status": "error",
                "message": f"Erro ao processar lead: {str(e)}",
            }

    # ------------- Internos -------------
    def _build_context(self, query: str, top_k: int = 4, char_limit: int = 2400) -> str:
        """
        Usa o RAG para montar um contexto concatenado com cabeçalho por chunk.
        Limita o tamanho total para evitar prompts muito grandes.
        """
        try:
            # lazy-load do RAG
            if self.rag is None:
                logger.info("Carregando RAG (lazy)...")
                self.rag = get_rag()

            ctx = self.rag.build_context(query=query, top_k=top_k)
            ctx = ctx.strip() if ctx else "Nenhuma informação específica encontrada."
            if len(ctx) > char_limit:
                ctx = ctx[:char_limit] + " ..."
            return ctx
        except Exception as e:
            logger.warning(f"Falha ao montar contexto do RAG: {e}")
            return "Nenhuma informação específica encontrada."

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """
        Tenta carregar o JSON diretamente; se vier com ruído, faz um fallback
        extraindo o primeiro objeto { ... }.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Remove blocos de markdown
            clean = re.sub(r"```(?:json)?|```", "", text).strip()
            try:
                return json.loads(clean)
            except json.JSONDecodeError:
                # Fallback: pega o primeiro objeto que parece JSON
                m = re.search(r"\{.*\}", clean, flags=re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except json.JSONDecodeError:
                        pass
        # Último fallback
        return {}


# Função pública usada pela API
def process_lead_conversation(lead_data: Dict[str, Any]) -> Dict[str, Any]:
    agent = ConversationAgent()
    return agent.process_lead(lead_data)
