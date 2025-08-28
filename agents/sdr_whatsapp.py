# agents/sdr_whatsapp.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional

from openai import OpenAI

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # fallback simples


SDR_SYSTEM_PROMPT_TMPL = """Você é ANA, SDR do CENAT. Sua missão é transformar interesse em ligação agendada com a consultora.

Regras-chave:
- Respostas curtas: no máximo 3 linhas, PT-BR, claras e diretas, sempre com CTA para o agendamento.
- Sempre use o nome do lead e se apresente: “Sou a Ana, do CENAT”.
- Foco somente no curso de interesse informado: {course_name}. Não sugerir outros cursos.
- Se não souber, encaminhe para humano (escreva apenas #HUMANO).
- Sempre busque data e hora para ligação telefônica (15–20 min). Ao confirmar, gere o marcador: 
  #AGENDAR|data=YYYY-MM-DD|hora=HH:MM|duracao=20|min_gap=5|lead={lead_name}|curso={course_name}|contato=whatsapp:{lead_phone}|email={lead_email}
- Confirme/cole o e-mail para envio do voucher de isenção da matrícula por e-mail.
- Timezone padrão: {timezone}. Hoje: {today}.

Contexto:
- Este é o primeiro contato do processo seletivo da pós-graduação em {course_name}.
- Linguagem acolhedora, técnica e direta, sem jargões.
- Mantenha o foco em coletar informações essenciais e agendar a ligação.

Fluxo do atendimento (siga, adaptando ao que o lead já respondeu):
1) Abertura: confirme que é do CENAT, mencione a aplicação na pós de {course_name} e diga que é o 1º contato do processo seletivo. Conduza para alinhar e agendar.
2) Formação:
   - Pergunte se já concluiu a graduação.
   - Se for estudante: explique que é necessário ter a graduação concluída para avançar e pergunte se possui outra graduação já concluída.
   - Se não tiver graduação concluída: encerre cordialmente e aguarde conclusão (se insistir em detalhes, responda curto e marque #HUMANO).
3) Atuação: reconheça a área onde atua e relacione com a pós de {course_name} em 1 frase.
4) Motivação: pergunte objetivamente o que motiva a fazer {course_name}.
5) Investimento: informe que o investimento é por volta de R$ 300/mês e pergunte direto se consegue investir esse valor.
6) Agendamento (objetivo principal):
   - Peça data e horário para a ligação (15–20 min) com a consultora.
   - Se o lead não propor, ofereça 2 janelas (ex.: “Hoje 16:30 ou amanhã 10:00?”).
   - Garanta que, quando o lead confirmar um horário, você inclua o marcador #AGENDAR na mesma resposta.
   - Confirme e-mail para envio do voucher de isenção da matrícula.
7) Confirmação final: confirme dia/hora; reforce a importância de atender no horário combinado e informe que a consultora detalhará conteúdo e condições.

Formato das respostas:
- Sempre até 3 linhas.
- Termine com uma pergunta que avance (disponibilidade, confirmação de e-mail ou escolha de horário).
- Quando o lead CONFIRMAR horário, inclua também a linha do marcador #AGENDAR (formato exato acima).

Exemplos concisos (apenas guia):
- Abertura: “Oi, {lead_name}! Sou a Ana, do CENAT. Recebi sua aplicação na pós de {course_name}. Este é o 1º contato do processo seletivo. Podemos alinhar rapidinho para agendarmos sua ligação?”
- Formação: “Você já concluiu sua graduação? Para avançarmos na seleção é necessário. Caso esteja cursando outra, tem alguma graduação já concluída?”
- Investimento: “O investimento é ~R$ 300/mês. Você consegue assumir esse valor para iniciar a especialização?”
- Agendamento: “Podemos falar por telefone (15–20 min). Pode hoje 16:30 ou amanhã 10:00? Confirma seu e-mail para eu enviar o voucher de isenção?”

Lembre-se: Responda curto (≤3 linhas), personalize com o nome, conduza sempre para o agendamento e use o marcador #AGENDAR quando houver confirmação de horário.
"""

@dataclass
class LeadContext:
    lead_name: str
    course_name: str
    lead_phone: str
    lead_email: str
    timezone: str = "America/Fortaleza"
    today: Optional[str] = None  # YYYY-MM-DD

    def as_dict(self) -> Dict[str, str]:
        today_str = self.today
        if not today_str:
            if ZoneInfo and self.timezone:
                now = datetime.now(ZoneInfo(self.timezone))
            else:
                now = datetime.now()
            today_str = now.strftime("%Y-%m-%d")
        return {
            "lead_name": self.lead_name,
            "course_name": self.course_name,
            "lead_phone": self.lead_phone,
            "lead_email": self.lead_email,
            "timezone": self.timezone,
            "today": today_str,
        }


class SdrWhatsappAgent:
    """
    Agente SDR para WhatsApp usando o prompt fixo.
    - Responde curto (≤3 linhas), orientado a agendamento.
    - Em caso de confirmação de horário, o modelo deve emitir o marcador #AGENDAR.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY não configurada.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def _render_system(self, lead_ctx: LeadContext) -> str:
        ctx = lead_ctx.as_dict()
        return SDR_SYSTEM_PROMPT_TMPL.format(**ctx)

    def reply(
        self,
        lead_ctx: LeadContext,
        last_user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 220,
    ) -> str:
        """
        Gera a próxima resposta para o WhatsApp.
        - lead_ctx: dados do lead/curso.
        - last_user_message: última mensagem do lead.
        - history: conversas anteriores no formato [{"role": "user"/"assistant", "content": "..."}]
        """
        messages: List[Dict[str, str]] = [{"role": "system", "content": self._render_system(lead_ctx)}]

        # histórico (opcional)
        if history:
            messages.extend(history)

        # última mensagem do lead
        messages.append({"role": "user", "content": last_user_message})

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
            messages=messages,
        )

        text = (resp.choices[0].message.content or "").strip()
        text = self._enforce_three_lines(text)
        return text

    # --------- util ---------
    @staticmethod
    def _enforce_three_lines(text: str) -> str:
        """
        Garante no máximo 3 linhas.
        - Remove espaços excessivos.
        - Se vier mais de 3 linhas, mantém só as 3 primeiras.
        """
        # normaliza quebras múltiplas
        text = re.sub(r"\n{3,}", "\n\n", text.strip())
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        # se o modelo devolveu um bloco longo com frases separadas por ponto e espaço
        if len(lines) <= 1 and len(text) > 0:
            # tenta quebrar por sentenças curtas
            sentences = re.split(r"(?<=[.!?])\s+", text)
            # junta em até 3 linhas, equilibrando tamanho
            new_lines: List[str] = []
            current = ""
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                candidate = f"{current} {s}".strip() if current else s
                # limite aproximado para manter a resposta enxuta
                if len(candidate) > 140 and current:
                    new_lines.append(current)
                    current = s
                else:
                    current = candidate
                if len(new_lines) == 3:
                    break
            if current and len(new_lines) < 3:
                new_lines.append(current)
            lines = new_lines or [text]

        # corta para no máximo 3
        lines = lines[:3]
        return "\n".join(lines)


# ---------------------------
# Exemplo rápido de uso local
# ---------------------------
if __name__ == "__main__":
    lead = LeadContext(
        lead_name="Maria",
        course_name="Psicologia Clínica",
        lead_phone="+55 11 99999-9999",
        lead_email="maria@email.com",
    )

    agent = SdrWhatsappAgent()
    # histórico hipotético (opcional)
    hist = [
        {"role": "assistant", "content": "Oi, Maria! Sou a Ana, do CENAT. Recebi sua aplicação na pós de Psicologia Clínica. Este é o 1º contato do processo seletivo. Podemos alinhar rapidinho para agendarmos sua ligação?"},
        {"role": "user", "content": "Oi, tudo bem. Pode ser, como funciona?"},
    ]

    print(agent.reply(lead, last_user_message="Tenho interesse, mas ainda estou cursando a graduação.", history=hist))
