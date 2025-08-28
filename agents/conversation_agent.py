import os
import json
from openai import OpenAI
from agents.simple_rag import get_rag
from loguru import logger

class ConversationAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.rag = get_rag()
    
    def process_lead(self, lead_data: dict) -> dict:
        """Processa lead e gera estratégia de conversão"""
        
        # Buscar informações relevantes no RAG
        query = f"{lead_data.get('interest', '')} {lead_data.get('utm_source', '')}"
        relevant_docs = self.rag.search_relevant_content(query)
        
        # Contexto para o AI
        context = "Nenhuma informação específica encontrada."
        if relevant_docs:
            context = "\n".join([doc['content'][:500] for doc in relevant_docs])
        
        # Prompt mais direto
        prompt = f"""
        Lead: {lead_data.get('name')} interessado em {lead_data.get('interest', 'serviços gerais')}
        Fonte: {lead_data.get('utm_source', 'site')}
        
        Informações da empresa: {context}
        
        Crie uma estratégia de conversão. Responda apenas JSON válido:
        {{
          "message": "mensagem personalizada para o lead",
          "next_action": "call/whatsapp/email", 
          "priority": "baixa/média/alta",
          "product_suggestion": "produto mais adequado"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Tentar extrair JSON da resposta
            try:
                # Remover markdown se houver
                if "```json" in ai_response:
                    ai_response = ai_response.split("```json")[1].split("```")[0]
                
                ai_json = json.loads(ai_response)
                
                return {
                    "status": "processed",
                    "lead_name": lead_data.get('name'),
                    "message": ai_json.get("message", "Lead será contatado em breve"),
                    "next_action": ai_json.get("next_action", "call"),
                    "priority": ai_json.get("priority", "média"),
                    "product_suggestion": ai_json.get("product_suggestion", "Consulta inicial")
                }
                
            except json.JSONDecodeError:
                # Fallback se não conseguir parsear JSON
                return {
                    "status": "processed",
                    "lead_name": lead_data.get('name'),
                    "message": f"Olá {lead_data.get('name')}, entraremos em contato sobre {lead_data.get('interest')}",
                    "next_action": "call",
                    "priority": "média",
                    "raw_ai_response": ai_response
                }
            
        except Exception as e:
            logger.error(f"Erro no conversation agent: {e}")
            return {
                "status": "error",
                "message": f"Erro ao processar lead: {str(e)}"
            }

def process_lead_conversation(lead_data: dict) -> dict:
    """Função pública para processar lead"""
    agent = ConversationAgent()
    return agent.process_lead(lead_data)
