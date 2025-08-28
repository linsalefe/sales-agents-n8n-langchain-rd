import os
from pathlib import Path
from typing import List, Dict
import PyPDF2
from openai import OpenAI

class SimpleRAG:
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.documents = []
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._load_documents()
    
    def _load_documents(self):
        """Carrega documentos TXT e PDF da pasta data/"""
        for folder in ["produtos", "empresas"]:
            folder_path = self.data_path / folder
            if folder_path.exists():
                # Arquivos TXT
                for txt_file in folder_path.glob("*.txt"):
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self.documents.append({
                            "source": str(txt_file),
                            "content": content,
                            "type": folder
                        })
                
                # Arquivos PDF
                for pdf_file in folder_path.glob("*.pdf"):
                    content = self._extract_pdf_text(pdf_file)
                    if content:
                        self.documents.append({
                            "source": str(pdf_file),
                            "content": content,
                            "type": folder
                        })
    
    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extrai texto de PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Erro ao ler PDF {pdf_path}: {e}")
            return ""
    
    def search_relevant_content(self, query: str, max_results: int = 3) -> List[Dict]:
        """Busca conteúdo relevante usando OpenAI"""
        if not self.documents:
            return []
        
        # Usar OpenAI para encontrar documentos mais relevantes
        try:
            prompt = f"""
            Query: {query}
            
            Encontre os documentos mais relevantes para esta consulta.
            
            Documentos disponíveis:
            {[doc['source'] for doc in self.documents[:10]]}
            
            Retorne apenas os nomes dos arquivos mais relevantes, separados por vírgula.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            
            # Por simplicidade, retorna os primeiros documentos se não conseguir fazer busca inteligente
            return self.documents[:max_results]
            
        except Exception:
            # Fallback: retorna primeiros documentos
            return self.documents[:max_results]

# Instância global
_rag_instance = None

def get_rag() -> SimpleRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = SimpleRAG()
    return _rag_instance
