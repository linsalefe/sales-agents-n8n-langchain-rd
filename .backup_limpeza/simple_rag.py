# agents/simple_rag.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import PyPDF2


@dataclass
class Chunk:
    source: str        # caminho do arquivo
    category: str      # "produtos" | "empresas"
    title: str         # nome amigável (arquivo)
    text: str          # conteúdo do chunk


class SimpleRAG:
    """
    RAG enxuto para TXT/PDF em data/{produtos,empresas}.
    - Sem embeddings, sem DB, sem chamadas à OpenAI.
    - Carrega, fragmenta em chunks e faz uma busca por sobreposição de tokens.
    API típica:
        rag = SimpleRAG().load()
        ctx = rag.build_context("consulta do lead", top_k=4)

    Compatibilidade:
        - Método extra `search_relevant_content(...)` retorna List[Dict] no formato
          similar ao código anterior do usuário (source, content, type, title).
    """

    def __init__(self, data_dir: Optional[Path] = None, chunk_size: int = 800, overlap: int = 120):
        root = Path(__file__).resolve().parents[1]  # .../project-root
        self.data_dir = data_dir or (root / "data")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[Chunk] = []

    # ---------- Público ----------
    def load(self) -> "SimpleRAG":
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Diretório de dados não encontrado: {self.data_dir}")

        for category in ("produtos", "empresas"):
            dirpath = self.data_dir / category
            if not dirpath.exists():
                continue

            for path in sorted(dirpath.glob("**/*")):
                if path.is_dir():
                    continue
                text = self._read_any(path)
                if not text or not text.strip():
                    continue

                title = path.stem.replace("_", " ").title()
                for part in self._split(text):
                    self.chunks.append(Chunk(str(path), category, title, part))

        return self

    def retrieve(self, query: str, top_k: int = 4) -> List[Chunk]:
        """Retorna os top_k chunks mais relevantes via sobreposição de tokens + boosts simples."""
        q_tokens = self._tokens(query)
        if not q_tokens:
            # Sem consulta: retorna 1 de produto + 1 de empresa (fallback)
            return self._fallback_minimal()

        scored: List[Tuple[float, Chunk]] = []

        for ch in self.chunks:
            tokens = self._tokens(ch.text)
            if not tokens:
                continue

            # sobreposição de tokens normalizada
            overlap = len(q_tokens & tokens)
            base = overlap / max(1, len(q_tokens))

            # boosts simples
            boost_cat = 1.2 if ch.category == "produtos" else 1.0
            boost_title = 1.5 if any(t in ch.title.lower() for t in q_tokens) else 1.0

            # leve TF para diferenciar empates
            tf_bonus = sum(ch.text.lower().count(t) for t in q_tokens) * 0.01

            score = base * boost_cat * boost_title + tf_bonus
            if score > 0:
                scored.append((score, ch))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for _, c in scored[:top_k]]

        # fallback: 1 produto + 1 empresa se nada casou
        if not top:
            top = self._fallback_minimal()

        return top

    def build_context(self, query: str, top_k: int = 4) -> str:
        """Monta um contexto concatenado com cabeçalho por chunk."""
        hits = self.retrieve(query, top_k=top_k)
        blocks = []
        for ch in hits:
            header = f"[{ch.category.upper()} • {ch.title} • {ch.source}]"
            blocks.append(f"{header}\n{ch.text.strip()}")
        return "\n\n---\n\n".join(blocks)

    # Compat: método no formato parecido com a versão enviada pelo usuário
    def search_relevant_content(self, query: str, max_results: int = 3) -> List[Dict]:
        """Wrapper que retorna dicionários no formato (source, content, type, title)."""
        hits = self.retrieve(query, top_k=max_results)
        return [
            {
                "source": ch.source,
                "content": ch.text,
                "type": ch.category,
                "title": ch.title,
            }
            for ch in hits
        ]

    # ---------- Internos ----------
    def _read_any(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            return self._extract_pdf_text(path)
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="ignore")

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        try:
            text_parts: List[str] = []
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text() or ""
                    text_parts.append(extracted)
            return "\n".join(text_parts)
        except Exception:
            return ""

    def _split(self, text: str) -> List[str]:
        # normaliza espaços e quebras
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                window = text[start:end]
                # tenta quebrar em ponto final/; ou início de seção
                last_break = max(window.rfind("\n#"), window.rfind(". "), window.rfind("; "))
                if last_break == -1:
                    last_break = end
                else:
                    last_break += 1
                chunk = text[start:last_break]
                chunks.append(chunk)
                start = max(0, last_break - self.overlap)
            else:
                chunks.append(text[start:])
                break
        return chunks

    def _tokens(self, s: str) -> set[str]:
        # inclui acentos pt-BR e números
        return set(re.findall(r"[a-zà-ú0-9]+", s.lower()))

    def _fallback_minimal(self) -> List[Chunk]:
        """Garante pelo menos 1 chunk de produto + 1 de empresa, se existirem."""
        pick = {"produtos": None, "empresas": None}
        for ch in self.chunks:
            if pick.get(ch.category) is None:
                pick[ch.category] = ch
            if all(pick.values()):
                break
        return [c for c in pick.values() if c]


# Instância global (lazy)
_rag_instance: Optional[SimpleRAG] = None


def get_rag() -> SimpleRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = SimpleRAG().load()
    return _rag_instance
