"""Limpezas de texto.

- limpar_leve: para embedding, LLM e exibição. Mantém acentos, pontuação e negações.
- limpar_bm25: para o índice lexical. Lowercase, sem acentos, abreviações expandidas,
  stopwords removidas EXCETO negações. Aplicar a mesma função à pergunta.
"""

NEGACOES = {"não", "nao", "nem", "nunca", "jamais", "nenhum", "nenhuma", "sem"}


def limpar_leve(texto: str) -> str:
    """Remove HTML, URLs e e-mails, normaliza unicode (NFC) e colapsa espaços."""
    raise NotImplementedError


def limpar_bm25(texto: str) -> list[str]:
    """Retorna tokens normalizados para o BM25."""
    raise NotImplementedError
