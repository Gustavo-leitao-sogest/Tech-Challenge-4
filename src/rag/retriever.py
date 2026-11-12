"""Recuperação híbrida (busca densa + BM25 + Reciprocal Rank Fusion).

Densa (Chroma, prefixo "query: ") e BM25 com os mesmos filtros -> RRF (k=60) -> top-k.
RRF implementado manualmente: dispensa normalizar scores de escalas diferentes.
"""


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Funde rankings de IDs: score(d) = soma de 1 / (k + posição de d em cada ranking)."""
    raise NotImplementedError


def recuperar(state: dict) -> dict:
    raise NotImplementedError


def recuperar_panorama(state: dict) -> dict:
    """Rota panorâmica: contagens por tópico (pandas) + evidências representativas."""
    raise NotImplementedError
