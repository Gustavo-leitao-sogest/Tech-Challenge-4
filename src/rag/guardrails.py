"""Suficiência de evidências e validação de citações."""


def avaliar_evidencias(state: dict) -> dict:
    raise NotImplementedError


def validar_citacoes(resposta: str, citacoes: list[int], n_evidencias: int) -> bool:
    """True se todo [n] presente no texto e na lista existe entre 1 e n_evidencias."""
    raise NotImplementedError
