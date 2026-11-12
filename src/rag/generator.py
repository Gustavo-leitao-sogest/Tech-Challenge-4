"""Montagem do contexto numerado e geração com saída estruturada (RespostaLLM)."""


def montar_contexto(evidencias: list[dict]) -> str:
    raise NotImplementedError


def gerar_resposta(state: dict) -> dict:
    raise NotImplementedError


def reescrever_consulta(state: dict) -> dict:
    """HyDE: gera uma avaliação hipotética e usa como nova consulta."""
    raise NotImplementedError


def recusar(state: dict) -> dict:
    raise NotImplementedError
