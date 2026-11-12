"""Orquestração com LangGraph.

Expor: build_graph() e answer(pergunta: str, filtros: dict | None = None) -> RagResult.
"""


def build_graph():
    raise NotImplementedError


def answer(pergunta: str, filtros: dict | None = None):
    raise NotImplementedError
