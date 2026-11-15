"""Suficiência de evidências e validação de citações."""
import re
from collections.abc import Collection

from src import config

# Só colchetes puramente numéricos: [1], [1, 3]; ignora [2016-2018], [nota 5/5], [ver abaixo]
_CITACAO = re.compile(r"\[(\s*\d+(?:\s*,\s*\d+)*\s*)\]")


def numeros_citados(texto: str) -> list[int]:
    """Extrai todos os números citados em colchetes: [1], [1, 3], [2][5]."""
    numeros: list[int] = []
    for grupo in _CITACAO.findall(texto):
        numeros += [int(n) for n in re.findall(r"\d+", grupo)]
    return numeros


def avaliar_evidencias(state: dict) -> dict:
    """Suficiente se ao menos MIN_EVIDENCES evidências têm score >= MIN_SCORE."""
    fortes = [e for e in state.get("evidencias", []) if e["score"] >= config.MIN_SCORE]
    return {"suficiente": len(fortes) >= config.MIN_EVIDENCES}


def citacoes_invalidas(resposta: str, citacoes: list[int], validos: int | Collection[int]) -> list[int]:
    """Números citados (texto e lista) fora do conjunto válido; int N equivale a 1..N."""
    conjunto = set(range(1, validos + 1)) if isinstance(validos, int) else set(validos)
    return sorted({n for n in numeros_citados(resposta) + list(citacoes) if n not in conjunto})


def validar_citacoes(resposta: str, citacoes: list[int], n_evidencias: int | Collection[int]) -> bool:
    """True se todo [n] do texto e da lista é válido (1..N ou conjunto de números dado).

    Com evidências disponíveis, a resposta precisa citar ao menos uma.
    """
    tem_evidencias = bool(n_evidencias) if not isinstance(n_evidencias, int) else n_evidencias > 0
    if tem_evidencias and not numeros_citados(resposta):
        return False
    return not citacoes_invalidas(resposta, citacoes, n_evidencias)
