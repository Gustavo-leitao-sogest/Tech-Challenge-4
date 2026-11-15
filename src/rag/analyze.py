"""Nó analisar_pergunta: escopo, tipo, consulta reescrita e filtros (saída AnalisePergunta)."""
import difflib
import logging
from functools import lru_cache

import pandas as pd

from src import config
from src.rag.llm import get_llm
from src.rag.prompts import MSG_FORA_DE_ESCOPO, PROMPT_ANALISE_V2
from src.rag.schemas import AnalisePergunta

log = logging.getLogger(__name__)

CUTOFF_CATEGORIA = 0.8
CHAVES_FILTRO = (
    "nota_min", "nota_max", "categoria", "apenas_atrasados", "ano_mes_ini", "ano_mes_fim",
)


@lru_cache(maxsize=1)
def listar_categorias() -> tuple[str, ...]:
    """Categorias reais do dataset (sem 'desconhecido'), lidas uma vez do parquet."""
    col = pd.read_parquet(config.REVIEWS_PARQUET, columns=["categoria"])["categoria"]
    return tuple(sorted(c for c in col.dropna().unique() if c != config.DESCONHECIDO))


def normalizar_categoria(categoria: str | None, validas: tuple[str, ...] | list[str]) -> str | None:
    """Mapeia a categoria para a lista válida (aproximação por difflib) ou descarta."""
    if not categoria:
        return None
    cat = categoria.strip().lower().replace(" ", "_")
    if cat in validas:
        return cat
    achadas = difflib.get_close_matches(cat, list(validas), n=1, cutoff=CUTOFF_CATEGORIA)
    return achadas[0] if achadas else None


def _nota_conflita(filtros: dict) -> bool:
    lo, hi = filtros.get("nota_min"), filtros.get("nota_max")
    return lo is not None and hi is not None and lo > hi


def montar_filtros(analise: AnalisePergunta, filtros_ui: dict | None,
                   validas: tuple[str, ...] | list[str]) -> dict:
    """Filtros extraídos (saneados) mesclados com os da UI, que têm precedência."""
    filtros = {
        "nota_min": analise.nota_min,
        "nota_max": analise.nota_max,
        "categoria": normalizar_categoria(analise.categoria, validas),
        "apenas_atrasados": analise.apenas_atrasados,
    }
    if _nota_conflita(filtros):
        filtros["nota_min"] = filtros["nota_max"] = None
    ui = {k: v for k, v in (filtros_ui or {}).items() if k in CHAVES_FILTRO and v is not None}
    mesclado = {**filtros, **ui}
    if _nota_conflita(mesclado):  # par do LLM conflita com a UI: vale só o que a UI mandou
        mesclado["nota_min"], mesclado["nota_max"] = ui.get("nota_min"), ui.get("nota_max")
    return {k: v for k, v in mesclado.items() if v is not None}


def _analise_padrao(pergunta: str) -> AnalisePergunta:
    """Fallback neutro: em escopo, específica, consulta = pergunta, sem filtros."""
    return AnalisePergunta(em_escopo=True, tipo="especifica", consulta_reescrita=pergunta)


def _invocar_analise(prompt: str, pergunta: str) -> tuple[AnalisePergunta, bool]:
    """Chama o LLM; (análise, extraiu_filtros). Cai no padrão se a saída vier vazia/inválida."""
    try:
        analise = get_llm().with_structured_output(AnalisePergunta).invoke(prompt)
    except ValueError as e:  # inclui ValidationError e OutputParserException
        log.warning("Saída estruturada inválida na análise: %s", e)
        analise = None
    if analise is None:
        log.warning("Análise indisponível; usando a pergunta como consulta, sem filtros extraídos.")
        return _analise_padrao(pergunta), False
    return analise, True


def analisar_pergunta(state: dict) -> dict:
    """Analisa a pergunta com o LLM e monta consulta, filtros e status de escopo."""
    validas = listar_categorias()
    prompt = PROMPT_ANALISE_V2.format(
        pergunta=state["pergunta"], categorias="\n".join(validas)
    )
    analise, _ = _invocar_analise(prompt, state["pergunta"])
    saida = {
        "analise": analise.model_dump(),
        "consulta": analise.consulta_reescrita or state["pergunta"],
        "filtros": montar_filtros(analise, state.get("filtros_ui"), validas),
        "tentativas": 0,
    }
    if not analise.em_escopo:
        saida.update(status="fora_de_escopo", resposta=MSG_FORA_DE_ESCOPO, citacoes=[])
    return saida
