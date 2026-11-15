"""Grafo LangGraph que orquestra análise, recuperação, guardrails e geração."""
import argparse
import sys
from collections.abc import Callable
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src import config
from src.rag import analyze, generator, guardrails, retriever
from src.rag.schemas import Evidencia, RagResult


class RagState(TypedDict, total=False):
    """Estado compartilhado entre os nós do grafo."""

    pergunta: str
    filtros_ui: dict | None
    modo: str | None
    analise: dict
    consulta: str
    consulta_score: str
    filtros: dict
    evidencias: list[dict]
    contagens: dict | None
    panorama_amostra: int
    tentativas: int
    suficiente: bool
    status: str
    resposta: str
    citacoes: list[int]
    modelo: str


# --- Rotas (funções puras) ---

def rota_escopo(state: dict) -> str:
    """Decide o ramo após a análise: fim, panorâmica ou específica."""
    if state.get("status") == "fora_de_escopo":
        return "fim"
    if (state.get("analise") or {}).get("tipo") == "panoramica":
        return "panoramica"
    return "especifica"


def rota_panorama(state: dict) -> str:
    """Panorâmica: sem evidências recusa; senão gera.

    Não há limiar de score aqui. Numa base de 42 mil avaliações, pergunta genérica sempre tem
    material, e os scores de consultas genéricas (0,88-0,89) não se distinguem dos de temas
    inexistentes. O risco é inventar frequência, tratado declarando a contagem sobre a amostra
    recuperada; a barreira é o suficiente=false do gerador.
    """
    return "gerar" if state.get("evidencias") else "recusar"


def rota_suficiencia(state: dict) -> str:
    """Específica: gera se suficiente; senão tenta HyDE enquanto houver tentativas; senão recusa."""
    if state.get("suficiente"):
        return "gerar"
    if state.get("tentativas", 0) < config.MAX_RETRIES:
        return "reescrever"
    return "recusar"


def _nos_padrao() -> dict[str, Callable[[dict], dict]]:
    return {
        "analisar": analyze.analisar_pergunta,
        "recuperar": retriever.recuperar,
        "recuperar_panorama": retriever.recuperar_panorama,
        "avaliar": guardrails.avaliar_evidencias,
        "reescrever": generator.reescrever_consulta,
        "gerar": generator.gerar_resposta,
        "recusar": generator.recusar,
    }


def build_graph(nos: dict[str, Callable[[dict], dict]] | None = None):
    """Monta e compila o grafo; `nos` permite injetar nós (testes)."""
    n = {**_nos_padrao(), **(nos or {})}
    g = StateGraph(RagState)
    for nome, fn in n.items():
        g.add_node(nome, fn)
    g.add_edge(START, "analisar")
    g.add_conditional_edges("analisar", rota_escopo, {
        "fim": END, "panoramica": "recuperar_panorama", "especifica": "recuperar"})
    g.add_conditional_edges("recuperar_panorama", rota_panorama, {
        "gerar": "gerar", "recusar": "recusar"})
    g.add_edge("recuperar", "avaliar")
    g.add_conditional_edges("avaliar", rota_suficiencia, {
        "gerar": "gerar", "reescrever": "reescrever", "recusar": "recusar"})
    g.add_edge("reescrever", "recuperar")
    g.add_edge("gerar", END)
    g.add_edge("recusar", END)
    return g.compile()


@lru_cache(maxsize=1)
def get_graph():
    """Grafo padrão compilado uma única vez."""
    return build_graph()


def montar_resultado(state: dict) -> RagResult:
    """Converte o state final em RagResult (amostra panorâmica vai em filtros)."""
    filtros = dict(state.get("filtros") or {})
    if state.get("panorama_amostra") is not None:
        filtros["panorama_amostra"] = state["panorama_amostra"]
    return RagResult(
        pergunta=state["pergunta"],
        status=state.get("status", "insuficiente"),
        resposta=state.get("resposta", ""),
        citacoes=state.get("citacoes", []),
        evidencias=[Evidencia(**e) for e in state.get("evidencias", [])],
        consulta_transformada=state.get("consulta"),
        filtros=filtros,
        contagens=state.get("contagens"),
        tentativas=state.get("tentativas", 0),
        modelo=state.get("modelo"),
    )


def answer(pergunta: str, filtros: dict | None = None, modo: str | None = None,
           grafo=None) -> RagResult:
    """Responde a uma pergunta executando o grafo completo."""
    entrada: RagState = {"pergunta": pergunta, "filtros_ui": filtros}
    if modo:
        entrada["modo"] = modo
    final = (grafo or get_graph()).invoke(entrada)
    return montar_resultado(final)


def mermaid() -> str:
    """Diagrama Mermaid do grafo (para o README)."""
    return build_graph().get_graph().draw_mermaid()


def main() -> None:
    """CLI: imprime a resposta e as evidências citadas, ou o diagrama."""
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Executa o pipeline RAG")
    ap.add_argument("pergunta", nargs="?")
    ap.add_argument("--mermaid", action="store_true")
    args = ap.parse_args()
    if args.mermaid:
        print(mermaid())
        return
    if not args.pergunta:
        ap.error("informe a pergunta ou --mermaid")
    r = answer(args.pergunta)
    print(f"status: {r.status}")
    print(f"modelo: {r.modelo}")
    print(f"consulta: {r.consulta_transformada}")
    print(f"filtros: {r.filtros}")
    print(f"\n{r.resposta}\n")
    for e in r.evidencias:
        if e.n in r.citacoes:
            print(f"[{e.n}] nota {e.nota} | {e.categoria} | {e.texto[:100]}")


if __name__ == "__main__":
    main()
