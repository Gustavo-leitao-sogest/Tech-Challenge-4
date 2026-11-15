"""Montagem do contexto numerado, geração com citações, reescrita HyDE e recusa."""
import logging

from src.rag.guardrails import citacoes_invalidas, numeros_citados, validar_citacoes
from src.rag.llm import get_llm
from src.rag.prompts import (
    AVISO_CITACOES_INVALIDAS, INSTRUCAO_CONTAGENS, MSG_INSUFICIENTE, PROMPT_GERACAO_V2,
    PROMPT_HYDE_V1,
)
from src.rag.schemas import RespostaLLM

log = logging.getLogger(__name__)

_ATRASO = {True: "sim", False: "não", "sim": "sim", "nao": "não", "não": "não"}


def _rotulo_atraso(valor) -> str:
    """Aceita bool/None ou texto sim/nao/desconhecido."""
    try:
        return _ATRASO.get(valor, "desconhecido")
    except TypeError:
        return "desconhecido"


def _neutralizar(texto: str) -> str:
    """Troca < e > por equivalentes visuais, para o texto não fechar tags do prompt."""
    return " ".join(str(texto).split()).replace("<", "‹").replace(">", "›")


def montar_contexto(evidencias: list[dict]) -> str:
    """Uma evidência por linha: [n] (nota · data · categoria · atraso) texto."""
    linhas = []
    for e in evidencias:
        meta = (f"nota {e['nota']}/5 · {e['data']} · {e['categoria']} · "
                f"entregue com atraso: {_rotulo_atraso(e.get('atraso'))}")
        linhas.append(f"[{e['n']}] ({meta}) {_neutralizar(e['texto'])}")
    return "\n".join(linhas)


def _bloco_contagens(contagens: dict | None) -> str:
    if not contagens:
        return ""
    itens = "\n".join(f"- {_neutralizar(t)}: {f}" for t, f in contagens.items())
    return f"\n<contagens>\n{itens}\n</contagens>\n"


def _montar_prompt(state: dict) -> str:
    contagens = state.get("contagens")
    return PROMPT_GERACAO_V2.format(
        contexto=montar_contexto(state.get("evidencias", [])),
        bloco_contagens=_bloco_contagens(contagens),
        instrucao_contagens=INSTRUCAO_CONTAGENS if contagens else "",
        pergunta=state["pergunta"],
    )


def _invocar(chain, prompt: str) -> RespostaLLM | None:
    """Chama o LLM; None se a saída estruturada vier vazia ou inválida."""
    try:
        return chain.invoke(prompt)
    except ValueError as e:  # inclui ValidationError e OutputParserException
        log.warning("Saída estruturada inválida na geração: %s", e)
        return None


def gerar_resposta(state: dict) -> dict:
    """Gera a resposta citada; refaz uma vez, avisando das citações inválidas."""
    validos = {e["n"] for e in state.get("evidencias", [])}
    chain = get_llm().with_structured_output(RespostaLLM)
    prompt = _montar_prompt(state)
    saida = None
    for tentativa in range(2):
        saida = _invocar(chain, prompt)
        if saida is not None and validar_citacoes(saida.resposta, saida.citacoes, validos):
            break
        if tentativa == 0 and saida is not None:
            ruins = citacoes_invalidas(saida.resposta, saida.citacoes, validos)
            aviso = AVISO_CITACOES_INVALIDAS.format(
                invalidas=ruins or "nenhuma citação [n] no texto",
                validos=f"1 a {max(validos)}" if validos else "nenhum",
            )
            prompt = _montar_prompt(state) + aviso
    else:
        return recusar(state)
    if not saida.suficiente:
        return recusar(state)
    citacoes = sorted(set(saida.citacoes) | set(numeros_citados(saida.resposta)))
    return {"status": "ok", "resposta": saida.resposta, "citacoes": citacoes}


def reescrever_consulta(state: dict) -> dict:
    """HyDE: avaliação hipotética como nova consulta; se vier vazia, mantém a anterior."""
    texto = (get_llm().invoke(PROMPT_HYDE_V1.format(pergunta=state["pergunta"])).text or "").strip()
    anterior = state.get("consulta") or state["pergunta"]
    saida = {"consulta": texto or anterior, "tentativas": state.get("tentativas", 0) + 1}
    # A primeira consulta (reescrita da análise) é a referência do score de suficiência:
    # o texto hipotético serve para buscar, mas inflaria o cosseno de temas inexistentes.
    if not state.get("consulta_score"):
        saida["consulta_score"] = anterior
    return saida


def recusar(state: dict) -> dict:
    """Recusa por falta de evidências; as evidências ficam no state para a UI."""
    return {"status": "insuficiente", "resposta": MSG_INSUFICIENTE, "citacoes": []}
