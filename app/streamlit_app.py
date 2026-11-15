"""Interface Streamlit: pergunta, filtros opcionais, resposta, evidências, consulta transformada e status.

Uso: streamlit run app/streamlit_app.py
"""
import sys
import time
from pathlib import Path

# `streamlit run app/streamlit_app.py` põe só `app/` no sys.path; a raiz do projeto
# precisa estar nele para que `import src...` funcione.
RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import streamlit as st  # noqa: E402

SEM_FILTRO = "Sem filtro"
TODAS = "Todas"
NOTA_MIN_PADRAO, NOTA_MAX_PADRAO = 1, 5
MES_INICIAL, MES_FINAL = (2016, 9), (2018, 10)  # cobertura aproximada da base
MODOS = {"Híbrida (densa + BM25)": "hibrida", "Densa": "densa"}

EXEMPLOS = (
    "Quais são os principais problemas relatados pelos clientes em suas avaliações?",
    "O que os clientes relatam sobre problemas relacionados à entrega?",
    "Quais aspectos da experiência de compra são mais elogiados pelos clientes?",
    "O que os clientes dizem sobre a qualidade dos produtos de cama, mesa e banho?",
)

STATUS_ROTULO = {
    "ok": ("Resposta fundamentada", "success"),
    "insuficiente": ("Evidências insuficientes", "warning"),
    "fora_de_escopo": ("Fora de escopo", "info"),
}
MARCADORES_SOBRECARGA = ("503", "504", "unavailable", "overloaded", "deadline", "timeout")


# --------------------------------------------------------------------------- lógica pura

def nome_legivel(categoria: str) -> str:
    """'cama_mesa_banho' -> 'cama mesa banho'."""
    return categoria.replace("_", " ")


def gerar_meses(ini: tuple[int, int] = MES_INICIAL, fim: tuple[int, int] = MES_FINAL) -> list[str]:
    """Meses 'AAAA-MM' do intervalo, em ordem."""
    meses, (ano, mes) = [], ini
    while (ano, mes) <= fim:
        meses.append(f"{ano:04d}-{mes:02d}")
        ano, mes = (ano + 1, 1) if mes == 12 else (ano, mes + 1)
    return meses


def montar_filtros_ui(nota: tuple[int, int] = (NOTA_MIN_PADRAO, NOTA_MAX_PADRAO),
                      categoria: str = TODAS, mes_ini: str = SEM_FILTRO,
                      mes_fim: str = SEM_FILTRO, apenas_atrasados: bool = False) -> dict:
    """Dicionário só com os filtros que o usuário alterou do padrão."""
    filtros: dict = {}
    if nota[0] > NOTA_MIN_PADRAO:
        filtros["nota_min"] = int(nota[0])
    if nota[1] < NOTA_MAX_PADRAO:
        filtros["nota_max"] = int(nota[1])
    if categoria and categoria != TODAS:
        filtros["categoria"] = categoria
    if mes_ini and mes_ini != SEM_FILTRO:
        filtros["ano_mes_ini"] = mes_ini
    if mes_fim and mes_fim != SEM_FILTRO:
        filtros["ano_mes_fim"] = mes_fim
    if apenas_atrasados:
        filtros["apenas_atrasados"] = True
    return filtros


def periodo_invalido(filtros: dict) -> bool:
    """True se o mês inicial vier depois do final (formato AAAA-MM compara como texto)."""
    ini, fim = filtros.get("ano_mes_ini"), filtros.get("ano_mes_fim")
    return bool(ini and fim and ini > fim)


ROTULOS_FILTRO = {
    "nota_min": "nota mínima", "nota_max": "nota máxima", "categoria": "categoria",
    "ano_mes_ini": "de", "ano_mes_fim": "até", "apenas_atrasados": "só com atraso",
}


def formatar_filtros(filtros: dict) -> list[str]:
    """Linhas 'rótulo: valor' dos filtros aplicados, sem valores vazios/falsos."""
    linhas = []
    for chave, valor in filtros.items():
        if valor is None or valor is False:
            continue
        if chave == "categoria":
            valor = nome_legivel(str(valor))
        elif valor is True:
            valor = "sim"
        linhas.append(f"{ROTULOS_FILTRO.get(chave, chave)}: {valor}")
    return linhas


def titulo_evidencia(ev) -> str:
    """Título do expander: '[n] nota X/5 · categoria · data'."""
    return f"[{ev.n}] nota {ev.nota}/5 · {nome_legivel(ev.categoria)} · {ev.data}"


def texto_atraso(atraso: bool | None) -> str:
    """Rótulo do indicador de atraso na entrega."""
    return "desconhecido" if atraso is None else ("sim" if atraso else "não")


def mensagem_erro(exc: Exception) -> str:
    """Mensagem amigável para falhas do modelo ou do pipeline."""
    if any(m in str(exc).lower() for m in MARCADORES_SOBRECARGA):
        return "O serviço do modelo está sobrecarregado no momento. Tente de novo em instantes."
    return "Não foi possível gerar a resposta agora. Tente de novo; se persistir, reformule a pergunta."


def filtros_para_exibir(filtros: dict) -> tuple[dict, int | None]:
    """Separa os filtros aplicados do tamanho da amostra panorâmica."""
    visiveis = dict(filtros)
    return visiveis, visiveis.pop("panorama_amostra", None)


# --------------------------------------------------------------------------- recursos

@st.cache_resource(show_spinner=False)
def aquecer() -> tuple[str, ...]:
    """Carrega modelo de embeddings, índices e categorias uma única vez por processo."""
    from src.rag import analyze, retriever

    retriever._embeddings()
    retriever._colecao()
    retriever._bm25()
    retriever._meta_df()
    return analyze.listar_categorias()


def consultar(pergunta: str, filtros: dict, modo: str):
    """Chama o pipeline e devolve (resultado, latência em segundos)."""
    from src.rag import graph

    t0 = time.perf_counter()
    resultado = graph.answer(pergunta, filtros or None, modo)
    return resultado, time.perf_counter() - t0


# --------------------------------------------------------------------------- UI

def sidebar(categorias: tuple[str, ...]) -> tuple[dict, str]:
    """Desenha os filtros e devolve (filtros alterados, modo de busca)."""
    st.sidebar.header("Filtros (opcionais)")
    nota = st.sidebar.slider("Faixa de nota", 1, 5, (NOTA_MIN_PADRAO, NOTA_MAX_PADRAO))
    categoria = st.sidebar.selectbox("Categoria", [TODAS, *categorias],
                                     format_func=lambda c: c if c == TODAS else nome_legivel(c))
    meses = [SEM_FILTRO, *gerar_meses()]
    mes_ini = st.sidebar.selectbox("Período: de", meses)
    mes_fim = st.sidebar.selectbox("Período: até", meses)
    atrasados = st.sidebar.checkbox("Só pedidos entregues com atraso")
    with st.sidebar.expander("Avançado"):
        rotulo = st.radio("Modo de busca", list(MODOS))
    filtros = montar_filtros_ui(nota, categoria, mes_ini, mes_fim, atrasados)
    if periodo_invalido(filtros):
        st.sidebar.warning("O mês inicial é posterior ao final. Ajuste o período.")
    return filtros, MODOS[rotulo]


def mostrar_evidencia(ev) -> None:
    """Expander com o texto e os metadados de uma avaliação."""
    with st.expander(titulo_evidencia(ev)):
        st.write(ev.texto)
        st.caption(f"review_id: {ev.review_id} · score: {ev.score:.3f} · "
                   f"entrega com atraso: {texto_atraso(ev.atraso)}")


def mostrar_rastreabilidade(res, latencia: float | None) -> None:
    """Expander com consulta transformada, filtros, rota e amostra."""
    filtros, amostra = filtros_para_exibir(res.filtros)
    with st.expander("Rastreabilidade"):
        st.markdown(f"**Consulta transformada:** {res.consulta_transformada or '(sem busca)'}")
        linhas = formatar_filtros(filtros)
        st.markdown("**Filtros aplicados:** " + ("" if linhas else "nenhum"))
        for linha in linhas:
            st.markdown(f"- {linha}")
        rota = "panorâmica (amostra diversificada)" if amostra is not None else "específica"
        st.markdown(f"**Rota:** {rota} · **Reescritas:** {res.tentativas}")
        st.markdown(f"**Evidências recuperadas:** {len(res.evidencias)}")
        if amostra is not None:
            st.warning(f"As contagens referem-se a uma amostra de {amostra} avaliações "
                       "recuperadas, não à base inteira.")
        if res.contagens:
            st.json(res.contagens)
        if latencia is not None:
            st.markdown(f"**Latência:** {latencia:.1f} s")


def mostrar_resultado(res, latencia: float | None) -> None:
    """Badge de status, resposta, evidências e rastreabilidade."""
    rotulo, tipo = STATUS_ROTULO.get(res.status, (res.status, "info"))
    getattr(st, tipo)(rotulo)
    if res.status == "insuficiente":
        st.markdown("Não há evidências suficientes na base para responder com segurança.")
    if res.resposta:
        st.markdown(res.resposta)
    citadas = [e for e in res.evidencias if e.n in res.citacoes]
    if res.status == "ok" and citadas:
        st.subheader("Evidências citadas")
        for ev in citadas:
            mostrar_evidencia(ev)
    elif res.status == "insuficiente" and res.evidencias:
        st.subheader("Avaliações mais próximas encontradas")
        for ev in res.evidencias:
            mostrar_evidencia(ev)
    mostrar_rastreabilidade(res, latencia)


def definir_exemplo(texto: str) -> None:
    """Callback dos exemplos: preenche o campo e agenda a execução."""
    st.session_state["pergunta"] = texto
    st.session_state["executar"] = True


def agendar_execucao() -> None:
    """Callback do botão Perguntar."""
    st.session_state["executar"] = True


def main() -> None:
    st.set_page_config(page_title="Voice of Customer — Olist", layout="wide")
    st.title("Voice of Customer — Olist")
    st.caption("Pergunte sobre as avaliações de clientes do e-commerce Olist (2016–2018, "
               "cerca de 42 mil avaliações com comentário em português). "
               "As respostas citam as avaliações reais que as sustentam.")

    try:
        with st.spinner("Carregando modelo e índices (primeira vez, ~20 s)..."):
            categorias = aquecer()
    except Exception as exc:  # noqa: BLE001 - qualquer falha de carga vira aviso na tela
        st.error(f"Não foi possível carregar os índices: {exc}")
        st.stop()

    filtros, modo = sidebar(categorias)

    st.text_input("Sua pergunta", key="pergunta",
                  placeholder="Ex.: o que reclamam sobre o prazo de entrega?")
    st.button("Perguntar", type="primary", on_click=agendar_execucao)
    st.caption("Exemplos:")
    for i, ex in enumerate(EXEMPLOS):
        st.button(ex, key=f"ex{i}", on_click=definir_exemplo, args=(ex,))

    if st.session_state.pop("executar", False):
        pergunta = st.session_state.get("pergunta", "").strip()
        if not pergunta:
            st.warning("Digite uma pergunta ou escolha um exemplo.")
        elif periodo_invalido(filtros):
            st.error("Período inválido: o mês inicial é posterior ao final. Corrija os filtros e tente de novo.")
        else:
            try:
                with st.spinner("Buscando avaliações e gerando a resposta..."):
                    st.session_state["resultado"] = consultar(pergunta, filtros, modo)
            except Exception as exc:  # noqa: BLE001
                st.session_state.pop("resultado", None)
                st.error(mensagem_erro(exc))

    if "resultado" in st.session_state:
        res, latencia = st.session_state["resultado"]
        st.divider()
        mostrar_resultado(res, latencia)


if __name__ == "__main__":
    main()
