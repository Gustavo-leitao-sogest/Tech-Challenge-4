"""Roda o eval: acurácia de status, precision@k e faithfulness por LLM-juiz, latência.

Salva os resultados em eval/results/ (ignorado no git) e imprime uma tabela para o README.
Uso:
    python -m eval.run_eval --config hibrida [--sem-juiz] [--ids pdf1,meta2]
    python -m eval.run_eval --comparar-filtros
    python -m eval.run_eval --calibrar
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from eval import prompts_juiz as pj
from src import config
from src.rag import guardrails

RAIZ = Path(__file__).resolve().parents[1]
QUESTIONS = RAIZ / "eval" / "questions.yaml"
RESULTS = RAIZ / "eval" / "results"
FIGURA = RAIZ / "images" / "calibracao_limiar.png"
IDS_FILTROS = ("meta1", "meta2", "meta3")
IDS_CALIB_OK = ("pdf2", "pdf5")  # só rota específica; panorâmicas são puladas na medição
IDS_CALIB_SEM = ("sem1", "sem2", "sem3")
MAX_CHARS_PRECISAO = 400    # trecho de cada evidência enviado ao juiz de precision@k
MAX_CHARS_FIDELIDADE = 600  # idem, para o juiz de faithfulness
RETRY_TENTATIVAS = 4
RETRY_ESPERA_S = 5.0
ERROS_TRANSITORIOS = ("503", "504", "429", "unavailable", "deadline", "timeout", "overloaded")


# --------------------------------------------------------------------------- funções puras

def carregar_perguntas(caminho: Path = QUESTIONS, secao: str = "perguntas",
                       ids: list[str] | None = None) -> list[dict]:
    """Lê uma seção do yaml; `ids` filtra e mantém a ordem pedida."""
    dados = yaml.safe_load(Path(caminho).read_text(encoding="utf-8")) or {}
    itens = dados.get(secao) or []
    if ids:
        por_id = {p["id"]: p for p in itens}
        faltam = [i for i in ids if i not in por_id]
        if faltam:
            raise KeyError(f"ids inexistentes em '{secao}': {faltam}")
        return [por_id[i] for i in ids]
    return itens


def acuracia_status(linhas: list[dict]) -> float:
    """Fração de linhas com status obtido igual ao esperado."""
    if not linhas:
        return 0.0
    return sum(1 for r in linhas if r["status"] == r["esperado"]) / len(linhas)


def checar_citacoes(resposta: str, citacoes: list[int], n_evidencias: int) -> dict:
    """Checagem determinística: todas as citações existem? há ao menos uma?"""
    usadas = set(citacoes) | set(guardrails.numeros_citados(resposta))
    return {
        "n_citacoes": len(usadas),
        "citacoes_validas": not guardrails.citacoes_invalidas(resposta, citacoes, n_evidencias),
        "tem_citacao": bool(usadas),
    }


def media(valores: list[float | None]) -> float | None:
    """Média ignorando None; None se não sobrar valor."""
    v = [x for x in valores if x is not None]
    return sum(v) / len(v) if v else None


def agregar(linhas: list[dict]) -> dict:
    """Métricas agregadas de uma execução (uma linha por pergunta)."""
    ok = [r for r in linhas if r["status"] == "ok"]
    lat = [r["latencia_s"] for r in linhas
           if r.get("latencia_s") is not None and not r.get("erro")]
    return {
        "n": len(linhas),
        "acuracia_status": acuracia_status(linhas),
        "citacoes_validas": media([1.0 if r["citacoes_validas"] else 0.0 for r in ok]),
        "ok_com_citacao": media([1.0 if r["tem_citacao"] else 0.0 for r in ok]),
        "precision_k": media([r.get("precision_k") for r in linhas if r["esperado"] == "ok"]),
        "precision_proximas": media([r.get("precision_proximas") for r in linhas
                                     if r["esperado"] == "insuficiente"]),
        "faithfulness": media([r.get("faithfulness") for r in linhas]),
        "latencia_n": len(lat),
        "latencia_media_s": statistics.fmean(lat) if lat else None,
        "latencia_mediana_s": statistics.median(lat) if lat else None,
        "erros": sum(1 for r in linhas if r.get("erro")),
    }


def fracao(marcas: list[bool]) -> float | None:
    """Fração de True; None se vazia."""
    return sum(marcas) / len(marcas) if marcas else None


def recomendar_limiar(positivos: list[float], negativos: list[float],
                      atual: float | None = None) -> dict:
    """Limiar que melhor separa positivos (devem passar) de negativos (devem ficar abaixo).

    Candidatos: pontos médios entre valores consecutivos (passa = score >= limiar). Maximiza a
    acurácia balanceada; no empate prefere `atual` se cair no intervalo empatado, senão o centro
    do intervalo (do primeiro ao último candidato empatado).
    """
    if not positivos or not negativos:
        return {"limiar": None, "separa": False, "acuracia_balanceada": None}
    valores = sorted(set(positivos) | set(negativos))
    cands = [(a + b) / 2 for a, b in zip(valores, valores[1:])] or [valores[0]]
    pontuados = []
    for c in cands:
        tpr = sum(p >= c for p in positivos) / len(positivos)
        tnr = sum(n < c for n in negativos) / len(negativos)
        pontuados.append((c, (tpr + tnr) / 2))
    melhor = max(b for _, b in pontuados)
    empatados = [c for c, b in pontuados if abs(b - melhor) < 1e-9]
    lo, hi = empatados[0], empatados[-1]
    escolhido = atual if atual is not None and lo <= atual <= hi else (lo + hi) / 2
    return {"limiar": round(escolhido, 3), "separa": min(positivos) > max(negativos),
            "acuracia_balanceada": melhor, "margem": round(min(positivos) - max(negativos), 3),
            "intervalo_empate": (round(lo, 3), round(hi, 3))}


def fmt(valor: float | None, casas: int = 2, pct: bool = False) -> str:
    """Formata número (ou '-' se None); pct multiplica por 100."""
    if valor is None:
        return "-"
    return f"{valor * 100:.0f}%" if pct else f"{valor:.{casas}f}"


def tabela_resumo(agg: dict, config_nome: str) -> str:
    """Tabela markdown com as métricas agregadas."""
    return "\n".join([
        f"| Métrica ({config_nome}) | Valor |",
        "|---|---|",
        f"| Acurácia de status | {fmt(agg['acuracia_status'], pct=True)} |",
        f"| Citações válidas (respostas ok) | {fmt(agg['citacoes_validas'], pct=True)} |",
        f"| Respostas ok com >= 1 citação | {fmt(agg['ok_com_citacao'], pct=True)} |",
        f"| Precision@k (juiz; perguntas com resposta esperada, recusa indevida pesa 0; "
        f"k=25 panorâmica, k=15 específica) | {fmt(agg['precision_k'])} |",
        f"| Precision das mais próximas, sem evidência (informativa) | "
        f"{fmt(agg.get('precision_proximas'))} |",
        f"| Faithfulness (juiz) | {fmt(agg['faithfulness'])} |",
        f"| Latência média / mediana (s) | {fmt(agg['latencia_media_s'], 1)} / "
        f"{fmt(agg['latencia_mediana_s'], 1)} (n sem erro = {agg['latencia_n']}) |",
        f"| Erros | {agg['erros']} |",
    ])


def neutralizar(texto: str) -> str:
    """Achata espaços e troca < e > para o texto não fechar as tags do prompt do juiz."""
    return " ".join(str(texto).split()).replace("<", "‹").replace(">", "›")


def formatar_evidencias(evidencias: list[dict], max_chars: int = MAX_CHARS_PRECISAO) -> str:
    """Evidências numeradas e neutralizadas para o prompt do juiz."""
    return "\n".join(f"[{e['n']}] (nota {e['nota']}) {neutralizar(e['texto'][:max_chars])}"
                     for e in evidencias)


# --------------------------------------------------------------------------- juiz (LLM)

def com_retry(fn, tentativas: int = RETRY_TENTATIVAS, espera: float = RETRY_ESPERA_S):
    """Chama fn(); repete em erros transitórios (503/504/429/timeout), senão propaga."""
    for i in range(tentativas):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - classifica pela mensagem do provedor
            transitorio = any(t in str(e).lower() for t in ERROS_TRANSITORIOS)
            if not transitorio or i == tentativas - 1:
                raise
            time.sleep(espera * (i + 1))


def _juiz(schema):
    from src.rag.llm import get_llm
    return get_llm(temperature=0).with_structured_output(schema)


def julgar_precisao(pergunta: str, evidencias: list[dict]) -> float | None:
    """Precision@k: fração das evidências recuperadas que o juiz marca como relevantes."""
    if not evidencias:
        return None
    prompt = pj.PROMPT_RELEVANCIA_V2.format(
        pergunta=pergunta, evidencias=formatar_evidencias(evidencias))
    res = com_retry(lambda: _juiz(pj.JulgamentoRelevancia).invoke(prompt))
    if res is None:
        return None
    por_n = {j.n: j.relevante for j in res.julgamentos}
    return fracao([por_n.get(e["n"], False) for e in evidencias])


def julgar_fidelidade(resposta: str, evidencias: list[dict], citacoes: list[int]) -> float | None:
    """Faithfulness: fração das afirmações sustentadas pelas evidências citadas (None sem citação)."""
    citadas = [e for e in evidencias if e["n"] in citacoes]
    if not citadas:
        return None
    prompt = pj.PROMPT_FIDELIDADE_V2.format(
        resposta=neutralizar(resposta),
        evidencias=formatar_evidencias(citadas, MAX_CHARS_FIDELIDADE))
    res = com_retry(lambda: _juiz(pj.JulgamentoFidelidade).invoke(prompt))
    if res is None:
        return None
    return fracao([a.sustentada for a in res.afirmacoes])


# --------------------------------------------------------------------------- execução

def precisao_da_linha(esperado: str, status: str, pergunta: str,
                      evidencias: list[dict]) -> float | None:
    """Precision@k da linha. Esperado ok e status diferente de ok (recusa indevida): 0.0, sem juiz.

    Nas demais, julga as evidências (None se não houver nenhuma).
    """
    if esperado == "ok" and status != "ok":
        return 0.0
    if not evidencias:
        return None
    return julgar_precisao(pergunta, evidencias)


def rodar_pergunta(p: dict, modo: str, juiz: bool) -> dict:
    """Executa o grafo para uma pergunta e mede tudo; erros viram o campo `erro`."""
    from src.rag import graph
    linha = {"id": p["id"], "esperado": p["esperado"], "status": "erro", "rota": "",
             "tentativas": 0, "n_evidencias": 0, "n_citacoes": 0, "citacoes_validas": False,
             "tem_citacao": False, "top1": None, "top3_media": None, "precision_k": None,
             "precision_proximas": None, "faithfulness": None, "latencia_s": None,
             "tentativas_eval": 0, "erro": ""}
    marca = {"t0": 0.0, "n": 0}

    def executar():
        marca["n"] += 1
        marca["t0"] = time.perf_counter()  # latência = só a última tentativa
        return graph.get_graph().invoke({"pergunta": p["pergunta"], "modo": modo})

    try:
        final = com_retry(executar)
    except Exception as e:  # noqa: BLE001
        linha["erro"] = f"{type(e).__name__}: {str(e)[:150]}"
        linha["latencia_s"] = time.perf_counter() - marca["t0"]
        linha["tentativas_eval"] = marca["n"]
        return linha
    linha["latencia_s"] = time.perf_counter() - marca["t0"]
    linha["tentativas_eval"] = marca["n"]
    r = graph.montar_resultado(final)
    evid = [e.model_dump() for e in r.evidencias]
    scores = sorted((e["score"] for e in evid), reverse=True)
    tipo = (final.get("analise") or {}).get("tipo", "")
    linha.update(
        status=r.status, tentativas=r.tentativas, n_evidencias=len(evid),
        rota="fora" if r.status == "fora_de_escopo" else tipo,
        top1=scores[0] if scores else None,
        top3_media=media(scores[:3]) if scores else None,
        **checar_citacoes(r.resposta, r.citacoes, len(evid)),
    )
    if juiz:
        erros = []
        # precision@k: toda pergunta com resposta esperada, qualquer status obtido (recusa
        # indevida pesa 0); nas "insuficiente" vira a métrica informativa separada.
        if p["esperado"] in ("ok", "insuficiente"):
            try:
                chave = "precision_k" if p["esperado"] == "ok" else "precision_proximas"
                linha[chave] = precisao_da_linha(p["esperado"], r.status, p["pergunta"], evid)
            except Exception as e:  # noqa: BLE001
                erros.append(f"precision {type(e).__name__}: {str(e)[:100]}")
        if r.status == "ok":
            try:
                linha["faithfulness"] = julgar_fidelidade(r.resposta, evid, r.citacoes)
            except Exception as e:  # noqa: BLE001
                erros.append(f"faithfulness {type(e).__name__}: {str(e)[:100]}")
        if erros:
            linha["erro"] = "juiz " + "; ".join(erros)
    return linha


def salvar(linhas: list[dict], agg: dict, nome: str) -> tuple[Path, Path]:
    """Grava CSV (uma linha por pergunta) e MD (tabela-resumo) em eval/results/."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_p, md_p = RESULTS / f"{nome}_{ts}.csv", RESULTS / f"{nome}_{ts}.md"
    with csv_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    md_p.write_text(tabela_resumo(agg, nome) + "\n", encoding="utf-8")
    return csv_p, md_p


def cmd_config(nome: str, juiz: bool, ids: list[str] | None) -> None:
    """Roda todas as perguntas numa configuração e salva CSV + MD."""
    perguntas = carregar_perguntas(ids=ids)
    linhas = []
    for p in perguntas:
        linha = rodar_pergunta(p, nome, juiz)
        print(f"{p['id']:6} esperado={p['esperado']:15} obtido={linha['status']:15} "
              f"{linha['latencia_s']:.1f}s {linha['erro']}", flush=True)
        linhas.append(linha)
    agg = agregar(linhas)
    csv_p, md_p = salvar(linhas, agg, nome)
    print("\n" + tabela_resumo(agg, nome))
    print(f"\nsalvo: {csv_p}\n       {md_p}")


# --------------------------------------------------------------------------- comparar filtros

def cmd_comparar_filtros() -> None:
    """precision@k com e sem os filtros extraídos pela análise (meta1-3), recuperação híbrida."""
    from src.rag import analyze, retriever
    linhas = []
    for p in carregar_perguntas(ids=list(IDS_FILTROS)):
        st = com_retry(lambda: analyze.analisar_pergunta({"pergunta": p["pergunta"]}))
        filtros = st["filtros"]
        res = {}
        for rotulo, f in (("sem", {}), ("com", filtros)):
            evid = retriever.recuperar({"consulta": st["consulta"], "filtros": f,
                                        "modo": "hibrida"})["evidencias"]
            res[rotulo] = (julgar_precisao(p["pergunta"], evid), len(evid))
        linhas.append({"id": p["id"], "filtros": str(filtros),
                       "p_sem": res["sem"][0], "n_sem": res["sem"][1],
                       "p_com": res["com"][0], "n_com": res["com"][1]})
        print(linhas[-1], flush=True)
    md = ["Mede a recuperação específica (híbrida, k=15); no pipeline meta1–3 vão pela rota "
          "panorâmica (k=25), então isto isola o efeito do filtro, não o desempenho ponta a ponta.",
          "",
          "| Pergunta | Filtros extraídos | Precision@k sem filtro | Precision@k com filtro |",
          "|---|---|---|---|"]
    for r in linhas:
        md.append(f"| {r['id']} | `{r['filtros']}` | {fmt(r['p_sem'])} (k={r['n_sem']}) | "
                  f"{fmt(r['p_com'])} (k={r['n_com']}) |")
    md.append(f"| **média** | | {fmt(media([r['p_sem'] for r in linhas]))} | "
              f"{fmt(media([r['p_com'] for r in linhas]))} |")
    texto = "\n".join(md)
    RESULTS.mkdir(parents=True, exist_ok=True)
    arq = RESULTS / f"filtros_{datetime.now():%Y%m%d_%H%M%S}.md"
    arq.write_text(texto + "\n", encoding="utf-8")
    print("\n" + texto + f"\n\nsalvo: {arq}")


# --------------------------------------------------------------------------- calibração

def _scores(evid: list[dict]) -> tuple[float, float, float]:
    """(top1, média do top3, 3º melhor score); 0 se faltar evidência."""
    s = sorted((e["score"] for e in evid), reverse=True) + [0.0, 0.0, 0.0]
    return s[0], sum(s[:3]) / 3, s[2]


def score_efetivo(r: dict) -> float:
    """3º melhor score (sempre contra a consulta original) na melhor tentativa: inicial ou HyDE."""
    return max(r["s3_antes"], r["s3_pipeline"])


def medir_consulta(p: dict) -> dict:
    """Scores do 3º melhor documento (híbrido, sem geração) por pergunta.

    antes: busca e score com a consulta original. pipeline: busca com o texto HyDE, score contra a
    consulta original (como o pipeline atual). hyde_antigo: busca e score contra o texto HyDE
    (critério antigo, mantido só para documentar a decisão).
    """
    from src.rag import analyze, generator, retriever
    st = com_retry(lambda: analyze.analisar_pergunta({"pergunta": p["pergunta"]}))
    if (st.get("analise") or {}).get("tipo") == "panoramica":
        return {"id": p["id"], "pulada": "panoramica"}
    base = {"pergunta": p["pergunta"], "consulta": st["consulta"],
            "filtros": st["filtros"], "modo": "hibrida"}
    antes = _scores(retriever.recuperar(base)["evidencias"])
    hyde = com_retry(lambda: generator.reescrever_consulta(base))["consulta"]
    pipe = _scores(retriever.recuperar({**base, "consulta": hyde,
                                        "consulta_score": st["consulta"]})["evidencias"])
    antigo = _scores(retriever.recuperar({**base, "consulta": hyde})["evidencias"])
    return {"id": p["id"], "grupo": "com_evidencia" if p["esperado"] == "ok" else "sem_evidencia",
            "top1_antes": antes[0], "top3_antes": antes[1], "s3_antes": antes[2],
            "s3_pipeline": pipe[2], "s3_hyde_antigo": antigo[2]}


def figura_calibracao(linhas: list[dict], limiar: float, recomendado: float | None,
                      caminho: Path = FIGURA) -> None:
    """Strip + box plot do 3º melhor score (pipeline atual) por grupo, com os limiares."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grupos = [("com_evidencia", "Com evidência"), ("sem_evidencia", "Sem evidência")]
    cores = {"com_evidencia": "#1b7f5c", "sem_evidencia": "#b5452d"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    dados = [[score_efetivo(r) for r in linhas if r["grupo"] == g] for g, _ in grupos]
    ax.boxplot(dados, tick_labels=[f"{n} (n={len(d)})" for (_, n), d in zip(grupos, dados)],
               widths=0.4, showfliers=False)
    for i, (g, _) in enumerate(grupos, start=1):
        ys = dados[i - 1]
        xs = [i + (k - (len(ys) - 1) / 2) * 0.05 for k in range(len(ys))]
        ax.scatter(xs, ys, color=cores[g], zorder=3, s=36)
    ax.axhline(limiar, color="black", linestyle="--", label=f"limiar atual ({limiar:.3f})")
    if recomendado is not None and abs(recomendado - limiar) > 1e-9:
        ax.axhline(recomendado, color="#2f6db5", linestyle=":",
                   label=f"recomendado ({recomendado:.3f})")
    ax.set_ylabel("Cosseno do 3º melhor documento com a pergunta original")
    ax.set_title("Calibração do limiar de suficiência (pipeline atual)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def cmd_calibrar() -> None:
    """Mede scores nos dois grupos, gera tabela e figura e recomenda o limiar (não altera config)."""
    extras = carregar_perguntas(secao="calibracao")
    ok = carregar_perguntas(ids=list(IDS_CALIB_OK)) + [p for p in extras if p["esperado"] == "ok"]
    sem = carregar_perguntas(ids=list(IDS_CALIB_SEM)) + [p for p in extras if p["esperado"] != "ok"]
    linhas, puladas = [], []
    for p in ok + sem:
        r = medir_consulta(p)
        if r.get("pulada"):
            puladas.append(p["id"])
        else:
            linhas.append(r)
        print(r, flush=True)
    pos = [score_efetivo(r) for r in linhas if r["grupo"] == "com_evidencia"]
    neg = [score_efetivo(r) for r in linhas if r["grupo"] == "sem_evidencia"]
    rec = recomendar_limiar(pos, neg, config.MIN_SCORE)
    md = ["| Pergunta | Grupo | (a) 3º, consulta original | (b) 3º, busca HyDE pontuada vs original "
          "| efetivo (máx a,b) | (c) 3º vs texto HyDE (critério antigo) |",
          "|---|---|---|---|---|---|"]
    for r in linhas:
        md.append(f"| {r['id']} | {r['grupo']} | {r['s3_antes']:.3f} | {r['s3_pipeline']:.3f} | "
                  f"{score_efetivo(r):.3f} | {r['s3_hyde_antigo']:.3f} |")
    antigo_pos = [max(r["s3_antes"], r["s3_hyde_antigo"]) for r in linhas
                  if r["grupo"] == "com_evidencia"]
    antigo_neg = [max(r["s3_antes"], r["s3_hyde_antigo"]) for r in linhas
                  if r["grupo"] == "sem_evidencia"]
    md += ["",
           f"Estatística de decisão: 3º melhor score (MIN_EVIDENCES={config.MIN_EVIDENCES}) contra a "
           "consulta original, máximo entre a tentativa inicial e a com HyDE.",
           f"Com evidência: min={min(pos):.3f} max={max(pos):.3f}; "
           f"sem evidência: min={min(neg):.3f} max={max(neg):.3f}.",
           f"Critério antigo (c, máx com a): com evidência min={min(antigo_pos):.3f}; "
           f"sem evidência max={max(antigo_neg):.3f}; "
           + ("há sobreposição entre os grupos." if max(antigo_neg) >= min(antigo_pos)
              else "os grupos não se sobrepõem."),
           f"Perguntas puladas (rota panorâmica, não usa o limiar): {', '.join(puladas) or 'nenhuma'}.",
           f"Limiar atual: {config.MIN_SCORE}. Recomendação: {rec}"]
    texto = "\n".join(md)
    RESULTS.mkdir(parents=True, exist_ok=True)
    arq = RESULTS / f"calibracao_{datetime.now():%Y%m%d_%H%M%S}.md"
    arq.write_text(texto + "\n", encoding="utf-8")
    figura_calibracao(linhas, config.MIN_SCORE, rec["limiar"])
    print("\n" + texto + f"\n\nsalvo: {arq}\nfigura: {FIGURA}")


# --------------------------------------------------------------------------- CLI

def main() -> None:
    """Ponto de entrada da linha de comando."""
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Eval do RAG")
    ap.add_argument("--config", choices=["hibrida", "densa"])
    ap.add_argument("--sem-juiz", action="store_true", help="pula precision@k e faithfulness")
    ap.add_argument("--ids", help="ids separados por vírgula (ex.: pdf1,meta2)")
    ap.add_argument("--comparar-filtros", action="store_true")
    ap.add_argument("--calibrar", action="store_true")
    a = ap.parse_args()
    if a.calibrar:
        cmd_calibrar()
    elif a.comparar_filtros:
        cmd_comparar_filtros()
    elif a.config:
        cmd_config(a.config, not a.sem_juiz, a.ids.split(",") if a.ids else None)
    else:
        ap.error("informe --config, --comparar-filtros ou --calibrar")


if __name__ == "__main__":
    main()
