"""Recuperação híbrida (busca densa + BM25 + Reciprocal Rank Fusion).

Densa (Chroma, prefixo "query: ") e BM25 com os mesmos filtros -> RRF (k=60) -> top-k.
RRF implementado manualmente: dispensa normalizar scores de escalas diferentes.
O score de cada evidência é a similaridade cosseno consulta x documento (vetores
normalizados), calculada igual para docs vindos da densa ou só do BM25.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from src import config
from src.data.text import limpar_bm25

log = logging.getLogger(__name__)
_RE_ANO_MES = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
MODOS = ("densa", "hibrida")
COLUNAS_FILTRO = ["review_id", "review_score", "categoria", "entregue_com_atraso", "ano_mes"]


# --------------------------------------------------------------------------- lógica pura

def rrf(rankings: list[list[str]], k: int = config.RRF_K) -> list[tuple[str, float]]:
    """Funde rankings de IDs: score(d) = soma de 1 / (k + posição de d em cada ranking)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for pos, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + pos)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def normalizar_filtros(filtros: dict | None) -> dict:
    """Descarta (com warning) limites de mês fora do formato AAAA-MM."""
    f = dict(filtros or {})
    for chave in ("ano_mes_ini", "ano_mes_fim"):
        valor = f.get(chave)
        if valor and not (isinstance(valor, str) and _RE_ANO_MES.match(valor)):
            log.warning("Filtro %s ignorado: '%s' não está no formato AAAA-MM", chave, valor)
            f.pop(chave)
    return f


def meses_intervalo(ini: str | None, fim: str | None) -> list[str]:
    """Lista os meses "AAAA-MM" de ini a fim (inclusive); ponta ausente usa o limite da base.

    Vazia se ini > fim ou se o intervalo cai fora do período da base.
    """
    ini = ini or config.ANO_MES_LIMITE_INF
    fim = fim or config.ANO_MES_LIMITE_SUP
    a, m = map(int, ini.split("-"))
    fa, fm = map(int, fim.split("-"))
    meses = []
    while (a, m) <= (fa, fm) and len(meses) < 1200:
        meses.append(f"{a:04d}-{m:02d}")
        a, m = (a + 1, 1) if m == 12 else (a, m + 1)
    return meses


def intervalo_vazio(filtros: dict | None) -> bool:
    """True se há filtro de mês válido cujo intervalo não contém nenhum mês."""
    f = normalizar_filtros(filtros)
    if not (f.get("ano_mes_ini") or f.get("ano_mes_fim")):
        return False
    return not meses_intervalo(f.get("ano_mes_ini"), f.get("ano_mes_fim"))


def filtros_para_where(filtros: dict | None) -> dict | None:
    """Converte os filtros no `where` do Chroma; sem filtros devolve None."""
    f = normalizar_filtros(filtros)
    conds: list[dict] = []
    if f.get("nota_min") is not None:
        conds.append({"review_score": {"$gte": int(f["nota_min"])}})
    if f.get("nota_max") is not None:
        conds.append({"review_score": {"$lte": int(f["nota_max"])}})
    if f.get("categoria"):
        conds.append({"categoria": {"$eq": f["categoria"]}})
    if f.get("apenas_atrasados"):
        conds.append({"entregue_com_atraso": {"$eq": "sim"}})
    if f.get("ano_mes_ini") or f.get("ano_mes_fim"):
        conds.append({"ano_mes": {"$in": meses_intervalo(f.get("ano_mes_ini"), f.get("ano_mes_fim"))}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def filtros_para_mascara(df: pd.DataFrame, filtros: dict | None) -> np.ndarray:
    """Máscara booleana (mesma semântica do `where`) sobre as linhas de `df`."""
    f = normalizar_filtros(filtros)
    m = np.ones(len(df), dtype=bool)
    if f.get("nota_min") is not None:
        m &= (df["review_score"] >= int(f["nota_min"])).to_numpy()
    if f.get("nota_max") is not None:
        m &= (df["review_score"] <= int(f["nota_max"])).to_numpy()
    if f.get("categoria"):
        m &= (df["categoria"] == f["categoria"]).to_numpy()
    if f.get("apenas_atrasados"):
        m &= (df["entregue_com_atraso"] == "sim").to_numpy()
    if f.get("ano_mes_ini") or f.get("ano_mes_fim"):
        meses = meses_intervalo(f.get("ano_mes_ini"), f.get("ano_mes_fim"))
        m &= df["ano_mes"].isin(meses).to_numpy()
    return m


def montar_evidencia(n: int, review_id: str, texto: str, meta: dict, score: float) -> dict:
    """Evidência no formato do schema; atraso: sim -> True, nao -> False, outro -> None."""
    atraso = {"sim": True, "nao": False}.get(meta.get("entregue_com_atraso"))
    return {
        "n": n,
        "review_id": review_id,
        "score": float(min(1.0, max(0.0, score))),
        "nota": int(meta["review_score"]),
        "data": str(meta.get("data_avaliacao", "")),
        "categoria": str(meta.get("categoria", config.DESCONHECIDO)),
        "atraso": atraso,
        "texto": texto,
    }


def mmr(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int, lambda_: float) -> list[int]:
    """Índices selecionados por Maximal Marginal Relevance (vetores normalizados)."""
    rel = doc_vecs @ query_vec
    restantes = list(range(len(doc_vecs)))
    escolhidos: list[int] = []
    while restantes and len(escolhidos) < k:
        if escolhidos:
            redund = (doc_vecs[restantes] @ doc_vecs[escolhidos].T).max(axis=1)
        else:
            redund = np.zeros(len(restantes))
        valores = lambda_ * rel[restantes] - (1 - lambda_) * redund
        melhor = restantes[int(np.argmax(valores))]
        escolhidos.append(melhor)
        restantes.remove(melhor)
    return escolhidos


# --------------------------------------------------------------------------- recursos (I/O)

@lru_cache(maxsize=1)
def _embeddings():
    from src.index.embeddings import E5Embeddings
    return E5Embeddings()


@lru_cache(maxsize=1)
def _colecao():
    """Coleção Chroma aberta uma vez, pelo caminho ASCII seguro."""
    import chromadb
    cliente = chromadb.PersistentClient(path=config.chroma_path())
    return cliente.get_collection(config.CHROMA_COLLECTION)


@lru_cache(maxsize=1)
def _bm25() -> tuple:
    with open(config.BM25_PATH, "rb") as f:
        dados = pickle.load(f)
    return dados["bm25"], dados["ids"]


@lru_cache(maxsize=1)
def _meta_df() -> pd.DataFrame:
    """Colunas de filtro alinhadas à ordem dos ids do BM25."""
    df = pd.read_parquet(config.REVIEWS_PARQUET, columns=COLUNAS_FILTRO)
    _, ids = _bm25()
    return df.set_index("review_id").loc[ids].reset_index()


def _vetor_consulta(consulta: str) -> np.ndarray:
    return np.asarray(_embeddings().embed_query(consulta), dtype=np.float32)


# --------------------------------------------------------------------------- buscas

def buscar_densa(consulta: str, filtros: dict | None = None,
                 k: int = config.TOP_K_DENSE) -> list[str]:
    """IDs dos k documentos mais próximos da consulta (com `where`), do melhor ao pior."""
    res = _colecao().query(query_embeddings=[_vetor_consulta(consulta).tolist()],
                           n_results=k, where=filtros_para_where(filtros), include=[])
    return list(res["ids"][0])


def buscar_bm25(consulta: str, filtros: dict | None = None,
                k: int = config.TOP_K_BM25) -> list[str]:
    """IDs dos k melhores pelo BM25 dentro dos filtros; docs sem pontuação ficam fora."""
    tokens = limpar_bm25(consulta)
    if not tokens:
        return []
    bm25, ids = _bm25()
    scores = bm25.get_scores(tokens)
    scores = np.where(filtros_para_mascara(_meta_df(), filtros), scores, 0.0)
    top = np.argsort(-scores)[:k]
    return [ids[i] for i in top if scores[i] > 0]


def _evidencias(consulta_vec: np.ndarray, ids: list[str]) -> list[dict]:
    """Busca docs/metadados/vetores no Chroma e monta evidências com score cosseno, na ordem de `ids`."""
    if not ids:
        return []
    r = _colecao().get(ids=ids, include=["documents", "metadatas", "embeddings"])
    por_id = {i: (d, m, np.asarray(e, dtype=np.float32))
              for i, d, m, e in zip(r["ids"], r["documents"], r["metadatas"], r["embeddings"])}
    saida = []
    for n, i in enumerate((x for x in ids if x in por_id), start=1):
        doc, meta, vec = por_id[i]
        saida.append(montar_evidencia(n, i, doc, meta, float(vec @ consulta_vec)))
    return saida


# --------------------------------------------------------------------------- nós

def _resolver_modo(modo: str | None) -> str:
    """Modo pedido, ou config.RETRIEVAL_MODE se ausente; modo desconhecido gera warning e usa o padrão."""
    if not modo:
        return config.RETRIEVAL_MODE
    if modo not in MODOS:
        log.warning("Modo '%s' desconhecido; usando '%s'", modo, config.RETRIEVAL_MODE)
        return config.RETRIEVAL_MODE
    return modo


def recuperar(state: dict) -> dict:
    """Nó: densa (+ BM25 se modo híbrido) -> RRF -> top TOP_K_CONTEXT com score cosseno."""
    consulta, filtros = state["consulta"], state.get("filtros") or {}
    if intervalo_vazio(filtros):
        return {"evidencias": []}
    modo = _resolver_modo(state.get("modo"))
    rankings = [buscar_densa(consulta, filtros)]
    if modo == "hibrida":
        rankings.append(buscar_bm25(consulta, filtros))
    ids = [i for i, _ in rrf(rankings)[:config.TOP_K_CONTEXT]]
    return {"evidencias": _evidencias(_vetor_consulta(consulta), ids)}


def recuperar_panorama(state: dict) -> dict:
    """Nó panorâmico (sem tópicos): MMR sobre os candidatos densos para diversificar."""
    consulta, filtros = state["consulta"], state.get("filtros") or {}
    if intervalo_vazio(filtros):
        return {"evidencias": [], "contagens": None, "panorama_amostra": 0}
    q = _vetor_consulta(consulta)
    res = _colecao().query(query_embeddings=[q.tolist()], n_results=config.PANORAMA_FETCH_K,
                           where=filtros_para_where(filtros), include=["embeddings"])
    ids = list(res["ids"][0])
    if not ids:
        return {"evidencias": [], "contagens": None, "panorama_amostra": 0}
    vecs = np.asarray(res["embeddings"][0], dtype=np.float32)
    escolhidos = mmr(q, vecs, config.PANORAMA_K, config.MMR_LAMBDA)
    evid = _evidencias(q, [ids[i] for i in escolhidos])
    return {"evidencias": evid, "contagens": None, "panorama_amostra": len(evid)}


# --------------------------------------------------------------------------- CLI

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Sanidade da recuperação")
    ap.add_argument("consulta")
    ap.add_argument("--nota-min", type=int)
    ap.add_argument("--nota-max", type=int)
    ap.add_argument("--mes-ini", help="AAAA-MM")
    ap.add_argument("--mes-fim", help="AAAA-MM")
    ap.add_argument("--atraso", action="store_true")
    ap.add_argument("--categoria")
    ap.add_argument("--modo", choices=["densa", "hibrida"], default=config.RETRIEVAL_MODE)
    ap.add_argument("--panorama", action="store_true")
    a = ap.parse_args()
    filtros = {"nota_min": a.nota_min, "nota_max": a.nota_max,
               "apenas_atrasados": a.atraso or None, "categoria": a.categoria,
               "ano_mes_ini": a.mes_ini, "ano_mes_fim": a.mes_fim}
    filtros = {k: v for k, v in filtros.items() if v is not None}
    state = {"consulta": a.consulta, "filtros": filtros, "modo": a.modo}
    no = recuperar_panorama if a.panorama else recuperar
    t0 = time.time()
    out = no(state)
    t1 = time.time() - t0
    t0 = time.time()
    no(state)  # segunda chamada: latência com recursos já carregados
    print(f"[1a {t1:.2f}s | 2a {time.time() - t0:.2f}s] filtros={filtros} modo={a.modo}")
    for e in out["evidencias"]:
        print(f"{e['n']:>2} {e['score']:.3f} nota={e['nota']} {e['categoria']} "
              f"atraso={e['atraso']} | {e['texto'][:80]}")
    if a.panorama:
        print(f"amostra={out['panorama_amostra']}")


if __name__ == "__main__":
    main()
