"""Constrói os índices: embeddings e5 (prefixo "passage: ") -> Chroma, e BM25 -> pickle.

Inserção no Chroma em lotes de config.CHROMA_INSERT_BATCH. Idempotente: a coleção é
apagada e recriada a cada execução.

Uso: python -m src.index.build_index [--sample N]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from tqdm import tqdm

from src import config
from src.index.embeddings import E5Embeddings

# Colunas que não viram metadado: texto vai como documento; texto_bm25 só serve ao BM25.
COLUNAS_FORA_METADADOS = {"texto", "texto_bm25"}


def montar_metadados(df: pd.DataFrame) -> list[dict]:
    """Metadados por linha, só str/int/float/bool (Chroma rejeita None e listas)."""
    cols = [c for c in df.columns if c not in COLUNAS_FORA_METADADOS]
    metas = []
    for linha in df[cols].to_dict("records"):
        meta = {}
        for k, v in linha.items():
            if hasattr(v, "item"):  # numpy -> tipo nativo
                v = v.item()
            if v is None or (isinstance(v, float) and v != v):
                v = config.DESCONHECIDO
            meta[k] = v
        metas.append(meta)
    return metas


def construir_bm25(textos_bm25: list[str], ids: list[str]) -> dict:
    """BM25Okapi sobre tokens separados por espaço.

    Doc sem tokens (ex.: só emoji) fica com lista vazia: o BM25Okapi aceita e o
    documento simplesmente nunca pontua. A ordem de `ids` acompanha o corpus.
    """
    from rank_bm25 import BM25Okapi
    corpus = [(t or "").split() for t in textos_bm25]
    return {"bm25": BM25Okapi(corpus), "ids": list(ids)}


def salvar_bm25(dados: dict, caminho: Path) -> None:
    """Grava o pickle {bm25, ids}, criando a pasta se preciso."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "wb") as f:
        pickle.dump(dados, f)


def construir_chroma(df: pd.DataFrame, embeddings: Embeddings, persist_dir: Path,
                     colecao: str, lote: int = config.CHROMA_INSERT_BATCH) -> Chroma:
    """Recria a coleção e insere os documentos em lotes."""
    import chromadb
    from chromadb.errors import NotFoundError

    cliente = chromadb.PersistentClient(path=config.chroma_path(persist_dir))
    try:
        cliente.delete_collection(colecao)
    except NotFoundError:  # coleção inexistente na primeira execução
        pass
    store = Chroma(
        client=cliente, collection_name=colecao, embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    textos = df["texto"].tolist()
    ids = df["review_id"].tolist()
    metas = montar_metadados(df)
    for ini in tqdm(range(0, len(df), lote), desc="Chroma (lotes)"):
        fim = ini + lote
        store.add_texts(textos[ini:fim], metadatas=metas[ini:fim], ids=ids[ini:fim])
    return store


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="usa só N docs (teste de tempo)")
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_parquet(config.REVIEWS_PARQUET)
    if args.sample is not None:
        df = df.sample(n=min(args.sample, len(df)), random_state=42).reset_index(drop=True)
    print(f"{len(df)} documentos")

    salvar_bm25(construir_bm25(df["texto_bm25"].tolist(), df["review_id"].tolist()),
                config.BM25_PATH)
    print(f"BM25 salvo em {config.BM25_PATH}")

    emb = E5Embeddings(show_progress=True)
    store = construir_chroma(df, emb, config.CHROMA_DIR, config.CHROMA_COLLECTION)
    print(f"Chroma: {store._collection.count()} docs em {config.CHROMA_DIR}")
    print(f"Tempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
