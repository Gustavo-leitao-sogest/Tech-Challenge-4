"""Gera data/processed/reviews.parquet a partir dos CSVs do Kaggle.

Etapas:
1. Carregar reviews, orders, order_items, products e a tradução de categorias.
2. Deduplicar review_id (cópias ligam a pedidos diferentes; guarda n_pedidos).
3. Texto = "{titulo}. {mensagem}" -> limpar_leve -> mascarar_pii.
4. Selecionar avaliações com texto útil (>= MIN_TEXT_ALNUM caracteres alfanuméricos).
5. Metadados: review_score, data, categoria, entregue_com_atraso, dias_entrega.
   Ausentes viram "desconhecido"/-1 (o Chroma não aceita None).
6. Gerar a coluna texto_bm25 com limpar_bm25.
7. Imprimir um resumo (linhas em cada etapa) para o README.

Uso: python -m src.data.prepare
"""
from __future__ import annotations

import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import config
from src.data.pii import mascarar_pii
from src.data.text import limpar_bm25, limpar_leve

DESC = config.DESCONHECIDO
PONTUACAO_FINAL = ".!?…:;"

COLUNAS = [
    "review_id", "order_id", "texto", "texto_bm25", "review_score",
    "data_avaliacao", "ano_mes", "ano", "categoria", "categoria_en",
    "n_categorias", "entregue_com_atraso", "dias_entrega", "order_status",
    "n_pedidos", "tem_titulo",
]


def carregar() -> dict[str, pd.DataFrame]:
    """Lê os CSVs necessários de data/raw."""
    raw = config.DATA_RAW

    def ler(nome: str, **kw) -> pd.DataFrame:
        # utf-8-sig: o CSV de tradução vem com BOM.
        return pd.read_csv(raw / nome, encoding="utf-8-sig", **kw)

    return {
        "reviews": ler("olist_order_reviews_dataset.csv"),
        "orders": ler(
            "olist_orders_dataset.csv",
            parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                         "order_estimated_delivery_date"],
        ),
        "items": ler("olist_order_items_dataset.csv", usecols=["order_id", "product_id"]),
        "products": ler("olist_products_dataset.csv",
                        usecols=["product_id", "product_category_name"]),
        "traducao": ler("product_category_name_translation.csv"),
    }


def deduplicar(reviews: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por review_id, mantendo o primeiro pedido; n_pedidos = nº de pedidos cobertos.

    A ordenação explícita (data, order_id) torna a escolha determinística.
    """
    ordenado = reviews.sort_values(
        ["review_id", "review_creation_date", "order_id"], kind="stable"
    )
    n = ordenado.groupby("review_id")["order_id"].nunique().rename("n_pedidos")
    unico = ordenado.drop_duplicates("review_id", keep="first")
    return unico.merge(n, left_on="review_id", right_index=True).reset_index(drop=True)


def montar_texto(titulo: object, mensagem: object) -> str:
    """Combina título e mensagem sem duplicar conteúdo nem pontuação."""
    t = "" if pd.isna(titulo) else str(titulo).strip()
    m = "" if pd.isna(mensagem) else str(mensagem).strip()
    if not t:
        return m
    if not m:
        return t
    if t.casefold() == m.casefold():
        return m
    sep = " " if t[-1] in PONTUACAO_FINAL else ". "
    return f"{t}{sep}{m}"


def categoria_principal(categorias: pd.Series) -> str:
    """Categoria conhecida mais frequente (empate: ordem alfabética)."""
    conhecidas = [c for c in categorias.dropna() if c and c != DESC]
    if not conhecidas:
        return DESC
    contagem = pd.Series(conhecidas).value_counts()
    return sorted(contagem[contagem == contagem.max()].index)[0]


def metadados_pedido(
    pedidos: pd.DataFrame, itens: pd.DataFrame, produtos: pd.DataFrame, traducao: pd.DataFrame
) -> pd.DataFrame:
    """Metadados por order_id: status, atraso, dias_entrega e categorias."""
    p = pedidos.drop_duplicates("order_id").set_index("order_id")
    entrega = p["order_delivered_customer_date"]
    estimada = p["order_estimated_delivery_date"]
    # Compara por dia: entrega no mesmo dia da estimativa não é atraso.
    atraso = (entrega.dt.normalize() > estimada.dt.normalize()).map({True: "sim", False: "nao"})
    atraso = atraso.astype(object).where(entrega.notna(), DESC)
    dias = (entrega.dt.normalize() - p["order_purchase_timestamp"].dt.normalize()).dt.days
    meta = pd.DataFrame({
        "order_status": p["order_status"].fillna(DESC).astype(str),
        "entregue_com_atraso": atraso.astype(str),
        "dias_entrega": dias.fillna(-1).astype(int),
    })

    cat = itens.merge(produtos, on="product_id", how="left")
    cat = cat[cat["order_id"].isin(meta.index)]
    # Vetorizado (contagem por pedido+categoria): groupby com função Python
    # por pedido levava minutos. Regra equivale a categoria_principal.
    conhecidas = cat.dropna(subset=["product_category_name"])
    conhecidas = conhecidas[conhecidas["product_category_name"] != DESC]
    cont = (conhecidas.groupby(["order_id", "product_category_name"]).size()
            .rename("n").reset_index())
    cont = cont.sort_values(["order_id", "n", "product_category_name"],
                            ascending=[True, False, True])
    por_pedido = pd.DataFrame({
        "categoria": cont.drop_duplicates("order_id").set_index("order_id")["product_category_name"],
        "n_categorias": cont.groupby("order_id").size(),
    })
    meta = meta.join(por_pedido)
    meta["categoria"] = meta["categoria"].fillna(DESC).astype(str)
    meta["n_categorias"] = meta["n_categorias"].fillna(0).astype(int)
    en = dict(zip(traducao["product_category_name"], traducao["product_category_name_english"]))
    # Sem tradução, repete o nome em português.
    meta["categoria_en"] = [en.get(c, c) for c in meta["categoria"]]
    return meta


def _alnum(texto: str) -> int:
    return sum(ch.isalnum() for ch in texto)


def selecionar(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém só textos com ao menos MIN_TEXT_ALNUM caracteres alfanuméricos."""
    return df[df["texto"].map(_alnum) >= config.MIN_TEXT_ALNUM].reset_index(drop=True)


def _validar(df: pd.DataFrame) -> None:
    assert df["review_id"].is_unique, "review_id duplicado"
    assert not df.isna().any().any(), "há valores nulos"
    for c in COLUNAS:
        tipos = {type(v) for v in df[c]}
        assert tipos <= {str, int, bool}, f"{c}: tipos {tipos}"


def main() -> None:
    t0 = time.time()
    d = carregar()
    funil = [("avaliações no CSV", len(d["reviews"]))]

    rev = deduplicar(d["reviews"])
    funil.append(("review_id únicos", len(rev)))

    rev["tem_titulo"] = rev["review_comment_title"].fillna("").str.strip().ne("")
    rev["texto"] = [
        mascarar_pii(limpar_leve(montar_texto(t, m)))
        for t, m in zip(rev["review_comment_title"], rev["review_comment_message"])
    ]
    rev = rev[rev["texto"].str.len() > 0]
    funil.append(("com texto", len(rev)))
    rev = selecionar(rev)
    funil.append((f"com >= {config.MIN_TEXT_ALNUM} caracteres alfanuméricos", len(rev)))

    meta = metadados_pedido(d["orders"], d["items"], d["products"], d["traducao"])
    rev = rev.join(meta, on="order_id")
    for c, vazio in [("order_status", DESC), ("entregue_com_atraso", DESC),
                     ("categoria", DESC), ("categoria_en", DESC),
                     ("dias_entrega", -1), ("n_categorias", 0)]:
        rev[c] = rev[c].fillna(vazio)

    data = pd.to_datetime(rev["review_creation_date"], errors="coerce")
    rev["data_avaliacao"] = data.dt.strftime("%Y-%m-%d").fillna(DESC)
    rev["ano_mes"] = data.dt.strftime("%Y-%m").fillna(DESC)
    rev["ano"] = data.dt.year.fillna(-1).astype(int)
    rev["texto_bm25"] = [" ".join(limpar_bm25(t)) for t in rev["texto"]]

    out = rev[COLUNAS].reset_index(drop=True)
    # object puro: garante str/int/bool nativos do Python (sem numpy) na validação.
    tipos_int = ["review_score", "ano", "n_categorias", "dias_entrega", "n_pedidos"]
    for c in out.columns:
        vals = out[c].tolist()
        if c in tipos_int:
            vals = [int(v) for v in vals]
        out[c] = pd.Series(vals, dtype=object)
    _validar(out)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), config.REVIEWS_PARQUET)

    print("Funil de preparação")
    for nome, n in funil:
        print(f"  {nome}: {n:,}".replace(",", "."))
    print("entregue_com_atraso:")
    for k, v in out["entregue_com_atraso"].value_counts().items():
        print(f"  {k}: {v:,} ({100 * v / len(out):.2f}%)".replace(",", "."))
    com_data = out[out["entregue_com_atraso"] != DESC]
    print(f"  atraso entre as com data de entrega: "
          f"{100 * (com_data['entregue_com_atraso'] == 'sim').mean():.2f}%")
    print(f"categoria desconhecida: {(out['categoria'] == DESC).sum()}")
    for tag in ("[TELEFONE]", "[CPF]", "[EMAIL]"):
        print(f"textos com {tag}: {out['texto'].str.contains(tag, regex=False).sum()}")
    print(f"Tempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
