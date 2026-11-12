"""Gera data/processed/reviews.parquet a partir dos CSVs do Kaggle.

Etapas:
1. Carregar reviews, orders, order_items, products e a tradução de categorias.
2. Selecionar avaliações com título ou mensagem; deduplicar review_id.
3. Texto = "{titulo}. {mensagem}" -> limpar_leve -> mascarar_pii.
4. Metadados: review_score, ano_mes, categoria, entregue_com_atraso, dias_entrega.
   Ausentes viram "desconhecido"/-1 (o Chroma não aceita None).
5. Gerar a coluna texto_bm25 com limpar_bm25.
6. Imprimir um resumo (quantas linhas entraram e saíram em cada etapa) para o README.

Uso: python -m src.data.prepare
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
