"""Constrói os índices: embeddings e5 (prefixo "passage: ") -> Chroma, e BM25 -> pickle.

Inserir no Chroma em lotes de config.CHROMA_INSERT_BATCH. Salvar também os embeddings
em data/index/embeddings.npy para reaproveitar no BERTopic.

Uso: python -m src.index.build_index
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
