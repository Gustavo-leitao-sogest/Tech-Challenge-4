"""Embeddings e5 com prefixos de consulta/passagem, compatíveis com langchain-core."""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from typing import TYPE_CHECKING

from src import config

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

PREFIXO_PASSAGEM = "passage: "
PREFIXO_CONSULTA = "query: "


class E5Embeddings(Embeddings):
    """Modelo e5 exige "passage: " nos documentos e "query: " nas consultas."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL,
                 batch_size: int = config.EMBEDDING_BATCH_SIZE,
                 show_progress: bool = False, model: SentenceTransformer | None = None) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress = show_progress
        self._model = model  # injetável em testes

    @property
    def model(self):
        """Carrega o modelo sob demanda (evita custo ao importar)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, textos: list[str]) -> list[list[float]]:
        vetores = self.model.encode(
            textos, batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=self.show_progress,
        )
        return [list(map(float, v)) for v in vetores]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vetores normalizados das passagens, com o prefixo "passage: "."""
        return self._encode([PREFIXO_PASSAGEM + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        """Vetor normalizado da consulta, com o prefixo "query: "."""
        return self._encode([PREFIXO_CONSULTA + text])[0]
