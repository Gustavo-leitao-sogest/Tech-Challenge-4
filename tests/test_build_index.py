"""Testes de src/index (embeddings e5, metadados, BM25) sem baixar modelo."""
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.index.build_index import construir_bm25, montar_metadados, salvar_bm25
from src.index.embeddings import E5Embeddings


class ModeloFalso:
    def __init__(self):
        self.chamadas = []

    def encode(self, textos, **kw):
        self.chamadas.append((list(textos), kw))
        return np.ones((len(textos), 3), dtype="float32")


class EmbeddingsTest(unittest.TestCase):
    def test_prefixos(self):
        m = ModeloFalso()
        e = E5Embeddings(model=m, batch_size=8)
        self.assertEqual(len(e.embed_documents(["a", "b"])), 2)
        self.assertEqual(m.chamadas[0][0], ["passage: a", "passage: b"])
        self.assertTrue(m.chamadas[0][1]["normalize_embeddings"])
        self.assertEqual(m.chamadas[0][1]["batch_size"], 8)
        self.assertEqual(len(e.embed_query("x")), 3)
        self.assertEqual(m.chamadas[1][0], ["query: x"])


class MetadadosTest(unittest.TestCase):
    def test_sem_none_e_sem_texto(self):
        df = pd.DataFrame({
            "review_id": ["1", "2"], "texto": ["a", "b"], "texto_bm25": ["a", "b"],
            "review_score": [5, 1], "categoria": ["x", None], "peso": [1.0, np.nan],
            "tem_titulo": [True, False],
        })
        metas = montar_metadados(df)
        for m in metas:
            self.assertNotIn("texto", m)
            self.assertNotIn("texto_bm25", m)
            for v in m.values():
                self.assertIsInstance(v, (str, int, float, bool))
        self.assertEqual(metas[1]["categoria"], "desconhecido")
        self.assertEqual(metas[1]["peso"], "desconhecido")
        self.assertIs(type(metas[0]["review_score"]), int)


class BM25Test(unittest.TestCase):
    def test_doc_vazio_e_pickle(self):
        d = construir_bm25(["entrega rapida", "", "produto quebrado"], ["a", "b", "c"])
        self.assertEqual(d["ids"], ["a", "b", "c"])
        scores = d["bm25"].get_scores(["quebrado"])
        self.assertEqual(int(np.argmax(scores)), 2)
        self.assertEqual(scores[1], 0)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sub" / "bm25.pkl"
            salvar_bm25(d, p)
            with open(p, "rb") as f:
                d2 = pickle.load(f)
        self.assertEqual(d2["ids"], ["a", "b", "c"])


class ChromaPathTest(unittest.TestCase):
    def test_ascii_retorna_igual(self):
        with tempfile.TemporaryDirectory() as tmp:
            alvo = Path(tmp) / "c"
            if not str(alvo.resolve()).isascii():
                self.skipTest("temp resolve para caminho não ASCII")
            self.assertEqual(config.chroma_path(alvo), str(alvo.resolve()))

    def test_nao_ascii_vira_ascii(self):
        self.assertTrue(config.chroma_path().isascii())


if __name__ == "__main__":
    unittest.main()
