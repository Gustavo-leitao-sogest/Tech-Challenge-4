import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src.rag import retriever as r


class TestRrf(unittest.TestCase):
    def test_caso_manual(self):
        out = dict(r.rrf([["a", "b", "c"], ["b", "a", "d"]], k=60))
        self.assertAlmostEqual(out["a"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(out["b"], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(out["c"], 1 / 63)
        self.assertAlmostEqual(out["d"], 1 / 63)

    def test_ordem_desc_e_vazio(self):
        res = r.rrf([["x", "y"], ["y"]])
        self.assertEqual(res[0][0], "y")
        self.assertEqual(r.rrf([]), [])


class TestFiltros(unittest.TestCase):
    def test_sem_filtros(self):
        self.assertIsNone(r.filtros_para_where({}))
        self.assertIsNone(r.filtros_para_where(None))
        self.assertIsNone(r.filtros_para_where({"nota_min": None, "apenas_atrasados": False}))

    def test_uma_condicao(self):
        self.assertEqual(r.filtros_para_where({"nota_max": 2}),
                         {"review_score": {"$lte": 2}})
        self.assertEqual(r.filtros_para_where({"apenas_atrasados": True}),
                         {"entregue_com_atraso": {"$eq": "sim"}})

    def test_varias_condicoes(self):
        w = r.filtros_para_where({"nota_min": 4, "categoria": "cama_mesa_banho"})
        self.assertEqual(w, {"$and": [{"review_score": {"$gte": 4}},
                                      {"categoria": {"$eq": "cama_mesa_banho"}}]})

    def test_intervalo_meses(self):
        self.assertEqual(r.meses_intervalo("2017-11", "2018-02"),
                         ["2017-11", "2017-12", "2018-01", "2018-02"])
        w = r.filtros_para_where({"ano_mes_ini": "2018-01", "ano_mes_fim": "2018-03"})
        self.assertEqual(w, {"ano_mes": {"$in": ["2018-01", "2018-02", "2018-03"]}})

    def test_intervalo_aberto(self):
        self.assertEqual(r.meses_intervalo("2018-11", None)[-1], "2018-12")

    def test_mascara(self):
        df = pd.DataFrame({
            "review_score": [1, 2, 5, 2],
            "categoria": ["a", "a", "b", "a"],
            "entregue_com_atraso": ["sim", "nao", "sim", "sim"],
            "ano_mes": ["2018-01", "2018-02", "2018-03", "2018-05"],
        })
        self.assertTrue(r.filtros_para_mascara(df, {}).all())
        m = r.filtros_para_mascara(df, {"nota_max": 2, "categoria": "a", "apenas_atrasados": True})
        self.assertEqual(m.tolist(), [True, False, False, True])
        m = r.filtros_para_mascara(df, {"nota_min": 2, "ano_mes_ini": "2018-02", "ano_mes_fim": "2018-03"})
        self.assertEqual(m.tolist(), [False, True, True, False])


class TestEvidenciaEMmr(unittest.TestCase):
    def test_atraso(self):
        for valor, esperado in [("sim", True), ("nao", False), ("desconhecido", None)]:
            e = r.montar_evidencia(1, "id", "texto", {
                "review_score": 3, "data_avaliacao": "2018-01-02", "categoria": "x",
                "entregue_com_atraso": valor}, 0.9)
            self.assertIs(e["atraso"], esperado)
        self.assertEqual(e["nota"], 3)

    def test_score_limitado(self):
        e = r.montar_evidencia(1, "i", "t", {"review_score": 1}, 1.2)
        self.assertEqual(e["score"], 1.0)

    def test_mmr_evita_duplicata(self):
        q = np.array([1.0, 0.0])
        v = np.array([[1.0, 0.0], [1.0, 0.0], [0.7071, 0.7071]])
        self.assertEqual(r.mmr(q, v, 2, 0.4), [0, 2])


class TestRobustez(unittest.TestCase):
    def test_mes_mal_formado_ignorado(self):
        for ruim in ("2018", "2018-13", "18-01", 201801):
            with self.assertLogs(r.log, "WARNING"):
                self.assertIsNone(r.filtros_para_where({"ano_mes_ini": ruim}))
        self.assertFalse(r.intervalo_vazio({"ano_mes_ini": "2018"}))

    def test_intervalo_vazio(self):
        self.assertTrue(r.intervalo_vazio({"ano_mes_ini": "2019-01"}))
        self.assertTrue(r.intervalo_vazio({"ano_mes_ini": "2018-05", "ano_mes_fim": "2018-02"}))
        self.assertFalse(r.intervalo_vazio({"ano_mes_ini": "2018-01"}))
        self.assertFalse(r.intervalo_vazio({}))

    def test_nos_com_intervalo_vazio_nao_consultam_indices(self):
        with mock.patch.object(r, "_colecao", side_effect=AssertionError("não deveria abrir")),              mock.patch.object(r, "_vetor_consulta", side_effect=AssertionError):
            st = {"consulta": "x", "filtros": {"ano_mes_ini": "2019-01"}}
            self.assertEqual(r.recuperar(st), {"evidencias": []})
            self.assertEqual(r.recuperar_panorama(st),
                             {"evidencias": [], "contagens": None, "panorama_amostra": 0})

    def test_modo_desconhecido_usa_padrao(self):
        with self.assertLogs(r.log, "WARNING"):
            self.assertEqual(r._resolver_modo("xyz"), r.config.RETRIEVAL_MODE)
        self.assertEqual(r._resolver_modo("densa"), "densa")
        self.assertEqual(r._resolver_modo(None), r.config.RETRIEVAL_MODE)

    def test_evidencias_get_fora_de_ordem(self):
        meta = lambda n: {"review_score": n, "categoria": "c", "entregue_com_atraso": "nao"}
        col = mock.Mock()
        col.get.return_value = {
            "ids": ["b", "c", "a"], "documents": ["tb", "tc", "ta"],
            "metadatas": [meta(2), meta(3), meta(1)],
            "embeddings": [[0.0, 1.0], [0.6, 0.8], [1.0, 0.0]],
        }
        with mock.patch.object(r, "_colecao", return_value=col):
            ev = r._evidencias(np.array([1.0, 0.0], dtype=np.float32), ["a", "b", "c"])
        self.assertEqual([e["review_id"] for e in ev], ["a", "b", "c"])
        self.assertEqual([e["n"] for e in ev], [1, 2, 3])
        self.assertEqual([e["texto"] for e in ev], ["ta", "tb", "tc"])
        self.assertAlmostEqual(ev[0]["score"], 1.0)
        self.assertAlmostEqual(ev[2]["score"], 0.6, places=5)


if __name__ == "__main__":
    unittest.main()
