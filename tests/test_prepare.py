"""Testes de src/data/prepare.py (DataFrames pequenos, sem ler os CSVs reais)."""
import unittest

import pandas as pd

from src.data.prepare import (
    categoria_principal, deduplicar, metadados_pedido, montar_texto, selecionar,
)


class MontarTextoTest(unittest.TestCase):
    def test_titulo_vazio(self):
        self.assertEqual(montar_texto(None, "ótimo"), "ótimo")
        self.assertEqual(montar_texto("  ", "ótimo"), "ótimo")

    def test_mensagem_vazia(self):
        self.assertEqual(montar_texto("Bom", float("nan")), "Bom")

    def test_iguais_caixa_diferente(self):
        self.assertEqual(montar_texto("BOM", "bom "), "bom")

    def test_ambos(self):
        self.assertEqual(montar_texto("Bom", "chegou rápido"), "Bom. chegou rápido")

    def test_titulo_com_pontuacao(self):
        self.assertEqual(montar_texto("Ótimo!", "chegou"), "Ótimo! chegou")


class DeduplicarTest(unittest.TestCase):
    def test_copias(self):
        df = pd.DataFrame({
            "review_id": ["r1", "r1", "r2"],
            "order_id": ["o2", "o1", "o3"],
            "review_creation_date": ["2018-01-01"] * 3,
        })
        a = deduplicar(df)
        b = deduplicar(df.iloc[::-1])
        self.assertEqual(len(a), 2)
        r1 = a[a["review_id"] == "r1"].iloc[0]
        self.assertEqual(r1["n_pedidos"], 2)
        self.assertEqual(r1["order_id"], "o1")
        self.assertEqual(a.to_dict("records"), b.to_dict("records"))


class CategoriaPrincipalTest(unittest.TestCase):
    def test_mais_frequente(self):
        self.assertEqual(categoria_principal(pd.Series(["b", "a", "b"])), "b")

    def test_empate_alfabetico(self):
        self.assertEqual(categoria_principal(pd.Series(["b", "a"])), "a")

    def test_conhecida_vence_desconhecido(self):
        s = pd.Series([None, None, "desconhecido", "x"])
        self.assertEqual(categoria_principal(s), "x")

    def test_nenhuma(self):
        self.assertEqual(categoria_principal(pd.Series([None, "desconhecido"])), "desconhecido")


class MetadadosPedidoTest(unittest.TestCase):
    def test_atraso_por_data(self):
        ts = pd.Timestamp
        pedidos = pd.DataFrame({
            "order_id": ["mesmo", "depois", "sem"],
            "order_status": ["delivered", "delivered", "shipped"],
            "order_purchase_timestamp": [ts("2018-01-01 10:00")] * 3,
            "order_delivered_customer_date": [ts("2018-01-10 23:00"), ts("2018-01-11 00:05"), pd.NaT],
            "order_estimated_delivery_date": [ts("2018-01-10 00:00")] * 3,
        })
        itens = pd.DataFrame({"order_id": ["mesmo"], "product_id": ["p1"]})
        produtos = pd.DataFrame({"product_id": ["p1"], "product_category_name": ["cat_a"]})
        trad = pd.DataFrame({"product_category_name": ["cat_a"],
                             "product_category_name_english": ["cat_a_en"]})
        m = metadados_pedido(pedidos, itens, produtos, trad)
        self.assertEqual(m.loc["mesmo", "entregue_com_atraso"], "nao")
        self.assertEqual(m.loc["depois", "entregue_com_atraso"], "sim")
        self.assertEqual(m.loc["sem", "entregue_com_atraso"], "desconhecido")
        self.assertEqual(m.loc["sem", "dias_entrega"], -1)
        self.assertEqual(m.loc["mesmo", "dias_entrega"], 9)
        self.assertEqual(m.loc["mesmo", "categoria_en"], "cat_a_en")
        self.assertEqual(m.loc["sem", "categoria"], "desconhecido")
        self.assertEqual(m.loc["sem", "n_categorias"], 0)


class CategoriaPorPedidoTest(unittest.TestCase):
    def test_frequente_empate_e_conhecida(self):
        ts = pd.Timestamp("2018-01-01")
        pedidos = pd.DataFrame({
            "order_id": ["freq", "empate", "mista"], "order_status": ["delivered"] * 3,
            "order_purchase_timestamp": [ts] * 3,
            "order_delivered_customer_date": [ts] * 3,
            "order_estimated_delivery_date": [ts] * 3,
        })
        itens = pd.DataFrame({
            "order_id": ["freq"] * 3 + ["empate"] * 2 + ["mista"] * 2,
            "product_id": ["pb", "pb", "pa", "pb", "pa", "pn", "pa"],
        })
        produtos = pd.DataFrame({"product_id": ["pa", "pb", "pn"],
                                 "product_category_name": ["a", "b", None]})
        trad = pd.DataFrame({"product_category_name": ["a"], "product_category_name_english": ["A"]})
        m = metadados_pedido(pedidos, itens, produtos, trad)
        self.assertEqual(m.loc["freq", "categoria"], "b")
        self.assertEqual(m.loc["freq", "n_categorias"], 2)
        self.assertEqual(m.loc["empate", "categoria"], "a")
        self.assertEqual(m.loc["mista", "categoria"], "a")
        self.assertEqual(m.loc["mista", "n_categorias"], 1)
        self.assertEqual(m.loc["freq", "categoria_en"], "b")  # sem tradução: repete o PT
        self.assertEqual(m.loc["mista", "categoria_en"], "A")


class SelecionarTest(unittest.TestCase):
    def test_minimo_alfanumerico(self):
        df = pd.DataFrame({"texto": ["...", "ok", "bom", "a b c"]})
        self.assertEqual(selecionar(df)["texto"].tolist(), ["bom", "a b c"])


if __name__ == "__main__":
    unittest.main()
