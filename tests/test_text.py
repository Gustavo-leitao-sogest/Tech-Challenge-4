"""Testes de src/data/text.py."""
import unicodedata
import unittest
from unittest import mock

from src import config
from src.data.text import limpar_bm25, limpar_leve


class LimparLeveTest(unittest.TestCase):
    def test_nfc(self):
        decomposto = unicodedata.normalize("NFD", "avaliação péssima")
        self.assertEqual(limpar_leve(decomposto), "avaliação péssima")

    def test_quebras_e_espacos(self):
        self.assertEqual(limpar_leve("  a\r\nb\n\tc   d  "), "a b c d")

    def test_html_e_url(self):
        r = limpar_leve("ótimo<br>produto<br/> veja http://x.com/a?b=1 e www.y.com ok")
        self.assertEqual(r, "ótimo produto veja e ok")

    def test_menor_que_nao_e_html(self):
        self.assertIn("<3", limpar_leve("adorei <3"))

    def test_mantem_acentos_caixa_pontuacao_negacao(self):
        s = "Não gostei, NUNCA mais! Está sem cor."
        self.assertEqual(limpar_leve(s), s)

    def test_email_preservado(self):
        self.assertIn("a@b.com", limpar_leve("fale com a@b.com"))

    def test_idempotente_entidades(self):
        for s in ["a&#771;o", "&amp;lt;br&amp;gt; ok", "<<br>br> ok", "htt<b>ps://x.com fim"]:
            with self.subTest(s=s):
                a = limpar_leve(s)
                self.assertEqual(limpar_leve(a), a)
        self.assertEqual(limpar_leve("a&#771;o"), "ão")

    def test_url_nao_engole_pontuacao(self):
        self.assertEqual(limpar_leve("veja (http://x.com). fim"), "veja ( ). fim")

    def test_idempotente(self):
        s = "Olá<br>\r\nmundo  http://a.b  Não!"
        self.assertEqual(limpar_leve(limpar_leve(s)), limpar_leve(s))

    def test_vazios(self):
        self.assertEqual(limpar_leve(None), "")
        self.assertEqual(limpar_leve(float("nan")), "")
        self.assertEqual(limpar_leve(""), "")


class LimparBm25Test(unittest.TestCase):
    def test_negacao_e_acento(self):
        t = limpar_bm25("Não gostei, vc é péssimo!")
        self.assertEqual(t, ["nao", "gostei", "pessimo"])

    def test_abreviacao_p_barra(self):
        # "para" é stopword e some; sem o tratamento sobraria "entrega" igual, então
        # confere também que "p/" colado não deixa resíduo e que "s/" vira "sem".
        self.assertEqual(limpar_bm25("p/ entrega"), ["entrega"])
        self.assertEqual(limpar_bm25("p/entrega"), ["entrega"])
        self.assertEqual(limpar_bm25("s/ defeito"), ["sem", "defeito"])
        self.assertNotIn("p", limpar_bm25("p/ mim"))
        self.assertNotIn("s", limpar_bm25("s/ defeito"))

    def test_indicador_numerico_nao_e_negacao(self):
        for s in ["n° 123 chegou", "nº 123", "N° do pedido", "n. 5"]:
            with self.subTest(s=s):
                self.assertNotIn("nao", limpar_bm25(s))
        self.assertIn("nao", limpar_bm25("n gostei"))

    def test_abreviacoes(self):
        self.assertEqual(limpar_bm25("n gostei pq chegou tbm mt atrasado"),
                         ["nao", "gostei", "chegou", "atrasado"])  # "tambem" e "muito" sao stopwords
        t = limpar_bm25("n gostei")
        self.assertEqual(t, ["nao", "gostei"])
        self.assertIn("beleza", limpar_bm25("blz"))

    def test_so_palavra_inteira(self):
        # "tb" dentro de "atb" não pode ser expandido
        self.assertIn("atb", limpar_bm25("atb"))

    def test_sem_pontuacao_sem_acento_minusculo(self):
        self.assertEqual(limpar_bm25("Entrega... RÁPIDA!!!"), ["entrega", "rapida"])

    def test_placeholders_pii(self):
        t = limpar_bm25("ligue [TELEFONE] ou [EMAIL] [CPF]")
        self.assertEqual(set(["telefone", "email", "cpf"]) - set(t), set())

    def test_token_de_um_char_descartado_mas_digito_fica(self):
        t = limpar_bm25("x 5 produto")
        self.assertNotIn("x", t)
        self.assertIn("5", t)

    def test_stemming(self):
        with mock.patch.object(config, "BM25_USE_STEMMING", False):
            off = limpar_bm25("atrasou atrasado")
        with mock.patch.object(config, "BM25_USE_STEMMING", True):
            on = limpar_bm25("atrasou atrasado")
        self.assertNotEqual(off[0], off[1])
        self.assertEqual(on[0], on[1])

    def test_vazios(self):
        self.assertEqual(limpar_bm25(None), [])
        self.assertEqual(limpar_bm25(""), [])


if __name__ == "__main__":
    unittest.main()
