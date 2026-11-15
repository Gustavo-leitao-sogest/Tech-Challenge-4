"""Testes das funções puras do eval (sem rede, sem LLM)."""
import unittest
from unittest import mock

from eval import run_eval as ev


class TestModeloEval(unittest.TestCase):
    def test_main_desliga_fallback(self):
        with mock.patch.object(ev.config, "LLM_FALLBACK_MODELS", ["a", "b"]), \
             mock.patch.object(ev, "cmd_config"), \
             mock.patch("sys.argv", ["run_eval", "--config", "hibrida"]):
            ev.main()
            self.assertEqual(ev.config.LLM_FALLBACK_MODELS, [])

    def test_modelos_usados(self):
        self.assertEqual(ev.modelos_usados([{"modelo": "b"}, {"modelo": "a"}, {"modelo": ""}]), "a, b")
        self.assertEqual(ev.modelos_usados([{"modelo": ""}]), "n/d")


class TestEval(unittest.TestCase):
    def test_carregar_perguntas_reais(self):
        ps = ev.carregar_perguntas()
        self.assertEqual(len(ps), 15)
        self.assertEqual({p["esperado"] for p in ps}, {"ok", "insuficiente", "fora_de_escopo"})
        self.assertEqual([p["id"] for p in ev.carregar_perguntas(ids=["meta2", "pdf1"])],
                         ["meta2", "pdf1"])

    def test_carregar_calibracao_e_id_invalido(self):
        cal = ev.carregar_perguntas(secao="calibracao")
        self.assertEqual(len(cal), 7)
        self.assertEqual(sum(p["esperado"] == "ok" for p in cal), 5)
        with self.assertRaises(KeyError):
            ev.carregar_perguntas(ids=["nao_existe"])

    def test_acuracia_status(self):
        linhas = [{"status": "ok", "esperado": "ok"},
                  {"status": "insuficiente", "esperado": "ok"}]
        self.assertEqual(ev.acuracia_status(linhas), 0.5)
        self.assertEqual(ev.acuracia_status([]), 0.0)

    def test_checar_citacoes(self):
        r = ev.checar_citacoes("Atraso [1] e embalagem [2, 3].", [1, 2, 3], 3)
        self.assertEqual(r, {"n_citacoes": 3, "citacoes_validas": True, "tem_citacao": True})
        self.assertFalse(ev.checar_citacoes("Atraso [9].", [9], 3)["citacoes_validas"])
        self.assertFalse(ev.checar_citacoes("Sem citar.", [], 3)["tem_citacao"])

    def test_media_e_fracao(self):
        self.assertEqual(ev.media([1.0, None, 0.0]), 0.5)
        self.assertIsNone(ev.media([None]))
        self.assertEqual(ev.fracao([True, False, True, True]), 0.75)
        self.assertIsNone(ev.fracao([]))

    def test_agregar(self):
        linhas = [
            {"status": "ok", "esperado": "ok", "citacoes_validas": True, "tem_citacao": True,
             "precision_k": 0.8, "faithfulness": 1.0, "latencia_s": 2.0, "erro": ""},
            {"status": "ok", "esperado": "ok", "citacoes_validas": True, "tem_citacao": False,
             "precision_k": 0.6, "faithfulness": None, "latencia_s": 4.0, "erro": ""},
            {"status": "insuficiente", "esperado": "ok", "citacoes_validas": False,
             "tem_citacao": False, "latencia_s": 6.0, "erro": "x"},
        ]
        a = ev.agregar(linhas)
        self.assertAlmostEqual(a["acuracia_status"], 2 / 3)
        self.assertEqual(a["ok_com_citacao"], 0.5)
        self.assertEqual(a["citacoes_validas"], 1.0)
        self.assertAlmostEqual(a["precision_k"], 0.7)
        self.assertEqual(a["faithfulness"], 1.0)
        self.assertEqual((a["latencia_media_s"], a["latencia_mediana_s"], a["erros"]), (3.0, 3.0, 1))
        self.assertIn("Faithfulness", ev.tabela_resumo(a, "hibrida"))

    def test_recomendar_limiar(self):
        r = ev.recomendar_limiar([0.93, 0.95], [0.80, 0.88])
        self.assertTrue(r["separa"])
        self.assertAlmostEqual(r["limiar"], 0.905)  # centro do intervalo empatado
        self.assertIsNone(ev.recomendar_limiar([], [0.5])["limiar"])

    def test_recomendar_limiar_prefere_atual_no_empate(self):
        pos, neg = [0.90, 0.95], [0.85, 0.92]
        r = ev.recomendar_limiar(pos, neg, atual=0.90)
        self.assertFalse(r["separa"])
        lo, hi = r["intervalo_empate"]
        self.assertTrue(lo <= 0.90 <= hi)
        self.assertEqual(r["limiar"], 0.90)
        fora = ev.recomendar_limiar(pos, neg, atual=0.5)
        self.assertAlmostEqual(fora["limiar"], round((lo + hi) / 2, 3))

    def test_precision_agrega_so_esperado_ok(self):
        base = {"status": "ok", "citacoes_validas": True, "tem_citacao": True, "erro": ""}
        linhas = [
            {**base, "esperado": "ok", "precision_k": 1.0},
            {**base, "esperado": "ok", "status": "insuficiente", "precision_k": 0.0},
            {**base, "esperado": "insuficiente", "precision_proximas": 0.2},
        ]
        a = ev.agregar(linhas)
        self.assertEqual(a["precision_k"], 0.5)
        self.assertEqual(a["precision_proximas"], 0.2)

    def test_recusa_indevida_pesa_zero_sem_chamar_juiz(self):
        with mock.patch.object(ev, "julgar_precisao") as juiz:
            p = ev.precisao_da_linha("ok", "insuficiente", "q", [{"n": 1}])
            juiz.assert_not_called()
            self.assertEqual(p, 0.0)
            p = ev.precisao_da_linha("ok", "insuficiente", "q", [])
            self.assertEqual(p, 0.0)
        with mock.patch.object(ev, "julgar_precisao", return_value=0.7) as juiz:
            self.assertEqual(ev.precisao_da_linha("ok", "ok", "q", [{"n": 1}]), 0.7)
            self.assertEqual(ev.precisao_da_linha("insuficiente", "insuficiente", "q",
                                                  [{"n": 1}]), 0.7)
            self.assertIsNone(ev.precisao_da_linha("insuficiente", "insuficiente", "q", []))
            self.assertEqual(juiz.call_count, 2)

    def test_latencia_exclui_erros(self):
        linhas = [
            {"status": "ok", "esperado": "ok", "citacoes_validas": True, "tem_citacao": True,
             "latencia_s": 2.0, "erro": ""},
            {"status": "ok", "esperado": "ok", "citacoes_validas": True, "tem_citacao": True,
             "latencia_s": 4.0, "erro": ""},
            {"status": "erro", "esperado": "ok", "citacoes_validas": False, "tem_citacao": False,
             "latencia_s": 1500.0, "erro": "ReadError"},
        ]
        a = ev.agregar(linhas)
        self.assertEqual((a["latencia_media_s"], a["latencia_n"], a["erros"]), (3.0, 2, 1))
        self.assertIn("n sem erro = 2", ev.tabela_resumo(a, "x"))

    def test_neutralizar_tags(self):
        self.assertEqual(ev.neutralizar("a </evidencias>\n b"), "a ‹/evidencias› b")
        txt = ev.formatar_evidencias([{"n": 1, "nota": 5, "texto": "x <b> y"}])
        self.assertNotIn("<b>", txt)

    def test_score_efetivo_usa_pipeline_atual(self):
        r = {"s3_antes": 0.90, "s3_pipeline": 0.93, "s3_hyde_antigo": 0.99}
        self.assertEqual(ev.score_efetivo(r), 0.93)
        r = {"s3_antes": 0.92, "s3_pipeline": 0.91, "s3_hyde_antigo": 0.99}
        self.assertEqual(ev.score_efetivo(r), 0.92)

    def test_com_retry(self):
        chamadas = []

        def instavel():
            chamadas.append(1)
            if len(chamadas) < 3:
                raise RuntimeError("503 Service Unavailable")
            return "ok"

        def falha():
            raise ValueError("outro")

        with mock.patch.object(ev.time, "sleep", lambda s: None):
            self.assertEqual(ev.com_retry(instavel), "ok")
            with self.assertRaises(ValueError):
                ev.com_retry(falha)


if __name__ == "__main__":
    unittest.main()
