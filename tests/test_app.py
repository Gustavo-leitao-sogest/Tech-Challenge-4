"""Testes da UI Streamlit (sem rede nem modelos): lógica pura e renderização com pipeline simulado."""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.rag.schemas import Evidencia, RagResult

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def _carregar_modulo():
    """Importa o módulo do app pelo caminho; fora de __main__, main() não executa."""
    spec = importlib.util.spec_from_file_location("streamlit_app_teste", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


app = _carregar_modulo()


def _evid(n: int, texto: str = "Chegou atrasado.") -> Evidencia:
    return Evidencia(n=n, review_id=f"r{n}", score=0.8123, nota=1, data="2018-03-02",
                     categoria="cama_mesa_banho", atraso=True, texto=texto)


def _resultado(status: str = "ok") -> RagResult:
    return RagResult(pergunta="p", status=status, resposta="Atrasos frequentes [1].",
                     citacoes=[1] if status == "ok" else [],
                     evidencias=[_evid(1), _evid(2, "Outro texto.")],
                     consulta_transformada="entrega atrasada", filtros={"panorama_amostra": 2},
                     tentativas=1, modelo="gemini-x")


def _rodar(resultado) -> tuple[AppTest, object]:
    """Roda o app com o pipeline simulado, pergunta preenchida e botão clicado."""
    efeito = {"side_effect": resultado} if isinstance(resultado, Exception) else {"return_value": resultado}
    with patch("src.rag.retriever._embeddings"), patch("src.rag.retriever._colecao"), \
            patch("src.rag.retriever._bm25"), patch("src.rag.retriever._meta_df"), \
            patch("src.rag.analyze.listar_categorias", return_value=("cama_mesa_banho",)), \
            patch("src.rag.graph.answer", **efeito) as mock:
        at = AppTest.from_file(str(APP), default_timeout=30).run()
        at.text_input(key="pergunta").set_value("Quais os problemas?")
        at.button[0].click().run()
    return at, mock


class TestLogicaPura(unittest.TestCase):
    def test_filtros_padrao_vazio(self):
        self.assertEqual(app.montar_filtros_ui(), {})

    def test_filtros_alterados(self):
        f = app.montar_filtros_ui((2, 4), "cama_mesa_banho", "2017-01", "2017-06", True)
        self.assertEqual(f, {"nota_min": 2, "nota_max": 4, "categoria": "cama_mesa_banho",
                             "ano_mes_ini": "2017-01", "ano_mes_fim": "2017-06",
                             "apenas_atrasados": True})

    def test_filtros_so_um_lado_da_nota(self):
        self.assertEqual(app.montar_filtros_ui((1, 2)), {"nota_max": 2})
        self.assertEqual(app.montar_filtros_ui((4, 5)), {"nota_min": 4})

    def test_gerar_meses(self):
        meses = app.gerar_meses()
        self.assertEqual((meses[0], meses[-1], len(meses)), ("2016-09", "2018-10", 26))
        self.assertEqual(app.gerar_meses((2016, 11), (2017, 2)),
                         ["2016-11", "2016-12", "2017-01", "2017-02"])

    def test_periodo_invalido(self):
        self.assertTrue(app.periodo_invalido({"ano_mes_ini": "2018-05", "ano_mes_fim": "2017-01"}))
        self.assertFalse(app.periodo_invalido({"ano_mes_ini": "2017-01", "ano_mes_fim": "2017-01"}))
        self.assertFalse(app.periodo_invalido({"ano_mes_ini": "2018-05"}))
        self.assertFalse(app.periodo_invalido({}))

    def test_formatar_filtros(self):
        linhas = app.formatar_filtros({"categoria": "cama_mesa_banho", "apenas_atrasados": True,
                                       "nota_min": 2, "nota_max": None, "x": False})
        self.assertEqual(linhas, ["categoria: cama mesa banho", "só com atraso: sim", "nota mínima: 2"])

    def test_titulo_e_erro(self):
        self.assertEqual(app.titulo_evidencia(_evid(3)), "[3] nota 1/5 · cama mesa banho · 2018-03-02")
        self.assertIn("sobrecarregado", app.mensagem_erro(RuntimeError("503 UNAVAILABLE")))
        self.assertNotIn("sobrecarregado", app.mensagem_erro(ValueError("x")))
        self.assertEqual(app.texto_atraso(None), "desconhecido")


class TestRenderizacao(unittest.TestCase):
    def test_status_ok_com_evidencia_citada(self):
        at, mock = _rodar(_resultado("ok"))
        self.assertFalse(at.exception)
        self.assertEqual([s.value for s in at.success], ["Resposta fundamentada"])
        titulos = [e.label for e in at.expander]
        self.assertIn("[1] nota 1/5 · cama mesa banho · 2018-03-02", titulos)
        self.assertFalse(any(t.startswith("[2]") for t in titulos))  # só as citadas
        self.assertIn("Rastreabilidade", titulos)
        self.assertTrue(any("amostra de 2" in w.value for w in at.warning))
        mock.assert_called_once()
        self.assertIsNone(mock.call_args.args[1])  # nenhum filtro alterado

    def test_mostra_modelo(self):
        at, _ = _rodar(_resultado())
        self.assertFalse(at.exception)
        self.assertIn("Modelo: gemini-x", [c.value for c in at.caption])
        self.assertTrue(any("**Modelo:** gemini-x" in m.value for m in at.markdown))

    def test_insuficiente_mostra_proximas(self):
        at, _ = _rodar(_resultado("insuficiente"))
        self.assertFalse(at.exception)
        self.assertTrue(any("Evidências" in w.value for w in at.warning))
        self.assertTrue(any(s.value == "Avaliações mais próximas encontradas" for s in at.subheader))
        self.assertEqual(sum(e.label.startswith("[") for e in at.expander), 2)

    def test_periodo_invalido_nao_chama_pipeline(self):
        with patch("src.rag.retriever._embeddings"), patch("src.rag.retriever._colecao"),                 patch("src.rag.retriever._bm25"), patch("src.rag.retriever._meta_df"),                 patch("src.rag.analyze.listar_categorias", return_value=("x",)),                 patch("src.rag.graph.answer") as mock:
            at = AppTest.from_file(str(APP), default_timeout=30).run()
            at.selectbox[1].select("2018-05")
            at.selectbox[2].select("2017-01")
            at.text_input(key="pergunta").set_value("Quais os problemas?")
            at.button[0].click().run()
        self.assertFalse(at.exception)
        mock.assert_not_called()
        self.assertTrue(at.sidebar.warning)
        self.assertIn("Período inválido", at.error[0].value)

    def test_erro_do_modelo_vira_mensagem_amigavel(self):
        at, _ = _rodar(RuntimeError("503 UNAVAILABLE"))
        self.assertFalse(at.exception)
        self.assertIn("sobrecarregado", at.error[0].value)


if __name__ == "__main__":
    unittest.main()
