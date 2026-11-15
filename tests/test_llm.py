"""Testes de src/rag/llm.py (sem rede: o pacote do provedor é simulado)."""
import sys
import unittest
from unittest import mock

import httpx
from langchain_core.exceptions import ModelRateLimitError
from pydantic import BaseModel

from src.rag import llm


def _chamar(provider, model="m-teste", temperature=None):
    with mock.patch.object(llm.config, "LLM_PROVIDER", provider), \
         mock.patch.object(llm.config, "LLM_MODEL", model):
        return llm.get_llm(temperature)


class GetLlmTest(unittest.TestCase):
    def _testa_provedor(self, provider, modulo, classe):
        fake_mod = mock.MagicMock()
        with mock.patch.dict(sys.modules, {modulo: fake_mod}):
            res = _chamar(provider, temperature=0.3)
        fabrica = getattr(fake_mod, classe)
        fabrica.assert_called_once()
        kwargs = fabrica.call_args.kwargs
        self.assertEqual(kwargs["model"], "m-teste")
        self.assertEqual(kwargs["temperature"], 0.3)
        self.assertIs(res, fabrica.return_value)

    def test_google(self):
        self._testa_provedor("google", "langchain_google_genai", "ChatGoogleGenerativeAI")

    def test_openai(self):
        self._testa_provedor("openai", "langchain_openai", "ChatOpenAI")

    def test_anthropic(self):
        self._testa_provedor("anthropic", "langchain_anthropic", "ChatAnthropic")

    def test_ollama(self):
        self._testa_provedor("ollama", "langchain_ollama", "ChatOllama")

    def test_temperatura_padrao_da_config(self):
        fake_mod = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"langchain_openai": fake_mod}):
            _chamar("openai")
        self.assertEqual(
            fake_mod.ChatOpenAI.call_args.kwargs["temperature"], llm.config.LLM_TEMPERATURE
        )

    def test_provedor_invalido(self):
        with self.assertRaises(ValueError):
            _chamar("xpto")

    def test_modelo_vazio(self):
        with self.assertRaises(ValueError):
            _chamar("google", model="")

    def test_pacote_nao_instalado(self):
        with mock.patch.dict(sys.modules, {"langchain_ollama": None}):
            with self.assertRaises(ImportError) as ctx:
                _chamar("ollama")
        self.assertIn("langchain-ollama", str(ctx.exception))


class FakeLLM:
    """Chat model falso: `acao` é resposta (devolvida) ou exceção (lançada)."""

    def __init__(self, acao):
        self.acao = acao

    def _res(self):
        if isinstance(self.acao, Exception):
            raise self.acao
        return self.acao

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _prompt):
        return self._res()


class Texto:
    def __init__(self, text):
        self.text = text


class Esquema(BaseModel):
    x: int


def _fabrica(acoes: dict):
    """get_llm falso: um FakeLLM por modelo; registra (modelo, max_retries)."""
    chamadas = []

    def fake(temperature=None, modelo=None, max_retries=None):
        chamadas.append((modelo, max_retries))
        return FakeLLM(acoes[modelo])

    return fake, chamadas


class FallbackTest(unittest.TestCase):
    def _ctx(self, acoes, fallbacks=("b", "c"), principal="a"):
        fake, chamadas = _fabrica(acoes)
        self.chamadas = chamadas
        return (mock.patch.object(llm, "get_llm", fake),
                mock.patch.object(llm.config, "LLM_MODEL", principal),
                mock.patch.object(llm.config, "LLM_FALLBACK_MODELS", list(fallbacks)))

    def _rodar(self, fn, acoes, **kw):
        p1, p2, p3 = self._ctx(acoes, **kw)
        with p1, p2, p3:
            return fn()

    def test_modelo_principal_responde(self):
        r = self._rodar(lambda: llm.invocar_texto("p"), {"a": Texto("ok")}, fallbacks=())
        self.assertEqual(r, ("ok", "a"))
        self.assertEqual(self.chamadas, [("a", None)])  # último/único modelo mantém os retries

    def test_troca_no_429_e_devolve_modelo(self):
        acoes = {"a": RuntimeError("429 RESOURCE_EXHAUSTED quota"), "b": Texto("ok"), "c": None}
        with self.assertLogs(llm.log, "WARNING"):
            r = self._rodar(lambda: llm.invocar_texto("p"), acoes)
        self.assertEqual(r, ("ok", "b"))
        self.assertEqual(self.chamadas[0], ("a", llm.config.LLM_MAX_RETRIES_COM_FALLBACK))

    def test_troca_no_503_estruturado(self):
        acoes = {"a": RuntimeError("503 UNAVAILABLE"), "b": TimeoutError("lento"),
                 "c": Esquema(x=1)}
        r = self._rodar(lambda: llm.invocar_estruturado(Esquema, "p"), acoes)
        self.assertEqual(r, (Esquema(x=1), "c"))

    def test_none_nao_troca(self):
        r = self._rodar(lambda: llm.invocar_estruturado(Esquema, "p"), {"a": None})
        self.assertEqual(r, (None, "a"))
        self.assertEqual(len(self.chamadas), 1)

    def test_validacao_nao_troca(self):
        try:
            Esquema(x="abc")
        except ValueError as e:
            erro = e
        r = self._rodar(lambda: llm.invocar_estruturado(Esquema, "p"), {"a": erro})
        self.assertEqual(r, (None, "a"))
        self.assertEqual(len(self.chamadas), 1)

    def test_erro_nao_transitorio_propaga(self):
        with self.assertRaises(KeyError):
            self._rodar(lambda: llm.invocar_texto("p"), {"a": KeyError("x")})
        self.assertEqual(len(self.chamadas), 1)

    def test_todos_falham_relanca_ultimo(self):
        acoes = {"a": RuntimeError("429"), "b": RuntimeError("503"), "c": RuntimeError("504 timeout")}
        with self.assertRaises(RuntimeError) as ctx, self.assertLogs(llm.log, "WARNING"):
            self._rodar(lambda: llm.invocar_texto("p"), acoes)
        self.assertIn("504", str(ctx.exception))
        self.assertEqual(len(self.chamadas), 3)

    def test_usar_fallback_false(self):
        with self.assertRaises(RuntimeError):
            self._rodar(lambda: llm.invocar_texto("p", usar_fallback=False),
                        {"a": RuntimeError("429")})
        self.assertEqual(self.chamadas, [("a", None)])

    def test_classificacao_de_erros(self):
        for e in (RuntimeError("429"), RuntimeError("RESOURCE_EXHAUSTED"), RuntimeError("402"),
                  RuntimeError("504 DEADLINE_EXCEEDED"), RuntimeError("503 UNAVAILABLE"),
                  TimeoutError(), ConnectionError(), RuntimeError("Read timed out")):
            self.assertTrue(llm.erro_transitorio(e), repr(e))
        class ErroHttp(Exception):
            def __init__(self, code):
                self.code = code

        for e in (ErroHttp(500), ErroHttp(429), RuntimeError("500 INTERNAL"),
                  httpx.ConnectError("[Errno 11001] getaddrinfo failed"),
                  httpx.ReadTimeout("lento"), ModelRateLimitError("limite")):
            self.assertTrue(llm.erro_transitorio(e), repr(e))
        for e in (ErroHttp(400), ErroHttp(401), ErroHttp(403), ErroHttp(404),
                  RuntimeError("404 NOT_FOUND")):
            self.assertFalse(llm.erro_transitorio(e), repr(e))
        for e in (ValueError("429"), KeyError("x"), RuntimeError("400 INVALID_ARGUMENT"),
                  RuntimeError("chave 14290 invalida")):
            self.assertFalse(llm.erro_transitorio(e), repr(e))

    def test_modelos_sem_repeticao(self):
        with mock.patch.object(llm.config, "LLM_MODEL", "a"), \
             mock.patch.object(llm.config, "LLM_FALLBACK_MODELS", ["a", "b"]):
            self.assertEqual(llm.modelos_configurados(), ["a", "b"])
            self.assertEqual(llm.modelos_configurados(False), ["a"])

    def test_get_llm_modelo_e_retries(self):
        fake_mod = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"langchain_openai": fake_mod}), \
             mock.patch.object(llm.config, "LLM_PROVIDER", "openai"), \
             mock.patch.object(llm.config, "LLM_MODEL", "base"):
            llm.get_llm(modelo="outro", max_retries=0)
        kw = fake_mod.ChatOpenAI.call_args.kwargs
        self.assertEqual((kw["model"], kw["max_retries"]), ("outro", 0))


if __name__ == "__main__":
    unittest.main()
