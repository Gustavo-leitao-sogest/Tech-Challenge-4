"""Testes de src/rag/llm.py (sem rede: o pacote do provedor é simulado)."""
import sys
import unittest
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
