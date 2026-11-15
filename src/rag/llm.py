"""Fábrica de LLM por provedor.

Importa só o pacote do provedor escolhido em LLM_PROVIDER. Modelo sempre vindo do .env.
"""
import logging
import re
import sys
from collections.abc import Callable

import httpx
from pydantic import BaseModel

from src import config

log = logging.getLogger(__name__)

# Erros transitórios/de cota que justificam trocar de modelo: 429/RESOURCE_EXHAUSTED (cota),
# 402 (crédito), 500/INTERNAL, 503/UNAVAILABLE e 504/DEADLINE_EXCEEDED (demanda), timeout e
# falha de conexão. O código HTTP e o tipo da exceção valem primeiro; o regex é só reserva.
_CODIGOS_TRANSITORIOS = {402, 429, 500, 503, 504}
_MARCADORES_TRANSITORIOS = re.compile(
    r"\b(429|402|500|503|504)\b|resource_exhausted|unavailable|deadline_exceeded|internal"
    r"|overloaded"
    r"|timed? ?out|timeout|connection|connect error", re.IGNORECASE)
_TIPOS_TRANSITORIOS = (TimeoutError, ConnectionError, httpx.ConnectError,
                       httpx.TimeoutException)
try:  # presente nas versões recentes do langchain-core
    from langchain_core.exceptions import ModelRateLimitError
    _TIPOS_TRANSITORIOS += (ModelRateLimitError,)
except ImportError:  # pragma: no cover
    pass

PROVEDORES = {
    "google": "langchain-google-genai",
    "openai": "langchain-openai",
    "anthropic": "langchain-anthropic",
    "ollama": "langchain-ollama",
}


def get_llm(temperature: float | None = None, modelo: str | None = None,
            max_retries: int | None = None):
    """Cria o chat model do provedor configurado (import só do pacote escolhido).

    `modelo` sobrescreve LLM_MODEL; `max_retries` sobrescreve as tentativas do cliente.
    """
    provider = (config.LLM_PROVIDER or "").strip().lower()
    model = ((config.LLM_MODEL if modelo is None else modelo) or "").strip()
    retries = config.LLM_MAX_RETRIES if max_retries is None else max_retries
    temp = config.LLM_TEMPERATURE if temperature is None else temperature

    if provider not in PROVEDORES:
        raise ValueError(
            f"LLM_PROVIDER invalido: {provider!r}. Use um de: {', '.join(PROVEDORES)}."
        )
    if not model:
        raise ValueError("LLM_MODEL vazio. Defina o modelo no .env.")

    try:
        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model, temperature=temp, max_retries=retries,
                timeout=config.LLM_TIMEOUT
            )
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model, temperature=temp, max_retries=retries,
                timeout=config.LLM_TIMEOUT
            )
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model, temperature=temp, max_retries=retries,
                timeout=config.LLM_TIMEOUT
            )
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model, temperature=temp, client_kwargs={"timeout": config.LLM_TIMEOUT}
        )
    except ImportError as e:
        raise ImportError(
            f"Pacote do provedor {provider!r} nao instalado. "
            f"Instale com: pip install {PROVEDORES[provider]}"
        ) from e


def erro_transitorio(exc: Exception) -> bool:
    """True para erros de cota/demanda/rede em que vale tentar outro modelo.

    Saída inválida (ValueError, inclui ValidationError) nunca conta: trocar de modelo não a
    corrige e o chamador já tem tratamento próprio. Detecção, em ordem: código HTTP do erro
    (`code`/`status_code`), tipo da exceção (timeout/conexão/limite de taxa) e, como reserva,
    o texto do erro. 400/401/403/404 não trocam.
    """
    if isinstance(exc, ValueError):
        return False
    for attr in ("code", "status_code"):
        codigo = getattr(exc, attr, None)
        if isinstance(codigo, int) and codigo in _CODIGOS_TRANSITORIOS:
            return True
    if isinstance(exc, _TIPOS_TRANSITORIOS):
        return True
    nomes = " ".join(c.__name__ for c in type(exc).__mro__)
    return bool(_MARCADORES_TRANSITORIOS.search(f"{nomes} {exc}"))


def modelos_configurados(usar_fallback: bool = True) -> list[str]:
    """Ordem de tentativa: [LLM_MODEL, *LLM_FALLBACK_MODELS], sem repetições."""
    lista = [(config.LLM_MODEL or "").strip()]
    if usar_fallback:
        lista += [m.strip() for m in config.LLM_FALLBACK_MODELS]
    return list(dict.fromkeys(m for m in lista if m)) or [""]


def _com_fallback(chamar: Callable, usar_fallback: bool, temperature: float | None):
    """Executa `chamar(llm)` por modelo, em ordem; devolve (resultado, modelo que respondeu).

    Mesmo padrão de fallback do LCEL (`with_fallbacks`), mas com laço próprio para informar
    qual modelo respondeu. Só erros transitórios trocam de modelo; se todos falharem,
    relança o último erro. Fora do último modelo o cliente não insiste (retries reduzidos).
    """
    modelos = modelos_configurados(usar_fallback)
    for i, modelo in enumerate(modelos):
        ultimo = i == len(modelos) - 1
        retries = None if ultimo else config.LLM_MAX_RETRIES_COM_FALLBACK
        llm = get_llm(temperature, modelo=modelo or None, max_retries=retries)
        try:
            return chamar(llm), modelo
        except Exception as e:  # noqa: BLE001 - classificado por erro_transitorio
            if ultimo or not erro_transitorio(e):
                raise
            log.warning("Modelo %s falhou (%s: %.120s); tentando %s.",
                        modelo, type(e).__name__, e, modelos[i + 1])
    raise RuntimeError("nenhum modelo configurado")  # inalcançável


def invocar_estruturado(schema: type[BaseModel], prompt: str, usar_fallback: bool = True,
                        temperature: float | None = None) -> tuple[BaseModel | None, str]:
    """Saída estruturada com fallback entre modelos; (objeto ou None, modelo usado).

    None/ValueError (saída vazia ou inválida) não troca de modelo: devolve (None, modelo).
    """
    def chamar(llm):
        estruturado = llm.with_structured_output(schema)
        try:
            return estruturado.invoke(prompt)
        except ValueError as e:  # inclui ValidationError e OutputParserException
            log.warning("Saída estruturada inválida: %s", e)
            return None

    return _com_fallback(chamar, usar_fallback, temperature)


def invocar_texto(prompt: str, usar_fallback: bool = True,
                  temperature: float | None = None) -> tuple[str, str]:
    """Texto livre com fallback entre modelos; (texto, modelo usado)."""
    return _com_fallback(lambda llm: llm.invoke(prompt).text or "", usar_fallback, temperature)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    texto, modelo_usado = invocar_texto("Responda apenas: ok")
    print(f"provedor: {config.LLM_PROVIDER}")
    print(f"modelos: {modelos_configurados()}")
    print(f"modelo: {modelo_usado}")
    print(f"resposta: {texto}")
