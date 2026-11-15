"""Fábrica de LLM por provedor.

Importa só o pacote do provedor escolhido em LLM_PROVIDER. Modelo sempre vindo do .env.
"""
import sys

from src import config

PROVEDORES = {
    "google": "langchain-google-genai",
    "openai": "langchain-openai",
    "anthropic": "langchain-anthropic",
    "ollama": "langchain-ollama",
}


def get_llm(temperature: float | None = None):
    """Cria o chat model do provedor configurado (import só do pacote escolhido)."""
    provider = (config.LLM_PROVIDER or "").strip().lower()
    model = (config.LLM_MODEL or "").strip()
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
                model=model, temperature=temp, max_retries=config.LLM_MAX_RETRIES,
                timeout=config.LLM_TIMEOUT
            )
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model, temperature=temp, max_retries=config.LLM_MAX_RETRIES,
                timeout=config.LLM_TIMEOUT
            )
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model, temperature=temp, max_retries=config.LLM_MAX_RETRIES,
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


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    resposta = get_llm().invoke("Responda apenas: ok")
    print(f"provedor: {config.LLM_PROVIDER}")
    print(f"modelo: {config.LLM_MODEL}")
    print(f"resposta: {resposta.text}")
